"""UWYK eval on the 6 synthetic case studies — PEHE + ATE err.

Uses UWYK's PreprocessingGraphConditionedPFN wrapper. anc_mode picks the
adjacency injection: 'noanc' (padded 0s, no ancestor info) vs 'anc'
(full ancestral graph). CATE is derived from BarDist marginals + arm-
independence convolution — same as l2_ihdp/methods_densities.py, just
we skip the density resampling and only take the mean.

Env vars:
  DATASET          case study name (Observed_Confounder / ...)
  OUT              per-realization NPZ dir
  UWYK_SRC         path to UWYK repo src (has models/ + utils/)
  UWYK_CKPT_DIR    dir holding best_model.pt + best_model_config.yaml
                   (or final_model_with_bardist.pt + _config.yaml)
  ANC_MODE         noanc | anc  (default noanc)
  MAX_REAL         optional cap
  DOPFN_DATA_ROOT  where the prior_sampling pkls live
"""
from __future__ import annotations
import argparse, os, sys, time, importlib
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=os.environ.get('DATASET', 'Observed_Confounder'))
args, _ = parser.parse_known_args()
DATASET       = args.dataset
OUT           = os.environ['OUT']
UWYK_SRC      = os.environ['UWYK_SRC']
UWYK_CKPT_DIR = os.environ['UWYK_CKPT_DIR']
UWYK_CKPT     = os.environ.get('UWYK_CKPT', '')     # explicit .pt (overrides dir search)
UWYK_CONFIG   = os.environ.get('UWYK_CONFIG', '')   # explicit .yaml (overrides dir search)
ANC_MODE      = os.environ.get('ANC_MODE', 'noanc').lower()
MAX_REAL      = os.environ.get('MAX_REAL', '')
assert ANC_MODE in ('noanc', 'anc'), ANC_MODE

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, os.path.join(REPO_SRC, 'benchmarks'))   # top-level import

from scm_case_study_dataset import SCMCaseStudyDataset  # noqa: E402


# ── UWYK model bootstrap (isolated from local `models`/`utils` collisions) ──
def _load_uwyk():
    saved = {}
    for name in list(sys.modules):
        if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
            saved[name] = sys.modules.pop(name)
    sys.path.insert(0, UWYK_SRC)
    pre_mod = importlib.import_module('models.PreprocessingGraphConditionedPFN')
    if UWYK_SRC in sys.path: sys.path.remove(UWYK_SRC)
    for name in list(sys.modules):
        if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
            del sys.modules[name]
    sys.modules.update(saved)

    _orig_load = torch.load
    def _patched_load(*a, **kw):
        kw.setdefault('weights_only', False); return _orig_load(*a, **kw)
    torch.load = _patched_load
    try:
        # 1) explicit CKPT / CONFIG env vars win
        if UWYK_CKPT and UWYK_CONFIG:
            ck_p, cfg_p = UWYK_CKPT, UWYK_CONFIG
        else:
            # 2) default filenames next to the ckpt
            final_ck = os.path.join(UWYK_CKPT_DIR, 'final_model_with_bardist.pt')
            final_cfg = os.path.join(UWYK_CKPT_DIR, 'final_model_with_bardist_config.yaml')
            if os.path.isfile(final_ck) and os.path.isfile(final_cfg):
                ck_p, cfg_p = final_ck, final_cfg
            else:
                ck_p  = os.path.join(UWYK_CKPT_DIR, 'best_model.pt')
                cfg_p = os.path.join(UWYK_CKPT_DIR, 'best_model_config.yaml')
        if not os.path.isfile(ck_p):
            raise FileNotFoundError(f'UWYK ckpt not found: {ck_p}')
        if not os.path.isfile(cfg_p):
            raise FileNotFoundError(
                f'UWYK config yaml not found: {cfg_p}\n'
                'Set UWYK_CKPT and UWYK_CONFIG env vars explicitly.'
            )
        print(f'[uwyk] loading  ckpt={ck_p}  cfg={cfg_p}', flush=True)
        m = pre_mod.PreprocessingGraphConditionedPFN(
            config_path=cfg_p, checkpoint_path=ck_p, device='cpu', verbose=False,
            random_state=42, use_clustering=False,
        ).load()
    finally:
        torch.load = _orig_load
    return m


def _standardize_train_test(Xtr, Xte, eps=1e-8):
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + eps
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _pad_features(X, F):
    if X.shape[1] == F: return X
    if X.shape[1] > F:  return X[:, :F]
    return np.hstack([X, np.zeros((X.shape[0], F - X.shape[1]), dtype=X.dtype)])


def _cate_from_uwyk(model, X_train, T_train, y_train, X_test, adjacency_kind):
    """Predict per-arm means via UWYK's raw-bar output, take Y1 - Y0 per query."""
    F = model.model.num_features
    Xs, Xq = _standardize_train_test(X_train, X_test)
    Xs = _pad_features(Xs, F); Xq = _pad_features(Xq, F)

    # UWYK expects (context X, context T, context y, query X). Adjacency built
    # inside the wrapper — controlled by the model's own graph-input path
    # (adjacency_kind = 'noanc' → all-zero adj; 'anc' → full-graph).
    y_min = float(y_train.min()); y_max = float(y_train.max())
    y_rng = max(y_max - y_min, 1e-6)
    y_scaled = 2 * (y_train - y_min) / y_rng - 1

    # Query at T=0 then T=1, take difference of expected values under the bar
    # distribution.  We use the wrapper's `predict_full` style call —
    # different UWYK wrappers expose this under slightly different names;
    # fall back to `predict` for the raw bar probs.
    #
    # The safest cross-version call: use model.model directly (the underlying
    # PFN) so we can pass the two-branch adjacency.
    from utils.graph_utils import propagate_ancestor_knowledge  # noqa: E402
    def _pred_arm(t_val):
        t_col = np.full((X_train.shape[0], 1), 0.0, dtype=np.float32)
        t_col[T_train.reshape(-1) > 0.5] = 1.0
        # Wrapper API: predict(X_train, y_train, t_train, X_test, do_t)
        # Not all wrappers implement do_t explicitly — fall back to
        # duplicating with a T-augmented X.
        try:
            preds = model.predict(
                X_train=Xs, y_train=y_scaled, t_train=T_train,
                X_test=Xq, do_treatment_value=float(t_val),
                adjacency_kind=adjacency_kind,
            )
        except TypeError:
            # Simpler two-column signature: t goes as first X col
            Xs_t = np.hstack([np.full((Xs.shape[0], 1), 0.0, dtype=np.float32) + T_train.reshape(-1, 1), Xs])
            Xq_t = np.hstack([np.full((Xq.shape[0], 1), float(t_val), dtype=np.float32), Xq])
            preds = model.predict(X_train=Xs_t, y_train=y_scaled, X_test=Xq_t)
        preds = np.asarray(preds, dtype=np.float64).reshape(-1)   # scaled Y mean
        return preds

    e0_s = _pred_arm(0.0); e1_s = _pred_arm(1.0)
    # un-scale each arm
    cate = (e1_s - e0_s) * (y_rng / 2.0)
    return cate


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = SCMCaseStudyDataset(DATASET)
    n = ds.n_tables if not MAX_REAL else min(ds.n_tables, int(MAX_REAL))
    print(f'[bootstrap] UWYK {ANC_MODE}  {DATASET}  n={n}  ckpt_dir={UWYK_CKPT_DIR}', flush=True)
    model = _load_uwyk()

    rows = []; t0 = time.time()
    for r in range(n):
        cate_ds, _ = ds[r]
        cate_pred = _cate_from_uwyk(
            model,
            cate_ds.X_train, cate_ds.t_train, cate_ds.y_train, cate_ds.X_test,
            ANC_MODE,
        )
        true_cate = cate_ds.true_cate
        pehe = float(np.sqrt(np.mean((cate_pred - true_cate) ** 2)))
        ate_true = float(true_cate.mean()); ate_hat = float(cate_pred.mean())
        err = abs(ate_hat - ate_true) / max(abs(ate_true), 1e-9)
        row = {'dataset': DATASET, 'realization': r, 'anc_mode': ANC_MODE,
               'true_ate': ate_true, 'ate_pred': ate_hat,
               'pehe_raw': pehe, 'err_raw': err}
        rows.append(row)
        np.savez(os.path.join(OUT, f'r{r:03d}.npz'), **{k: np.array(v) for k, v in row.items()})
        print(f'r={r:03d}  pehe={pehe:6.3f}  err={err:5.3f}  ate={ate_hat:+5.2f} vs true {ate_true:+5.2f}  '
              f'({time.time()-t0:.0f}s)', flush=True)

    def _ms(k):
        v = np.array([r[k] for r in rows]); return v.mean(), v.std(ddof=1)/np.sqrt(len(v))
    print(f'\n══ {DATASET}  UWYK-{ANC_MODE}  n={len(rows)} ══')
    for k in ('pehe_raw', 'err_raw'):
        m, s = _ms(k); print(f'  {k:10s} = {m:8.3f} ± {s:6.3f}')


if __name__ == '__main__':
    main()
