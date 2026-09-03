"""UWYK eval on the 6 synthetic case studies — PEHE + ATE err.

FAST path: single `PreprocessingGraphConditionedPFN.predict(prediction_type='mean')`
per realization, both do(0) and do(1) queries concatenated → one forward
pass, no density resampling.

Env vars:
  DATASET          case study name
  OUT              per-realization NPZ dir
  UWYK_SRC         path to UWYK repo src
  UWYK_CKPT        explicit .pt path
  UWYK_CONFIG      explicit .yaml path
  UWYK_CKPT_DIR    fallback dir when UWYK_CKPT/UWYK_CONFIG unset
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
UWYK_CKPT_DIR = os.environ.get('UWYK_CKPT_DIR', '')
UWYK_CKPT     = os.environ.get('UWYK_CKPT', '')
UWYK_CONFIG   = os.environ.get('UWYK_CONFIG', '')
ANC_MODE      = os.environ.get('ANC_MODE', 'noanc').lower()
MAX_REAL      = os.environ.get('MAX_REAL', '')
assert ANC_MODE in ('noanc', 'anc'), ANC_MODE

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, os.path.join(REPO_SRC, 'benchmarks'))

from scm_case_study_dataset import SCMCaseStudyDataset  # noqa: E402


# ── UWYK model bootstrap (isolated from local models/utils collisions) ──
def _load_uwyk():
    saved = {}
    for name in list(sys.modules):
        if name in ('models', 'utils') or name.startswith('models.') or name.startswith('utils.'):
            saved[name] = sys.modules.pop(name)
    sys.path.insert(0, UWYK_SRC)
    pre_mod = importlib.import_module('models.PreprocessingGraphConditionedPFN')
    if UWYK_SRC in sys.path: sys.path.remove(UWYK_SRC)
    for name in list(sys.modules):
        if name in ('models', 'utils') or name.startswith('models.') or name.startswith('utils.'):
            del sys.modules[name]
    sys.modules.update(saved)

    _orig_load = torch.load
    def _patched(*a, **kw):
        kw.setdefault('weights_only', False); return _orig_load(*a, **kw)
    torch.load = _patched
    try:
        if UWYK_CKPT and UWYK_CONFIG:
            ck_p, cfg_p = UWYK_CKPT, UWYK_CONFIG
        else:
            fin_ck  = os.path.join(UWYK_CKPT_DIR, 'final_model_with_bardist.pt')
            fin_cfg = os.path.join(UWYK_CKPT_DIR, 'final_model_with_bardist_config.yaml')
            if os.path.isfile(fin_ck) and os.path.isfile(fin_cfg):
                ck_p, cfg_p = fin_ck, fin_cfg
            else:
                ck_p  = os.path.join(UWYK_CKPT_DIR, 'best_model.pt')
                cfg_p = os.path.join(UWYK_CKPT_DIR, 'best_model_config.yaml')
        assert os.path.isfile(ck_p),  f'UWYK ckpt missing: {ck_p}'
        assert os.path.isfile(cfg_p), f'UWYK config missing: {cfg_p}'
        _dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'[uwyk] loading  ckpt={ck_p}  cfg={cfg_p}  device={_dev}', flush=True)
        m = pre_mod.PreprocessingGraphConditionedPFN(
            config_path=cfg_p, checkpoint_path=ck_p, device=_dev, verbose=False,
            random_state=42, use_clustering=False,
        ).load()
    finally:
        torch.load = _orig_load
    return m


def _pad_X(X, F):
    """Pad X columns to F with zeros (if <F) or truncate (if >F). Matches
    training-time padding convention of UWYK's fixed-num_features backbone."""
    X = np.asarray(X, dtype=np.float32)
    if X.shape[1] == F: return X
    if X.shape[1] > F:  return X[:, :F]
    return np.hstack([X, np.zeros((X.shape[0], F - X.shape[1]), dtype=np.float32)])


def _cate_uwyk(model, X_train, T_train, y_train, X_test, anc_mode):
    """Single forward for both arms via predict(prediction_type='mean').
    Concatenates do(0) and do(1) queries → one transformer pass.
    Returns cate_pred = E[Y|do(1), x] - E[Y|do(0), x] per query, in raw Y units.
    """
    F = model.model.num_features            # e.g. 50 for reproduce ckpt
    # Explicit padding — wrapper's auto-pad may not exist or may fill with the
    # wrong constant, causing the model to see OOD input → collapse to prior.
    X_train_p = _pad_X(X_train, F)
    X_test_p  = _pad_X(X_test,  F)

    M = X_test_p.shape[0]
    X_intv = np.vstack([X_test_p, X_test_p]).astype(np.float32)
    T_intv = np.concatenate([np.zeros(M, dtype=np.float32),
                              np.ones(M, dtype=np.float32)])

    # Adjacency sized to model's fixed (F+2, F+2). Real edges only for the
    # first L=X_train.shape[1] X-features (positions 2 .. 2+L in the adj);
    # padded feature slots stay 0 = unknown.
    if anc_mode == 'noanc':
        adj = np.zeros((F + 2, F + 2), dtype=np.float32)
    else:
        adj = None  # wrapper auto-builds partial adjacency

    X_obs_ = X_train_p
    T_obs_ = T_train.astype(np.float32)
    Y_obs_ = y_train.astype(np.float32)

    model.fit(X_obs_, T_obs_, Y_obs_)
    preds = model.predict(
        X_obs=X_obs_, T_obs=T_obs_, Y_obs=Y_obs_,
        X_intv=X_intv,
        T_intv=T_intv,
        adjacency_matrix=adj,
        prediction_type='mean',
        inverse_transform=True,
    )
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    e0 = preds[:M]; e1 = preds[M:]
    return (e1 - e0).astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = SCMCaseStudyDataset(DATASET)
    n = ds.n_tables if not MAX_REAL else min(ds.n_tables, int(MAX_REAL))
    print(f'[bootstrap] UWYK {ANC_MODE}  {DATASET}  n={n}', flush=True)
    uwyk_model = _load_uwyk()

    rows = []; t0 = time.time()
    for r in range(n):
        cate_ds, _ = ds[r]
        try:
            cate_pred = _cate_uwyk(uwyk_model,
                                    cate_ds.X_train, cate_ds.t_train, cate_ds.y_train,
                                    cate_ds.X_test, ANC_MODE)
        except Exception as e:
            print(f'r={r:03d}  ERROR: {type(e).__name__}: {e}', flush=True)
            continue
        true_cate = np.asarray(cate_ds.true_cate, dtype=np.float32).reshape(-1)
        pehe = float(np.sqrt(np.mean((cate_pred - true_cate) ** 2)))
        ate_true = float(true_cate.mean()); ate_hat = float(cate_pred.mean())
        err = abs(ate_hat - ate_true) / max(abs(ate_true), 0.1)
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
