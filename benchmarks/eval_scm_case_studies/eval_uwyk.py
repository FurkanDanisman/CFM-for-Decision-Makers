"""UWYK eval on the 6 synthetic case studies — PEHE + ATE err.

VERBATIM copy of UWYK's own dofm_full_conditioning.py pipeline
(reproduced in benchmarks/uwyk_direct_repro.py). Key details we were
missing before:

- T is TARGET-ENCODED: t_train = where(T==0, mean_y_t0, mean_y_t1).
  NOT 0/1. Same for T_intv (query treatment is the encoded value).
- Two separate wrapper.predict(prediction_type='mean') calls (T=1 arm,
  T=0 arm). Wrapper handles X standardization + padding + Y scaling
  internally.
- Adjacency mode: 'all_unknown' (= noanc) or 'full_graph' (= anc).

Env: DATASET, OUT, UWYK_SRC, UWYK_CKPT, UWYK_CONFIG (or UWYK_CKPT_DIR),
     ANC_MODE (noanc|anc), MAX_REAL, DOPFN_DATA_ROOT
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

# Map our anc_mode → UWYK's graph_mode
GRAPH_MODE = 'all_unknown' if ANC_MODE == 'noanc' else 'full_graph'

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, os.path.join(REPO_SRC, 'benchmarks'))

from scm_case_study_dataset import SCMCaseStudyDataset  # noqa: E402


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
        wrapper = pre_mod.PreprocessingGraphConditionedPFN(
            config_path=cfg_p, checkpoint_path=ck_p, device=_dev, verbose=False,
            random_state=42, use_clustering=False,
        ).load()
    finally:
        torch.load = _orig_load
    return wrapper


def build_adjacency_matrix(model_n_features, n_real_features, graph_mode):
    """VERBATIM copy of dofm_full_conditioning.py::build_adjacency_matrix."""
    A = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
    T_idx, Y_idx, feat_off = 0, 1, 2

    if graph_mode == 'all_unknown':
        pass
    elif graph_mode == 'full_graph':
        A[T_idx, Y_idx] = 1.0
        for i in range(n_real_features):
            A[feat_off + i, T_idx] = 1.0
            A[feat_off + i, Y_idx] = 1.0
    else:
        raise ValueError(graph_mode)

    for i in range(n_real_features, model_n_features):
        idx = feat_off + i
        A[idx, :] = -1.0
        A[:, idx] = -1.0
        A[idx, idx] = -1.0
    return A


def _cate_uwyk_paper_pipeline(model, cate_dataset, graph_mode):
    """VERBATIM copy of dofm_full_conditioning.py::dofm_full_conditioning_pipeline
    (see benchmarks/uwyk_direct_repro.py:49-99)."""
    X_train = np.asarray(cate_dataset.X_train, dtype=np.float32)
    t_train_orig = np.asarray(cate_dataset.t_train, dtype=np.float32)
    t_train_orig = t_train_orig.reshape(-1, 1) if t_train_orig.ndim == 1 else t_train_orig
    y_train_orig = np.asarray(cate_dataset.y_train, dtype=np.float32)
    y_train_orig = y_train_orig.reshape(-1, 1) if y_train_orig.ndim == 1 else y_train_orig
    X_test = np.asarray(cate_dataset.X_test, dtype=np.float32)
    y_train = y_train_orig

    n_train = X_train.shape[0]; n_test = X_test.shape[0]

    # Target-encode T with mean Y per arm
    t_flat = t_train_orig.flatten(); y_flat = y_train.flatten()
    mean_y_t0 = float(y_flat[t_flat == 0].mean())
    mean_y_t1 = float(y_flat[t_flat == 1].mean())
    t_train = np.where(t_train_orig == 0, mean_y_t0, mean_y_t1).astype(np.float32)

    t_intv_0_encoded = mean_y_t0
    t_intv_1_encoded = mean_y_t1

    n_features_orig = X_train.shape[1]
    model_n_features = model.model.num_features
    n_real_features = min(n_features_orig, model_n_features)

    # Wrapper does its own fit → internal preprocessing state
    model.fit(X_train, t_train, y_train)

    adjacency_matrix = build_adjacency_matrix(model_n_features, n_real_features, graph_mode)

    T_intv_1 = np.full((n_test, 1), t_intv_1_encoded, dtype=np.float32)
    y_pred_1 = model.predict(
        X_obs=X_train, T_obs=t_train, Y_obs=y_train,
        X_intv=X_test, T_intv=T_intv_1,
        adjacency_matrix=adjacency_matrix,
        prediction_type='mean', inverse_transform=True,
    )
    T_intv_0 = np.full((n_test, 1), t_intv_0_encoded, dtype=np.float32)
    y_pred_0 = model.predict(
        X_obs=X_train, T_obs=t_train, Y_obs=y_train,
        X_intv=X_test, T_intv=T_intv_0,
        adjacency_matrix=adjacency_matrix,
        prediction_type='mean', inverse_transform=True,
    )
    cate_pred = np.asarray(y_pred_1 - y_pred_0, dtype=np.float32).reshape(-1)
    return cate_pred


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = SCMCaseStudyDataset(DATASET)
    n = ds.n_tables if not MAX_REAL else min(ds.n_tables, int(MAX_REAL))
    print(f'[bootstrap] UWYK-paper-pipeline  graph_mode={GRAPH_MODE}  '
          f'{DATASET}  n={n}', flush=True)
    wrapper = _load_uwyk()

    rows = []; t0 = time.time()
    for r in range(n):
        cate_ds, _ = ds[r]
        try:
            cate_pred = _cate_uwyk_paper_pipeline(wrapper, cate_ds, GRAPH_MODE)
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
