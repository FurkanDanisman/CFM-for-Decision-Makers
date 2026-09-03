"""UWYK RealCause eval — sweeps 5 adjacency variants (v3a/v3b/v3c/v3d/v6a).

Uses the reproduce-branch verbatim pipeline (target-encoded T,
wrapper.fit + two predict() calls with target-encoded T_intv,
inverse_transform=True). Only the adjacency matrix changes per variant.

Env:
  DATASET       IHDP | ACIC | CPS | PSID | PSID_bal
  OUT           per-realization NPZ dir
  UWYK_SRC      path to UWYK repo src
  UWYK_CKPT     required — the CORRECT full_conditioned_model ckpt
  UWYK_CONFIG   sibling .yaml
  ANC_VARIANT   v3a | v3b | v3c | v3d | v6a | noanc
  CAUSALPFN     required (for dataset loaders)
  MAX_REAL      optional cap
"""
from __future__ import annotations
import argparse, os, sys, time, importlib
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default=os.environ.get('DATASET', 'IHDP'),
                    choices=('IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal'))
args, _ = parser.parse_known_args()
DATASET       = args.dataset
OUT           = os.environ['OUT']
UWYK_SRC      = os.environ['UWYK_SRC']
UWYK_CKPT     = os.environ['UWYK_CKPT']
UWYK_CONFIG   = os.environ['UWYK_CONFIG']
CAUSALPFN     = os.environ['CAUSALPFN']
ANC_VARIANT   = os.environ.get('ANC_VARIANT', 'noanc').lower()
MAX_REAL      = os.environ.get('MAX_REAL', '')
PSID_BAL_SEED = int(os.environ.get('PSID_BAL_SEED', '42'))

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, CAUSALPFN); sys.path.insert(0, CAUSALPFN + '/src')

from benchmarks import (  # noqa: E402
    IHDPDataset, ACIC2016Dataset,
    RealCauseLalondeCPSDataset, RealCauseLalondePSIDDataset,
)


def get_dataset(name):
    if name == 'IHDP':                return IHDPDataset()
    if name == 'ACIC':                return ACIC2016Dataset()
    if name == 'CPS':                 return RealCauseLalondeCPSDataset()
    if name in ('PSID', 'PSID_bal'):  return RealCauseLalondePSIDDataset()
    raise ValueError(name)


# ── UWYK loader (isolated from local models/utils collisions) ────────────
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
        _dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'[uwyk] loading  ckpt={UWYK_CKPT}  cfg={UWYK_CONFIG}  device={_dev}', flush=True)
        wrapper = pre_mod.PreprocessingGraphConditionedPFN(
            config_path=UWYK_CONFIG, checkpoint_path=UWYK_CKPT, device=_dev,
            verbose=False, random_state=42, use_clustering=False,
        ).load()
    finally:
        torch.load = _orig_load
    return wrapper


# ── Adj builders (verbatim from eval_graph2d_realcause.py) ───────────────
def _padded_neg1(F, n_real):
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    for i in range(n_real, F):
        A[2 + i, :] = -1.0; A[:, 2 + i] = -1.0; A[2 + i, 2 + i] = -1.0
    return A


def build_anc_none(F, n_real): return _padded_neg1(F, n_real)


def build_anc_v3a(F, n_real):
    A = _padded_neg1(F, n_real); A[0, 1] = 1.0
    for i in range(n_real):
        A[2 + i, 0] = 1.0; A[2 + i, 1] = 1.0
    return A


def build_anc_v3b(F, n_real):
    A = _padded_neg1(F, n_real); A[0, 1] = 1.0; A[1, 0] = -1.0
    for i in range(n_real):
        A[2 + i, 0] = 1.0; A[2 + i, 1] = 1.0
        A[0, 2 + i] = -1.0; A[1, 2 + i] = -1.0
    return A


def build_anc_v3c(F, n_real):
    A = build_anc_v3b(F, n_real)
    for i in range(2 + n_real): A[i, i] = -1.0
    return A


def build_anc_v3d(F, n_real):
    """v3d: v3b + diag=+1."""
    A = build_anc_v3b(F, n_real)
    for i in range(2 + n_real): A[i, i] = 1.0
    return A


def build_anc_v3e(F, n_real):
    """v3e: v3a + diag=-1 (no reverse edges asserted)."""
    A = build_anc_v3a(F, n_real)
    for i in range(2 + n_real): A[i, i] = -1.0
    return A


def build_anc_v3f(F, n_real):
    """v3f: v3a + diag=+1 (no reverse edges asserted)."""
    A = build_anc_v3a(F, n_real)
    for i in range(2 + n_real): A[i, i] = 1.0
    return A


def _build_anc_v6_core(F, n_real):
    """v6 core: reverse-only -1 edges from unconfoundedness (Y→T, T→X, Y→X = -1).
    Padded slots -1. Diagonal NOT set here — subclasses (v6a/v6b/v6c) set it."""
    A = _padded_neg1(F, n_real)
    A[1, 0] = -1.0                                # Y → T = -1
    for i in range(n_real):
        A[0, 2 + i] = -1.0                        # T → X = -1
        A[1, 2 + i] = -1.0                        # Y → X = -1
    return A


def build_anc_v6a(F, n_real):
    """v6a: v6-core + diag=-1."""
    A = _build_anc_v6_core(F, n_real)
    for i in range(2 + n_real): A[i, i] = -1.0
    return A


def build_anc_v6b(F, n_real):
    """v6b: v6-core + diag=0."""
    return _build_anc_v6_core(F, n_real)


def build_anc_v6c(F, n_real):
    """v6c: v6-core + diag=+1."""
    A = _build_anc_v6_core(F, n_real)
    for i in range(2 + n_real): A[i, i] = 1.0
    return A


_ADJ_BUILDERS = {
    'v3a':   build_anc_v3a,
    'v3b':   build_anc_v3b,
    'v3c':   build_anc_v3c,
    'v3d':   build_anc_v3d,
    'v3e':   build_anc_v3e,
    'v3f':   build_anc_v3f,
    'v6a':   build_anc_v6a,
    'v6b':   build_anc_v6b,
    'v6c':   build_anc_v6c,
    'noanc': build_anc_none,
}


def psid_balance_subsample(X, t, y):
    t_flat = t.reshape(-1); tr = (t_flat == 1); ct = (t_flat == 0)
    X_tr = X[tr]; t_tr = t[tr]; y_tr = y[tr]
    X_ct = X[ct]; t_ct = t[ct]; y_ct = y[ct]
    n_ct = X_ct.shape[0]; n_keep = min(500, n_ct)
    if n_ct > n_keep:
        np.random.seed(PSID_BAL_SEED)
        idx = np.random.choice(n_ct, n_keep, replace=False)
        X_ct = X_ct[idx]; t_ct = t_ct[idx]; y_ct = y_ct[idx]
    X = np.vstack([X_tr, X_ct]); t = np.concatenate([t_tr, t_ct]); y = np.concatenate([y_tr, y_ct])
    perm = np.random.RandomState(PSID_BAL_SEED).permutation(X.shape[0])
    return X[perm], t[perm], y[perm]


def _cate_uwyk(model, cate_ds, variant, apply_psid_bal):
    X_train = np.asarray(cate_ds.X_train, dtype=np.float32)
    t_train_orig = np.asarray(cate_ds.t_train, dtype=np.float32)
    t_train_orig = t_train_orig.reshape(-1, 1) if t_train_orig.ndim == 1 else t_train_orig
    y_train_orig = np.asarray(cate_ds.y_train, dtype=np.float32)
    y_train_orig = y_train_orig.reshape(-1, 1) if y_train_orig.ndim == 1 else y_train_orig
    X_test = np.asarray(cate_ds.X_test, dtype=np.float32)
    y_train = y_train_orig

    if apply_psid_bal:
        X_train, t_train_orig, y_train = psid_balance_subsample(X_train, t_train_orig, y_train)

    n_test = X_test.shape[0]

    # Target-encode T with mean(Y|T)
    t_flat = t_train_orig.flatten(); y_flat = y_train.flatten()
    mean_y_t0 = float(y_flat[t_flat == 0].mean())
    mean_y_t1 = float(y_flat[t_flat == 1].mean())
    t_train = np.where(t_train_orig == 0, mean_y_t0, mean_y_t1).astype(np.float32)

    n_features_orig = X_train.shape[1]
    F = model.model.num_features
    n_real = min(n_features_orig, F)

    model.fit(X_train, t_train, y_train)

    adj = _ADJ_BUILDERS[variant](F, n_real)

    T_intv_1 = np.full((n_test, 1), mean_y_t1, dtype=np.float32)
    T_intv_0 = np.full((n_test, 1), mean_y_t0, dtype=np.float32)
    y_pred_1 = model.predict(
        X_obs=X_train, T_obs=t_train, Y_obs=y_train,
        X_intv=X_test, T_intv=T_intv_1,
        adjacency_matrix=adj,
        prediction_type='mean', inverse_transform=True,
    )
    y_pred_0 = model.predict(
        X_obs=X_train, T_obs=t_train, Y_obs=y_train,
        X_intv=X_test, T_intv=T_intv_0,
        adjacency_matrix=adj,
        prediction_type='mean', inverse_transform=True,
    )
    return np.asarray(y_pred_1 - y_pred_0, dtype=np.float32).reshape(-1)


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = get_dataset(DATASET)
    apply_psid_bal = (DATASET == 'PSID_bal')
    n = ds.n_tables if not MAX_REAL else min(ds.n_tables, int(MAX_REAL))
    print(f'[bootstrap] UWYK RealCause  DATASET={DATASET}  ANC_VARIANT={ANC_VARIANT}  n={n}', flush=True)
    model = _load_uwyk()

    rows = []; t0 = time.time()
    for r in range(n):
        cate_ds = ds[r][0]
        try:
            cate_pred = _cate_uwyk(model, cate_ds, ANC_VARIANT, apply_psid_bal)
        except Exception as e:
            print(f'r={r:03d}  ERROR: {type(e).__name__}: {e}', flush=True)
            continue
        true_cate = np.asarray(cate_ds.true_cate, dtype=np.float32).reshape(-1)
        pehe = float(np.sqrt(np.mean((cate_pred - true_cate) ** 2)))
        ate_true = float(true_cate.mean()); ate_hat = float(cate_pred.mean())
        err_l1 = abs(ate_hat - ate_true)
        row = {'dataset': DATASET, 'realization': r, 'anc_variant': ANC_VARIANT,
               'true_ate': ate_true, 'ate_pred': ate_hat,
               'pehe_raw': pehe, 'err_l1': err_l1}
        rows.append(row)
        np.savez(os.path.join(OUT, f'r{r:03d}.npz'), **{k: np.array(v) for k, v in row.items()})
        print(f'r={r:03d}  pehe={pehe:6.3f}  L1={err_l1:6.3f}  ate={ate_hat:+7.3f} vs true {ate_true:+7.3f}  '
              f'({time.time()-t0:.0f}s)', flush=True)

    def _ms(k):
        v = np.array([r[k] for r in rows]); return v.mean(), v.std(ddof=1)/np.sqrt(len(v))
    print(f'\n══ {DATASET}  UWYK-{ANC_VARIANT}  n={len(rows)} ══')
    for k in ('pehe_raw', 'err_l1'):
        m, s = _ms(k); print(f'  {k:10s} = {m:8.3f} ± {s:6.3f}')


if __name__ == '__main__':
    main()
