"""Native Do-PFN eval on the 6 synthetic case studies.

Uses DoPFNRegressor from the dopfn upstream repo. Per realization:
  - fit on training rows (T is col 0 of X)
  - predict_cate on test X
  - report PEHE + ATE_err (matching our other eval scripts)

Env vars:
  DATASET     one of the 6 case-study names (Observed_Confounder, ...)
  OUT         per-realization NPZ dir
  DOPFN_ROOT  path to dopfn_upstream repo (has scripts/, artifacts/)
  DOPFN_DATA_ROOT  parent of prior_sampling/  (default: $DOPFN_ROOT/data/prior_sampling)
  MAX_REAL    optional cap
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=os.environ.get('DATASET', 'Observed_Confounder'))
args, _ = parser.parse_known_args()
DATASET   = args.dataset
OUT       = os.environ['OUT']
DOPFN_ROOT = os.environ['DOPFN_ROOT']
CAUSALPFN  = os.environ.get('CAUSALPFN', '')  # required for RealCause loaders (from `benchmarks import IHDPDataset`)
MAX_REAL  = os.environ.get('MAX_REAL', '')

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, os.path.join(REPO_SRC, 'benchmarks'))   # top-level import
sys.path.insert(0, DOPFN_ROOT)
# CausalPFN owns the `benchmarks` package that ships IHDPDataset/ACIC2016Dataset/...
if CAUSALPFN:
    sys.path.insert(0, CAUSALPFN)
    sys.path.insert(0, CAUSALPFN + '/src')

# DoPFN's base.py calls sklearn.utils.check_array with keyword
# `ensure_all_finite=` which was removed in sklearn ≥1.6 (replaced by
# `ensure_all_finite=` → `ensure_2d=`/`force_all_finite=`). Monkey-patch
# check_array to accept and drop the removed kwarg.
import sklearn.utils as _sku  # noqa: E402
_orig_ca = _sku.check_array
def _patched_check_array(*a, **kw):
    if 'ensure_all_finite' in kw:
        # Map to the current equivalent `force_all_finite`
        kw.setdefault('force_all_finite', kw.pop('ensure_all_finite'))
    return _orig_ca(*a, **kw)
_sku.check_array = _patched_check_array
# Also patch in the sklearn.utils.validation namespace (where check_array lives)
import sklearn.utils.validation as _skuv  # noqa: E402
_skuv.check_array = _patched_check_array

from scm_case_study_dataset import SCMCaseStudyDataset  # noqa: E402

# RealCause loaders (importing here is safe — they're lightweight).
_REALCAUSE = ('IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal')

def _get_dataset(name: str):
    if name in _REALCAUSE:
        from benchmarks import (IHDPDataset, ACIC2016Dataset,
                                 RealCauseLalondeCPSDataset,
                                 RealCauseLalondePSIDDataset)
        return {
            'IHDP':     IHDPDataset(),
            'ACIC':     ACIC2016Dataset(),
            'CPS':      RealCauseLalondeCPSDataset(),
            'PSID':     RealCauseLalondePSIDDataset(),
            'PSID_bal': RealCauseLalondePSIDDataset(),
        }[name]
    return SCMCaseStudyDataset(name)


def _cate_ds_from(ds, r: int, name: str):
    """Return the CATE-style dataset for realization r."""
    if name in _REALCAUSE:
        # RealCause loaders: ds[r] returns (cate_ds, meta) or just cate_ds.
        got = ds[r]
        return got[0] if isinstance(got, tuple) else got
    # SCMCaseStudyDataset: ds[r] returns (cate_ds, meta).
    return ds[r][0]


def _n_tables(ds, name: str):
    return int(getattr(ds, 'n_tables', len(ds)))

# DoPFNRegressor loads artifacts by relative path, so cwd matters.
_prev_cwd = os.getcwd()
os.chdir(DOPFN_ROOT)
try:
    from scripts.transformer_prediction_interface.base import DoPFNRegressor  # noqa: E402
    # DoPFNRegressor defaults to device='cpu' — force CUDA when available so
    # inference runs on the requested GPU instead of falling back to CPU.
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DoPFNRegressor(device=_device)
    print(f'[dopfn_native] instantiated on device={_device}', flush=True)
finally:
    os.chdir(_prev_cwd)


def evaluate(r: int, ds):
    cate_ds = _cate_ds_from(ds, r, DATASET)
    X_train_full = np.hstack([cate_ds.t_train.reshape(-1, 1), cate_ds.X_train])
    y_train = cate_ds.y_train
    X_test_full = np.hstack([np.zeros((cate_ds.X_test.shape[0], 1), dtype=np.float32),
                              cate_ds.X_test])   # T-col placeholder (predict_cate handles both)

    # Optional random context subsampling (matches cpfn1d/graph2d convention).
    _eval_cap = os.environ.get('EVAL_MAX_CONTEXT', '')
    if _eval_cap:
        cap = int(_eval_cap)
        n_ctx = X_train_full.shape[0]
        if cap < n_ctx:
            _seed = int(os.environ.get('EVAL_CONTEXT_SEED', '0'))
            rng = np.random.default_rng(_seed + r)
            idx = rng.choice(n_ctx, size=cap, replace=False)
            X_train_full = X_train_full[idx]
            y_train = y_train[idx]

    # DoPFNRegressor.fit() expects X with T in col 0; predict_cate does the do(1)-do(0) diff.
    # predict_cate internally calls X.cpu().detach().numpy(), so pass a torch tensor.
    os.chdir(DOPFN_ROOT)
    try:
        model.fit(X_train_full, y_train)
        X_test_t = torch.from_numpy(X_test_full.astype(np.float32))
        cate_pred = model.predict_cate(X_test_t)
        # For density dump: get raw bin probs via predict_full on both arms.
        dens = None
        if os.environ.get('DENSITY_DUMP', '0') == '1':
            X0 = X_test_full.copy(); X0[:, 0] = 0.0
            X1 = X_test_full.copy(); X1[:, 0] = 1.0
            X0_t = torch.from_numpy(X0.astype(np.float32))
            X1_t = torch.from_numpy(X1.astype(np.float32))
            full0 = model.predict_full(X0_t)
            full1 = model.predict_full(X1_t)
            logits0 = np.asarray(full0['logits'])                 # (N_q, nbins)
            logits1 = np.asarray(full1['logits'])
            edges = np.asarray(full0['criterion'].borders)        # (nbins+1,) — natural Y units
            p_y0 = np.exp(logits0 - logits0.max(axis=-1, keepdims=True))
            p_y0 = p_y0 / p_y0.sum(axis=-1, keepdims=True)
            p_y1 = np.exp(logits1 - logits1.max(axis=-1, keepdims=True))
            p_y1 = p_y1 / p_y1.sum(axis=-1, keepdims=True)
            # DoPFN emits densities on RAW Y edges — set y_shift=0, y_scale=1 (identity)
            dens = dict(
                edges=edges.astype(np.float32),
                p_y0_scaled=p_y0.astype(np.float32),
                p_y1_scaled=p_y1.astype(np.float32),
                y_shift=np.float32(0.0),
                y_scale=np.float32(1.0),
            )
    finally:
        os.chdir(_prev_cwd)

    cate_pred = np.asarray(cate_pred, dtype=np.float32).reshape(-1)
    true_cate = np.asarray(cate_ds.true_cate, dtype=np.float32).reshape(-1)
    pehe = float(np.sqrt(np.mean((cate_pred - true_cate) ** 2)))
    ate_true = float(true_cate.mean())
    ate_hat  = float(cate_pred.mean())
    # Case-study datasets: report err as pure L1. RealCause: relative-error.
    if DATASET not in _REALCAUSE:
        err = abs(ate_hat - ate_true)
    else:
        err = abs(ate_hat - ate_true) / max(abs(ate_true), 0.1)
    row = {'dataset': DATASET, 'realization': r,
           'true_ate': ate_true, 'ate_pred': ate_hat,
           'pehe_raw': pehe, 'err_raw': err}
    if dens is not None:
        row.update(dens)
        row['true_cate_per_query'] = true_cate.astype(np.float32)
    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = _get_dataset(DATASET)
    n_full = _n_tables(ds, DATASET)
    n = n_full if not MAX_REAL else min(n_full, int(MAX_REAL))
    print(f'[bootstrap] native DoPFN  {DATASET}  n={n}', flush=True)
    rows = []; t0 = time.time()
    for r in range(n):
        row = evaluate(r, ds)
        rows.append(row)
        np.savez(os.path.join(OUT, f'r{r:03d}.npz'), **{k: np.array(v) for k, v in row.items()})
        print(f'r={r:03d}  pehe={row["pehe_raw"]:6.3f}  err={row["err_raw"]:5.3f}  '
              f'ate={row["ate_pred"]:+5.2f} vs true {row["true_ate"]:+5.2f}  '
              f'({time.time()-t0:.0f}s)', flush=True)
    def _ms(k):
        v = np.array([r[k] for r in rows]); return v.mean(), v.std(ddof=1)/np.sqrt(len(v))
    print(f'\n══ {DATASET}  native DoPFN  n={len(rows)} ══')
    for k in ('pehe_raw', 'err_raw'):
        m, s = _ms(k); print(f'  {k:10s} = {m:8.3f} ± {s:6.3f}')


if __name__ == '__main__':
    main()
