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
MAX_REAL  = os.environ.get('MAX_REAL', '')

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, os.path.join(REPO_SRC, 'benchmarks'))   # top-level import
sys.path.insert(0, DOPFN_ROOT)

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

# DoPFNRegressor loads artifacts by relative path, so cwd matters.
_prev_cwd = os.getcwd()
os.chdir(DOPFN_ROOT)
try:
    from scripts.transformer_prediction_interface.base import DoPFNRegressor  # noqa: E402
    model = DoPFNRegressor()
finally:
    os.chdir(_prev_cwd)


def evaluate(r: int, ds: SCMCaseStudyDataset):
    cate_ds, _ = ds[r]
    X_train_full = np.hstack([cate_ds.t_train.reshape(-1, 1), cate_ds.X_train])
    y_train = cate_ds.y_train
    X_test_full = np.hstack([np.zeros((cate_ds.X_test.shape[0], 1), dtype=np.float32),
                              cate_ds.X_test])   # T-col placeholder (predict_cate handles both)

    # DoPFNRegressor.fit() expects X with T in col 0; predict_cate does the do(1)-do(0) diff.
    # predict_cate internally calls X.cpu().detach().numpy(), so pass a torch tensor.
    os.chdir(DOPFN_ROOT)
    try:
        model.fit(X_train_full, y_train)
        X_test_t = torch.from_numpy(X_test_full.astype(np.float32))
        cate_pred = model.predict_cate(X_test_t)
    finally:
        os.chdir(_prev_cwd)

    cate_pred = np.asarray(cate_pred, dtype=np.float32).reshape(-1)
    true_cate = np.asarray(cate_ds.true_cate, dtype=np.float32).reshape(-1)
    pehe = float(np.sqrt(np.mean((cate_pred - true_cate) ** 2)))
    ate_true = float(true_cate.mean())
    ate_hat  = float(cate_pred.mean())
    err = abs(ate_hat - ate_true) / max(abs(ate_true), 1e-9)
    return {'dataset': DATASET, 'realization': r,
            'true_ate': ate_true, 'ate_pred': ate_hat,
            'pehe_raw': pehe, 'err_raw': err}


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = SCMCaseStudyDataset(DATASET)
    n = ds.n_tables if not MAX_REAL else min(ds.n_tables, int(MAX_REAL))
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
