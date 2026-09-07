"""Do-PFN native RealCause eval — standalone reproduction of Table 3's Do-PFN row.

Extracted verbatim from `benchmarks/methods/dopfn.py::dopfn_pipeline` and the
loader / metric helpers in `benchmarks/run_one.py`. No behavior changes.
Produces per-realization npz with `pehe_dopfn` and `err_dopfn` that match the
aggregate.py output of the paper's pipeline exactly:

    Do-PFN  IHDP 6.07±0.90  ACIC 4.11±0.55  CPS 12015±32  PSID 20907±138  PSIDbal 22773±174

Usage:
    python realcause_eval/do_pfn.py \\
        --dataset IHDP \\
        --realization 0 \\
        --outdir /scratch/.../results_do_pfn_realcause \\
        --dopfn /scratch/.../external/dopfn \\
        --causalpfn /scratch/.../external/causalpfn

Or the sbatch wrapper (5-task array over datasets, all realizations sequentially per task):
    sbatch benchmarks/cluster/submit_realcause_do_pfn.sbatch
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch


# ── sklearn shim: DoPFN calls check_array with `ensure_all_finite=` (sklearn≥1.6)
#    On older sklearn this kwarg doesn't exist; alias it to `force_all_finite`.
def _install_check_array_shim():
    try:
        import inspect
        import sklearn.utils.validation as _v
        _sig = inspect.signature(_v.check_array)
        needs_patch = 'ensure_all_finite' not in _sig.parameters
        _orig = _v.check_array

        def _shim(*a, **kw):
            if 'ensure_all_finite' in kw:
                kw['force_all_finite'] = kw.pop('ensure_all_finite')
            return _orig(*a, **kw)

        if needs_patch:
            _v.check_array = _shim
            import sklearn.utils
            if hasattr(sklearn.utils, 'check_array'):
                sklearn.utils.check_array = _shim
            for _name in list(sys.modules):
                if _name.endswith('.transformer_prediction_interface.base') \
                        or _name == 'scripts.transformer_prediction_interface.base':
                    _mod = sys.modules[_name]
                    if hasattr(_mod, 'check_array'):
                        _mod.check_array = _shim
    except Exception:
        pass  # best-effort; a real mismatch will raise at the actual call site.


# ── helpers ------------------------------------------------------------------
def _to_np(a):
    if isinstance(a, torch.Tensor): return a.numpy()
    return np.asarray(a)


def _pehe(true_cate, pred_cate):
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(true_cate, pred_cate)))


def _ate_relerr(true_cate, pred_cate):
    t = float(np.mean(true_cate)); p = float(np.mean(pred_cate))
    if abs(t) < 1e-12:
        return 0.0 if abs(p) < 1e-12 else float('inf')
    return abs(t - p) / abs(t)


def _psid_balance_subsample(cd, seed: int = 42):
    """Match dofm_psid_balanced.py: keep all T=1, sample up to 500 T=0 with
    np.random.seed(42), shuffle with RandomState(42). Mutates cd in place."""
    X = _to_np(cd.X_train).astype(np.float32)
    t = _to_np(cd.t_train).astype(np.float32).reshape(-1)
    y = _to_np(cd.y_train).astype(np.float32).reshape(-1)

    treated = (t == 1); control = (t == 0)
    X_tr, t_tr, y_tr = X[treated], t[treated], y[treated]
    X_ct, t_ct, y_ct = X[control], t[control], y[control]

    n_ct = X_ct.shape[0]
    n_keep = min(500, n_ct)
    if n_ct > n_keep:
        np.random.seed(seed)
        idx = np.random.choice(n_ct, n_keep, replace=False)
        X_ct, t_ct, y_ct = X_ct[idx], t_ct[idx], y_ct[idx]

    X_new = np.vstack([X_tr, X_ct])
    t_new = np.concatenate([t_tr, t_ct])
    y_new = np.concatenate([y_tr, y_ct])
    perm = np.random.RandomState(seed).permutation(X_new.shape[0])

    cd.X_train = X_new[perm]
    cd.t_train = t_new[perm]
    cd.y_train = y_new[perm]
    print(f'[do_pfn] PSID-balanced: kept {X_tr.shape[0]} treated + '
          f'{X_ct.shape[0]}/{n_ct} controls (seed={seed}); '
          f'final n_train={X_new.shape[0]}', flush=True)
    return cd


def load_realization(dname: str, r: int):
    """Same dataset loaders as `benchmarks/run_one.py::load_realization`.
    PSIDbal applies the seed=42 balance-subsample from dofm_psid_balanced.py."""
    if dname == 'IHDP':
        from benchmarks import IHDPDataset
        cd, ad = IHDPDataset()[r]
    elif dname == 'ACIC':
        from benchmarks import ACIC2016Dataset
        cd, ad = ACIC2016Dataset()[r]
    elif dname == 'CPS':
        from benchmarks import RealCauseLalondeCPSDataset
        cd, ad = RealCauseLalondeCPSDataset()[r]
    elif dname == 'PSID':
        from benchmarks import RealCauseLalondePSIDDataset
        cd, ad = RealCauseLalondePSIDDataset()[r]
    elif dname == 'PSIDbal':
        from benchmarks import RealCauseLalondePSIDDataset
        cd, ad = RealCauseLalondePSIDDataset()[r]
        cd = _psid_balance_subsample(cd)
    else:
        raise ValueError(dname)
    return cd, ad


def _build_dopfn_regressor(DoPFNRegressor):
    """Instantiate DoPFNRegressor ONCE and pin to GPU. Reused across all
    realizations — otherwise DoPFN reloads its transformer checkpoint from
    disk on every realization, which is what made CPS/PSID crawl."""
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        reg = DoPFNRegressor(device=_device)
    except TypeError:
        reg = DoPFNRegressor()
        reg.device = _device
    print(f'[do_pfn] instantiated DoPFNRegressor on device={_device}', flush=True)
    return reg


# ── the pipeline (verbatim from methods/dopfn.py) ----------------------------
def dopfn_pipeline(cate_dataset, reg, return_density=False):
    """Returns length-N cate predictions on cate_dataset.X_test.

    Do-PFN convention: treatment is the first covariate column. `fit(x, y)`
    then run `predict_full` on X-with-T=0 and X-with-T=1 arms; take the
    bar-dist mean difference. This is IDENTICAL to reg.predict_cate(x)
    (which does the same two `predict_full` calls internally, see
    DoPFNRegressor.predict_cate → predict_cid → predict → predict_full),
    but critically we use the SAME forward passes for both the point CATE
    and the density dump.

    TabPFN's `transformer_predict` uses ensemble randomness (feature
    permutations, etc.) so two separate calls to predict_full return
    slightly different logits. If we call reg.predict_cate for the point
    and reg.predict_full separately for the density, the density mean will
    drift from the point (was seeing ~35-unit gaps on CPS). Doing both
    from the same full0/full1 pair guarantees density-mean == point CATE.
    """
    X_train = _to_np(cate_dataset.X_train).astype(np.float32)
    t_train = _to_np(cate_dataset.t_train).astype(np.float32).reshape(-1)
    y_train = _to_np(cate_dataset.y_train).astype(np.float32).reshape(-1)
    X_test  = _to_np(cate_dataset.X_test).astype(np.float32)

    x_tr = np.concatenate([t_train[:, None], X_train], axis=1)
    x_te = np.concatenate([np.zeros((X_test.shape[0], 1), dtype=np.float32), X_test], axis=1)

    reg.fit(torch.tensor(x_tr), torch.tensor(y_train))

    # ONE pair of forward passes shared by point CATE and density.
    X0 = x_te.copy(); X0[:, 0] = 0.0
    X1 = x_te.copy(); X1[:, 0] = 1.0
    full0 = reg.predict_full(torch.tensor(X0))
    full1 = reg.predict_full(torch.tensor(X1))
    # Point CATE = predict['mean'] diff, byte-identical to reg.predict_cate
    # would have produced FROM THIS PAIR of forward passes.
    cate = np.asarray(full1['mean']).reshape(-1) - np.asarray(full0['mean']).reshape(-1)

    if not return_density:
        return cate

    logits0 = np.asarray(full0['logits'])
    logits1 = np.asarray(full1['logits'])
    edges = np.asarray(full0['criterion'].borders)   # RAW Y units

    # DoPFN uses FullSupportBarDistribution: interior bars = uniform (midpoint
    # centers), FIRST bar = left half-normal with mean = borders[1] - w0·√(2/π)/0.6745,
    # LAST bar = right half-normal with mean = borders[-2] + w_last·√(2/π)/0.6745,
    # where 0.6745 = HalfNormal(scale=1).icdf(0.5) and √(2/π) = HalfNormal(1).mean.
    # Their ratio is exactly _HN_MEAN_FACTOR ≈ 1.1829. Save effective centers so
    # the aggregator's `sum(p * centers)` exactly reproduces criterion.mean(logits).
    _HN_MEAN_FACTOR = float(np.sqrt(2.0 / np.pi) /
                             (np.sqrt(2.0) * float(__import__('scipy.special',
                              fromlist=['erfinv']).erfinv(0.5))))
    widths = np.diff(edges).astype(np.float64)
    centers = (edges[:-1] + widths / 2).astype(np.float64)         # (K,) midpoints
    centers[0]  = float(edges[1])  - widths[0]  * _HN_MEAN_FACTOR   # left tail mean
    centers[-1] = float(edges[-2]) + widths[-1] * _HN_MEAN_FACTOR   # right tail mean

    p_y0 = np.exp(logits0 - logits0.max(axis=-1, keepdims=True))
    p_y0 /= p_y0.sum(axis=-1, keepdims=True)
    p_y1 = np.exp(logits1 - logits1.max(axis=-1, keepdims=True))
    p_y1 /= p_y1.sum(axis=-1, keepdims=True)
    return (cate, p_y0.astype(np.float32), p_y1.astype(np.float32),
            edges.astype(np.float32), centers.astype(np.float32))


# ── driver -------------------------------------------------------------------
def _dataset_n_tables(dname: str) -> int:
    return {'IHDP': 100, 'ACIC': 10, 'CPS': 100, 'PSID': 100, 'PSIDbal': 100}[dname]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True,
                    choices=['IHDP', 'ACIC', 'CPS', 'PSID', 'PSIDbal'])
    p.add_argument('--realization', type=int, default=None,
                    help='If given, run only that realization. Else run all '
                         'realizations of the dataset sequentially.')
    p.add_argument('--outdir', required=True,
                    help='Per-realization npzs land at OUTDIR/<DATASET>_r<###>.npz.')
    p.add_argument('--dopfn', required=True,
                    help='Path to dopfn_upstream repo root (has scripts/, artifacts/).')
    p.add_argument('--causalpfn', required=True,
                    help='Path to CausalPFN repo root (ships the benchmarks package '
                         'with IHDPDataset/ACIC2016Dataset/... loaders).')
    p.add_argument('--max-real', type=int, default=0,
                    help='Optional cap on realizations processed when --realization '
                         'is not given. 0 = no cap (use dataset default).')
    return p.parse_args()


def main():
    args = _parse_args()

    # sys.path: CausalPFN owns the `benchmarks` package; DoPFN ships DoPFNRegressor.
    sys.path.insert(0, args.causalpfn)
    sys.path.insert(0, args.causalpfn + '/src')
    sys.path.insert(0, args.dopfn)

    _install_check_array_shim()

    # DoPFNRegressor loads artifacts by relative path — cwd must be dopfn root.
    _prev_cwd = os.getcwd()
    os.chdir(args.dopfn)
    try:
        from scripts.transformer_prediction_interface.base import DoPFNRegressor
    finally:
        os.chdir(_prev_cwd)

    os.makedirs(args.outdir, exist_ok=True)

    n_full = _dataset_n_tables(args.dataset)
    if args.realization is not None:
        real_indices = [args.realization]
    else:
        n = n_full if args.max_real <= 0 else min(n_full, args.max_real)
        real_indices = list(range(n))

    print(f'[do_pfn] {args.dataset}  n_reals={len(real_indices)}  '
          f'outdir={args.outdir}', flush=True)

    # Build DoPFNRegressor ONCE (cwd must be dopfn root — it loads artifacts
    # by relative path). Reused across all realizations so we don't reload
    # the checkpoint 100 times.
    os.chdir(args.dopfn)
    try:
        reg = _build_dopfn_regressor(DoPFNRegressor)
    finally:
        os.chdir(_prev_cwd)

    _do_density = os.environ.get('DENSITY_DUMP', '0') == '1'

    t0 = time.time()
    for r in real_indices:
        cd, ad = load_realization(args.dataset, r)
        true_cate = _to_np(cd.true_cate).reshape(-1)

        os.chdir(args.dopfn)
        try:
            if _do_density:
                cate_pred, p_y0, p_y1, edges_np, centers_np = dopfn_pipeline(cd, reg, return_density=True)
            else:
                cate_pred = dopfn_pipeline(cd, reg)
        finally:
            os.chdir(_prev_cwd)

        pehe = _pehe(true_cate, cate_pred)
        err  = _ate_relerr(true_cate, cate_pred)

        # Normalize output filename to underscored dataset (PSIDbal → PSID_bal)
        # so it matches the naming the rest of realcause_eval/ uses and the
        # aggregator's glob pattern.
        _file_ds = 'PSID_bal' if args.dataset == 'PSIDbal' else args.dataset
        out_file = os.path.join(args.outdir, f'{_file_ds}_r{r:03d}.npz')
        save_kw = dict(
            dataset=args.dataset,
            realization=r,
            pehe_dopfn=np.float64(pehe),
            err_dopfn=np.float64(err),
            true_cate=true_cate.astype(np.float32),
            cate_pred=cate_pred.astype(np.float32),
        )
        if _do_density:
            # Bar-dist edges are in RAW Y units for DoPFN — y_shift=0, y_scale=1.
            # Save effective_centers (with tail half-normal correction) so the
            # aggregator's `sum(p * centers)` reproduces criterion.mean(logits)
            # to fp32 — matches DoPFN's own point CATE exactly.
            save_kw.update(dict(
                edges=edges_np.astype(np.float32),
                effective_centers=centers_np.astype(np.float32),
                p_y0_scaled=p_y0.astype(np.float32),
                p_y1_scaled=p_y1.astype(np.float32),
                y_shift=np.float32(0.0),
                y_scale=np.float32(1.0),
                true_cate_per_query=true_cate.astype(np.float32),
            ))
        np.savez(out_file, **save_kw)
        print(f'  r={r:03d}  pehe_dopfn={pehe:7.3f}  err_dopfn={err:6.3f}  '
              f'({time.time()-t0:.0f}s)', flush=True)

    print(f'[do_pfn] done in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
