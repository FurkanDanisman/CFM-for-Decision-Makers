"""Shared loaders + CATE predictors for Do-PFN and DoPFN-backbone-with-2D-head.

Used by fig2_pehe_l2.py, d_scaling_linear.py, d_n_grid.py to extend the
existing UWYK-NoAnc vs Ours(fn=50) comparisons with a Do-PFN vs
Ours-DoPFN-bb(200K) comparison.

The two loaders return objects analogous to load_ours() in
rho_scaling_linear.py so the caller can swap them in with minimal changes.
"""
from __future__ import annotations
import os
import sys
import types

import numpy as np
import torch


DEVICE = torch.device('cpu')


def load_dopfn_bb(args, checkpoint_path):
    """Load DoPFNBackboneWith2DHead. Returns the same tuple shape as
    load_ours() in rho_scaling_linear.py:
        (model, edges, J, bin_width, centers, NUM_FEATURES, wb_fn)
    NUM_FEATURES = -1 (per-feature attention: no padding cap).

    Requires args.dopfn (path to DoPFN root) so DoPFN's PerFeatureTransformer
    can be imported by the backbone."""
    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    sys.path.insert(0, os.path.join(args.repo, 'training_dopfn_base'))

    _orig = torch.load
    def _p_load(*a, **kw):
        kw.setdefault('weights_only', False); return _orig(*a, **kw)
    torch.load = _p_load
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    torch.load = _orig

    cfg = ckpt['config']; J = cfg['J']
    edges = ckpt['edges'].cpu().numpy()
    bin_width = float(edges[1] - edges[0])
    centers = 0.5 * (edges[:-1] + edges[1:])

    from dopfn_backbone_head import DoPFNBackboneWith2DHead
    m = DoPFNBackboneWith2DHead(dopfn_root=args.dopfn, K=J).to(DEVICE).eval()
    m.load_state_dict(ckpt['model_state_dict'])

    ot_dir = os.path.join(args.repo, 'MALC', 'Optimal_Transport')
    if ot_dir not in sys.path: sys.path.insert(0, ot_dir)
    from ot_barycenter import wasserstein_barycenter_1d
    return m, edges, J, bin_width, centers, -1, wasserstein_barycenter_1d


def load_dopfn(args):
    """Import DoPFN's DoPFNRegressor with the sklearn check_array shim
    already installed. Returns the DoPFNRegressor class.

    Requires args.dopfn (root of DoPFN repo) and args.repo (so
    benchmarks/methods/dopfn.py shim is on the path).

    DoPFN's semi-real dataset loader gets imported by
    `datasets/__init__.py` at package-import time and is expensive; we
    only need the DoPFNRegressor class here, not the datasets, so we
    install a minimal `datasets` shim that skips load_semi_real."""
    _bench_methods = os.path.join(args.repo, 'benchmarks', 'methods')
    if _bench_methods not in sys.path:
        sys.path.insert(0, _bench_methods)
    import dopfn as _dopfn_shim  # noqa: F401  — sklearn check_array shim install

    if args.dopfn not in sys.path:
        sys.path.insert(0, args.dopfn)

    # Datasets shim — only needed to satisfy DoPFN's import graph. Skip
    # load_semi_real() (which walks the case-studies dir and is slow).
    if 'datasets' not in sys.modules:
        ds_mod = types.ModuleType('datasets')
        with open(os.path.join(args.dopfn, 'datasets/__init__.py')) as fp:
            src = fp.read().split('def load_semi_real')[0]
        exec(src, ds_mod.__dict__)
        sys.modules['datasets'] = ds_mod

    _cwd = os.getcwd()
    try:
        os.chdir(args.dopfn)
        from scripts.transformer_prediction_interface.base import DoPFNRegressor
    finally:
        os.chdir(_cwd)

    # Late-bound check_array reference inside DoPFN's base module — patch it.
    import dopfn as _dopfn_shim
    if hasattr(_dopfn_shim, '_repatch_dopfn_check_array'):
        _dopfn_shim._repatch_dopfn_check_array()
    return DoPFNRegressor


def dopfn_predict_cate(DoPFNRegressor, cate_dataset):
    """Point-estimate CATE prediction using DoPFN. Mirrors
    benchmarks/methods/dopfn.py::dopfn_pipeline exactly."""
    def _to_np(a):
        if isinstance(a, torch.Tensor): return a.numpy()
        return np.asarray(a)

    X_train = _to_np(cate_dataset.X_train).astype(np.float32)
    t_train = _to_np(cate_dataset.t_train).astype(np.float32).reshape(-1)
    y_train = _to_np(cate_dataset.y_train).astype(np.float32).reshape(-1)
    X_test  = _to_np(cate_dataset.X_test).astype(np.float32)

    # Do-PFN convention: treatment is the first covariate column.
    x_tr = np.concatenate([t_train[:, None], X_train], axis=1)
    x_te = np.concatenate([np.zeros((X_test.shape[0], 1), dtype=np.float32),
                            X_test], axis=1)

    reg = DoPFNRegressor()
    reg.fit(torch.tensor(x_tr), torch.tensor(y_train))
    cate = reg.predict_cate(torch.tensor(x_te))
    return np.asarray(cate).reshape(-1)


def dopfn_predict_ymean(DoPFNRegressor, cate_dataset):
    """Per-arm point predictions AND residual sigma for
    Gaussian-approximation density derivation. Returns
        yhat0 (n_test,), yhat1 (n_test,), sigma (float)
    sigma is estimated from OOF-style training residuals."""
    def _to_np(a):
        if isinstance(a, torch.Tensor): return a.numpy()
        return np.asarray(a)

    X_train = _to_np(cate_dataset.X_train).astype(np.float32)
    t_train = _to_np(cate_dataset.t_train).astype(np.float32).reshape(-1)
    y_train = _to_np(cate_dataset.y_train).astype(np.float32).reshape(-1)
    X_test  = _to_np(cate_dataset.X_test).astype(np.float32)

    x_tr = np.concatenate([t_train[:, None], X_train], axis=1)
    x_te0 = np.concatenate([np.zeros((X_test.shape[0], 1), dtype=np.float32),
                             X_test], axis=1)
    x_te1 = np.concatenate([np.ones((X_test.shape[0], 1), dtype=np.float32),
                             X_test], axis=1)

    reg = DoPFNRegressor()
    reg.fit(torch.tensor(x_tr), torch.tensor(y_train))
    yhat0 = np.asarray(reg.predict(torch.tensor(x_te0))).reshape(-1)
    yhat1 = np.asarray(reg.predict(torch.tensor(x_te1))).reshape(-1)
    yhat_tr = np.asarray(reg.predict(torch.tensor(x_tr))).reshape(-1)
    resid = y_train - yhat_tr
    sigma = float(max(resid.std(ddof=1), 1e-3))
    return yhat0, yhat1, sigma
