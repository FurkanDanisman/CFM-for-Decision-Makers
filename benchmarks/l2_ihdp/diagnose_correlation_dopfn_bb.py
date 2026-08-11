"""Measure predicted correlation for the DoPFN-backbone R-PFN checkpoint on IHDP.

Analogue of diagnose_correlation.py but loads a DoPFNBackboneWith2DHead
checkpoint. Reports Pearson correlation of the model's raw p_mat output
and its MALC-smoothed density for every IHDP test query on one realization.

Truth ρ on IHDP = 0 exactly (Hill 2011, independent per-arm noise).
Compare against fn=50's ρ ≈ 0.05 (correctly learned) and fn=10's ρ ≈ 0.94
(learned false correlation) — the earlier diagnose_correlation.py output.

Usage
-----
    python R-PFN/benchmarks/l2_ihdp/diagnose_correlation_dopfn_bb.py \\
        --repo /scratch/furkanbd/rpfn_bench_kit/R-PFN \\
        --causalpfn /scratch/furkanbd/rpfn_bench_kit/external/causalpfn \\
        --dopfn /scratch/furkanbd/rpfn_bench_kit/external/dopfn \\
        --checkpoint /scratch/furkanbd/rpfn_bench_kit/checkpoints_dopfn_backbone_j10/step_35000.pt \\
        --realization 0 --malc-B 60 --n-eval 200 --malc-max-K 3
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import types

import numpy as np
import torch


def _corr_from_2d(p, y0_grid, y1_grid, dy0, dy1):
    """Pearson rho from a 2D density p[i, j] where i=y1_index, j=y0_index."""
    m_y0 = p.sum(axis=0) * dy1
    m_y1 = p.sum(axis=1) * dy0
    E_y0 = float((y0_grid * m_y0).sum() * dy0)
    E_y1 = float((y1_grid * m_y1).sum() * dy1)
    E_y0y1 = float((y1_grid[:, None] * y0_grid[None, :] * p).sum() * dy0 * dy1)
    Var_y0 = max(float(((y0_grid - E_y0) ** 2 * m_y0).sum() * dy0), 1e-16)
    Var_y1 = max(float(((y1_grid - E_y1) ** 2 * m_y1).sum() * dy1), 1e-16)
    Cov = E_y0y1 - E_y0 * E_y1
    return Cov / (np.sqrt(Var_y0) * np.sqrt(Var_y1))


def _install_dopfn_shim(dopfn_dir):
    if 'datasets' in sys.modules:
        return
    sys.path.insert(0, dopfn_dir)
    ds_mod = types.ModuleType('datasets')
    with open(os.path.join(dopfn_dir, 'datasets/__init__.py')) as fp:
        src = fp.read().split('def load_semi_real')[0]
    exec(src, ds_mod.__dict__)
    sys.modules['datasets'] = ds_mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo',       required=True)
    ap.add_argument('--causalpfn',  required=True)
    ap.add_argument('--dopfn',      required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--realization', type=int, default=0)
    ap.add_argument('--malc-B',     type=int, default=60)
    ap.add_argument('--n-eval',     type=int, default=200)
    ap.add_argument('--malc-max-K', type=int, default=3)
    args = ap.parse_args()

    _install_dopfn_shim(args.dopfn)
    sys.path.insert(0, args.causalpfn)
    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    sys.path.insert(0, os.path.join(args.repo, 'training_dopfn_base'))

    from benchmarks import IHDPDataset
    from dopfn_backbone_head import DoPFNBackboneWith2DHead
    from losses.BarDistribution2D import unpack_pred, fit_malc_inner
    from malc_2d import dmalc_2d

    cd, _ = IHDPDataset()[args.realization]
    y_train_full = cd.y_train.numpy() if hasattr(cd.y_train, 'numpy') else np.asarray(cd.y_train)
    X_train_full = cd.X_train.numpy() if hasattr(cd.X_train, 'numpy') else np.asarray(cd.X_train)
    t_train_full = cd.t_train.numpy() if hasattr(cd.t_train, 'numpy') else np.asarray(cd.t_train)
    X_test = cd.X_test.numpy() if hasattr(cd.X_test, 'numpy') else np.asarray(cd.X_test)
    true_cate = cd.true_cate.numpy() if hasattr(cd.true_cate, 'numpy') else np.asarray(cd.true_cate)
    true_cate = true_cate.reshape(-1)

    y_min = float(y_train_full.min()); y_max = float(y_train_full.max())
    y_rng = max(y_max - y_min, 1e-6)
    true_cate_scaled = true_cate * (2.0 / y_rng)

    # Split queries by |true tau|
    order = np.argsort(np.abs(true_cate_scaled))
    easy_idx = order[:len(order) // 2].tolist()
    hard_idx = order[len(order) // 2:].tolist()

    print(f'\n=== DoPFN-backbone checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['config']; J = cfg['J']
    edges_np = ckpt['edges'].cpu().numpy()
    bin_width = float(edges_np[1] - edges_np[0])
    bin_centres = 0.5 * (edges_np[:-1] + edges_np[1:])
    print(f'  cfg: J={J}  edges=[{edges_np[0]:.2f}, {edges_np[-1]:.2f}]  bin_width={bin_width:.4f}')

    model = DoPFNBackboneWith2DHead(dopfn_root=args.dopfn, K=J).eval()
    model.load_state_dict(ckpt['model_state_dict'])

    # Prepare inputs (no truncation — model uses per-feature attention).
    N = X_train_full.shape[0]
    X_context = X_train_full[:N].astype(np.float32)
    T_context = t_train_full[:N].astype(np.float32).reshape(-1, 1)
    Y_context = y_train_full[:N].astype(np.float32).reshape(-1, 1)
    Y_context = ((Y_context - y_min) / y_rng * 2.0 - 1.0).astype(np.float32)

    Xc = torch.from_numpy(X_context).unsqueeze(0)
    Tc = torch.from_numpy(T_context).unsqueeze(0)
    Yc = torch.from_numpy(Y_context).unsqueeze(0)
    Xq = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        pred = model(Xc, Tc, Yc, Xq)['predictions'][0]

    xs = np.linspace(edges_np[0], edges_np[-1], args.n_eval)
    ys = np.linspace(edges_np[0], edges_np[-1], args.n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    eval_pts = np.column_stack([XX.ravel(), YY.ravel()])
    dxs = float(xs[1] - xs[0]); dys = float(ys[1] - ys[0])

    rho_raw_all = []
    rho_smooth_all = []
    n_test = X_test.shape[0]
    for q in range(n_test):
        p_mat, *_ = unpack_pred(pred[q], J, bin_width)
        p_mat_np = p_mat.detach().cpu().numpy()
        p_dens = p_mat_np / (p_mat_np.sum() * bin_width * bin_width)
        p_dens_yx = p_dens.T
        rho_raw = _corr_from_2d(p_dens_yx, bin_centres, bin_centres,
                                bin_width, bin_width)

        seed = int(hashlib.md5(f'q{q}'.encode()).hexdigest()[:8], 16) % 10**8
        try:
            fit = fit_malc_inner(p_mat_np.T, edges_np, edges_np,
                                 B_fit=args.malc_B, B_select=args.malc_B,
                                 max_K=args.malc_max_K, seed=seed, parallel=False)
            density_2d = dmalc_2d(fit, eval_pts).reshape(args.n_eval, args.n_eval)
            rho_smooth = _corr_from_2d(density_2d, xs, ys, dxs, dys)
        except Exception:
            rho_smooth = np.nan

        rho_raw_all.append(rho_raw)
        rho_smooth_all.append(rho_smooth)

    rho_raw_all = np.array(rho_raw_all)
    rho_smooth_all = np.array(rho_smooth_all)

    def _pr(name, arr):
        arr = arr[np.isfinite(arr)]
        print(f'  {name:28s} n={arr.size:>3d}   mean={arr.mean():+.3f}   '
              f'median={np.median(arr):+.3f}   std={arr.std(ddof=1):.3f}   '
              f'min={arr.min():+.3f}  max={arr.max():+.3f}')

    print(f'  --- rho from raw p_mat ---')
    _pr('all queries',   rho_raw_all)
    _pr('easy (small tau)', rho_raw_all[easy_idx])
    _pr('hard (large tau)', rho_raw_all[hard_idx])
    print(f'  --- rho from MALC-smoothed density (K={args.malc_max_K}) ---')
    _pr('all queries',   rho_smooth_all)
    _pr('easy (small tau)', rho_smooth_all[easy_idx])
    _pr('hard (large tau)', rho_smooth_all[hard_idx])

    print()
    print('Reference (from prior diagnose_correlation.py runs on IHDP r=0):')
    print('  Truth rho on IHDP = 0 exactly')
    print('  Ours(fn=50):  raw ~ +0.03  smoothed ~ +0.05    (correctly learned)')
    print('  Ours(fn=10):  raw ~ +0.80  smoothed ~ +0.94    (learned false)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
