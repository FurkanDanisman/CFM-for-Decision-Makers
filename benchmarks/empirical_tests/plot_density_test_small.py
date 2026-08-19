"""Plot the y0, y1, τ, ATE density predictions from Do-PFN and DoPFN-bb
against the analytic Gaussian truth for a single polynomial-SCM seed.

Same SCM / same methods as density_test_small.py; adds a 4-panel PNG.

Usage:
    python plot_density_test_small.py \\
        --repo $DEPLOY_ROOT/R-PFN \\
        --dopfn $DEPLOY_ROOT/external/dopfn \\
        --checkpoint-dopfn-bb $DEPLOY_ROOT/checkpoints_dopfn_backbone_realj10/step_150000.pt \\
        --d 6 --N 200 --seed 6 --q 0 \\
        --out density_test_small_d6_seed6_q0.png
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--dopfn', required=True)
    ap.add_argument('--checkpoint-dopfn-bb', required=True)
    ap.add_argument('--d', type=int, default=6)
    ap.add_argument('--N', type=int, default=200)
    ap.add_argument('--n-test', type=int, default=25)
    ap.add_argument('--seed', type=int, default=6)
    ap.add_argument('--q', type=int, default=0,
                     help='Which of the n_test queries to plot for y0/y1/τ.')
    ap.add_argument('--rho', type=float, default=0.0)
    ap.add_argument('--sigma-eps', type=float, default=1.0)
    ap.add_argument('--degree', type=int, default=3)
    ap.add_argument('--malc-B', type=int, default=60)
    ap.add_argument('--malc-max-K', type=int, default=3)
    ap.add_argument('--n-eval', type=int, default=200)
    ap.add_argument('--out', required=True,
                     help='Output PNG path.')
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'empirical_tests'))
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks'))
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'l2_ihdp'))
    sys.path.insert(0, os.path.join(args.repo, 'training_dopfn_base'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC', 'Optimal_Transport'))
    sys.path.insert(0, args.repo)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    from fig2_pehe_l2 import (
        make_polynomial_scm, truth_marginals, truth_cate,
        Y_GRID, TAU_GRID, Y_DX, TAU_DX, l2_1d, wass_bary_of_grid,
    )
    from ot_barycenter import wasserstein_barycenter_1d
    from dopfn_backbone_head import DoPFNBackboneWith2DHead
    from losses.BarDistribution2D import fit_malc_inner
    from malc_2d import dmalc_2d
    from methods_densities import ours_densities, dopfn_densities
    from dopfn_helpers import load_dopfn
    from true_ihdp import Y_CENTERS, TAU_CENTERS

    print(f'[load] DoPFN-bb {args.checkpoint_dopfn_bb}', flush=True)
    ckpt = torch.load(args.checkpoint_dopfn_bb, map_location='cpu', weights_only=False)
    cfg = ckpt['config']; J = cfg['J']
    edges_np = ckpt['edges'].cpu().numpy()
    bin_width = float(edges_np[1] - edges_np[0])
    model_bb = DoPFNBackboneWith2DHead(dopfn_root=args.dopfn, K=J).eval()
    model_bb.load_state_dict(ckpt['model_state_dict'])

    DoPFNRegressor = load_dopfn(args)

    cd = make_polynomial_scm(seed=args.seed, n_context=args.N, n_test=args.n_test,
                              rho_eff=min(args.rho, 0.99),
                              x_dim=args.d, degree=args.degree,
                              sigma_eps=args.sigma_eps)
    p_y0_true = truth_marginals(cd)[0]
    p_y1_true = truth_marginals(cd)[1]
    p_tau_true = truth_cate(cd)
    p_ate_true = wass_bary_of_grid(p_tau_true, TAU_GRID, wasserstein_barycenter_1d)

    y_ctx_np = cd.y_train.numpy()
    y_min = float(y_ctx_np.min()); y_max = float(y_ctx_np.max())
    y_rng = max(y_max - y_min, 1e-6)

    print(f'[run] Do-PFN ...', flush=True)
    d_dopfn = dopfn_densities(cd, DoPFNRegressor, y_min=y_min, y_rng=y_rng,
                                dopfn_root=args.dopfn, n_context=args.N)
    print(f'[run] DoPFN-bb ...', flush=True)
    d_bb = ours_densities(
        cd, model_bb, edges_np, J, bin_width, -1,
        y_min=y_min, y_rng=y_rng,
        malc_B=args.malc_B, malc_max_K=args.malc_max_K, n_eval=args.n_eval,
        n_context=args.N,
        fit_malc_inner=fit_malc_inner, dmalc_2d=dmalc_2d,
        y_scaling='min_max',
    )

    raw_Y = y_min + (Y_CENTERS + 1.0) * y_rng / 2.0
    raw_TAU = TAU_CENTERS * y_rng / 2.0
    scale_y = 2.0 / y_rng

    def _to_ygrid(p_scaled, is_tau=False):
        src = raw_TAU if is_tau else raw_Y
        p_raw = np.interp(TAU_GRID if is_tau else Y_GRID, src, p_scaled * scale_y,
                          left=0.0, right=0.0)
        dx = TAU_DX if is_tau else Y_DX
        s = p_raw.sum() * dx
        return p_raw / s if s > 0 else p_raw

    q = args.q
    # Compute ATE for each method via barycenter of per-q τ densities
    def _ate_of(d_out):
        return wass_bary_of_grid(
            np.stack([_to_ygrid(d_out['p_tau'][k], is_tau=True) for k in range(args.n_test)]),
            TAU_GRID, wasserstein_barycenter_1d)

    p_ate_dopfn = _ate_of(d_dopfn)
    p_ate_bb    = _ate_of(d_bb)

    # 4-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        ('p(Y|do(0)) — query q=%d' % q, axes[0,0], Y_GRID,
            _to_ygrid(d_dopfn['p_y0'][q]), _to_ygrid(d_bb['p_y0'][q]), p_y0_true[q]),
        ('p(Y|do(1)) — query q=%d' % q, axes[0,1], Y_GRID,
            _to_ygrid(d_dopfn['p_y1'][q]), _to_ygrid(d_bb['p_y1'][q]), p_y1_true[q]),
        ('p(τ) — query q=%d' % q, axes[1,0], TAU_GRID,
            _to_ygrid(d_dopfn['p_tau'][q], is_tau=True), _to_ygrid(d_bb['p_tau'][q], is_tau=True), p_tau_true[q]),
        ('p(ATE) — barycenter across q', axes[1,1], TAU_GRID,
            p_ate_dopfn, p_ate_bb, p_ate_true),
    ]
    for title, ax, grid, dpfn, bb, truth in panels:
        ax.plot(grid, truth, 'k--', lw=2, label='truth', alpha=0.85)
        ax.plot(grid, dpfn, color='#8A4FBE', lw=1.5, label='Do-PFN')
        ax.plot(grid, bb,   color='#0F8A3C', lw=1.5, label='DoPFN-bb (MALC-LOGLIN)')
        ax.set_title(title)
        ax.set_xlabel('value'); ax.set_ylabel('density')
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    fig.suptitle(f'Density test — polynomial SCM d={args.d} N={args.N} seed={args.seed}',
                 y=1.005, fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'[save] {args.out}', flush=True)


if __name__ == '__main__':
    main()
