"""d=6 polynomial-SCM density-L2 sweep matching the IHDP/ACIC LEFT-edge
convention. For each seed, runs Do-PFN + DoPFN-bb, computes L2 at J=10
and J=100 resolutions for y0/y1 (LEFT-edge) and τ/ATE (center).

Reports mean±SEM pooled across n_seeds SCMs × n_test queries.

Usage:
    python d6_l2_left_edge_sweep.py --repo $DEPLOY_ROOT/R-PFN \\
        --dopfn $DEPLOY_ROOT/external/dopfn \\
        --checkpoint-dopfn-bb $DEPLOY_ROOT/checkpoints_dopfn_backbone_realj10/step_150000.pt \\
        --n-seeds 15
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
    ap.add_argument('--n-seeds', type=int, default=15)
    ap.add_argument('--rho', type=float, default=0.0)
    ap.add_argument('--sigma-eps', type=float, default=1.0)
    ap.add_argument('--degree', type=int, default=3)
    ap.add_argument('--malc-B', type=int, default=100)
    ap.add_argument('--malc-max-K', type=int, default=1)
    ap.add_argument('--n-eval', type=int, default=200)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'empirical_tests'))
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks'))
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'l2_ihdp'))
    sys.path.insert(0, os.path.join(args.repo, 'training_dopfn_base'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC', 'Optimal_Transport'))
    sys.path.insert(0, args.repo)

    import torch
    from fig2_pehe_l2 import (make_polynomial_scm, truth_marginals, truth_cate,
                                Y_GRID, TAU_GRID, Y_DX, TAU_DX,
                                wass_bary_of_grid)
    from ot_barycenter import wasserstein_barycenter_1d
    from dopfn_backbone_head import DoPFNBackboneWith2DHead
    from losses.BarDistribution2D import fit_malc_inner
    from malc_2d import dmalc_2d
    from methods_densities import ours_densities, dopfn_densities
    from dopfn_helpers import load_dopfn
    from true_ihdp import Y_CENTERS as IHDP_YC, TAU_CENTERS as IHDP_TC

    print(f'[load] DoPFN-bb {args.checkpoint_dopfn_bb}', flush=True)
    ckpt = torch.load(args.checkpoint_dopfn_bb, map_location='cpu', weights_only=False)
    cfg = ckpt['config']; J = cfg['J']
    edges_np = ckpt['edges'].cpu().numpy()
    bin_width = float(edges_np[1] - edges_np[0])
    model_bb = DoPFNBackboneWith2DHead(dopfn_root=args.dopfn, K=J).eval()
    model_bb.load_state_dict(ckpt['model_state_dict'])
    DoPFNRegressor = load_dopfn(args)

    Y_grid_np = np.asarray(Y_GRID);  TAU_grid_np = np.asarray(TAU_GRID)
    Y_DX_np = float(Y_grid_np[1] - Y_grid_np[0]); TAU_DX_np = float(TAU_grid_np[1] - TAU_grid_np[0])
    # J=10 edges in raw-Y space depend on per-seed y_min/y_rng; we recompute per seed
    def l2_dx(p, q, dx): return float(np.sqrt(np.sum((np.asarray(p)-np.asarray(q))**2)*dx))

    METRICS = ['y0_j100','y1_j100','tau_j100','ate_j100',
               'y0_j10','y1_j10','tau_j10','ate_j10']
    LABELS = ['Do-PFN','BB LOGLIN','BB RAW','BB OLD']
    acc = {lbl: {m: [] for m in METRICS} for lbl in LABELS}

    print(f'[cfg] d={args.d} N={args.N} n_test={args.n_test} n_seeds={args.n_seeds}', flush=True)
    for seed in range(args.n_seeds):
        t0 = time.time()
        cd = make_polynomial_scm(seed=seed, n_context=args.N, n_test=args.n_test,
                                  rho_eff=min(args.rho, 0.99), x_dim=args.d,
                                  degree=args.degree, sigma_eps=args.sigma_eps)
        p_y0_true = truth_marginals(cd)[0]; p_y1_true = truth_marginals(cd)[1]
        p_tau_true = truth_cate(cd)
        p_ate_true = wass_bary_of_grid(p_tau_true, TAU_grid_np, wasserstein_barycenter_1d)

        y_ctx_np = cd.y_train.numpy()
        y_min = float(y_ctx_np.min()); y_rng = max(float(y_ctx_np.max()) - y_min, 1e-6)

        try:
            d_dopfn = dopfn_densities(cd, DoPFNRegressor, y_min=y_min, y_rng=y_rng,
                                        dopfn_root=args.dopfn, n_context=args.N)
            d_bb = ours_densities(cd, model_bb, edges_np, J, bin_width, -1,
                                    y_min=y_min, y_rng=y_rng,
                                    malc_B=args.malc_B, malc_max_K=args.malc_max_K,
                                    n_eval=args.n_eval, n_context=args.N,
                                    fit_malc_inner=fit_malc_inner, dmalc_2d=dmalc_2d,
                                    y_scaling='min_max')
        except Exception as e:
            print(f'  seed={seed} SKIP: {type(e).__name__}: {e}', flush=True); continue

        # Convert IHDP-scaled outputs onto raw Y_GRID / TAU_GRID
        raw_Y   = y_min + (np.asarray(IHDP_YC) + 1.0) * y_rng / 2.0
        raw_TAU = np.asarray(IHDP_TC) * y_rng / 2.0
        scale_y = 2.0 / y_rng
        def to_ygrid(p_scaled):
            p = np.interp(Y_grid_np, raw_Y, p_scaled*scale_y, left=0., right=0.)
            s = p.sum()*Y_DX_np; return p/s if s>0 else p
        def to_tgrid(p_scaled):
            p = np.interp(TAU_grid_np, raw_TAU, p_scaled*scale_y, left=0., right=0.)
            s = p.sum()*TAU_DX_np; return p/s if s>0 else p

        # J=10 grid on raw Y: use edges_np (scaled J=10) unscaled to raw Y
        edges_J10_raw = y_min + (edges_np + 1.0) * y_rng / 2.0
        tau_edges_J10_raw = edges_np * y_rng / 2.0   # τ = Y1-Y0 scale
        bin_w_J10_raw = edges_J10_raw[1] - edges_J10_raw[0]
        tau_bin_w_J10_raw = tau_edges_J10_raw[1] - tau_edges_J10_raw[0]
        shift_left = -bin_w_J10_raw / 2.0

        def shift_y(d):
            p = np.interp(Y_grid_np, Y_grid_np - shift_left, d, left=0., right=0.)
            s = p.sum()*Y_DX_np; return p/s if s>0 else p
        # τ = Y1 - Y0: LEFT shifts on Y0/Y1 cancel → no net τ shift. Keep center.
        shift_left_tau = 0.0
        def shift_tau(d):
            if shift_left_tau == 0.0: return d
            p = np.interp(TAU_grid_np, TAU_grid_np - shift_left_tau, d, left=0., right=0.)
            s = p.sum()*TAU_DX_np; return p/s if s>0 else p
        def to_j10_y(d):
            p_bin = np.zeros(J)
            for j in range(J):
                mask = (Y_grid_np >= edges_J10_raw[j]) & (Y_grid_np < edges_J10_raw[j+1])
                p_bin[j] = np.array(d)[mask].sum() * Y_DX_np
            t = p_bin.sum(); return p_bin/t if t>0 else p_bin
        def to_j10_tau(d):
            p_bin = np.zeros(J)
            for j in range(J):
                mask = (TAU_grid_np >= tau_edges_J10_raw[j]) & (TAU_grid_np < tau_edges_J10_raw[j+1])
                p_bin[j] = np.array(d)[mask].sum() * TAU_DX_np
            t = p_bin.sum(); return p_bin/t if t>0 else p_bin

        variants = [
            ('Do-PFN',    d_dopfn['p_y0'], d_dopfn['p_y1'], d_dopfn['p_tau']),
            ('BB LOGLIN', d_bb['p_y0'],    d_bb['p_y1'],    d_bb['p_tau']),
            ('BB RAW',    d_bb['p_y0_raw'], d_bb['p_y1_raw'], d_bb['p_tau']),
            ('BB OLD',    d_bb['p_y0_malc_old'], d_bb['p_y1_malc_old'], d_bb['p_tau']),
        ]

        for lbl, m_p_y0, m_p_y1, m_p_tau in variants:
            per_ate = wass_bary_of_grid(
                np.stack([to_tgrid(m_p_tau[q]) for q in range(args.n_test)]),
                TAU_grid_np, wasserstein_barycenter_1d)
            per_ate_L = shift_tau(per_ate)
            acc[lbl]['ate_j100'].append(l2_dx(per_ate_L, p_ate_true, TAU_DX_np))
            acc[lbl]['ate_j10'].append(l2_dx(to_j10_tau(per_ate_L)/tau_bin_w_J10_raw,
                                              to_j10_tau(p_ate_true)/tau_bin_w_J10_raw,
                                              tau_bin_w_J10_raw))
            for q in range(args.n_test):
                py0_raw = to_ygrid(m_p_y0[q]); py1_raw = to_ygrid(m_p_y1[q])
                ptau_raw = to_tgrid(m_p_tau[q])
                # LEFT-edge shift for marginals AND τ (raw-Y space, tau space)
                py0_L = shift_y(py0_raw); py1_L = shift_y(py1_raw)
                ptau_L = shift_tau(ptau_raw)
                acc[lbl]['y0_j100'].append(l2_dx(py0_L, p_y0_true[q], Y_DX_np))
                acc[lbl]['y1_j100'].append(l2_dx(py1_L, p_y1_true[q], Y_DX_np))
                acc[lbl]['y0_j10'].append(l2_dx(to_j10_y(py0_L)/bin_w_J10_raw,
                                                 to_j10_y(p_y0_true[q])/bin_w_J10_raw,
                                                 bin_w_J10_raw))
                acc[lbl]['y1_j10'].append(l2_dx(to_j10_y(py1_L)/bin_w_J10_raw,
                                                 to_j10_y(p_y1_true[q])/bin_w_J10_raw,
                                                 bin_w_J10_raw))
                acc[lbl]['tau_j100'].append(l2_dx(ptau_L, p_tau_true[q], TAU_DX_np))
                acc[lbl]['tau_j10'].append(l2_dx(to_j10_tau(ptau_L)/tau_bin_w_J10_raw,
                                                  to_j10_tau(p_tau_true[q])/tau_bin_w_J10_raw,
                                                  tau_bin_w_J10_raw))
        print(f'  seed={seed:2d}  done in {time.time()-t0:.1f}s', flush=True)

    def _agg(vs):
        arr = np.asarray(vs, dtype=np.float64); arr = arr[np.isfinite(arr)]
        if arr.size == 0: return 'na'
        m = arr.mean(); sem = arr.std(ddof=1)/np.sqrt(arr.size) if arr.size>1 else 0.
        return f'{m:.4f}±{sem:.4f}'

    print()
    print(f'══ d={args.d} — LEFT-edge y0/y1, center τ/ATE  '
          f'(n_seeds={args.n_seeds}) ══')
    header = f'{"variant":15s}'
    for m in METRICS: header += f'  {m:>16s}'
    print(header)
    for lbl in LABELS:
        row = f'{lbl:15s}'
        for m in METRICS: row += f'  {_agg(acc[lbl][m]):>16s}'
        print(row)


if __name__ == '__main__':
    main()
