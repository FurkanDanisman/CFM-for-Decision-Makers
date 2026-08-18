"""Small controlled density-L2 test: polynomial SCM at fixed d, N vs known
analytic truth. Compares only Do-PFN and DoPFN-bb on:
    p(Y_do0), p(Y_do1), p(τ), p(ATE)

Reuses fig2_pehe_l2.py's SCM generator + analytic truth so we don't fork
the math. Prints per-metric mean ± SEM across n_seeds SCMs.

Purpose: isolate whether DoPFN-bb loses to Do-PFN on density in general
(→ J=10 limit, retrain at higher J) or only on IHDP (→ IHDP-specific
issue like the y1 asymmetry we saw).

Usage:
    python density_test_small.py \\
        --repo $DEPLOY_ROOT/R-PFN \\
        --dopfn $DEPLOY_ROOT/external/dopfn \\
        --checkpoint-dopfn-bb $DEPLOY_ROOT/checkpoints_dopfn_backbone_realj10/step_150000.pt \\
        --d 3 --N 200 --n-seeds 15
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--dopfn', required=True)
    ap.add_argument('--checkpoint-dopfn-bb', required=True)
    ap.add_argument('--d', type=int, default=3)
    ap.add_argument('--N', type=int, default=200,
                     help='Context size per SCM.')
    ap.add_argument('--n-test', type=int, default=25,
                     help='Test queries per SCM (for per-query densities).')
    ap.add_argument('--n-seeds', type=int, default=15)
    ap.add_argument('--rho', type=float, default=0.0,
                     help='True DGP ρ between Y0 and Y1 residuals.')
    ap.add_argument('--sigma-eps', type=float, default=1.0)
    ap.add_argument('--degree', type=int, default=3,
                     help='Polynomial degree of the SCM.')
    ap.add_argument('--malc-B', type=int, default=60)
    ap.add_argument('--malc-max-K', type=int, default=3)
    ap.add_argument('--n-eval', type=int, default=200)
    args = ap.parse_args()

    # ── sys.path setup ────────────────────────────────────────────────
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'empirical_tests'))
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks'))
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'l2_ihdp'))
    sys.path.insert(0, os.path.join(args.repo, 'training_dopfn_base'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC', 'Optimal_Transport'))
    sys.path.insert(0, args.repo)

    import torch
    from fig2_pehe_l2 import (
        make_polynomial_scm, truth_marginals, truth_cate,
        Y_GRID, TAU_GRID, Y_DX, TAU_DX, l2_1d,
        _to_common_grid, _discrete_to_density, wass_bary_of_grid,
    )
    from ot_barycenter import wasserstein_barycenter_1d
    from dopfn_backbone_head import DoPFNBackboneWith2DHead
    from losses.BarDistribution2D import fit_malc_inner, unpack_pred
    from malc_2d import dmalc_2d
    from methods_densities import ours_densities, dopfn_densities
    from dopfn_helpers import load_dopfn

    # ── Build DoPFN-bb model ONCE ─────────────────────────────────────
    print(f'[load] DoPFN-bb from {args.checkpoint_dopfn_bb}', flush=True)
    ckpt = torch.load(args.checkpoint_dopfn_bb, map_location='cpu', weights_only=False)
    cfg = ckpt['config']; J = cfg['J']
    edges_np = ckpt['edges'].cpu().numpy()
    bin_width = float(edges_np[1] - edges_np[0])
    model_bb = DoPFNBackboneWith2DHead(dopfn_root=args.dopfn, K=J).eval()
    model_bb.load_state_dict(ckpt['model_state_dict'])

    DoPFNRegressor = load_dopfn(args)

    # ── Accumulators (per-seed averages, then agg across seeds) ───────
    # Four ours variants: MALC-OLD (resample_onto), MALC-LOGLIN (log-linear),
    # RAW (no MALC), plus Do-PFN reference. Same p_tau across all ours
    # variants (2D MALC diagonal integration doesn't depend on marginal path).
    METHOD_NAMES = ['dopfn', 'bb_malc_old', 'bb_malc_loglin', 'bb_raw']
    l2_y0 = {m: [] for m in METHOD_NAMES}
    l2_y1 = {m: [] for m in METHOD_NAMES}
    l2_tau = {m: [] for m in METHOD_NAMES}
    l2_ate = {m: [] for m in METHOD_NAMES}

    print(f'[cfg] d={args.d}  N={args.N}  n_test={args.n_test}  '
          f'n_seeds={args.n_seeds}  rho={args.rho}  sigma_eps={args.sigma_eps}', flush=True)
    print()

    for seed in range(args.n_seeds):
        t0 = time.time()
        cd = make_polynomial_scm(seed=seed, n_context=args.N, n_test=args.n_test,
                                  rho_eff=min(args.rho, 0.99),
                                  x_dim=args.d, degree=args.degree,
                                  sigma_eps=args.sigma_eps)
        # Analytic truth on the shared Y_GRID / TAU_GRID (from fig2)
        p_y0_true = truth_marginals(cd)[0]   # (n_test, len(Y_GRID))
        p_y1_true = truth_marginals(cd)[1]
        p_tau_true = truth_cate(cd)          # (n_test, len(TAU_GRID))
        p_ate_true = wass_bary_of_grid(p_tau_true, TAU_GRID, wasserstein_barycenter_1d)

        # ── Do-PFN ────────────────────────────────────────────────────
        # dopfn_densities uses Y_CENTERS internally; we need to remap to Y_GRID.
        # Simplest: call it, get its output on IHDP Y_CENTERS ([-1.5, 1.5],
        # 100 pts), and interp to fig2's Y_GRID ([-8, 8], 501 pts).
        # But the SCM's Y range depends on the poly; use fig2's convention
        # of unscaled truth on Y_GRID and let dopfn_densities operate on
        # its own raw-Y scale (y_min/y_rng match cd's actual Y range).
        y_ctx_np = cd.y_train.numpy()
        y_min = float(y_ctx_np.min()); y_max = float(y_ctx_np.max())
        y_rng = max(y_max - y_min, 1e-6)
        try:
            d_dopfn = dopfn_densities(cd, DoPFNRegressor, y_min=y_min, y_rng=y_rng,
                                        dopfn_root=args.dopfn, n_context=args.N)
        except Exception as e:
            print(f'[warn] seed={seed} dopfn failed: {type(e).__name__}: {e}', flush=True)
            continue

        # ── Ours DoPFN-bb ─────────────────────────────────────────────
        try:
            d_bb = ours_densities(
                cd, model_bb, edges_np, J, bin_width, -1,
                y_min=y_min, y_rng=y_rng,
                malc_B=args.malc_B, malc_max_K=args.malc_max_K, n_eval=args.n_eval,
                n_context=args.N,
                fit_malc_inner=fit_malc_inner, dmalc_2d=dmalc_2d,
                y_scaling='min_max',
            )
        except Exception as e:
            print(f'[warn] seed={seed} dopfn_bb failed: {type(e).__name__}: {e}', flush=True)
            continue

        # dopfn_densities / ours_densities return densities on Y_CENTERS
        # (IHDP-scaled space). We need to unscale to raw Y and interp to Y_GRID
        # for comparison against truth. The scaled→raw factor is (y_rng/2)
        # for density (since Y_raw = y_min + (Y_scaled+1)*y_rng/2).
        from true_ihdp import Y_CENTERS, TAU_CENTERS
        # Scaled Y_CENTERS ∈ [-1.5, 1.5] → raw Y ∈ [y_min - y_rng*0.25, y_min + y_rng*1.25]
        raw_Y_from_ycenters = y_min + (Y_CENTERS + 1.0) * y_rng / 2.0
        raw_TAU_from_tcenters = TAU_CENTERS * y_rng / 2.0
        scale_factor_y = 2.0 / y_rng   # density transforms inversely to bin width

        # Method variants: each gives its own (p_y0, p_y1); τ / ATE share
        # d_bb's p_tau across bb variants (2D-MALC diagonal integration is
        # unchanged by marginal path). Do-PFN uses its own p_tau.
        variants = [
            ('dopfn',           d_dopfn['p_y0'],         d_dopfn['p_y1'],         d_dopfn['p_tau']),
            ('bb_malc_old',     d_bb['p_y0_malc_old'],   d_bb['p_y1_malc_old'],   d_bb['p_tau']),
            ('bb_malc_loglin',  d_bb['p_y0'],            d_bb['p_y1'],            d_bb['p_tau']),
            ('bb_raw',          d_bb['p_y0_raw'],        d_bb['p_y1_raw'],        d_bb['p_tau']),
        ]
        for method_name, m_p_y0, m_p_y1, m_p_tau in variants:
            per_q_l2_y0 = []
            per_q_l2_y1 = []
            per_q_l2_tau = []
            for q in range(args.n_test):
                # Interp method's density on Y_CENTERS-derived raw grid onto Y_GRID
                py0_raw = np.interp(Y_GRID, raw_Y_from_ycenters,
                                     m_p_y0[q] * scale_factor_y,
                                     left=0.0, right=0.0)
                py1_raw = np.interp(Y_GRID, raw_Y_from_ycenters,
                                     m_p_y1[q] * scale_factor_y,
                                     left=0.0, right=0.0)
                # Renormalise on Y_GRID
                s0 = py0_raw.sum() * Y_DX; py0_raw = py0_raw / s0 if s0 > 0 else py0_raw
                s1 = py1_raw.sum() * Y_DX; py1_raw = py1_raw / s1 if s1 > 0 else py1_raw
                per_q_l2_y0.append(l2_1d(py0_raw, p_y0_true[q], Y_DX))
                per_q_l2_y1.append(l2_1d(py1_raw, p_y1_true[q], Y_DX))
                # τ
                ptau_raw = np.interp(TAU_GRID, raw_TAU_from_tcenters,
                                      m_p_tau[q] * scale_factor_y,
                                      left=0.0, right=0.0)
                st = ptau_raw.sum() * TAU_DX
                if st > 0: ptau_raw = ptau_raw / st
                per_q_l2_tau.append(l2_1d(ptau_raw, p_tau_true[q], TAU_DX))
            # ATE via barycenter of per-query τ densities
            p_ate_hat = wass_bary_of_grid(
                np.stack([np.interp(TAU_GRID, raw_TAU_from_tcenters,
                                     m_p_tau[q] * scale_factor_y,
                                     left=0.0, right=0.0)
                          for q in range(args.n_test)]),
                TAU_GRID, wasserstein_barycenter_1d)
            l2_y0[method_name].append(np.mean(per_q_l2_y0))
            l2_y1[method_name].append(np.mean(per_q_l2_y1))
            l2_tau[method_name].append(np.mean(per_q_l2_tau))
            l2_ate[method_name].append(l2_1d(p_ate_hat, p_ate_true, TAU_DX))

        dt = time.time() - t0
        print(f'  seed={seed:2d}  done in {dt:.1f}s', flush=True)
        for m in METHOD_NAMES:
            print(f'    {m:18s}  y0={l2_y0[m][-1]:.3f}  y1={l2_y1[m][-1]:.3f}  '
                  f'tau={l2_tau[m][-1]:.3f}  ate={l2_ate[m][-1]:.3f}', flush=True)

    # ── Aggregate mean ± SEM ──────────────────────────────────────────
    def _stat(vs):
        arr = np.asarray(vs, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0: return 'na', 'na', 0
        m = arr.mean()
        sem = arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else 0.0
        return f'{m:.4f}', f'{sem:.4f}', arr.size

    print()
    print(f'══ Density L2 — polynomial SCM  d={args.d}  N={args.N}  '
          f'ρ={args.rho}  n_seeds={args.n_seeds} ══')
    LABELS = {
        'dopfn':          'Do-PFN',
        'bb_malc_old':    'DoPFN-bb MALC-OLD',
        'bb_malc_loglin': 'DoPFN-bb MALC-LOGLIN',
        'bb_raw':         'DoPFN-bb RAW',
    }
    header = f'{"metric":<6s}  ' + '  '.join(f'{LABELS[m]:>22s}' for m in METHOD_NAMES)
    print(header)
    print('-' * len(header))
    for metric_name, d_dict in [('y0', l2_y0), ('y1', l2_y1),
                                   ('tau', l2_tau), ('ate', l2_ate)]:
        row = f'{metric_name:<6s}  '
        best_m, best_v = None, float('inf')
        cells = {}
        for m in METHOD_NAMES:
            mn, sem, n = _stat(d_dict[m])
            cells[m] = (mn, sem, n)
            try:
                v = float(mn)
                if v < best_v: best_v, best_m = v, m
            except Exception:
                pass
        for m in METHOD_NAMES:
            mn, sem, _ = cells[m]
            marker = '*' if m == best_m else ' '
            cell = f'{marker}{mn:>10s} ± {sem:<8s}'
            row += f'  {cell:>22s}'
        print(row)
    print()
    print('(* marks the lowest L2 in each row)')


if __name__ == '__main__':
    main()
