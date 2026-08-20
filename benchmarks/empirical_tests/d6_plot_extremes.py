"""d=6 (ihdp_like) — save all per-query densities, then plot the extreme
cases where one method beats the other by the largest margin on CATE τ.

Two figures written:
  - `<out>_dopfn_wins.png`  — 4-panel: seed / query where Do-PFN beats BB most on τ
  - `<out>_bb_wins.png`     — 4-panel: seed / query where BB beats Do-PFN most on τ

Panels: p(Y_do0), p(Y_do1), p(τ|x), p(τ_ATE) — truth, Do-PFN, and BB overlaid.

Densities are cached to <out>_densities.npz so re-plotting is free.

Usage:
    python benchmarks/empirical_tests/d6_plot_extremes.py \\
      --repo $DEPLOY_ROOT/R-PFN --dopfn $DEPLOY_ROOT/external/dopfn \\
      --checkpoint-dopfn-bb $CKPT_BB \\
      --n-seeds 15 --n-test 25 --malc-B 1000 \\
      --out $DEPLOY_ROOT/d6_extremes
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
    ap.add_argument('--sigma-eps', type=float, default=1.0)
    ap.add_argument('--degree', type=int, default=3)
    ap.add_argument('--malc-B', type=int, default=1000)
    ap.add_argument('--malc-max-K', type=int, default=1)
    ap.add_argument('--n-eval', type=int, default=200)
    ap.add_argument('--y-scale', type=float, default=0.5)
    ap.add_argument('--tau-base', type=float, default=5.0)
    ap.add_argument('--tau-scale', type=float, default=0.5)
    ap.add_argument('--out', default='/tmp/d6_extremes')
    ap.add_argument('--recompute', action='store_true',
                     help='Ignore any cached densities npz and re-run the sweep.')
    ap.add_argument('--plot-all', action='store_true',
                     help='In addition to the two extreme plots, save one 4-panel PNG '
                          'per (seed, query) pair as <out>_seed{S}_q{Q}.png')
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'empirical_tests'))
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'l2_ihdp'))
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks'))
    sys.path.insert(0, os.path.join(args.repo, 'training_dopfn_base'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC', 'Optimal_Transport'))
    sys.path.insert(0, args.repo)

    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from fig2_pehe_l2 import make_ihdp_like_scm
    from ot_barycenter import wasserstein_barycenter_1d
    from dopfn_backbone_head import DoPFNBackboneWith2DHead
    from losses.BarDistribution2D import fit_malc_inner
    from malc_2d import dmalc_2d
    from methods_densities import ours_densities, dopfn_densities
    from dopfn_helpers import load_dopfn
    from true_ihdp import Y_CENTERS as YC

    cache = f'{args.out}_densities.npz'
    if os.path.exists(cache) and not args.recompute:
        print(f'[cache] loading {cache}', flush=True)
        d = np.load(cache, allow_pickle=True)
        Y_GRID     = d['Y_GRID']
        TAU_GRID   = d['TAU_GRID']
        mu0_all    = d['mu0_all']       # (n_seeds, n_test)
        mu1_all    = d['mu1_all']
        sig_all    = d['sig_all']       # (n_seeds,)
        p_y0_dopfn = d['p_y0_dopfn']    # (n_seeds, n_test, len(Y_GRID))
        p_y1_dopfn = d['p_y1_dopfn']
        p_ta_dopfn = d['p_ta_dopfn']    # (n_seeds, n_test, len(TAU_GRID))
        p_y0_bb    = d['p_y0_bb']
        p_y1_bb    = d['p_y1_bb']
        p_ta_bb    = d['p_ta_bb']
        p_ate_dopfn = d['p_ate_dopfn']  # (n_seeds, len(TAU_GRID))
        p_ate_bb    = d['p_ate_bb']
    else:
        # Load models
        ckpt = torch.load(args.checkpoint_dopfn_bb, map_location='cpu', weights_only=False)
        cfg = ckpt['config']; J = int(cfg['J'])
        edges_np = ckpt['edges'].cpu().numpy()
        bin_width = float(edges_np[1] - edges_np[0])
        model_bb = DoPFNBackboneWith2DHead(dopfn_root=args.dopfn, K=J).eval()
        model_bb.load_state_dict(ckpt['model_state_dict'])
        DoPFNRegressor = load_dopfn(args)

        # Shared plotting grid
        Y_GRID   = np.linspace(-1.5, 1.5, 601)
        TAU_GRID = np.linspace(-2.0, 2.0, 801)
        Y_CENTERS = np.asarray(YC)
        Y_BIN     = float(Y_CENTERS[1] - Y_CENTERS[0])
        TAU_FINE_C_len = 600
        TAU_FINE_C = 0.5 * (np.linspace(-3.0, 3.0, 601)[:-1] + np.linspace(-3.0, 3.0, 601)[1:])
        TAU_FINE_BIN = float(TAU_FINE_C[1] - TAU_FINE_C[0])

        mu0_all = np.zeros((args.n_seeds, args.n_test))
        mu1_all = np.zeros((args.n_seeds, args.n_test))
        sig_all = np.zeros(args.n_seeds)
        p_y0_dopfn = np.zeros((args.n_seeds, args.n_test, len(Y_GRID)))
        p_y1_dopfn = np.zeros((args.n_seeds, args.n_test, len(Y_GRID)))
        p_ta_dopfn = np.zeros((args.n_seeds, args.n_test, len(TAU_GRID)))
        p_y0_bb    = np.zeros((args.n_seeds, args.n_test, len(Y_GRID)))
        p_y1_bb    = np.zeros((args.n_seeds, args.n_test, len(Y_GRID)))
        p_ta_bb    = np.zeros((args.n_seeds, args.n_test, len(TAU_GRID)))
        p_ate_dopfn = np.zeros((args.n_seeds, len(TAU_GRID)))
        p_ate_bb    = np.zeros((args.n_seeds, len(TAU_GRID)))

        for seed in range(args.n_seeds):
            t0 = time.time()
            cd = make_ihdp_like_scm(seed=seed, n_context=args.N, n_test=args.n_test,
                                     x_dim=args.d, degree=args.degree,
                                     sigma_eps=args.sigma_eps,
                                     y_scale=args.y_scale, tau_base=args.tau_base,
                                     tau_scale=args.tau_scale)
            y_ctx = cd.y_train.numpy() if hasattr(cd.y_train, 'numpy') else np.asarray(cd.y_train)
            y_min = float(y_ctx.min()); y_rng = max(float(y_ctx.max()) - y_min, 1e-6)
            mu0_raw = np.asarray(cd._mu0_test).reshape(-1)
            mu1_raw = np.asarray(cd._mu1_test).reshape(-1)
            sigma_sc = float(cd._sigma_eps) * 2.0 / y_rng
            mu0 = (mu0_raw - y_min) / y_rng * 2.0 - 1.0
            mu1 = (mu1_raw - y_min) / y_rng * 2.0 - 1.0
            mu0_all[seed] = mu0; mu1_all[seed] = mu1; sig_all[seed] = sigma_sc

            d_dopfn = dopfn_densities(cd, DoPFNRegressor, y_min=y_min, y_rng=y_rng,
                                        dopfn_root=args.dopfn, n_context=args.N,
                                        edges_j10_scaled=edges_np)
            d_bb = ours_densities(cd, model_bb, edges_np, J, bin_width, -1,
                                    y_min=y_min, y_rng=y_rng,
                                    malc_B=args.malc_B, malc_max_K=args.malc_max_K,
                                    n_eval=args.n_eval, n_context=args.N,
                                    fit_malc_inner=fit_malc_inner, dmalc_2d=dmalc_2d,
                                    y_scaling='min_max', marginals_from_2d=True)
            # Interp densities onto shared plotting grid
            def _to_Y(d_native, native_grid=Y_CENTERS):
                out = np.interp(Y_GRID, native_grid, d_native, left=0, right=0)
                s = out.sum() * (Y_GRID[1]-Y_GRID[0])
                return out / s if s > 0 else out
            def _to_TAU(d_native):
                out = np.interp(TAU_GRID, TAU_FINE_C, d_native, left=0, right=0)
                s = out.sum() * (TAU_GRID[1]-TAU_GRID[0])
                return out / s if s > 0 else out
            for q in range(args.n_test):
                p_y0_dopfn[seed, q] = _to_Y(d_dopfn['p_y0'][q])
                p_y1_dopfn[seed, q] = _to_Y(d_dopfn['p_y1'][q])
                p_ta_dopfn[seed, q] = _to_TAU(d_dopfn['p_tau'][q])
                p_y0_bb[seed, q]    = _to_Y(d_bb['p_y0'][q])
                p_y1_bb[seed, q]    = _to_Y(d_bb['p_y1'][q])
                p_ta_bb[seed, q]    = _to_TAU(d_bb['p_tau'][q])
            # Per-seed ATE via Wasserstein barycentre of query-CATE densities
            dopfn_ta = np.stack([p_ta_dopfn[seed, q] for q in range(args.n_test)])
            bb_ta    = np.stack([p_ta_bb[seed, q]    for q in range(args.n_test)])
            p_ate_dopfn[seed] = wasserstein_barycenter_1d(dopfn_ta, TAU_GRID)
            p_ate_bb[seed]    = wasserstein_barycenter_1d(bb_ta,    TAU_GRID)
            print(f'  seed={seed:2d} done in {time.time()-t0:.1f}s', flush=True)

        np.savez_compressed(cache, Y_GRID=Y_GRID, TAU_GRID=TAU_GRID,
                              mu0_all=mu0_all, mu1_all=mu1_all, sig_all=sig_all,
                              p_y0_dopfn=p_y0_dopfn, p_y1_dopfn=p_y1_dopfn, p_ta_dopfn=p_ta_dopfn,
                              p_y0_bb=p_y0_bb,       p_y1_bb=p_y1_bb,       p_ta_bb=p_ta_bb,
                              p_ate_dopfn=p_ate_dopfn, p_ate_bb=p_ate_bb)
        print(f'\n[cache] saved {cache}', flush=True)

    # ─── Compute per-(seed, query) truth densities on the shared grid ───
    n_seeds, n_test = mu0_all.shape[:2]
    truth_y0 = np.zeros((n_seeds, n_test, len(Y_GRID)))
    truth_y1 = np.zeros((n_seeds, n_test, len(Y_GRID)))
    truth_ta = np.zeros((n_seeds, n_test, len(TAU_GRID)))
    truth_ate = np.zeros((n_seeds, len(TAU_GRID)))
    dy = float(Y_GRID[1] - Y_GRID[0])
    dt = float(TAU_GRID[1] - TAU_GRID[0])
    for s in range(n_seeds):
        sigma_sc = float(sig_all[s])
        sigma_tau = float(np.sqrt(2.0)) * sigma_sc
        per_q_ta = []
        for q in range(n_test):
            truth_y0[s, q] = norm.pdf(Y_GRID, mu0_all[s, q], sigma_sc)
            truth_y1[s, q] = norm.pdf(Y_GRID, mu1_all[s, q], sigma_sc)
            mu_tau = float(mu1_all[s, q] - mu0_all[s, q])
            truth_ta[s, q] = norm.pdf(TAU_GRID, mu_tau, sigma_tau)
            per_q_ta.append(truth_ta[s, q])
        truth_ate[s] = wasserstein_barycenter_1d(np.stack(per_q_ta), TAU_GRID)

    # ─── Distance functions: L2, W1, W2 ───────────────────────
    def _normalize(p, dx):
        s = p.sum() * dx
        return (p / s) if s > 0 else p

    def _l1(p, t, dx):
        """L1 density distance = ∫|p - t| dy  (bounded [0, 2] for densities;
        equals 2·TotalVariation)."""
        pn, tn = _normalize(p, dx), _normalize(t, dx)
        return float(np.sum(np.abs(pn - tn)) * dx)

    def _l2(p, t, dx):
        pn, tn = _normalize(p, dx), _normalize(t, dx)
        return float(np.sqrt(np.sum((pn - tn)**2) * dx))

    def _w1(p, t, grid):
        """W1 = ∫ |CDF_p(y) - CDF_t(y)| dy  (earth-mover on 1D)."""
        dx = float(grid[1] - grid[0])
        pn, tn = _normalize(p, dx), _normalize(t, dx)
        Fp = np.cumsum(pn) * dx
        Ft = np.cumsum(tn) * dx
        return float(np.sum(np.abs(Fp - Ft)) * dx)

    _U = np.linspace(1e-4, 1 - 1e-4, 2000)
    def _w2(p, t, grid):
        """W2 = L2 of inverse CDFs (quantile functions)."""
        dx = float(grid[1] - grid[0])
        pn, tn = _normalize(p, dx), _normalize(t, dx)
        Fp = np.cumsum(pn) * dx; Fp = np.clip(Fp / max(Fp[-1], 1e-12), 0, 1)
        Ft = np.cumsum(tn) * dx; Ft = np.clip(Ft / max(Ft[-1], 1e-12), 0, 1)
        Fp_inv = np.interp(_U, Fp, grid)
        Ft_inv = np.interp(_U, Ft, grid)
        return float(np.sqrt(np.mean((Fp_inv - Ft_inv) ** 2)))

    # ─── Per-query metrics for all four density families ───
    metrics = {name: {who: {'y0': np.zeros((n_seeds, n_test)),
                              'y1': np.zeros((n_seeds, n_test)),
                              'tau': np.zeros((n_seeds, n_test)),
                              'ate': np.zeros(n_seeds)}
                       for who in ('dopfn', 'bb')}
                 for name in ('L1', 'L2', 'W1', 'W2')}

    for s in range(n_seeds):
        for q in range(n_test):
            for who, pY0, pY1, pTa in [('dopfn', p_y0_dopfn, p_y1_dopfn, p_ta_dopfn),
                                          ('bb',    p_y0_bb,    p_y1_bb,    p_ta_bb)]:
                metrics['L1'][who]['y0'][s, q]  = _l1(pY0[s, q], truth_y0[s, q], dy)
                metrics['L1'][who]['y1'][s, q]  = _l1(pY1[s, q], truth_y1[s, q], dy)
                metrics['L1'][who]['tau'][s, q] = _l1(pTa[s, q], truth_ta[s, q], dt)
                metrics['L2'][who]['y0'][s, q]  = _l2(pY0[s, q], truth_y0[s, q], dy)
                metrics['L2'][who]['y1'][s, q]  = _l2(pY1[s, q], truth_y1[s, q], dy)
                metrics['L2'][who]['tau'][s, q] = _l2(pTa[s, q], truth_ta[s, q], dt)
                metrics['W1'][who]['y0'][s, q]  = _w1(pY0[s, q], truth_y0[s, q], Y_GRID)
                metrics['W1'][who]['y1'][s, q]  = _w1(pY1[s, q], truth_y1[s, q], Y_GRID)
                metrics['W1'][who]['tau'][s, q] = _w1(pTa[s, q], truth_ta[s, q], TAU_GRID)
                metrics['W2'][who]['y0'][s, q]  = _w2(pY0[s, q], truth_y0[s, q], Y_GRID)
                metrics['W2'][who]['y1'][s, q]  = _w2(pY1[s, q], truth_y1[s, q], Y_GRID)
                metrics['W2'][who]['tau'][s, q] = _w2(pTa[s, q], truth_ta[s, q], TAU_GRID)
        for who, pAte in [('dopfn', p_ate_dopfn), ('bb', p_ate_bb)]:
            metrics['L1'][who]['ate'][s] = _l1(pAte[s], truth_ate[s], dt)
            metrics['L2'][who]['ate'][s] = _l2(pAte[s], truth_ate[s], dt)
            metrics['W1'][who]['ate'][s] = _w1(pAte[s], truth_ate[s], TAU_GRID)
            metrics['W2'][who]['ate'][s] = _w2(pAte[s], truth_ate[s], TAU_GRID)

    # Keep legacy tau_L2_* names for the "who wins" gap below
    tau_L2_dopfn = metrics['L2']['dopfn']['tau']
    tau_L2_bb    = metrics['L2']['bb']['tau']

    # ─── Per-seed summary tables (one per metric) ───
    for mname in ('L1', 'L2', 'W1', 'W2'):
        print(f'\nper-seed mean {mname} — Do-PFN | Do-PFN-bb (2D-marg, B=1000)')
        print(f'  {"seed":>4s}  '
              f'{"y0 Do-PFN":>10s} {"y0 BB":>10s}   '
              f'{"y1 Do-PFN":>10s} {"y1 BB":>10s}   '
              f'{"τ  Do-PFN":>10s} {"τ  BB":>10s}   '
              f'{"ATE Do-PFN":>10s} {"ATE BB":>10s}')
        d = metrics[mname]
        for s in range(n_seeds):
            print(f'  {s:>4d}  '
                  f'{d["dopfn"]["y0"][s].mean():>10.4f} {d["bb"]["y0"][s].mean():>10.4f}   '
                  f'{d["dopfn"]["y1"][s].mean():>10.4f} {d["bb"]["y1"][s].mean():>10.4f}   '
                  f'{d["dopfn"]["tau"][s].mean():>10.4f} {d["bb"]["tau"][s].mean():>10.4f}   '
                  f'{d["dopfn"]["ate"][s]:>10.4f} {d["bb"]["ate"][s]:>10.4f}')

    # ─── Pick extremes on τ ────────────────────────────────────
    gap = tau_L2_bb - tau_L2_dopfn      # >0 means Do-PFN wins on τ
    idx_dopfn_wins = np.unravel_index(np.argmax(gap), gap.shape)
    idx_bb_wins    = np.unravel_index(np.argmin(gap), gap.shape)
    print(f'\n[Do-PFN wins]  seed={idx_dopfn_wins[0]}  query={idx_dopfn_wins[1]}  '
          f'τ_L2: Do-PFN={tau_L2_dopfn[idx_dopfn_wins]:.4f}  BB={tau_L2_bb[idx_dopfn_wins]:.4f}')
    print(f'[BB wins]      seed={idx_bb_wins[0]}  query={idx_bb_wins[1]}  '
          f'τ_L2: Do-PFN={tau_L2_dopfn[idx_bb_wins]:.4f}  BB={tau_L2_bb[idx_bb_wins]:.4f}')

    # ─── Plot ───
    def _plot(seed, q, tag, outpng):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        panels = [
            ('p(Y_do(0))', Y_GRID, truth_y0[seed, q], p_y0_dopfn[seed, q], p_y0_bb[seed, q], 'scaled Y'),
            ('p(Y_do(1))', Y_GRID, truth_y1[seed, q], p_y1_dopfn[seed, q], p_y1_bb[seed, q], 'scaled Y'),
            ('p(τ|x)  (CATE)',      TAU_GRID, truth_ta[seed, q], p_ta_dopfn[seed, q], p_ta_bb[seed, q], 'scaled τ'),
            ('p(τ_ATE) (seed avg)', TAU_GRID, truth_ate[seed],    p_ate_dopfn[seed],    p_ate_bb[seed],    'scaled τ'),
        ]
        for ax, (title, xg, tr, dp, bb, xl) in zip(axes.ravel(), panels):
            ax.plot(xg, tr, color='k',       lw=2.0, label='truth')
            ax.plot(xg, dp, color='#1f77b4', lw=1.8, label='Do-PFN')
            ax.plot(xg, bb, color='#d62728', lw=1.8, label='Do-PFN-bb (B=1000 2D-marg)')
            ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel('density')
            ax.legend(fontsize=8, frameon=False)
            ax.axvline(0.0, color='k', lw=0.5, alpha=0.3)
        fig.suptitle(f'{tag}  —  seed={seed}, query={q}  '
                      f'(τ L2: Do-PFN={tau_L2_dopfn[seed, q]:.4f}, '
                      f'BB={tau_L2_bb[seed, q]:.4f})', fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(outpng, dpi=140)
        print(f'[saved] {outpng}')

    _plot(idx_dopfn_wins[0], idx_dopfn_wins[1], 'Do-PFN wins on τ', f'{args.out}_dopfn_wins.png')
    _plot(idx_bb_wins[0],    idx_bb_wins[1],    'BB wins on τ',     f'{args.out}_bb_wins.png')

    if args.plot_all:
        for s in range(n_seeds):
            for q in range(n_test):
                _plot(s, q, f'seed={s} query={q}',
                      f'{args.out}_seed{s}_q{q}.png')


if __name__ == '__main__':
    main()
