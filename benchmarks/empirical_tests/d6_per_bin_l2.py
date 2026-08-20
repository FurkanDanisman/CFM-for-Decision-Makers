"""d=6 synthetic — per-bin probability L2 for Do-PFN and BB (B=1000, 2D-marg).

Identical recipe to benchmarks/l2_ihdp/l2_per_bin_prob.py, but on the
polynomial-SCM d=6 synthetic instead of IHDP. Only two rows:
  - Do-PFN [J=10]
  - Do-PFN-bb MALC (2D-τ) 2D-marg [B=1000, J=10]

Truth per-query is analytic Gaussian in raw Y (cd._mu{0,1}_test, cd._sigma_eps),
converted to scaled Y ([-1, 1]) via the same y_min/y_rng scaling the models use.
Truth per J=10 bin = Φ((e_{j+1}-μ)/σ) - Φ((e_j-μ)/σ). BB uses exact-CDF bin
probs (p_y{0,1}_bins_j10) direct from ours_densities; Do-PFN aggregates its
Y_CENTERS density onto J=10 bins.

Usage:
  python d6_per_bin_l2.py \\
    --repo $DEPLOY_ROOT/R-PFN \\
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
    ap.add_argument('--malc-B', type=int, default=1000)
    ap.add_argument('--malc-max-K', type=int, default=1)
    ap.add_argument('--n-eval', type=int, default=200)
    ap.add_argument('--flavor', choices=['polynomial', 'ihdp_like'], default='polynomial',
                     help='polynomial = default zero-mean-effect SCM; '
                          'ihdp_like = tuned to match IHDP truth density stats')
    ap.add_argument('--tau-base', type=float, default=5.0)
    ap.add_argument('--tau-scale', type=float, default=0.5)
    ap.add_argument('--y-scale', type=float, default=0.5)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'empirical_tests'))
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks'))
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'l2_ihdp'))
    sys.path.insert(0, os.path.join(args.repo, 'training_dopfn_base'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC', 'Optimal_Transport'))
    sys.path.insert(0, args.repo)

    import torch
    from scipy.stats import norm
    from fig2_pehe_l2 import make_polynomial_scm, make_ihdp_like_scm
    from ot_barycenter import wasserstein_barycenter_1d
    from dopfn_backbone_head import DoPFNBackboneWith2DHead
    from losses.BarDistribution2D import fit_malc_inner
    from malc_2d import dmalc_2d
    from methods_densities import ours_densities, dopfn_densities
    from dopfn_helpers import load_dopfn
    from true_ihdp import Y_CENTERS as YC

    print(f'[load] DoPFN-bb {args.checkpoint_dopfn_bb}', flush=True)
    ckpt = torch.load(args.checkpoint_dopfn_bb, map_location='cpu', weights_only=False)
    cfg = ckpt['config']; J = int(cfg['J'])
    edges_np = ckpt['edges'].cpu().numpy()             # scaled Y J=10 edges
    bin_width = float(edges_np[1] - edges_np[0])
    model_bb = DoPFNBackboneWith2DHead(dopfn_root=args.dopfn, K=J).eval()
    model_bb.load_state_dict(ckpt['model_state_dict'])
    DoPFNRegressor = load_dopfn(args)

    # J=10 evaluation grid in SCALED Y (matches IHDP recipe exactly).
    edges_Y   = edges_np                                # [-1, ..., 1]
    bin_w_Y   = float(edges_Y[1] - edges_Y[0])          # 0.2
    tau_min   = 2.0 * edges_Y[0]; tau_max = 2.0 * edges_Y[-1]
    n_tau_bins = int(round((tau_max - tau_min) / bin_w_Y))
    edges_tau = np.linspace(tau_min, tau_max, n_tau_bins + 1)
    bin_w_tau = float(edges_tau[1] - edges_tau[0])      # = 0.2

    Y_CENTERS = np.asarray(YC)                          # 100 midpts on [-1.5, 1.5]
    Y_BIN     = float(Y_CENTERS[1] - Y_CENTERS[0])

    # τ fine grid used for barycenter aggregation and ATE integration.
    TAU_FINE   = np.linspace(-3.0, 3.0, 601)
    TAU_FINE_C = 0.5 * (TAU_FINE[:-1] + TAU_FINE[1:])
    TAU_FINE_BIN = float(TAU_FINE[1] - TAU_FINE[0])

    def truth_probs_y(mu, sigma):
        cdf = norm.cdf(edges_Y, loc=mu, scale=max(sigma, 1e-8))
        return np.diff(cdf)

    def truth_probs_tau(mu_tau, sigma_tau):
        cdf = norm.cdf(edges_tau, loc=mu_tau, scale=max(sigma_tau, 1e-8))
        return np.diff(cdf)

    def y_probs_from_stored(d):
        p = np.zeros(J)
        for j in range(J):
            mask = (Y_CENTERS >= edges_Y[j]) & (Y_CENTERS < edges_Y[j+1])
            p[j] = float(d[mask].sum() * Y_BIN)
        s = p.sum()
        return p / s if s > 0 else p

    def tau_probs_from_stored(d_tau):
        p = np.zeros(n_tau_bins)
        for k in range(n_tau_bins):
            mask = (TAU_FINE_C >= edges_tau[k]) & (TAU_FINE_C < edges_tau[k+1])
            p[k] = float(d_tau[mask].sum() * TAU_FINE_BIN)
        s = p.sum()
        return p / s if s > 0 else p

    def l2(p, t, bw):
        return float(np.sqrt(np.sum((np.asarray(p) - np.asarray(t))**2) / bw))

    acc = {'dopfn':         {q: [] for q in ('y0','y1','tau','ate')},
           'bb_2dmarg_b1000': {q: [] for q in ('y0','y1','tau','ate')}}
    seen = {m: set() for m in acc}

    print(f'[cfg] d={args.d} N={args.N} n_test={args.n_test} n_seeds={args.n_seeds}  '
          f'malc_B={args.malc_B} K={args.malc_max_K}', flush=True)

    for seed in range(args.n_seeds):
        t0 = time.time()
        if args.flavor == 'polynomial':
            cd = make_polynomial_scm(seed=seed, n_context=args.N, n_test=args.n_test,
                                      rho_eff=min(args.rho, 0.99), x_dim=args.d,
                                      degree=args.degree, sigma_eps=args.sigma_eps)
        else:
            cd = make_ihdp_like_scm(seed=seed, n_context=args.N, n_test=args.n_test,
                                     x_dim=args.d, degree=args.degree,
                                     sigma_eps=args.sigma_eps,
                                     y_scale=args.y_scale, tau_base=args.tau_base,
                                     tau_scale=args.tau_scale)
        y_ctx = cd.y_train.numpy() if hasattr(cd.y_train, 'numpy') else np.asarray(cd.y_train)
        y_min = float(y_ctx.min()); y_rng = max(float(y_ctx.max()) - y_min, 1e-6)

        # Truth (scaled Y) per query — analytic Gaussian
        mu0_raw = np.asarray(cd._mu0_test).reshape(-1)
        mu1_raw = np.asarray(cd._mu1_test).reshape(-1)
        sigma_raw = float(cd._sigma_eps)
        mu0 = (mu0_raw - y_min) / y_rng * 2.0 - 1.0
        mu1 = (mu1_raw - y_min) / y_rng * 2.0 - 1.0
        sigma_scaled = sigma_raw * 2.0 / y_rng
        n_q = args.n_test

        try:
            d_dopfn = dopfn_densities(cd, DoPFNRegressor, y_min=y_min, y_rng=y_rng,
                                        dopfn_root=args.dopfn, n_context=args.N,
                                        edges_j10_scaled=edges_np)
            d_bb = ours_densities(cd, model_bb, edges_np, J, bin_width, -1,
                                    y_min=y_min, y_rng=y_rng,
                                    malc_B=args.malc_B, malc_max_K=args.malc_max_K,
                                    n_eval=args.n_eval, n_context=args.N,
                                    fit_malc_inner=fit_malc_inner, dmalc_2d=dmalc_2d,
                                    y_scaling='min_max',
                                    marginals_from_2d=True)
        except Exception as e:
            print(f'  seed={seed:2d} SKIP: {type(e).__name__}: {e}', flush=True); continue

        # Truth ATE density on TAU_FINE
        true_tau_dens = np.stack([
            norm.pdf(TAU_FINE_C, loc=float(mu1[q]-mu0[q]),
                      scale=float(np.sqrt(2.0)*sigma_scaled))
            for q in range(n_q)])
        true_ate_dens = wasserstein_barycenter_1d(true_tau_dens, TAU_FINE_C)
        t_ate_bins = tau_probs_from_stored(true_ate_dens)

        for q in range(n_q):
            t_y0 = truth_probs_y(mu0[q], sigma_scaled)
            t_y1 = truth_probs_y(mu1[q], sigma_scaled)
            mu_tau = float(mu1[q] - mu0[q])
            sigma_tau = float(np.sqrt(2.0) * sigma_scaled)
            t_tau = truth_probs_tau(mu_tau, sigma_tau)

            # Do-PFN: recipe-strict per-J=10-bin probability via CDF-interp
            # of the native bar distribution at the J=10 edges (piecewise-
            # uniform density assumption inside each Do-PFN bar).
            pd_y0 = np.asarray(d_dopfn['p_y0_bins_j10'][q], dtype=np.float64)
            pd_y1 = np.asarray(d_dopfn['p_y1_bins_j10'][q], dtype=np.float64)
            pd_tau = tau_probs_from_stored(d_dopfn['p_tau'][q])
            acc['dopfn']['y0'].append(l2(pd_y0, t_y0, bin_w_Y))
            acc['dopfn']['y1'].append(l2(pd_y1, t_y1, bin_w_Y))
            acc['dopfn']['tau'].append(l2(pd_tau, t_tau, bin_w_tau))

            # BB 2D-marg: exact-CDF bin probs from ours_densities
            pb_y0 = np.asarray(d_bb['p_y0_bins_j10'][q], dtype=np.float64)
            pb_y1 = np.asarray(d_bb['p_y1_bins_j10'][q], dtype=np.float64)
            pb_tau = tau_probs_from_stored(d_bb['p_tau'][q])
            acc['bb_2dmarg_b1000']['y0'].append(l2(pb_y0, t_y0, bin_w_Y))
            acc['bb_2dmarg_b1000']['y1'].append(l2(pb_y1, t_y1, bin_w_Y))
            acc['bb_2dmarg_b1000']['tau'].append(l2(pb_tau, t_tau, bin_w_tau))

        # ATE per seed (one number per seed, both methods)
        pd_tau_bary = wasserstein_barycenter_1d(np.stack([d_dopfn['p_tau'][q] for q in range(n_q)]),
                                                  TAU_FINE_C)
        acc['dopfn']['ate'].append(l2(tau_probs_from_stored(pd_tau_bary), t_ate_bins, bin_w_tau))
        pb_tau_bary = wasserstein_barycenter_1d(np.stack([d_bb['p_tau'][q] for q in range(n_q)]),
                                                  TAU_FINE_C)
        acc['bb_2dmarg_b1000']['ate'].append(l2(tau_probs_from_stored(pb_tau_bary),
                                                  t_ate_bins, bin_w_tau))
        seen['dopfn'].add(seed); seen['bb_2dmarg_b1000'].add(seed)
        print(f'  seed={seed:2d} done in {time.time()-t0:.1f}s', flush=True)

    def _stat(vs):
        arr = np.asarray(vs, dtype=np.float64); arr = arr[np.isfinite(arr)]
        if arr.size == 0: return 'na'
        m = arr.mean(); sem = arr.std(ddof=1)/np.sqrt(arr.size) if arr.size > 1 else 0.
        return f'{m:.4f}±{sem:.4f}'

    print()
    print(f'══ d={args.d} SYNTHETIC — per-bin probability L2 '
          f'(J=10 y-bins, {n_tau_bins} τ-bins @{bin_w_tau:.2f}) ══')
    print(f'{"method":52s}  {"cov":>8s}  {"y0":>16s}  {"y1":>16s}  {"τ (CATE)":>16s}  {"ATE":>16s}')
    for m_key, m_label in [('dopfn',            'Do-PFN [J=10]'),
                            ('bb_2dmarg_b1000',  f'Do-PFN-bb MALC (2D-τ) 2D-marg [B={args.malc_B}, J=10]')]:
        cov = f'{len(seen[m_key])}/{args.n_seeds}'
        row = f'{m_label:52s}  {cov:>8s}'
        for metric in ['y0', 'y1', 'tau', 'ate']:
            row += f'  {_stat(acc[m_key][metric]):>16s}'
        print(row)


if __name__ == '__main__':
    main()
