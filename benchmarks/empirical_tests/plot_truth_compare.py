"""4-panel truth density comparison: IHDP vs d=6 (ihdp_like flavour).

Panels:
  (0) p(Y_do(0)) — mixture across queries + Wasserstein barycentre
  (1) p(Y_do(1))
  (2) p(τ|x)     — CATE
  (3) p(τ_ATE)   — Wasserstein barycentre of per-query CATEs

Each dataset drawn in a distinct colour; light per-query traces + bold
mixture/barycentre. All densities in scaled Y ∈ [-1.5, 1.5] so shapes
are directly comparable across datasets.

Usage:
    python benchmarks/empirical_tests/plot_truth_compare.py \\
      --repo $DEPLOY_ROOT/R-PFN \\
      --causalpfn $DEPLOY_ROOT/external/causalpfn \\
      --out /tmp/truth_compare.png \\
      --n-ihdp 10 --n-d6-seeds 10 \\
      --d6-flavor ihdp_like
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--causalpfn', required=True)
    ap.add_argument('--out', default='/tmp/truth_compare.png')
    ap.add_argument('--n-ihdp', type=int, default=10)
    ap.add_argument('--n-d6-seeds', type=int, default=10)
    ap.add_argument('--d6-N', type=int, default=200)
    ap.add_argument('--d6-n-test', type=int, default=25)
    ap.add_argument('--d6-flavor', choices=['polynomial', 'ihdp_like'],
                     default='ihdp_like')
    ap.add_argument('--tau-base', type=float, default=5.0)
    ap.add_argument('--tau-scale', type=float, default=0.5)
    ap.add_argument('--y-scale', type=float, default=0.5)
    ap.add_argument('--show-traces', action='store_true',
                     help='Draw per-query Gaussian traces (light)')
    ap.add_argument('--max-traces', type=int, default=100)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'l2_ihdp'))
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'empirical_tests'))
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC', 'Optimal_Transport'))
    sys.path.insert(0, args.repo)
    sys.path.insert(0, args.causalpfn)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from ot_barycenter import wasserstein_barycenter_1d

    # Shared plotting grids (SCALED Y and SCALED τ)
    Y_GRID   = np.linspace(-1.5, 1.5, 601)
    TAU_GRID = np.linspace(-2.0, 2.0, 801)

    def collect(mu0_list, mu1_list, sigma_scaled):
        """Given per-query scaled means (mu0, mu1) and a single scaled σ,
        return (mixture_y0, mixture_y1, mixture_tau, ate_barycentre)."""
        p_y0 = np.stack([norm.pdf(Y_GRID, m, sigma_scaled) for m in mu0_list])
        p_y1 = np.stack([norm.pdf(Y_GRID, m, sigma_scaled) for m in mu1_list])
        sigma_tau = float(np.sqrt(2.0) * sigma_scaled)
        p_tau = np.stack([norm.pdf(TAU_GRID, m1 - m0, sigma_tau)
                           for m0, m1 in zip(mu0_list, mu1_list)])
        mix_y0 = p_y0.mean(axis=0);  mix_y0 /= max(mix_y0.sum() * (Y_GRID[1]-Y_GRID[0]), 1e-12)
        mix_y1 = p_y1.mean(axis=0);  mix_y1 /= max(mix_y1.sum() * (Y_GRID[1]-Y_GRID[0]), 1e-12)
        mix_tau = p_tau.mean(axis=0); mix_tau /= max(mix_tau.sum() * (TAU_GRID[1]-TAU_GRID[0]), 1e-12)
        ate = wasserstein_barycenter_1d(p_tau, TAU_GRID)
        ate = ate / max(ate.sum() * (TAU_GRID[1]-TAU_GRID[0]), 1e-12)
        return p_y0, p_y1, p_tau, mix_y0, mix_y1, mix_tau, ate

    # ---- IHDP ----
    from true_ihdp import load_ihdp_truth
    from benchmarks import IHDPDataset
    ihdp_mu0, ihdp_mu1, ihdp_sig = [], [], []
    for r in range(args.n_ihdp):
        cd, _ = IHDPDataset()[r]
        y = np.asarray(cd.y_train.detach().cpu()
                        if hasattr(cd.y_train, 'detach') else cd.y_train).reshape(-1)
        truth = load_ihdp_truth(r, args.causalpfn, y)
        ihdp_mu0.extend(np.asarray(truth.mu0_test_scaled).reshape(-1))
        ihdp_mu1.extend(np.asarray(truth.mu1_test_scaled).reshape(-1))
        ihdp_sig.append(float(truth.sigma_scaled))
    ihdp_sigma_scaled = float(np.mean(ihdp_sig))
    print(f'[ihdp] {len(ihdp_mu0)} queries, mean scaled σ = {ihdp_sigma_scaled:.4f}')
    ihdp_bundle = collect(ihdp_mu0, ihdp_mu1, ihdp_sigma_scaled)

    # ---- d=6 ----
    from fig2_pehe_l2 import make_polynomial_scm, make_ihdp_like_scm
    d6_mu0, d6_mu1, d6_sig = [], [], []
    for seed in range(args.n_d6_seeds):
        if args.d6_flavor == 'polynomial':
            cd = make_polynomial_scm(seed=seed, n_context=args.d6_N, n_test=args.d6_n_test,
                                      rho_eff=0.0, x_dim=6, degree=3, sigma_eps=1.0)
        else:
            cd = make_ihdp_like_scm(seed=seed, n_context=args.d6_N, n_test=args.d6_n_test,
                                     x_dim=6, degree=3, sigma_eps=1.0,
                                     y_scale=args.y_scale, tau_base=args.tau_base,
                                     tau_scale=args.tau_scale)
        y_ctx = cd.y_train.numpy() if hasattr(cd.y_train, 'numpy') else np.asarray(cd.y_train)
        y_min = float(y_ctx.min()); y_rng = max(float(y_ctx.max()) - y_min, 1e-6)
        m0 = np.asarray(cd._mu0_test).reshape(-1)
        m1 = np.asarray(cd._mu1_test).reshape(-1)
        d6_mu0.extend((m0 - y_min) / y_rng * 2 - 1)
        d6_mu1.extend((m1 - y_min) / y_rng * 2 - 1)
        d6_sig.append(float(cd._sigma_eps) * 2 / y_rng)
    d6_sigma_scaled = float(np.mean(d6_sig))
    print(f'[d=6]  {len(d6_mu0)} queries, mean scaled σ = {d6_sigma_scaled:.4f}')
    d6_bundle = collect(d6_mu0, d6_mu1, d6_sigma_scaled)

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    IHDP_COLOR = '#1f77b4'; D6_COLOR = '#d62728'
    for ax, (title, xgrid, ihdp_arr, d6_arr, ihdp_mix, d6_mix, xlabel) in zip(
        axes.ravel(),
        [('p(Y_do(0) | x)',        Y_GRID,   ihdp_bundle[0], d6_bundle[0],
                                    ihdp_bundle[3], d6_bundle[3], 'scaled Y'),
         ('p(Y_do(1) | x)',        Y_GRID,   ihdp_bundle[1], d6_bundle[1],
                                    ihdp_bundle[4], d6_bundle[4], 'scaled Y'),
         ('p(τ | x)  (CATE)',      TAU_GRID, ihdp_bundle[2], d6_bundle[2],
                                    ihdp_bundle[5], d6_bundle[5], 'scaled τ'),
         ('p(τ_ATE)  (Wass. bary)', TAU_GRID, None,          None,
                                    ihdp_bundle[6], d6_bundle[6], 'scaled τ')]):
        if args.show_traces and ihdp_arr is not None:
            k = min(args.max_traces, len(ihdp_arr))
            for t in ihdp_arr[:k]:
                ax.plot(xgrid, t, color=IHDP_COLOR, alpha=0.05, lw=0.5)
            for t in d6_arr[:k]:
                ax.plot(xgrid, t, color=D6_COLOR, alpha=0.05, lw=0.5)
        ax.plot(xgrid, ihdp_mix, color=IHDP_COLOR, lw=2.2, label=f'IHDP  σ_sc={ihdp_sigma_scaled:.3f}')
        ax.plot(xgrid, d6_mix,   color=D6_COLOR,   lw=2.2, label=f'd=6 ({args.d6_flavor})  σ_sc={d6_sigma_scaled:.3f}')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel); ax.set_ylabel('density')
        ax.legend(loc='upper right', fontsize=8, frameon=False)
        ax.axvline(0.0, color='k', lw=0.5, alpha=0.3)

    fig.suptitle(f'Truth density comparison — IHDP vs d=6 ({args.d6_flavor}, '
                  f'tau_base={args.tau_base}, y_scale={args.y_scale})',
                  fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=140)
    print(f'\n[saved] {args.out}')


if __name__ == '__main__':
    main()
