"""Sanity check for the CATE-density derivation pipeline.

Construct a KNOWN-PERFECT 2D joint p_mat by discretising a 2D Gaussian with
known (μ0, μ1, σ, ρ), pass it through both derivations, and compare against
the analytic truth CATE density:
      p(τ) = N(μ1 − μ0,  2σ²(1 − ρ))

If the pipeline is correct we should see:
  - RAW diagonal-projection L2 → small, dominated by J-bin discretisation.
  - MALC-fit-then-integrate L2 → small, dominated by log-concave-family
    approximation error (MALC's inherent smoothing).

If MALC-L2 is much LARGER than raw-L2 → the MALC integration step has a bug.
If both are large → the truth grid / normalisation / support is off.

Usage:
    python sanity_cate_density.py \
        --repo $DEPLOY_ROOT/R-PFN \
        --J 100 --y-range 6.0 \
        --mu0 0.0 --mu1 0.5 --sigma 1.0 --rho 0.6 \
        --malc-B 100 --malc-max-K 2

  Change --J, --rho to explore different regimes.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np


def _analytic_2d_gauss_p_mat(mu0, mu1, sigma, rho, edges):
    """Discretise the 2D Gaussian into J×J bin masses by evaluating the
    density at bin centers and multiplying by bin area (Riemann approx of
    the integral over each bin). p_mat[i, j] = mass in (y0_bin_i, y1_bin_j).

    edges: shape (J+1,), same for both y0 and y1. Assumes uniform grid.
    Returns:
      p_mat (J, J) with p_mat[i, j] = P(Y0 in bin i, Y1 in bin j),  sums to ~1
    """
    centers = 0.5 * (edges[:-1] + edges[1:])                    # (J,)
    bw = float(edges[1] - edges[0])
    Y0, Y1 = np.meshgrid(centers, centers, indexing='ij')       # (J, J)
    # 2D Gaussian density
    inv_det = 1.0 / (2 * np.pi * sigma * sigma * np.sqrt(1 - rho ** 2))
    z0 = (Y0 - mu0) / sigma; z1 = (Y1 - mu1) / sigma
    q = z0 ** 2 - 2 * rho * z0 * z1 + z1 ** 2
    dens_2d = inv_det * np.exp(-q / (2 * (1 - rho ** 2)))
    p_mat = dens_2d * (bw ** 2)                                  # mass = density * bin area
    # Renormalise (truncation error since Gaussian tails extend beyond [edges])
    p_mat = p_mat / p_mat.sum()
    return p_mat


def _analytic_true_cate_density(mu0, mu1, sigma, rho, tau_grid):
    """p(τ) = N(μ1 − μ0, 2σ²(1 − ρ))  evaluated at tau_grid."""
    mu_tau = mu1 - mu0
    sd_tau = float(np.sqrt(2.0 * sigma * sigma * (1 - rho)))
    d = np.exp(-0.5 * ((tau_grid - mu_tau) / sd_tau) ** 2) / (sd_tau * np.sqrt(2 * np.pi))
    return d


def _raw_diagonal_projection(p_mat, centers, tau_grid):
    """Sum p_mat[i,j] over cells where centers[j]-centers[i] falls in bin k.
    Then convert to density (/ tau_bin_width)."""
    tau_ij = centers[None, :] - centers[:, None]                 # (J, J)
    tau_min = float(tau_grid[0]); tau_dx = float(tau_grid[1] - tau_grid[0])
    idx = np.round((tau_ij - tau_min) / tau_dx).astype(int)
    valid = (idx >= 0) & (idx < tau_grid.shape[0])
    hist = np.zeros(tau_grid.shape[0])
    np.add.at(hist, idx[valid], p_mat[valid])
    return hist / max(tau_dx, 1e-12)                             # density


def _l2(f, g, dx):
    return float(np.sqrt(np.sum((f - g) ** 2) * dx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo',       required=True)
    ap.add_argument('--J',          type=int,   default=100)
    ap.add_argument('--y-range',    type=float, default=6.0,
                     help='half-range of edges: [-y_range, +y_range]')
    ap.add_argument('--mu0',        type=float, default=0.0)
    ap.add_argument('--mu1',        type=float, default=0.5)
    ap.add_argument('--sigma',      type=float, default=1.0)
    ap.add_argument('--rho',        type=float, default=0.6)
    ap.add_argument('--malc-B',     type=int,   default=100)
    ap.add_argument('--malc-max-K', type=int,   default=2)
    ap.add_argument('--malc-n-eval',type=int,   default=100)
    ap.add_argument('--tau-bins',   type=int,   default=401)
    ap.add_argument('--seed',       type=int,   default=42)
    args = ap.parse_args()

    # ── Build edges and grids ────────────────────────────────────────────
    edges = np.linspace(-args.y_range, args.y_range, args.J + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bw = float(edges[1] - edges[0])
    tau_grid = np.linspace(-2 * args.y_range, 2 * args.y_range, args.tau_bins)
    tau_dx = float(tau_grid[1] - tau_grid[0])

    print(f'Setup: J={args.J}  y ∈ [{-args.y_range:+.1f}, {args.y_range:+.1f}]  bin_width={bw:.4f}')
    print(f'       μ0={args.mu0}  μ1={args.mu1}  σ={args.sigma}  ρ={args.rho}')
    print(f'       τ-grid: {args.tau_bins} bins on [{tau_grid[0]:+.1f}, {tau_grid[-1]:+.1f}]  dτ={tau_dx:.4f}')

    # ── Construct the perfect p_mat ─────────────────────────────────────
    p_mat = _analytic_2d_gauss_p_mat(args.mu0, args.mu1, args.sigma, args.rho, edges)
    print(f'\n[perfect p_mat] shape={p_mat.shape}  sum={p_mat.sum():.6f}  '
          f'max={p_mat.max():.4e}  min={p_mat.min():.4e}')

    # Truth CATE density
    d_true = _analytic_true_cate_density(args.mu0, args.mu1, args.sigma, args.rho, tau_grid)
    print(f'[truth CATE] μ_τ={args.mu1 - args.mu0}  σ_τ={np.sqrt(2*args.sigma**2*(1-args.rho)):.4f}  '
          f'mass={d_true.sum() * tau_dx:.6f}')

    # ── (a) Raw diagonal projection ─────────────────────────────────────
    t0 = time.time()
    d_raw = _raw_diagonal_projection(p_mat, centers, tau_grid)
    dt_raw = time.time() - t0
    l2_raw = _l2(d_raw, d_true, tau_dx)
    print(f'\n[RAW diagonal projection] mass={d_raw.sum()*tau_dx:.4f}  '
          f'L2_vs_truth={l2_raw:.4f}   ({dt_raw*1000:.1f}ms)')

    # ── (b) MALC-fit-then-integrate ─────────────────────────────────────
    # Set up MALC's _GLOBAL like ours._fit_and_marginalize expects.
    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks'))
    from methods import ours as ours_mod
    from losses.BarDistribution2D import fit_malc_inner
    from malc_2d import dmalc_2d

    G = ours_mod._GLOBAL
    G.clear()
    G['fit'] = fit_malc_inner; G['dmalc'] = dmalc_2d
    G['edges'] = edges; G['J'] = args.J; G['bw'] = bw
    G['MALC_B'] = args.malc_B; G['MALC_MAX_K'] = args.malc_max_K
    xs = np.linspace(edges[0], edges[-1], args.malc_n_eval)
    ys = np.linspace(edges[0], edges[-1], args.malc_n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    G['xs'] = xs; G['ys'] = ys
    G['eval_pts'] = np.column_stack([XX.ravel(), YY.ravel()])
    G['dy0'] = xs[1] - xs[0]; G['dy1'] = ys[1] - ys[0]
    G['tau'] = tau_grid                                         # use OUR tau grid
    G['dtau'] = tau_dx

    t0 = time.time()
    d_malc = np.asarray(ours_mod._fit_and_marginalize(p_mat, seed=args.seed))
    dt_malc = time.time() - t0
    l2_malc = _l2(d_malc, d_true, tau_dx)
    print(f'[MALC-fit + integrate] mass={d_malc.sum()*tau_dx:.4f}  '
          f'L2_vs_truth={l2_malc:.4f}   ({dt_malc*1000:.0f}ms, B={args.malc_B}, K≤{args.malc_max_K})')

    # ── Verdict ─────────────────────────────────────────────────────────
    print()
    print(f'RAW  L2 / truth-peak: {l2_raw / d_true.max():.4f}  '
          f'(dimensionless, 0=perfect)')
    print(f'MALC L2 / truth-peak: {l2_malc / d_true.max():.4f}')
    print()
    if l2_malc > 2.0 * l2_raw:
        print('*** MALC is >2× worse than RAW — likely a bug in fit_and_marginalize ***')
    elif l2_malc < 0.5 * l2_raw:
        print('*** MALC is >2× better than RAW — smoothing recovering the truth ***')
    else:
        print('MALC and RAW are within factor of 2 — likely correct, differ by smoothing bias.')


if __name__ == '__main__':
    try: main()
    except Exception:
        import traceback; traceback.print_exc(); sys.exit(1)
