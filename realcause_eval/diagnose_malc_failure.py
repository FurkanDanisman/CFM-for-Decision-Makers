"""Instrument MALC's _fit_component_2d to pin down which failure path
fires on each query. Runs on a single (method, dataset, realization)
cell and prints, per query:
  path 1: insufficient mass (<1e-10 sum or <2 nonzero cells)
  path 2: beta_x explodes (min(α ± β_x) ≤ 0)
  path 3: beta_y explodes
  path 4: LCMLE _mlelcd_2d threw exception
  fits: neither None-branch fired → LCMLE returned a fit

Usage:
    python realcause_eval/diagnose_malc_failure.py \\
        --density $OUT_1D/cpfn1d/PSID/PSID_r000.npz \\
        --repo   $DEPLOY_ROOT/R-PFN \\
        --n-queries 30
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import numpy as np


def _inspect_fit(p_mat, grid_x, grid_y, B, alpha, rng):
    """Reimplementation of MALC's _fit_component_2d that returns a
    diagnostic dict instead of ComponentFit2D/None.

    Diagnostic keys returned:
      status: one of {'path1_mass', 'path2_beta_x', 'path3_beta_y',
                       'path4_lcmle', 'ok'}
      s, n_nonzero, sigma_x, sigma_y, mu_low_x, mu_mid_x, mu_n_x,
      beta_x, beta_y, xstar_unique_count, xstar_range_x, xstar_range_y,
      exc_msg (if path4).
    """
    from losses.BarDistribution2D import fit_malc_inner  # noqa
    from malc_2d import _em_mean_2d, mlelcd_2d

    p_mat = np.maximum(p_mat, 0.0)
    s = float(p_mat.sum())
    n_nonzero = int(np.sum(p_mat > 1e-10))
    if s < 1e-10 or n_nonzero < 2:
        return dict(status='path1_mass', s=s, n_nonzero=n_nonzero)
    p_mat = p_mat / s

    n_y, n_x = p_mat.shape
    delta_x = float(grid_x[1] - grid_x[0])
    delta_y = float(grid_y[1] - grid_y[0])
    p_x = p_mat.sum(axis=0)
    p_y = p_mat.sum(axis=1)
    grid_left_x = grid_x[:-1]
    grid_left_y = grid_y[:-1]
    centers_x = 0.5 * (grid_left_x + grid_x[1:])
    centers_y = 0.5 * (grid_left_y + grid_y[1:])

    mu_low_x = float(np.sum(p_x * grid_left_x))
    mu_low_y = float(np.sum(p_y * grid_left_y))
    mu_mid_x = 0.5 * (mu_low_x + float(np.sum(p_x * grid_x[1:])))
    mu_mid_y = 0.5 * (mu_low_y + float(np.sum(p_y * grid_y[1:])))

    sigma_x = float(np.sqrt(np.sum(p_x * (centers_x - mu_mid_x) ** 2) + delta_x ** 2 / 12.0))
    sigma_y = float(np.sqrt(np.sum(p_y * (centers_y - mu_mid_y) ** 2) + delta_y ** 2 / 12.0))
    if not np.isfinite(sigma_x) or sigma_x <= 0: sigma_x = delta_x
    if not np.isfinite(sigma_y) or sigma_y <= 0: sigma_y = delta_y

    mu_n_x = _em_mean_2d(p_x, grid_x, sigma=sigma_x, start=mu_mid_x)
    mu_n_y = _em_mean_2d(p_y, grid_y, sigma=sigma_y, start=mu_mid_y)

    beta_x = 2.0 * alpha * ((mu_n_x - mu_low_x) / delta_x - 0.5)
    beta_y = 2.0 * alpha * ((mu_n_y - mu_low_y) / delta_y - 0.5)

    base = dict(s=s, n_nonzero=n_nonzero, sigma_x=sigma_x, sigma_y=sigma_y,
                mu_low_x=mu_low_x, mu_mid_x=mu_mid_x, mu_n_x=mu_n_x,
                delta_x=delta_x, beta_x=beta_x, beta_y=beta_y,
                alpha=alpha)

    if not np.isfinite(beta_x) or min(alpha + beta_x, alpha - beta_x) <= 0:
        return dict(status='path2_beta_x', **base)
    if not np.isfinite(beta_y) or min(alpha + beta_y, alpha - beta_y) <= 0:
        return dict(status='path3_beta_y', **base)

    prob_vec = p_mat.flatten(order='F')
    bin_idx = rng.choice(len(prob_vec), size=B, p=prob_vec, replace=True)
    bi_j = bin_idx // n_y
    bi_i = bin_idx % n_y
    zstar_x = delta_x * rng.beta(alpha + beta_x, alpha - beta_x, size=B)
    zstar_y = delta_y * rng.beta(alpha + beta_y, alpha - beta_y, size=B)
    xstar = np.column_stack([grid_left_x[bi_j] + zstar_x, grid_left_y[bi_i] + zstar_y])

    xstar_unique = int(np.unique(xstar, axis=0).shape[0])
    xstar_range_x = float(xstar[:, 0].max() - xstar[:, 0].min())
    xstar_range_y = float(xstar[:, 1].max() - xstar[:, 1].min())
    base.update(xstar_unique=xstar_unique,
                xstar_range_x=xstar_range_x, xstar_range_y=xstar_range_y)

    try:
        _ = mlelcd_2d(xstar, jitter=1e-10, seed=int(rng.integers(2**31 - 1)),
                       tol_gap=1e-8, tol_feas=1e-8)
    except Exception as e:
        return dict(status='path4_lcmle', exc_msg=str(e)[:120], **base)

    return dict(status='ok', **base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--density', required=True,
                    help='Path to a 1D density-dump NPZ (e.g. .../PSID_r000.npz)')
    ap.add_argument('--repo', required=True)
    ap.add_argument('--n-queries', type=int, default=30,
                    help='Number of queries to inspect (mix of failed + succeeded).')
    ap.add_argument('--B', type=int, default=500)
    ap.add_argument('--alpha', type=float, default=1.0,
                    help='MALC alpha parameter (default 1.0).')
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.join(args.repo, 'MALC'))

    # Load 1D density + prior MALC dump (to know which queries failed).
    from compute_malc_ci_cell_1d import _load_1d_joint  # applies effective_centers etc.
    p_joint, edges, y_scale, y_shift, true_cate = _load_1d_joint(
        args.density, downsample_max_J=0)
    N_q, J, _ = p_joint.shape
    print(f'density: {args.density}')
    print(f'  N_q={N_q}  J={J}  edges=[{edges[0]:.4f}, {edges[-1]:.4f}]  '
          f'delta={edges[1]-edges[0]:.4g}  y_scale={y_scale:.4g}')

    # Load MALC dump to know which queries failed (if it exists).
    cell = os.path.dirname(args.density)
    r_tag = os.path.basename(args.density).split('_')[-1].replace('.npz', '')
    malc_path = os.path.join(cell, f'malc_ci_B500_{r_tag}.npz')
    if os.path.isfile(malc_path):
        with np.load(malc_path) as z:
            p_taus = np.asarray(z['p_taus_scaled'])
        row_sums = p_taus.sum(axis=-1)
        failed_prev = np.where(row_sums < 1e-8)[0]
        succ_prev = np.where(row_sums > 1e-8)[0]
        print(f'  prior MALC dump: {failed_prev.size} failed / {N_q}')
    else:
        failed_prev = np.arange(0)
        succ_prev = np.arange(N_q)
        print(f'  no prior MALC dump — inspecting first {args.n_queries} queries')

    # Sample queries: half from failed, half from succeeded (if both exist).
    n_take = min(args.n_queries // 2, failed_prev.size, succ_prev.size)
    q_show = list(failed_prev[:n_take]) + list(succ_prev[:n_take])
    if not q_show:
        q_show = list(range(min(args.n_queries, N_q)))

    # Inspect each.
    stats = {}
    print()
    print(f'{"q":>4s} {"prior":>7s} {"status":>15s} {"beta_x":>10s} {"beta_y":>10s}  '
          f'{"nnz":>5s} {"σ_x/δ_x":>10s} {"xstar_unq":>10s} {"note":<50s}')
    for q in q_show:
        seed = int(hashlib.md5(f'q{q}malcK1'.encode()).hexdigest()[:8], 16) % (10 ** 8)
        rng = np.random.default_rng(seed)
        info = _inspect_fit(p_joint[q], edges, edges, args.B, args.alpha, rng)
        status = info['status']
        stats[status] = stats.get(status, 0) + 1
        prior = 'FAIL' if q in set(failed_prev) else 'OK'
        bx = info.get('beta_x', float('nan'))
        by = info.get('beta_y', float('nan'))
        nnz = info.get('n_nonzero', -1)
        sr  = info.get('sigma_x', 0) / max(info.get('delta_x', 1), 1e-30)
        xun = info.get('xstar_unique', -1)
        note = info.get('exc_msg', '')
        print(f'{q:>4d} {prior:>7s} {status:>15s} {bx:>+10.3g} {by:>+10.3g}  '
              f'{nnz:>5d} {sr:>10.3g} {xun:>10d} {note:<50s}')

    print()
    print(f'=== Summary over {len(q_show)} inspected queries ===')
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f'  {k:<15s} : {v:3d}  ({100.0*v/len(q_show):.1f}%)')


if __name__ == '__main__':
    main()
