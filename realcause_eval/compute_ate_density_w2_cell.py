"""Per-realization ATE density via 1D 2-Wasserstein barycenter over queries.

Given per-query p(τ | x_q) densities from any of three sources:

  --source malc       Read malc_ci_{tag}_r<###>.npz's `p_taus_scaled` (built
                      by compute_malc_ci_cell.py). Requires --malc-tag.
                      Applies to cpfn2d, graph2d, dopfnbb.
  --source joint      Read density NPZ's `p_joint_scaled` (2D methods),
                      project p(τ) via anti-diagonal sums (no independence
                      assumption). Applies to cpfn2d, graph2d, dopfnbb.
  --source marginals  Read density NPZ's `p_y0_scaled`, `p_y1_scaled`
                      (1D methods) and convolve under Y|do(0) ⊥ Y|do(1).
                      Applies to cpfn1d, dopfn, uwyk1d.

Per-realization pipeline:
  1. Load p(τ | x_q) for q = 1..N_q on the realization's shared τ grid.
     Convert grid to RAW Y units (τ_raw = τ_scaled · y_scale; density in
     raw units = mass / (bin_width · y_scale)).
  2. Wasserstein barycenter over queries (uniform weights) → p_ATE(τ).
  3. Point estimates: ATE mode (argmax p_ATE), ATE mean (∫ τ p_ATE dτ).
  4. 95% CI: linear-interp inverse-CDF of p_ATE at 0.025 / 0.975.
  5. Save p_ATE + derived quantities into ate_w2_{tag}_r<###>.npz.

Layouts supported (same as compute_malc_ci_cell.py):
  inline (cpfn1d/cpfn2d/dopfn/graph2d/uwyk1d):
    <method_dir>/<DATASET>/<DATASET>_r<###>.npz    (density source)
    <method_dir>/<DATASET>/malc_ci_{tag}_r<###>.npz (MALC source)
  split  (dopfnbb):
    <method_dir>/<DATASET>/density_r<###>.npz       (density source)
    <method_dir>/<DATASET>/malc_ci_{tag}_r<###>.npz (MALC source)

Usage:
    python realcause_eval/compute_ate_density_w2_cell.py \\
        --method-dir /scratch/.../results_rc_2d_density/graph2d \\
        --dataset IHDP --source malc --malc-tag B100
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np


def _import_barycenter(repo_root):
    ot_dir = os.path.join(repo_root, 'MALC', 'Optimal_Transport')
    if ot_dir not in sys.path:
        sys.path.insert(0, ot_dir)
    from ot_barycenter import wasserstein_barycenter_1d
    return wasserstein_barycenter_1d


def _p_tau_from_joint(p_joint):
    """(N_q, J, J) → (N_q, 2J-1) probability mass per antidiagonal.

    Axis 1 = Y0 bin, axis 2 = Y1 bin (matches every 2D emitter's convention).
    τ index k corresponds to j - i = k - (J - 1).
    """
    N_q, J, _ = p_joint.shape
    p_tau = np.zeros((N_q, 2 * J - 1), dtype=np.float64)
    for k in range(2 * J - 1):
        p_tau[:, k] = np.trace(p_joint, axis1=-2, axis2=-1, offset=k - (J - 1))
    return p_tau


def _p_tau_from_marginals(p_y0, p_y1):
    """(N_q, K), (N_q, K) → (N_q, 2K-1) via 1D convolution under independence.

    Assumes centers are equispaced. Atom k corresponds to τ = c[j] - c[i]
    with j - i = k - (K - 1)."""
    from numpy.fft import rfft, irfft
    N_q, K = p_y0.shape
    n_out = 2 * K - 1
    n_fft = 1 << (n_out - 1).bit_length()
    F1 = rfft(p_y1, n=n_fft, axis=-1)
    F0 = rfft(p_y0[:, ::-1], n=n_fft, axis=-1)
    p_tau = irfft(F1 * F0, n=n_fft, axis=-1)[:, :n_out]
    return np.clip(p_tau, 0.0, None)


def _p_tau_from_marginals_general(p_y0, p_y1, centers, tau_grid):
    """Non-uniform-centers path: enumerate K² pairwise atoms per query at
    positions c[j] - c[i] with masses p_y0[i] * p_y1[j], then rasterize
    onto the uniform tau_grid via nearest-bin assignment.

    Used when centers have non-uniform spacing (e.g. dopfn's
    effective_centers where tail bars carry half-Gaussian corrections
    that don't fit the interior uniform grid).

    Returns (N_q, T) mass per bin on tau_grid.
    """
    N_q, K = p_y0.shape
    T = tau_grid.size
    dtau = float(tau_grid[1] - tau_grid[0])
    tau_min = float(tau_grid[0])
    tau_pairs = (centers[None, :] - centers[:, None]).ravel()      # (K²,) — τ = c[j] - c[i]
    idx = np.clip(np.round((tau_pairs - tau_min) / dtau).astype(int), 0, T - 1)
    out = np.zeros((N_q, T), dtype=np.float64)
    for q in range(N_q):
        mass = np.outer(p_y0[q], p_y1[q]).ravel()                  # (K²,)
        np.add.at(out[q], idx, mass)
    return out


def _is_uniform(centers, rtol=1e-4):
    if centers.size < 3: return True
    d = np.diff(centers)
    return bool(np.max(np.abs(d - d.mean())) < rtol * abs(d.mean()) + 1e-12)


def _load_ptau_raw(density_path, source, malc_path=None):
    """Return (p_tau_raw, tau_raw, true_cate_per_query, y_scale, y_shift).

    p_tau_raw shape (N_q, T)  — density in raw Y units, ∫ p_τ dτ ≈ 1 per query.
    tau_raw   shape (T,)      — uniformly spaced τ grid in raw Y units.
    """
    z_den = np.load(density_path, allow_pickle=True)
    y_scale = float(z_den['y_scale'])
    y_shift = float(z_den['y_shift'])
    true_cate = np.asarray(z_den['true_cate_per_query'], dtype=np.float64)
    edges = np.asarray(z_den['edges'], dtype=np.float64)

    if source == 'malc':
        assert malc_path is not None, 'source=malc requires malc_path'
        z_malc = np.load(malc_path, allow_pickle=True)
        p_tau_scaled = np.asarray(z_malc['p_taus_scaled'], dtype=np.float64)
        tau_scaled   = np.asarray(z_malc['tau_scaled'],    dtype=np.float64)
        # p_τ_raw(τ) = p_τ_scaled(τ_scaled) / y_scale (change of variables)
        p_tau_raw = p_tau_scaled / max(y_scale, 1e-12)
        tau_raw   = tau_scaled * y_scale
    elif source == 'joint':
        p_joint = np.asarray(z_den['p_joint_scaled'], dtype=np.float64)
        # Normalize per query (defensive against fp32 drift).
        s = p_joint.sum(axis=(1, 2), keepdims=True)
        p_joint = p_joint / np.where(s > 0, s, 1.0)
        J = p_joint.shape[1]
        w_scaled = float(edges[1] - edges[0])
        mass_tau = _p_tau_from_joint(p_joint)       # probability mass per atom
        # Atoms at τ_scaled = (k - (J-1)) * w_scaled
        tau_scaled = (np.arange(2 * J - 1) - (J - 1)).astype(np.float64) * w_scaled
        tau_raw    = tau_scaled * y_scale
        dtau_raw   = w_scaled * y_scale
        # Mass → density: divide by bin width (raw units).
        p_tau_raw = mass_tau / max(dtau_raw, 1e-12)
    elif source == 'marginals':
        p_y0 = np.asarray(z_den['p_y0_scaled'], dtype=np.float64)
        p_y1 = np.asarray(z_den['p_y1_scaled'], dtype=np.float64)
        p_y0 = p_y0 / p_y0.sum(axis=-1, keepdims=True).clip(min=1e-12)
        p_y1 = p_y1 / p_y1.sum(axis=-1, keepdims=True).clip(min=1e-12)
        K = p_y0.shape[1]
        # Prefer effective_centers when present (dopfn's tail-adjusted per-bar
        # centers — first/last bar carry half-Gaussian offsets, NOT plain
        # edge midpoints; earlier density-CI aggregator has the same fix).
        # cpfn1d / uwyk1d don't save it → fall back to edges midpoints.
        if 'effective_centers' in z_den.files:
            centers_scaled = np.asarray(z_den['effective_centers'], dtype=np.float64)
            assert centers_scaled.size == K, (
                f'effective_centers has {centers_scaled.size} entries but density has '
                f'{K} bins — schema mismatch in {density_path}')
        elif edges.size >= K + 1:
            centers_scaled = 0.5 * (edges[:K] + edges[1:K + 1])
        else:
            w = float(edges[1] - edges[0])
            centers_scaled = np.arange(K) * w + float(edges[0])
        centers_raw = centers_scaled * y_scale
        if _is_uniform(centers_raw):
            # Uniform grid → FFT convolution is exact and fast. τ atom k maps
            # to c[j] - c[i] with j - i = k - (K - 1), giving 2K-1 atoms
            # centred at zero with spacing w_raw.
            w_raw = float(centers_raw[1] - centers_raw[0])
            mass_tau = _p_tau_from_marginals(p_y0, p_y1)   # (N_q, 2K-1)
            tau_raw  = (np.arange(2 * K - 1) - (K - 1)).astype(np.float64) * w_raw
            dtau_raw = w_raw
            p_tau_raw = mass_tau / max(dtau_raw, 1e-12)
        else:
            # Non-uniform (dopfn tail atoms) → enumerate K² atoms and
            # rasterize onto a uniform τ grid. Grid extent = full pairwise
            # τ range; grid resolution matches the interior bar spacing.
            tau_pairs_flat = (centers_raw[None, :] - centers_raw[:, None]).ravel()
            tau_min_ = float(tau_pairs_flat.min())
            tau_max_ = float(tau_pairs_flat.max())
            # Grid resolution: use median interior spacing so tail atoms
            # get isolated bins but interior stays well-resolved.
            interior_step = float(np.median(np.diff(centers_raw)))
            T = max(2 * K - 1, int(np.ceil((tau_max_ - tau_min_) / max(interior_step, 1e-12))) + 1)
            tau_raw = np.linspace(tau_min_, tau_max_, T)
            mass_tau = _p_tau_from_marginals_general(p_y0, p_y1, centers_raw, tau_raw)
            dtau_raw = float(tau_raw[1] - tau_raw[0])
            p_tau_raw = mass_tau / max(dtau_raw, 1e-12)
    else:
        raise ValueError(f'unknown source: {source}')
    return p_tau_raw, tau_raw, true_cate, y_scale, y_shift


def _quantile_from_density(p_tau, tau, level):
    """Linear-interp inverse-CDF at `level` for a single density on `tau` grid."""
    dtau = tau[1] - tau[0]
    F = np.concatenate([[0.0], np.cumsum(0.5 * (p_tau[1:] + p_tau[:-1]) * dtau)])
    if F[-1] <= 0:
        return float('nan')
    F = F / F[-1]
    return float(np.interp(level, F, tau))


def _process_realization(density_path, out_path, source, malc_path, barycenter_fn,
                          n_tau_barycenter=4001):
    p_tau_raw, tau_raw, true_cate, y_scale, y_shift = _load_ptau_raw(
        density_path, source, malc_path=malc_path)
    N_q = p_tau_raw.shape[0]

    # Per-query normalize (density integrates to 1 on this grid).
    dtau_raw = tau_raw[1] - tau_raw[0]
    norm = (p_tau_raw.sum(axis=-1, keepdims=True) * dtau_raw).clip(min=1e-12)
    p_tau_raw = p_tau_raw / norm

    # ── W2 barycenter across queries (uniform weights).
    p_ate = barycenter_fn(p_tau_raw, tau_raw, n_tau=n_tau_barycenter)
    # Renormalize (barycenter fn does but be defensive).
    s = p_ate.sum() * dtau_raw
    if s > 0: p_ate = p_ate / s

    # ── Point + CI.
    ate_mean = float((tau_raw * p_ate).sum() * dtau_raw)
    ate_mode = float(tau_raw[int(np.argmax(p_ate))])
    tau_lo   = _quantile_from_density(p_ate, tau_raw, 0.025)
    tau_hi   = _quantile_from_density(p_ate, tau_raw, 0.975)
    ci_length = float(tau_hi - tau_lo)
    true_ate = float(true_cate.mean())
    covered = int((tau_lo <= true_ate) and (true_ate <= tau_hi))

    np.savez(out_path,
             p_ate_raw=p_ate.astype(np.float32),
             tau_raw=tau_raw.astype(np.float32),
             ate_mean=np.float32(ate_mean),
             ate_mode=np.float32(ate_mode),
             tau_lo=np.float32(tau_lo),
             tau_hi=np.float32(tau_hi),
             ci_length=np.float32(ci_length),
             true_ate=np.float32(true_ate),
             covered=np.int32(covered),
             y_scale=np.float32(y_scale),
             y_shift=np.float32(y_shift),
             N_q=int(N_q),
             source=str(source))
    return {'ate_mean': ate_mean, 'ate_mode': ate_mode, 'true_ate': true_ate,
            'tau_lo': tau_lo, 'tau_hi': tau_hi, 'ci_length': ci_length,
            'covered': covered}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method-dir', required=True,
                    help='<OUT_ROOT>/<method>; script picks up <method-dir>/<dataset>/...')
    ap.add_argument('--dataset', required=True,
                    help='Dataset dir name under method-dir. For dopfnbb use PSIDbal (legacy).')
    ap.add_argument('--source', required=True, choices=['malc', 'joint', 'marginals'])
    ap.add_argument('--is-split', action='store_true',
                    help='dopfnbb layout (density_r*.npz + summary.npz).')
    ap.add_argument('--malc-tag', default='',
                    help='Required with --source malc. E.g. B100, B500, B1000.')
    ap.add_argument('--out-tag', default=None,
                    help='Prefix inserted into output filename '
                         '(ate_w2_{tag}_r<###>.npz). Defaults to '
                         "the --source name (e.g. 'malc_B100', 'joint', 'marginals').")
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--max-realizations', type=int, default=None)
    ap.add_argument('--n-tau-barycenter', type=int, default=4001,
                    help='τ grid resolution inside the barycenter fn (default 4001).')
    ap.add_argument('--repo', default=None)
    args = ap.parse_args()

    repo = args.repo or os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    barycenter_fn = _import_barycenter(repo)

    dataset_dir = os.path.join(args.method_dir, args.dataset)
    if args.is_split:
        density_paths = sorted(glob.glob(os.path.join(dataset_dir, 'density_r*.npz')))
    else:
        density_paths = sorted(glob.glob(os.path.join(dataset_dir, f'{args.dataset}_r*.npz')))
    if not density_paths:
        sys.exit(f'FATAL: no density NPZs under {dataset_dir}')
    if args.max_realizations:
        density_paths = density_paths[:args.max_realizations]

    if args.source == 'malc':
        assert args.malc_tag, '--source malc requires --malc-tag (e.g. B100)'

    out_tag = args.out_tag or (
        f'malc_{args.malc_tag}' if args.source == 'malc' else args.source
    )

    print(f'[bootstrap] method_dir={args.method_dir}  dataset={args.dataset}  '
          f'n_realizations={len(density_paths)}  source={args.source}'
          + (f'  malc_tag={args.malc_tag}' if args.source == 'malc' else '')
          + f'  out_tag={out_tag}',
          flush=True)

    stats = []
    for path in density_paths:
        base = os.path.basename(path)
        if base.startswith('density_r'):
            r_tag = base[len('density_'):-len('.npz')]
        else:
            r_tag = base[len(args.dataset) + 1:-len('.npz')]
        out_path = os.path.join(dataset_dir, f'ate_w2_{out_tag}_{r_tag}.npz')
        if os.path.isfile(out_path) and not args.overwrite:
            continue
        malc_path = None
        if args.source == 'malc':
            malc_path = os.path.join(dataset_dir, f'malc_ci_{args.malc_tag}_{r_tag}.npz')
            if not os.path.isfile(malc_path):
                print(f'  [skip] missing MALC dump: {malc_path}', flush=True)
                continue
        t0 = time.time()
        try:
            s = _process_realization(path, out_path, args.source, malc_path, barycenter_fn,
                                       n_tau_barycenter=args.n_tau_barycenter)
        except Exception as e:
            print(f'  [warn] {path}: {e}', file=sys.stderr, flush=True); continue
        stats.append(s)
        print(f'  [{time.strftime("%H:%M:%S")}] {r_tag}  '
              f'ate_mean={s["ate_mean"]:+.4g}  true={s["true_ate"]:+.4g}  '
              f'CI=[{s["tau_lo"]:+.4g}, {s["tau_hi"]:+.4g}]  cov={s["covered"]}  '
              f'({time.time() - t0:.2f}s)', flush=True)

    if stats:
        n = len(stats)
        cov  = np.mean([s['covered']   for s in stats])
        ln   = np.mean([s['ci_length'] for s in stats])
        bias = np.mean([s['ate_mean'] - s['true_ate'] for s in stats])
        abs_err = np.mean([abs(s['ate_mean'] - s['true_ate']) for s in stats])
        print(f'\n[done] n={n}  coverage={cov:.3f}  ci_length_mean={ln:.4g}  '
              f'ate_bias_mean={bias:+.4g}  ate_abs_err_mean={abs_err:.4g}',
              flush=True)


if __name__ == '__main__':
    main()
