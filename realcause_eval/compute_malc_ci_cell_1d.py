"""Per-cell 2D-MALC CI computation for 1D-method density dumps.

Sibling of compute_malc_ci_cell.py, but reads the 1D density-dump NPZ
(schemas: bar for cpfn1d/dopfn, UWYK's K+2 for uwyk1d) and constructs
the independence-implied joint p_joint[q, i, j] = p_y0[q, i] * p_y1[q, j]
before handing off to the same MALC path (fit_malc_inner → dmalc_2d →
diagonal-integrate → CDF quantiles).

Reuses the worker init + fit + quantile helpers from compute_malc_ci_cell
so any change to the MALC recipe stays in one place.

UWYK schema quirk: p_y0/p_y1 have (K + 2) columns (K bars + 2 tail
atoms) with only K+1 edges. Trim the two tail columns; drop the small
tail mass; renormalize. Same mild-bias tradeoff as eval_density_metrics.

UWYK bin-count: raw K is 1000 which makes p_joint 1M entries per query
— MALC's cvxpy solve would blow up. --downsample-bins-to N mean-pools
p_y0/p_y1 (sum adjacent K/N bars, keep every (K/N)-th edge). Preserves
mass; MALC-tractable. Default 100.

Usage:
    python realcause_eval/compute_malc_ci_cell_1d.py \\
        --method-dir /scratch/.../results_1d_density/cpfn1d \\
        --dataset IHDP \\
        --n-workers 32 --malc-K 1 --malc-B 100 --n-eval 50 \\
        [--downsample-bins-to 100] [--overwrite]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

from compute_malc_ci_cell import (
    _init_worker,
    _fit_one_query,
    _quantile_from_sorted,
)


def _pick_downsample_factor(K, max_J):
    """Return the smallest factor f ≥ 1 such that K % f == 0 and K/f ≤ max_J.

    Preserves exact mass conservation (only divisors of K allowed). For
    cpfn1d K=1024 with max_J=100 → f=16 (J_out=64); for uwyk1d K=1000
    with max_J=100 → f=10 (J_out=100); dopfn K=100 with max_J=100 →
    f=1 (no downsample).
    """
    if not max_J or K <= max_J: return 1
    for f in range(int(np.ceil(K / max_J)), K + 1):
        if K % f == 0: return f
    return 1        # unreachable — f=K always satisfies


def _is_uniform_grid(x, tol=1e-4):
    """True if consecutive gaps in x agree to within `tol` relative to mean."""
    w = np.diff(x)
    if w.size == 0: return True
    return (w.max() - w.min()) / max(w.mean(), 1e-30) < tol


def _rasterize_atoms_to_uniform(p_y, centers, J_uniform=None, pad_frac=0.02):
    """Treat each bar as a point mass at centers[i] with weight p_y[:, i].
    Nearest-bin rasterize onto a uniform grid covering [centers.min(),
    centers.max()] with a small pad so extreme atoms don't collide with
    the boundary. Mass-conservative (multinomial → multinomial).

    p_y:       (N_q, K)   per-bar probability (sums to 1 per row)
    centers:   (K,)       atom positions (possibly non-uniform, e.g.
                          dopfn's effective_centers with tail-adjusted mids)
    J_uniform: uniform target bin count (default K)
    returns:   (p_out (N_q, J_uniform), edges_uniform (J_uniform+1,))

    Matches the convention in eval_density_metrics._load_1d_ptau_raw_on_grid:
    bars ARE point masses at centers, NOT uniform-density boxes over edges.
    """
    N_q, K = p_y.shape
    J = J_uniform or K
    c_lo, c_hi = float(centers.min()), float(centers.max())
    pad = pad_frac * (c_hi - c_lo) if c_hi > c_lo else 1.0
    edges_uniform = np.linspace(c_lo - pad, c_hi + pad, J + 1)
    dtau = float(edges_uniform[1] - edges_uniform[0])
    # Nearest-bin index for each atom's center.
    idx = np.round((centers - edges_uniform[0]) / dtau).astype(int)
    idx = np.clip(idx, 0, J - 1)
    p_out = np.zeros((N_q, J), dtype=np.float64)
    for i in range(K):
        p_out[:, idx[i]] += p_y[:, i]
    return p_out, edges_uniform


def _load_1d_joint(npz_path, downsample_max_J):
    """Return (p_joint (N_q, J, J), edges (J+1,), y_scale, y_shift, true_cate).

    UWYK trim: if p_y0 has K+2 columns and edges has K+1, drop the two
    tail columns and renormalize (drops small tail mass; matches the
    convention in eval_density_metrics._load_1d_ptau_raw_on_grid).

    Center convention (aligns with cate_ci_from_density and
    eval_density_metrics): bars are POINT masses at `centers`, where
    centers = z['effective_centers'] if present (dopfn's tail-adjusted
    first/last-bar means), else 0.5*(edges[:-1]+edges[1:]).

    Uniform-grid enforcement: MALC's _fit_component_2d uses
    `delta_x = grid_x[1] - grid_x[0]` as if bins were uniform. When
    `centers` are non-uniform (dopfn), we nearest-bin rasterize p_y0/p_y1
    onto a uniform grid spanning [centers.min(), centers.max()] with the
    same J. Mass-conservative. Without this, MALC's beta computation
    explodes and all queries fail (verified on dopfn IHDP smoke).

    Downsample: if K_bars > downsample_max_J, mean-pool p_y0/p_y1 by
    the largest factor f such that K/f ≤ downsample_max_J and K%f==0
    (preserves total mass exactly; keeps every f-th edge). Applied
    AFTER uniformization so both cases share the same downsample path.
    """
    with np.load(npz_path, allow_pickle=True) as z:
        p_y0    = np.asarray(z['p_y0_scaled'], dtype=np.float64)
        p_y1    = np.asarray(z['p_y1_scaled'], dtype=np.float64)
        edges   = np.asarray(z['edges'],       dtype=np.float64)
        y_scale = float(z['y_scale'])
        y_shift = float(z['y_shift'])
        true_cate = np.asarray(z['true_cate_per_query'], dtype=np.float64)
        eff_centers = (np.asarray(z['effective_centers'], dtype=np.float64)
                        if 'effective_centers' in z.files else None)

    N_q, K = p_y0.shape
    # UWYK: K bars + 2 tail atoms, K+1 edges. Trim tails; effective_centers
    # aren't written by UWYK so eff_centers stays None → falls through to
    # bar-mid centers below.
    if K == edges.size + 1:
        p_y0 = p_y0[:, 1:-1]
        p_y1 = p_y1[:, 1:-1]
        K = p_y0.shape[1]
    assert K == edges.size - 1, (
        f'{npz_path}: shape mismatch: p_y0 has {p_y0.shape[1]} bars but '
        f'{edges.size - 1} bar edges expected')

    # Centers: prefer effective_centers when the writer stored them (dopfn).
    centers = eff_centers if (eff_centers is not None and eff_centers.size == K) \
               else 0.5 * (edges[:-1] + edges[1:])

    # Uniformize if centers are non-uniform (dopfn's effective_centers span
    # a data-driven interior range; cpfn1d/uwyk1d bar mids are already
    # uniform → this is a no-op there).
    if not _is_uniform_grid(centers):
        p_y0, edges = _rasterize_atoms_to_uniform(p_y0, centers)
        p_y1, _     = _rasterize_atoms_to_uniform(p_y1, centers)
        # After rasterization, the effective K is unchanged and edges are
        # uniform over [centers.min(), centers.max()] + a small pad.
        K = p_y0.shape[1]

    # Optional downsample by mean-pool. Choose the largest divisor of K
    # that keeps J ≤ downsample_max_J. Exact mass conservation.
    factor = _pick_downsample_factor(K, downsample_max_J)
    if factor > 1:
        J_out = K // factor
        p_y0 = p_y0.reshape(N_q, J_out, factor).sum(axis=-1)
        p_y1 = p_y1.reshape(N_q, J_out, factor).sum(axis=-1)
        edges = edges[::factor]
        K = J_out
        assert edges.size == K + 1, (
            f'downsample produced {edges.size} edges for K={K}; expected {K+1}')

    # Renormalize per-arm per-query (defensive; tail-trim drops mass).
    p_y0 = p_y0 / p_y0.sum(axis=-1, keepdims=True).clip(min=1e-12)
    p_y1 = p_y1 / p_y1.sum(axis=-1, keepdims=True).clip(min=1e-12)

    p_joint = p_y0[:, :, None] * p_y1[:, None, :]        # (N_q, K, K)
    return p_joint, edges, y_scale, y_shift, true_cate


def _process_realization_1d(density_npz_path, out_npz_path, pool, n_workers,
                              n_eval, malc_K, malc_B, downsample_max_J,
                              tau_grid_size=401):
    p_joint, edges, y_scale, y_shift, true_cate = _load_1d_joint(
        density_npz_path, downsample_max_J)
    N_q, J, _ = p_joint.shape

    # Normalize the joint per query — outer product of already-normalized
    # marginals sums to 1, but fp32 renorm on-load could nudge it.
    s = p_joint.sum(axis=(1, 2), keepdims=True)
    p_joint = p_joint / np.where(s > 0, s, 1.0)

    p_taus = np.zeros((N_q, tau_grid_size), dtype=np.float64)
    fails = 0
    t0 = time.time()
    args = [(i, p_joint[i]) for i in range(N_q)]
    if pool is None:
        for a in args:
            i, p_tau = _fit_one_query(a)
            if p_tau is None: fails += 1
            else: p_taus[i] = p_tau
    else:
        for i, p_tau in pool.imap_unordered(_fit_one_query, args, chunksize=1):
            if p_tau is None: fails += 1
            else: p_taus[i] = p_tau

    xs = np.linspace(edges[0], edges[-1], n_eval)
    ys = xs.copy()
    tau_axis = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], tau_grid_size)
    dtau     = tau_axis[1] - tau_axis[0]
    cdf = np.cumsum(p_taus * dtau, axis=-1)
    cdf = cdf / cdf[:, -1:].clip(min=1e-12)
    tau_lo_axis = _quantile_from_sorted(tau_axis[None, :], cdf, 0.025)
    tau_hi_axis = _quantile_from_sorted(tau_axis[None, :], cdf, 0.975)
    tau_lo = tau_lo_axis * y_scale
    tau_hi = tau_hi_axis * y_scale
    coverage_per_query = ((true_cate >= tau_lo) & (true_cate <= tau_hi)).astype(np.float32)
    length_per_query   = (tau_hi - tau_lo).astype(np.float32)

    cate_malc_axis = (tau_axis[None, :] * p_taus * dtau).sum(axis=-1)
    cate_malc      = cate_malc_axis * y_scale
    ate_malc       = float(cate_malc.mean())

    np.savez(out_npz_path,
             p_taus_scaled=p_taus.astype(np.float32),
             tau_scaled=tau_axis.astype(np.float32),
             cate_malc_per_query=cate_malc.astype(np.float32),
             tau_lo=tau_lo.astype(np.float32),
             tau_hi=tau_hi.astype(np.float32),
             coverage_per_query=coverage_per_query,
             length_per_query=length_per_query,
             ate_malc=np.float32(ate_malc),
             y_scale=np.float32(y_scale),
             y_shift=np.float32(y_shift),
             true_cate_per_query=true_cate.astype(np.float32),
             n_workers=n_workers, n_fails=fails,
             malc_K=int(malc_K), malc_B=int(malc_B), n_eval=int(n_eval),
             J_effective=int(J))
    elapsed = time.time() - t0
    print(f'  [{time.strftime("%H:%M:%S")}] wrote {out_npz_path}  '
          f'N_q={N_q}  J={J}  fails={fails}  ate_malc={ate_malc:+.4g}  '
          f'coverage={coverage_per_query.mean():.3f}  '
          f'length_mean={length_per_query.mean():.4g}  ({elapsed:.1f}s)',
          flush=True)
    return {'N_q': N_q, 'J': J, 'fails': fails, 'ate_malc': ate_malc,
            'coverage': float(coverage_per_query.mean()),
            'length':   float(length_per_query.mean()),
            'elapsed_s': elapsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method-dir', required=True,
                    help='<OUT_ROOT>/<method>; script picks up <method-dir>/<dataset>/...')
    ap.add_argument('--dataset', required=True,
                    help='Dataset dir name under method-dir.')
    ap.add_argument('--n-workers', type=int, default=32)
    ap.add_argument('--malc-K', type=int, default=1,
                    help='Force MALC component count (default 1).')
    ap.add_argument('--malc-max-K', type=int, default=0,
                    help='If > 0, BIC-select K in {1..MAX_K} per query.')
    ap.add_argument('--malc-B', type=int, default=100,
                    help='MALC bandwidth (default 100).')
    ap.add_argument('--n-eval', type=int, default=50)
    ap.add_argument('--downsample-max-J', type=int, default=100,
                    help='Upper bound on effective J after mean-pool. Actual '
                         'J_out is the largest divisor of K that is ≤ this. '
                         'Preserves total mass exactly (only divisor factors '
                         'allowed). Needed to keep MALC tractable — uwyk1d '
                         'K=1000 → J=100, cpfn1d K=1024 → J=64. Set to 0 '
                         'to disable.')
    ap.add_argument('--repo', default=None)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--max-realizations', type=int, default=None)
    ap.add_argument('--out-tag', default='',
                    help="Filename tag → malc_ci_{TAG}_r<###>.npz.")
    args = ap.parse_args()

    repo = args.repo or os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    malc_dir = os.path.join(repo, 'MALC')
    assert os.path.isdir(malc_dir), f'MALC dir not found: {malc_dir}'

    dataset_dir = os.path.join(args.method_dir, args.dataset)
    density_paths = sorted(glob.glob(os.path.join(dataset_dir, f'{args.dataset}_r*.npz')))
    if not density_paths:
        sys.exit(f'FATAL: no 1D density NPZs under {dataset_dir}')
    if args.max_realizations:
        density_paths = density_paths[:args.max_realizations]

    # Peek by ACTUALLY running _load_1d_joint on the first NPZ — this
    # captures uniformization (dopfn's non-uniform effective_centers →
    # uniform edges) + downsample together, so pool init sees the same
    # edges the workers will see.
    with np.load(density_paths[0], allow_pickle=True) as z0:
        K_raw = int(z0['p_y0_scaled'].shape[1])
        edges_raw = np.asarray(z0['edges'], dtype=np.float64)
    K_bars = K_raw - 2 if K_raw == edges_raw.size + 1 else K_raw
    factor = _pick_downsample_factor(K_bars, args.downsample_max_J)
    p_joint0, edges0, _, _, _ = _load_1d_joint(density_paths[0], args.downsample_max_J)
    J = int(p_joint0.shape[1])
    bin_width = float(edges0[1] - edges0[0])

    _raw_desc = (f'[raw edges=[{edges_raw[0]:.4f}, {edges_raw[-1]:.4f}] '
                 f'δ_first={edges_raw[1]-edges_raw[0]:.4g}]')
    _grid_kind = ('uniform' if _is_uniform_grid(edges_raw)
                  else f'non-uniform → rasterized on effective_centers')
    print(f'[bootstrap] method_dir={args.method_dir}  dataset={args.dataset}  '
          f'n_realizations={len(density_paths)}  K_bars={K_bars}  '
          f'downsample_factor={factor}  J_effective={J}  '
          f'grid={_grid_kind}  '
          f'edges_used=[{edges0[0]:.4f}, {edges0[-1]:.4f}]  '
          f'{_raw_desc}', flush=True)
    print(f'[bootstrap] MALC K={args.malc_K}  B={args.malc_B}  n_eval={args.n_eval}  '
          f'workers={args.n_workers}  downsample_max_J={args.downsample_max_J}',
          flush=True)

    ctx = get_context('spawn')
    init_args = (edges0, J, bin_width, args.n_eval, args.malc_K, args.malc_B, repo, malc_dir,
                  args.malc_max_K)

    if args.n_workers > 1:
        pool = ctx.Pool(processes=args.n_workers, initializer=_init_worker, initargs=init_args)
    else:
        _init_worker(*init_args)
        pool = None

    tag_prefix = f'{args.out_tag}_' if args.out_tag else ''
    try:
        stats = []
        for path in density_paths:
            base = os.path.basename(path)
            r_tag = base[len(args.dataset) + 1:-len('.npz')]     # r<###>
            out_path = os.path.join(dataset_dir, f'malc_ci_{tag_prefix}{r_tag}.npz')
            if os.path.isfile(out_path) and not args.overwrite:
                print(f'  [skip] exists: {out_path}', flush=True)
                continue
            s = _process_realization_1d(path, out_path, pool, args.n_workers,
                                          args.n_eval, args.malc_K, args.malc_B,
                                          args.downsample_max_J)
            if s: stats.append(s)
    finally:
        if pool is not None:
            pool.close(); pool.join()

    if stats:
        cov = np.array([s['coverage'] for s in stats])
        ln  = np.array([s['length']   for s in stats])
        tot_fails = int(sum(s['fails'] for s in stats))
        tot_time  = float(sum(s['elapsed_s'] for s in stats))
        print(f'[done] realizations={len(stats)}  fails={tot_fails}  '
              f'coverage={cov.mean():.3f} ± {cov.std(ddof=1)/np.sqrt(cov.size):.3f}  '
              f'length_mean={ln.mean():.4g} ± {ln.std(ddof=1)/np.sqrt(ln.size):.4g}  '
              f'total_time={tot_time:.1f}s', flush=True)


if __name__ == '__main__':
    main()
