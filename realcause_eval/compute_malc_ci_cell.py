"""Per-cell 2D-MALC CI computation for the 2D density-CI pipeline.

Reads every density NPZ in a (method, dataset) cell, fits 2D-MALC per query
via a persistent multiprocessing Pool (init once, reused across realizations
— MALC's numpy/scipy imports are the expensive part), computes 95% CI on
τ = Y1 − Y0 from the MALC-smoothed p(τ), and writes malc_ci_r<###>.npz
next to each density NPZ.

The aggregator (cate_ci_from_density_malc.py) then reads those.

Recipe (matches benchmarks/methods/ours.py::_fit_and_marginalize):
    1. fit_malc_inner(p_mat.T, edges, edges, K=1, B=100, seed=..., parallel=False)
    2. dmalc_2d(fit, eval_pts).reshape(N_EVAL, N_EVAL)  — smooth 2D density
    3. Diagonal-integrate → p(τ) on 401-bin scaled tau grid
    4. Per-query CDF → linear-interp quantile at 0.025 / 0.975
    5. Un-scale by y_scale to raw units (y_shift cancels in τ)

Layouts supported:
  inline  (cpfn2d, graph2d):  <method_dir>/<DATASET>/<DATASET>_r<###>.npz
  split   (dopfnbb):          <method_dir>/<DATASET>/density_r<###>.npz

Usage:
    python realcause_eval/compute_malc_ci_cell.py \\
        --method-dir /scratch/.../results_rc_2d_density/graph2d \\
        --dataset IHDP \\
        --n-workers 32 --malc-K 1 --malc-B 100 --n-eval 50 [--overwrite]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
import time
from multiprocessing import get_context

import numpy as np


# ── Worker plumbing (mirrors benchmarks/methods/ours.py::_init_worker but
#    stripped to just what we need for CI).
_GLOBAL = {}


def _init_worker(edges_np, J, bin_width, n_eval, malc_K, malc_B, repo, malc_dir):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    if repo     and repo     not in sys.path: sys.path.insert(0, repo)
    if malc_dir and malc_dir not in sys.path: sys.path.insert(0, malc_dir)
    from losses.BarDistribution2D import fit_malc_inner
    from malc_2d import dmalc_2d
    _GLOBAL['fit']   = fit_malc_inner
    _GLOBAL['dmalc'] = dmalc_2d
    _GLOBAL['edges'] = edges_np
    _GLOBAL['J']     = J
    _GLOBAL['bw']    = bin_width
    _GLOBAL['K']     = malc_K       # K passed DIRECTLY (skips BIC scan)
    _GLOBAL['B']     = malc_B
    xs = np.linspace(edges_np[0], edges_np[-1], n_eval)
    ys = np.linspace(edges_np[0], edges_np[-1], n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    _GLOBAL['xs'] = xs
    _GLOBAL['ys'] = ys
    _GLOBAL['eval_pts'] = np.column_stack([XX.ravel(), YY.ravel()])
    _GLOBAL['dy0']  = xs[1] - xs[0]
    _GLOBAL['dy1']  = ys[1] - ys[0]
    _GLOBAL['tau']  = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], 401)
    _GLOBAL['dtau'] = _GLOBAL['tau'][1] - _GLOBAL['tau'][0]


def _fit_one_query(args):
    """Return (i, p_tau) or (i, None) on solver failure.

    p_tau is the MALC-smoothed density on the 401-bin scaled tau grid,
    normalized so p_tau.sum() * dtau == 1."""
    i, p_mat_np = args
    seed = int(hashlib.md5(f'q{i}malcK1'.encode()).hexdigest()[:8], 16) % (10 ** 8)
    try:
        fit = _GLOBAL['fit'](p_mat_np.T, _GLOBAL['edges'], _GLOBAL['edges'],
                              K=_GLOBAL['K'], B_fit=_GLOBAL['B'],
                              seed=seed, parallel=False)
    except Exception as e:
        print(f'  [worker] fit failed q={i}: {e}', file=sys.stderr, flush=True)
        return i, None
    try:
        density = _GLOBAL['dmalc'](fit, _GLOBAL['eval_pts']).reshape(
            len(_GLOBAL['xs']), len(_GLOBAL['ys']))
    except Exception as e:
        print(f'  [worker] dmalc failed q={i}: {e}', file=sys.stderr, flush=True)
        return i, None
    tau = _GLOBAL['tau']
    xs, ys = _GLOBAL['xs'], _GLOBAL['ys']
    dy0, dy1 = _GLOBAL['dy0'], _GLOBAL['dy1']
    out = np.zeros_like(tau)
    for k, t in enumerate(tau):
        y1 = xs + t
        v = (y1 >= ys[0]) & (y1 <= ys[-1])
        if not v.any(): continue
        col = np.clip(np.searchsorted(xs, xs[v]), 0, len(xs) - 1)
        rf  = (y1[v] - ys[0]) / dy1
        rlo = np.clip(np.floor(rf).astype(int), 0, len(ys) - 2)
        rhi = rlo + 1
        whi = rf - rlo; wlo = 1.0 - whi
        f   = wlo * density[rlo, col] + whi * density[rhi, col]
        out[k] = f.sum() * dy0
    s = out.sum() * _GLOBAL['dtau']
    if s > 0: out = out / s
    return i, out


def _quantile_from_sorted(tau, cdf, level):
    """Linear-interp inverse-CDF at `level`. tau (1, K) or (N, K), cdf (N, K)."""
    below = cdf < level
    idx   = np.argmax(~below, axis=-1)
    idx   = np.where(cdf[..., -1] < level, cdf.shape[-1] - 1, idx)
    row   = np.arange(cdf.shape[0])
    c_hi  = cdf[row, idx]
    c_lo  = np.where(idx > 0, cdf[row, np.maximum(idx - 1, 0)], 0.0)
    t_row = np.broadcast_to(tau, cdf.shape)
    t_hi  = t_row[row, idx]
    t_lo  = np.where(idx > 0, t_row[row, np.maximum(idx - 1, 0)], t_row[row, 0])
    w     = np.where(c_hi > c_lo, (level - c_lo) / (c_hi - c_lo), 0.0)
    return t_lo + w * (t_hi - t_lo)


def _process_realization(density_npz_path, out_npz_path, pool, n_workers,
                          n_eval, malc_K, malc_B, tau_grid_size=401):
    z = np.load(density_npz_path, allow_pickle=True)
    if 'p_joint_scaled' not in z.files:
        print(f'  [skip] no p_joint_scaled in {density_npz_path}', file=sys.stderr)
        return None
    p_joint = np.asarray(z['p_joint_scaled'], dtype=np.float64)
    edges   = np.asarray(z['edges'], dtype=np.float64)
    y_scale = float(z['y_scale']); y_shift = float(z['y_shift'])
    true_cate = np.asarray(z['true_cate_per_query'], dtype=np.float64)
    N_q, J, _ = p_joint.shape

    # Normalize the joint per query (density-dump code already does this,
    # but be defensive against fp32 drift).
    s = p_joint.sum(axis=(1, 2), keepdims=True)
    p_joint = p_joint / np.where(s > 0, s, 1.0)

    p_taus = np.zeros((N_q, tau_grid_size), dtype=np.float64)
    fails = 0
    t0 = time.time()
    args = [(i, p_joint[i]) for i in range(N_q)]
    if pool is None:
        # Serial fallback (mostly for debug / n_workers=1). Requires main
        # process to have already called _init_worker.
        for a in args:
            i, p_tau = _fit_one_query(a)
            if p_tau is None: fails += 1
            else: p_taus[i] = p_tau
    else:
        for i, p_tau in pool.imap_unordered(_fit_one_query, args, chunksize=1):
            if p_tau is None: fails += 1
            else: p_taus[i] = p_tau

    # ── p(τ) → CI. tau axis matches what workers use in _init_worker.
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

    # MALC-mean CATE per query (for the density-vs-stored sanity).
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
             malc_K=int(malc_K), malc_B=int(malc_B), n_eval=int(n_eval))
    elapsed = time.time() - t0
    print(f'  [{time.strftime("%H:%M:%S")}] wrote {out_npz_path}  '
          f'N_q={N_q}  fails={fails}  ate_malc={ate_malc:+.4g}  '
          f'coverage={coverage_per_query.mean():.3f}  '
          f'length_mean={length_per_query.mean():.4g}  ({elapsed:.1f}s)',
          flush=True)
    return {'N_q': N_q, 'fails': fails, 'ate_malc': ate_malc,
            'coverage': float(coverage_per_query.mean()),
            'length':   float(length_per_query.mean()),
            'elapsed_s': elapsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method-dir', required=True,
                    help='<OUT_ROOT>/<method>; script picks up <method-dir>/<dataset>/...')
    ap.add_argument('--dataset', required=True,
                    help='Dataset dir name under method-dir. For dopfnbb use PSIDbal (legacy).')
    ap.add_argument('--is-split', action='store_true',
                    help='Use dopfnbb layout (glob density_r*.npz instead of <DATASET>_r*.npz).')
    ap.add_argument('--n-workers', type=int, default=32,
                    help='Multiprocessing pool workers (default 32).')
    ap.add_argument('--malc-K', type=int, default=1,
                    help='Force MALC component count. Default 1 (skips BIC scan).')
    ap.add_argument('--malc-B', type=int, default=100,
                    help='MALC bandwidth parameter (default 100).')
    ap.add_argument('--n-eval', type=int, default=50,
                    help='Fine 2D grid resolution for dmalc_2d evaluation (default 50).')
    ap.add_argument('--repo', default=None,
                    help='R-PFN repo root (needed by worker imports). Defaults to '
                         'the parent of this script.')
    ap.add_argument('--overwrite', action='store_true',
                    help='Redo realizations whose output NPZ already exists.')
    ap.add_argument('--max-realizations', type=int, default=None,
                    help='Cap number of realizations (mostly for smoke tests).')
    ap.add_argument('--out-tag', default='',
                    help='Prefix inserted into the output filename so multiple '
                         'hyperparameter sweeps can coexist: '
                         "'' → malc_ci_r<###>.npz (default, back-compat); "
                         "'B200' → malc_ci_B200_r<###>.npz. Also propagates to "
                         "the skip check so tagged runs don't collide with untagged ones.")
    args = ap.parse_args()

    repo = args.repo or os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    malc_dir = os.path.join(repo, 'MALC')
    assert os.path.isdir(malc_dir), f'MALC dir not found: {malc_dir}'

    dataset_dir = os.path.join(args.method_dir, args.dataset)
    if args.is_split:
        density_paths = sorted(glob.glob(os.path.join(dataset_dir, 'density_r*.npz')))
    else:
        density_paths = sorted(glob.glob(os.path.join(dataset_dir, f'{args.dataset}_r*.npz')))
    if not density_paths:
        sys.exit(f'FATAL: no density NPZs under {dataset_dir}')
    if args.max_realizations:
        density_paths = density_paths[:args.max_realizations]

    # Peek at the first NPZ to grab edges + J for pool init.
    z0 = np.load(density_paths[0], allow_pickle=True)
    if 'p_joint_scaled' not in z0.files:
        sys.exit(f'FATAL: {density_paths[0]} has no p_joint_scaled '
                 f'(is this a 2D density dump?)')
    edges0 = np.asarray(z0['edges'], dtype=np.float64)
    J = int(z0['p_joint_scaled'].shape[1])
    bin_width = float(edges0[1] - edges0[0])

    print(f'[bootstrap] method_dir={args.method_dir}  dataset={args.dataset}  '
          f'n_realizations={len(density_paths)}  J={J}  edges=[{edges0[0]:.2f}, {edges0[-1]:.2f}]',
          flush=True)
    print(f'[bootstrap] MALC K={args.malc_K}  B={args.malc_B}  n_eval={args.n_eval}  '
          f'workers={args.n_workers}', flush=True)

    ctx = get_context('spawn')
    init_args = (edges0, J, bin_width, args.n_eval, args.malc_K, args.malc_B, repo, malc_dir)

    if args.n_workers > 1:
        pool = ctx.Pool(processes=args.n_workers, initializer=_init_worker, initargs=init_args)
    else:
        # Init the main process so the serial-fallback _fit_one_query works.
        _init_worker(*init_args)
        pool = None

    tag_prefix = f'{args.out_tag}_' if args.out_tag else ''
    try:
        stats = []
        for path in density_paths:
            base = os.path.basename(path)
            if base.startswith('density_r'):
                r_tag = base[len('density_'):-len('.npz')]           # r000
            else:
                # <DATASET>_r<###>.npz → r<###>
                r_tag = base[len(args.dataset) + 1:-len('.npz')]
            out_path = os.path.join(dataset_dir, f'malc_ci_{tag_prefix}{r_tag}.npz')
            if os.path.isfile(out_path) and not args.overwrite:
                print(f'  [skip] exists: {out_path}', flush=True)
                continue
            s = _process_realization(path, out_path, pool, args.n_workers,
                                       args.n_eval, args.malc_K, args.malc_B)
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
