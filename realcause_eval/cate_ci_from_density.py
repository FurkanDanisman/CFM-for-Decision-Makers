"""CATE 95% CI / coverage / length from 1D density dumps.

Reads per-realization NPZs written by realcause_eval/{do_pfn,cpfn1d,uwyk1d}
runs with DENSITY_DUMP=1. Each NPZ must contain (see the emitters):

    edges          shape (nbins+1,)   bar-dist edges in scaled Y units
                                       (RAW Y for DoPFN — y_shift=0, y_scale=1)
    p_y0_scaled    shape (N_q, nbins) softmax marginal p(Y|do(0)) per query
    p_y1_scaled    shape (N_q, nbins) softmax marginal p(Y|do(1)) per query
    y_shift        scalar             raw = scaled*y_scale + y_shift
    y_scale        scalar             (drops out in τ = Y_1 - Y_0)
    true_cate_per_query shape (N_q,)  ground-truth CATE per query

Method:
    1. Assume Y|do(0) ⊥ Y|do(1). Then p(τ = Y_1 - Y_0) = p_y1 * flip(p_y0)
       via 1D discrete convolution (2·nbins-1 output bins).
    2. Convolution mean equals E[Y_1] - E[Y_0] — SAME as the point CATE the
       models already report. So point estimates are guaranteed consistent.
    3. Per-query CI: find τ_lo, τ_hi via linear-interp on the discrete CDF
       at levels 0.025 and 0.975.
    4. Un-scale τ_lo, τ_hi to raw units by multiplying by y_scale (y_shift
       cancels in the difference).
    5. Coverage = mean_q 1[τ_true[q] ∈ [τ_lo[q], τ_hi[q]]].
       Length   = mean_q (τ_hi[q] - τ_lo[q]).
    6. Aggregate across realizations: mean of coverages, mean of lengths.

Usage:
    python realcause_eval/cate_ci_from_density.py \\
        --out-root /scratch/.../results_realcause_all \\
        --methods cpfn1d dopfn uwyk1d
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


DATASETS = ('IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal')


def _load_density(npz_path: str):
    """Return (edges, p_y0, p_y1, y_shift, y_scale, true_cate_pq) or None."""
    try:
        with np.load(npz_path, allow_pickle=True) as z:
            required = {'edges', 'p_y0_scaled', 'p_y1_scaled', 'y_shift',
                        'y_scale', 'true_cate_per_query'}
            if not required.issubset(set(z.files)):
                return None
            edges = np.asarray(z['edges'], dtype=np.float64)
            p_y0  = np.asarray(z['p_y0_scaled'], dtype=np.float64)
            p_y1  = np.asarray(z['p_y1_scaled'], dtype=np.float64)
            y_shift = float(z['y_shift'])
            y_scale = float(z['y_scale'])
            true_cate_pq = np.asarray(z['true_cate_per_query'], dtype=np.float64)
        # Ensure per-query densities normalized.
        p_y0 /= p_y0.sum(axis=-1, keepdims=True).clip(min=1e-12)
        p_y1 /= p_y1.sum(axis=-1, keepdims=True).clip(min=1e-12)
        return edges, p_y0, p_y1, y_shift, y_scale, true_cate_pq
    except Exception as e:
        print(f'  [warn] {npz_path}: {e}', file=sys.stderr)
        return None


def _tau_grid(edges: np.ndarray) -> np.ndarray:
    """Given nbins+1 edges (uniform-spaced), return the τ bin centers on which
    the convolution p_y1 ⊗ flip(p_y0) is defined — length 2·nbins-1."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    nbins = centers.size
    width = float(centers[1] - centers[0])
    k = np.arange(2 * nbins - 1) - (nbins - 1)
    return k.astype(np.float64) * width


def _ci_from_pmf(tau_centers: np.ndarray, p_tau: np.ndarray, lo: float, hi: float):
    """Linear-interp CDF at levels lo, hi. p_tau shape (N_q, 2·nbins-1),
    tau_centers shape (2·nbins-1,). Returns (tau_lo, tau_hi) both (N_q,)."""
    p_tau = p_tau / p_tau.sum(axis=-1, keepdims=True).clip(min=1e-12)
    cdf = np.cumsum(p_tau, axis=-1)
    def _q(level):
        # Per query, find first index where cdf >= level, then linear interp.
        # cdf shape (N_q, K); tau_centers shape (K,).
        below = cdf < level
        # index of first True in "cdf >= level" per row
        idx = np.argmax(~below, axis=-1)
        # If entire cdf < level, argmax returns 0 → clip to K-1.
        idx = np.where(cdf[:, -1] < level, cdf.shape[-1] - 1, idx)
        row = np.arange(cdf.shape[0])
        c_hi = cdf[row, idx]
        c_lo = np.where(idx > 0, cdf[row, np.maximum(idx - 1, 0)], 0.0)
        t_hi = tau_centers[idx]
        t_lo = np.where(idx > 0, tau_centers[np.maximum(idx - 1, 0)], tau_centers[0])
        w = np.where(c_hi > c_lo, (level - c_lo) / (c_hi - c_lo), 0.0)
        return t_lo + w * (t_hi - t_lo)
    return _q(lo), _q(hi)


def process_npz(npz_path: str) -> tuple[float, float, int] | None:
    """→ (per-realization coverage, per-realization avg length, n_queries)."""
    loaded = _load_density(npz_path)
    if loaded is None:
        return None
    edges, p_y0, p_y1, y_shift, y_scale, true_cate_pq = loaded

    # Convolve p_y1 with flip(p_y0) per query → p(τ_scaled) of length 2N-1.
    # np.apply_along_axis is slow; vectorize with FFT since nbins is small anyway.
    from numpy.fft import rfft, irfft
    nbins = p_y0.shape[-1]
    n_out = 2 * nbins - 1
    n_fft = 1 << (n_out - 1).bit_length()   # next power of 2
    F1 = rfft(p_y1, n=n_fft, axis=-1)
    F0 = rfft(p_y0[:, ::-1], n=n_fft, axis=-1)   # flip so that convolve = correlate-with-original
    p_tau_scaled = irfft(F1 * F0, n=n_fft, axis=-1)[:, :n_out]
    p_tau_scaled = np.clip(p_tau_scaled, 0.0, None)

    tau_centers_scaled = _tau_grid(edges)
    tau_lo_s, tau_hi_s = _ci_from_pmf(tau_centers_scaled, p_tau_scaled, 0.025, 0.975)

    # τ_raw = τ_scaled * y_scale (y_shift cancels in the difference).
    tau_lo = tau_lo_s * y_scale
    tau_hi = tau_hi_s * y_scale

    inside = (true_cate_pq >= tau_lo) & (true_cate_pq <= tau_hi)
    coverage = float(np.mean(inside))
    length   = float(np.mean(tau_hi - tau_lo))
    return coverage, length, int(true_cate_pq.size)


def summarize_method_dataset(method_dir: str, dataset: str):
    """Walk NPZs, compute per-realization coverage/length, return means + SE."""
    paths = sorted(glob.glob(os.path.join(method_dir, dataset, f'{dataset}_r*.npz')))
    if not paths:
        return None
    covs, lens = [], []
    for p in paths:
        got = process_npz(p)
        if got is None:
            continue
        covs.append(got[0]); lens.append(got[1])
    if not covs:
        return None
    covs = np.asarray(covs); lens = np.asarray(lens)
    n = covs.size
    return (
        float(covs.mean()), float(covs.std(ddof=1) / np.sqrt(n)) if n > 1 else float('nan'),
        float(lens.mean()), float(lens.std(ddof=1) / np.sqrt(n)) if n > 1 else float('nan'),
        n,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', required=True,
                    help='Parent dir; expects <out-root>/<method>/<DATASET>/<D>_r<###>.npz.')
    ap.add_argument('--methods', nargs='+', default=['cpfn1d', 'dopfn', 'uwyk1d'],
                    help='Methods to include (one row each).')
    ap.add_argument('--out-md', default=None,
                    help='Also write the markdown table to this path.')
    args = ap.parse_args()

    if not os.path.isdir(args.out_root):
        sys.exit(f'FATAL: --out-root not found: {args.out_root}')

    big_len = {'CPS', 'PSID', 'PSID_bal'}

    header = '| Method | ' + ' | '.join(DATASETS) + ' |'
    sep    = '|' + '|'.join(['---'] * (1 + len(DATASETS))) + '|'
    lines = [f'\nCATE 95% CI (assume Y|do(0) ⊥ Y|do(1)) — {args.out_root}', '',
             '(each cell: coverage on top, length below; n = realizations)',
             '', header, sep]

    for method in args.methods:
        method_dir = os.path.join(args.out_root, method)
        cells = [method]
        for d in DATASETS:
            got = summarize_method_dataset(method_dir, d)
            if got is None:
                cells.append('—')
                continue
            cov_m, cov_se, len_m, len_se, n = got
            cov_str = f'{cov_m:.3f} ± {cov_se:.3f}' if np.isfinite(cov_se) else f'{cov_m:.3f}'
            if d in big_len:
                len_str = f'{len_m:,.0f} ± {len_se:,.0f}' if np.isfinite(len_se) else f'{len_m:,.0f}'
            else:
                len_str = f'{len_m:.3f} ± {len_se:.3f}' if np.isfinite(len_se) else f'{len_m:.3f}'
            cells.append(f'{cov_str}<br>{len_str} (n={n})')
        lines.append('| ' + ' | '.join(cells) + ' |')

    md = '\n'.join(lines) + '\n'
    print(md)
    if args.out_md:
        with open(args.out_md, 'w') as f:
            f.write(md)
        print(f'wrote {args.out_md}')


if __name__ == '__main__':
    main()
