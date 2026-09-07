"""Compute 95% CI coverage + interval length for CATE from density-dump npzs.

For each per-realization npz under a `<model>/<dataset>/` dir (or the flatter
layout used by some sbatches) this script:

  1. Reads edges + p_y0_scaled + p_y1_scaled + y_shift + y_scale.
  2. Per query, inverts the CDF at α=0.025 and 1-α=0.975 (linear-interp on the
     cumsum) → (L0, U0) and (L1, U1) in SCALED Y space.
  3. Un-scales to RAW Y with y_raw = y_scaled * y_scale + y_shift.
  4. Minkowski-sum CATE interval: [L1 - U0, U1 - L0].
     - coverage = 1[L1-U0 <= true_cate <= U1-L0]
     - length   = (U1-L0) - (L1-U0)
  5. For 2D models with `p_joint_scaled` (N, J, J): also computes DIRECT
     p(τ) via diagonal projection → percentile → CATE interval.

Aggregation:
  - Per realization: mean coverage rate, mean interval length.
  - Per dataset: mean ± SE across realizations for both metrics.

Usage:
    python compute_ci95_coverage.py \
        --results-root /scratch/.../results_ci95 \
        --out          /scratch/.../ci95_summary.csv

`--results-root` should contain one subdir per (model, dataset) pair (or
per-model subdir with per-dataset children). See --pattern to control glob.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np


# ─── Per-query intervals from a histogram ────────────────────────────────────
def interval_from_hist(edges: np.ndarray, p: np.ndarray, level: float = 0.95):
    """Inverse-CDF interval per row of `p` (histogram probabilities).

    edges : (K+1,) bin edges (float32/float64)
    p     : (N, K)  bin probabilities (assumed non-negative, roughly sum to 1)
    level : coverage level (0.95 → 2.5th / 97.5th pct)

    Returns
    -------
    L, U : (N,) each — lower / upper bin-edge coordinates.
    """
    alpha = 0.5 * (1.0 - level)
    p = np.clip(p.astype(np.float64), 0.0, None)
    row_sums = p.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    p = p / row_sums
    cdf = np.cumsum(p, axis=1)                                    # (N, K)
    cdf = np.concatenate([np.zeros((cdf.shape[0], 1)), cdf], axis=1)  # (N, K+1)
    # np.interp is 1D; loop rows.
    N = p.shape[0]
    L = np.empty(N, dtype=np.float64)
    U = np.empty(N, dtype=np.float64)
    for i in range(N):
        L[i] = np.interp(alpha,       cdf[i], edges)
        U[i] = np.interp(1.0 - alpha, cdf[i], edges)
    return L, U


def unscale(y_scaled: np.ndarray, y_shift: float, y_scale: float) -> np.ndarray:
    return y_scaled * y_scale + y_shift


# ─── Direct p(τ) from a 2D joint (diagonal projection) ───────────────────────
def p_tau_from_joint(p_joint: np.ndarray, edges: np.ndarray):
    """Diagonal-integrate p_joint to get p(τ).

    p_joint : (N, J, J) with axes (Y0-bin, Y1-bin).
    edges   : (J+1,) shared for both axes.

    Returns
    -------
    tau_edges : (2J-1,) bin edges for τ = Y1 - Y0. Uniform spacing.
    p_tau     : (N, 2J-2) probabilities per τ-bin.
    """
    N, J, _ = p_joint.shape
    # τ-index k = i1 - i0, ranging from -(J-1) to +(J-1). We use bin centers
    # of Y0 and Y1 to define τ-centers ≈ i1 - i0 in bin-index space, then
    # convert to Y units via the shared bin width.
    dy = float(edges[1] - edges[0])
    # Sum along diagonals i1 - i0 = k.
    K = 2 * J - 1                       # number of τ centers
    tau_centers = np.arange(-(J - 1), J) * dy   # (K,)
    tau_edges = np.concatenate([
        [tau_centers[0] - dy / 2.0],
        (tau_centers[:-1] + tau_centers[1:]) / 2.0,
        [tau_centers[-1] + dy / 2.0],
    ]).astype(np.float64)              # (K+1,) — uniform spacing dy
    p_tau = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        i0_min = max(0, -(k - (J - 1)))
        i0_max = min(J, J - (k - (J - 1)))
        for i0 in range(i0_min, i0_max):
            i1 = i0 + (k - (J - 1))
            p_tau[:, k] += p_joint[:, i0, i1]
    return tau_edges, p_tau


# ─── Main per-realization computation ────────────────────────────────────────
def process_npz(path: str, direct_tau: bool = False):
    """Return dict of per-realization aggregate metrics, or None if not usable."""
    try:
        with np.load(path) as z:
            keys = set(z.files)
            if not {'edges', 'p_y0_scaled', 'p_y1_scaled', 'true_cate_per_query'} <= keys:
                return None
            edges   = z['edges'].astype(np.float64)
            p_y0    = z['p_y0_scaled'].astype(np.float64)
            p_y1    = z['p_y1_scaled'].astype(np.float64)
            y_shift = float(z['y_shift'])
            y_scale = float(z['y_scale'])
            true_cate = z['true_cate_per_query'].astype(np.float64)
            p_joint = z['p_joint_scaled'].astype(np.float64) if 'p_joint_scaled' in keys else None
    except Exception as e:
        print(f'[warn] {path}: {type(e).__name__}: {e}', file=sys.stderr)
        return None

    L0s, U0s = interval_from_hist(edges, p_y0)
    L1s, U1s = interval_from_hist(edges, p_y1)
    L0 = unscale(L0s, y_shift, y_scale)
    U0 = unscale(U0s, y_shift, y_scale)
    L1 = unscale(L1s, y_shift, y_scale)
    U1 = unscale(U1s, y_shift, y_scale)

    # Minkowski-sum CATE interval
    cate_lo = L1 - U0
    cate_hi = U1 - L0
    covered = ((cate_lo <= true_cate) & (true_cate <= cate_hi)).astype(np.float64)
    length  = (cate_hi - cate_lo)

    result = {
        'n_queries': int(true_cate.size),
        'coverage_mink':    float(covered.mean()),
        'length_mink_mean': float(length.mean()),
    }

    if direct_tau and p_joint is not None:
        tau_edges, p_tau_arr = p_tau_from_joint(p_joint, edges)
        Ltau_s, Utau_s = interval_from_hist(tau_edges, p_tau_arr)
        # τ scaling: dy_raw = dy_scaled * y_scale. τ has no shift.
        cate_lo_d = Ltau_s * y_scale
        cate_hi_d = Utau_s * y_scale
        covered_d = ((cate_lo_d <= true_cate) & (true_cate <= cate_hi_d)).astype(np.float64)
        length_d  = (cate_hi_d - cate_lo_d)
        result['coverage_direct']    = float(covered_d.mean())
        result['length_direct_mean'] = float(length_d.mean())

    return result


# ─── Aggregation over realizations ───────────────────────────────────────────
def aggregate_dataset(per_real_rows: list):
    if not per_real_rows:
        return None
    keys = ('coverage_mink', 'length_mink_mean', 'coverage_direct', 'length_direct_mean')
    out = {'n_realizations': len(per_real_rows)}
    for k in keys:
        vals = np.array([r[k] for r in per_real_rows if k in r], dtype=np.float64)
        if vals.size == 0:
            continue
        out[k + '_mean'] = float(vals.mean())
        out[k + '_se']   = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
    return out


# ─── CLI ─────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results-root', required=True,
                   help='Root dir with model×dataset subtrees. --pattern controls layout.')
    p.add_argument('--pattern', default='{model}/{dataset}/*.npz',
                   help='Glob pattern with {model} {dataset} placeholders. '
                        'Default matches OUT_ROOT/<model>/<dataset>/*.npz.')
    p.add_argument('--models', nargs='+', required=True,
                   help='Model subdirs to scan (e.g. cpfn1d cpfn2d graph2d uwyk).')
    p.add_argument('--datasets', nargs='+',
                   default=['IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal'])
    p.add_argument('--direct-tau', action='store_true', default=True,
                   help='Also compute the direct p(τ) interval for 2D models '
                        '(uses p_joint_scaled if present).')
    p.add_argument('--out', required=True, help='CSV output path.')
    return p.parse_args()


def main():
    args = _parse_args()

    rows_out = []
    for model in args.models:
        for dataset in args.datasets:
            glob_pat = os.path.join(
                args.results_root,
                args.pattern.format(model=model, dataset=dataset)
            )
            paths = sorted(glob.glob(glob_pat))
            if not paths:
                print(f'[skip] {model:10s} {dataset:10s} no npzs matched {glob_pat}',
                      file=sys.stderr)
                continue
            per_real = []
            for p in paths:
                r = process_npz(p, direct_tau=args.direct_tau)
                if r is not None:
                    per_real.append(r)
            agg = aggregate_dataset(per_real)
            if agg is None:
                print(f'[skip] {model:10s} {dataset:10s} no usable npzs (missing keys?)',
                      file=sys.stderr)
                continue
            rows_out.append({'model': model, 'dataset': dataset, **agg})
            summary = (f'coverage={agg.get("coverage_mink_mean", float("nan")):.3f} '
                       f'length={agg.get("length_mink_mean_mean", float("nan")):.3f}')
            if 'coverage_direct_mean' in agg:
                summary += (f'  |  direct: coverage={agg["coverage_direct_mean"]:.3f} '
                            f'length={agg["length_direct_mean_mean"]:.3f}')
            print(f'{model:10s} {dataset:10s} R={agg["n_realizations"]:4d}  {summary}',
                  flush=True)

    if not rows_out:
        print('[error] no rows to write.', file=sys.stderr)
        sys.exit(1)

    # Write CSV.
    all_cols = sorted({k for r in rows_out for k in r.keys()},
                       key=lambda k: (k not in ('model', 'dataset'), k))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(','.join(all_cols) + '\n')
        for r in rows_out:
            f.write(','.join(str(r.get(c, '')) for c in all_cols) + '\n')
    print(f'\n[ok] wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
