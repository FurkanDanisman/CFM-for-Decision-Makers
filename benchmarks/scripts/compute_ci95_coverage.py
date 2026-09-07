"""Compute 95% CATE-interval coverage + length via full p(τ).

For each per-realization density-dump npz this script:

  1. Reads edges + p_y0_scaled + p_y1_scaled + y_shift + y_scale +
     true_cate_per_query.  If p_joint_scaled is also present the model is
     treated as 2D; otherwise 1D (independence assumption).
  2. Builds p(τ) per query:
       - 2D: diagonal projection of the model's joint p(Y0, Y1) → p(τ).
       - 1D: independence, p_joint = outer(p_y0, p_y1), then diagonal projection.
  3. Extracts the 2.5% / 97.5% percentiles of p(τ) → CATE interval in scaled
     τ units. Un-scales via τ_raw = τ_scaled * y_scale (τ has no shift term).
  4. Per query: covered = 1[true_cate ∈ interval], length = interval width.
  5. Aggregates per realization (mean) and across realizations (mean ± SE).

Assumes shared bin edges for both arms (true for UWYK, DoPFN-bb, graph2d).
DoPFN-native uses its own criterion.borders with y_shift=0/y_scale=1, so
the same code path works — just no un-scaling.

Usage:
    python compute_ci95_coverage.py \
        --results-root /scratch/.../results_ci95 \
        --pattern '{model}/{dataset}/*.npz' \
        --models uwyk dopfn_native dopfn_bb graph2d \
        --out /scratch/.../ci95_ptau_summary.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


# ─── p(τ) via diagonal projection of a joint p(Y0, Y1) ───────────────────────
def p_tau_from_joint(p_joint: np.ndarray, edges: np.ndarray):
    """p_joint: (J, J) with axes (Y0-bin, Y1-bin).  edges: (J+1,) shared.

    Returns (tau_edges: (2J,), p_tau: (2J-1,)) — uniform-width bins of dy each.
    """
    J = p_joint.shape[0]
    dy = float(edges[1] - edges[0])
    K = 2 * J - 1
    p_tau = np.zeros(K, dtype=np.float64)
    # k = i1 - i0 shifted so k index in [0, 2J-2]; k=J-1 is the central τ=0 bin.
    for k_idx in range(K):
        k = k_idx - (J - 1)                     # i1 - i0
        i0_min = max(0, -k)
        i0_max = min(J, J - k)
        for i0 in range(i0_min, i0_max):
            i1 = i0 + k
            p_tau[k_idx] += p_joint[i0, i1]
    # τ centers = (i1 - i0) * dy; edges = center ± dy/2.
    tau_centers = np.arange(-(J - 1), J) * dy
    tau_edges = np.concatenate([
        [tau_centers[0] - dy / 2.0],
        (tau_centers[:-1] + tau_centers[1:]) / 2.0,
        [tau_centers[-1] + dy / 2.0],
    ]).astype(np.float64)
    return tau_edges, p_tau


def p_tau_batch(p_y0: np.ndarray, p_y1: np.ndarray,
                 edges: np.ndarray, p_joint: np.ndarray | None):
    """Compute p(τ) for a batch of queries.

    p_y0, p_y1 : (N, J)
    edges      : (J+1,)
    p_joint    : (N, J, J) or None
        If None → 1D independence: joint = outer(p_y0, p_y1).
        If given → 2D direct: use model's joint.

    Returns (tau_edges: (2J,), p_tau_batch: (N, 2J-1)).
    """
    N, J = p_y0.shape
    dy = float(edges[1] - edges[0])
    K = 2 * J - 1
    p_tau_all = np.zeros((N, K), dtype=np.float64)

    if p_joint is None:
        # Independence — outer product per query.
        for i0 in range(J):
            for i1 in range(J):
                k_idx = (i1 - i0) + (J - 1)
                p_tau_all[:, k_idx] += p_y0[:, i0] * p_y1[:, i1]
    else:
        for i0 in range(J):
            for i1 in range(J):
                k_idx = (i1 - i0) + (J - 1)
                p_tau_all[:, k_idx] += p_joint[:, i0, i1]

    tau_centers = np.arange(-(J - 1), J) * dy
    tau_edges = np.concatenate([
        [tau_centers[0] - dy / 2.0],
        (tau_centers[:-1] + tau_centers[1:]) / 2.0,
        [tau_centers[-1] + dy / 2.0],
    ]).astype(np.float64)
    return tau_edges, p_tau_all


# ─── Per-query inverse-CDF interval from a batch of histograms ───────────────
def interval_from_hist_batch(edges: np.ndarray, p: np.ndarray,
                              level: float = 0.95):
    """edges: (K+1,), p: (N, K).  Returns L, U : (N,) each."""
    alpha = 0.5 * (1.0 - level)
    p = np.clip(p.astype(np.float64), 0.0, None)
    row_sums = p.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    p = p / row_sums
    cdf = np.cumsum(p, axis=1)
    cdf = np.concatenate([np.zeros((cdf.shape[0], 1)), cdf], axis=1)  # (N, K+1)
    N = p.shape[0]
    L = np.empty(N, dtype=np.float64)
    U = np.empty(N, dtype=np.float64)
    for i in range(N):
        L[i] = np.interp(alpha,       cdf[i], edges)
        U[i] = np.interp(1.0 - alpha, cdf[i], edges)
    return L, U


# ─── Per-realization computation ─────────────────────────────────────────────
def process_npz(path: str):
    try:
        with np.load(path) as z:
            keys = set(z.files)
            if not {'edges', 'p_y0_scaled', 'p_y1_scaled', 'true_cate_per_query'} <= keys:
                return None
            edges     = z['edges'].astype(np.float64)
            p_y0      = z['p_y0_scaled'].astype(np.float64)
            p_y1      = z['p_y1_scaled'].astype(np.float64)
            y_shift   = float(z['y_shift'])
            y_scale   = float(z['y_scale'])
            true_cate = z['true_cate_per_query'].astype(np.float64)
            p_joint   = z['p_joint_scaled'].astype(np.float64) if 'p_joint_scaled' in keys else None
    except Exception as e:
        print(f'[warn] {path}: {type(e).__name__}: {e}', file=sys.stderr)
        return None

    tau_edges, p_tau = p_tau_batch(p_y0, p_y1, edges, p_joint=p_joint)
    L_scaled, U_scaled = interval_from_hist_batch(tau_edges, p_tau, level=0.95)
    # τ has no shift term (mean of Y1 - Y0 cancels y_shift). Un-scale by y_scale only.
    L = L_scaled * y_scale
    U = U_scaled * y_scale
    covered = ((L <= true_cate) & (true_cate <= U)).astype(np.float64)
    length  = U - L

    result = {
        'n_queries':      int(true_cate.size),
        'is_2d':          bool(p_joint is not None),
        'coverage':       float(covered.mean()),
        'length_mean':    float(length.mean()),
        'true_cate_mean': float(true_cate.mean()),
        'true_cate_std':  float(true_cate.std()),
    }
    return result


# ─── Aggregation across realizations ─────────────────────────────────────────
def aggregate_dataset(per_real_rows: list):
    if not per_real_rows:
        return None
    out = {
        'n_realizations': len(per_real_rows),
        'model_type':     '2D' if any(r['is_2d'] for r in per_real_rows) else '1D',
    }
    for k in ('coverage', 'length_mean'):
        vals = np.array([r[k] for r in per_real_rows], dtype=np.float64)
        out[k + '_mean'] = float(vals.mean())
        out[k + '_se']   = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
    return out


# ─── CLI ─────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results-root', required=True)
    p.add_argument('--pattern', default='{model}/{dataset}/*.npz',
                   help='Glob with {model} and {dataset} placeholders.')
    p.add_argument('--models', nargs='+', required=True,
                   help='Model subdir names to scan.')
    p.add_argument('--datasets', nargs='+',
                   default=['IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal'])
    p.add_argument('--out', required=True, help='CSV output path.')
    return p.parse_args()


def main():
    args = _parse_args()

    rows_out = []
    for model in args.models:
        for dataset in args.datasets:
            glob_pat = os.path.join(
                args.results_root,
                args.pattern.format(model=model, dataset=dataset),
            )
            paths = sorted(glob.glob(glob_pat))
            if not paths:
                print(f'[skip] {model:14s} {dataset:10s} no npzs matched {glob_pat}',
                      file=sys.stderr)
                continue
            per_real = []
            for p in paths:
                r = process_npz(p)
                if r is not None:
                    per_real.append(r)
            agg = aggregate_dataset(per_real)
            if agg is None:
                print(f'[skip] {model:14s} {dataset:10s} no usable npzs (missing keys?)',
                      file=sys.stderr)
                continue
            rows_out.append({'model': model, 'dataset': dataset, **agg})
            print(f'{model:14s} {dataset:10s} type={agg["model_type"]}  '
                  f'R={agg["n_realizations"]:4d}  '
                  f'coverage={agg["coverage_mean"]:.3f}±{agg["coverage_se"]:.3f}  '
                  f'length={agg["length_mean_mean"]:.3f}±{agg["length_mean_se"]:.3f}',
                  flush=True)

    if not rows_out:
        print('[error] no rows.', file=sys.stderr); sys.exit(1)

    all_cols = sorted({k for r in rows_out for k in r.keys()},
                       key=lambda k: (k not in ('model', 'dataset', 'model_type'), k))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(','.join(all_cols) + '\n')
        for r in rows_out:
            f.write(','.join(str(r.get(c, '')) for c in all_cols) + '\n')
    print(f'\n[ok] wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
