"""Post-hoc compute PEHE(raw-mean) + eps_ATE(raw-mean) from existing IHDP L2
shards that were produced BEFORE methods_densities.py emitted `cate_raw_scaled`.

Uses per-query marginals already saved in the shard:
    <key>__p_y0[q, :]   (density on Y_CENTERS = 100 bins on [-1.5, 1.5])
    <key>__p_y1[q, :]
along with top-level `true_cate_raw`, `true_ate_raw`, `y_rng`.

Per query q:
    E[Y_0]_scaled  = Σ Y_CENTERS · p_y0[q] · Y_BIN
    E[Y_1]_scaled  = Σ Y_CENTERS · p_y1[q] · Y_BIN
    CATE_raw_q     = (E[Y_1]_scaled - E[Y_0]_scaled) · (y_rng / 2)

Aggregates to PEHE(raw) and eps_ATE(raw) per realization, then reports
mean ± std across realizations. On IHDP the per-query truth is Gaussian,
so mean-of-per-query-means equals OT-mean-of-barycenter exactly, and the
eps_ATE reported here is the raw-OT-mean.

Usage
-----
    python R-PFN/benchmarks/l2_ihdp/posthoc_raw_pehe.py \\
        --shards-glob "/scratch/.../out_dopfn_bb_j10_B500_step55000_K1.r*.npz" \\
        --key ours_dopfn_bb \\
        --label "Ours(DoPFN-bb J=10 55K K=1)"
"""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np

Y_EDGES = np.linspace(-1.5, 1.5, 101)
Y_CENTERS = 0.5 * (Y_EDGES[:-1] + Y_EDGES[1:])
Y_BIN = float(Y_CENTERS[1] - Y_CENTERS[0])


def _agg(vals):
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float('nan'), float('nan'), 0
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0), arr.size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards-glob', required=True)
    ap.add_argument('--key',   default='ours_dopfn_bb',
                    help='method key in the shard (default ours_dopfn_bb)')
    ap.add_argument('--label', default=None,
                    help='display label (defaults to --key)')
    args = ap.parse_args()
    label = args.label or args.key

    shards = sorted(glob.glob(args.shards_glob))
    if not shards:
        print(f'[fatal] no shards match {args.shards_glob}'); return 2

    pehe_malc, eps_malc = [], []
    pehe_raw,  eps_raw  = [], []
    n_missing_raw_input = 0

    for path in shards:
        with np.load(path) as f:
            files = set(f.files)
            need = {f'{args.key}__p_y0', f'{args.key}__p_y1',
                    'true_cate_raw', 'true_ate_raw', 'y_rng'}
            if not need <= files:
                n_missing_raw_input += 1
                continue

            p_y0 = np.asarray(f[f'{args.key}__p_y0'], dtype=np.float64)  # (Q, J_y)
            p_y1 = np.asarray(f[f'{args.key}__p_y1'], dtype=np.float64)  # (Q, J_y)
            true_cate_raw = np.asarray(f['true_cate_raw'], dtype=np.float64).reshape(-1)
            true_ate_raw  = float(f['true_ate_raw'])
            y_rng = float(f['y_rng'])

            if f'{args.key}__pehe' in files:
                pehe_malc.append(float(f[f'{args.key}__pehe']))
            if f'{args.key}__eps_ate' in files:
                eps_malc.append(float(f[f'{args.key}__eps_ate']))

        y_rng_over_2 = y_rng / 2.0
        E_y0 = (Y_CENTERS[None, :] * p_y0).sum(axis=1) * Y_BIN
        E_y1 = (Y_CENTERS[None, :] * p_y1).sum(axis=1) * Y_BIN
        cate_raw = (E_y1 - E_y0) * y_rng_over_2       # (Q,) in raw Y units
        ate_raw  = float(cate_raw.mean())
        pehe = float(np.sqrt(np.mean((cate_raw - true_cate_raw) ** 2)))
        eps  = float(abs(ate_raw - true_ate_raw) / max(abs(true_ate_raw), 1e-9))
        pehe_raw.append(pehe); eps_raw.append(eps)

    if n_missing_raw_input:
        print(f'[warn] {n_missing_raw_input} shard(s) missing required fields — skipped')

    pm_m, ps_m, n_m = _agg(pehe_malc)
    em_m, es_m, _   = _agg(eps_malc)
    pm_r, ps_r, n_r = _agg(pehe_raw)
    em_r, es_r, _   = _agg(eps_raw)

    print()
    print('── Post-hoc raw-mean PEHE / eps_ATE — IHDP ────────────────────────')
    print(f'{"Method":40s} {"sqrt(PEHE)":>16s}   {"eps_ATE":>14s}')
    print('-' * 78)
    if n_m:
        print(f'{label + " (MALC-mean)":40s} {pm_m:>6.2f} ± {ps_m:<6.2f}   '
              f'{em_m:>5.2f} ± {es_m:<5.2f}   (n={n_m})')
    if n_r:
        print(f'{label + " (raw-mean)":40s} {pm_r:>6.2f} ± {ps_r:<6.2f}   '
              f'{em_r:>5.2f} ± {es_r:<5.2f}   (n={n_r})')
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
