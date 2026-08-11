"""Two-row-per-dataset summary for Ours(fn=50) at MALC_B=500.

Row 1  ─  raw  : PEHE(raw-mean)   | eps_ATE(raw-OT-mean)
Row 2  ─  MALC : PEHE(MALC-mean)  | eps_ATE(MALC-OT-mean)

Fields expected in each shard:
    pehe_ours_mean          (per-query CATE from raw p_mat marginals -> PEHE)
    pehe_ours_malc_mean     (per-query CATE from MALC-smoothed p(τ)  -> PEHE)
    err_ours_ot_mean_raw    (mean of W2 barycenter of raw per-query p(τ))
    err_ours_ot_mean        (mean of W2 barycenter of MALC per-query p(τ))

Usage
-----
    python R-PFN/benchmarks/summary_table3_two_row.py \\
        --results /scratch/furkanbd/.../results_B500
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

DATASETS = ['IHDP', 'ACIC', 'CPS', 'PSID', 'PSIDbal']

FIELDS = {
    'pehe_raw':       'pehe_ours_mean',
    'pehe_malc':      'pehe_ours_malc_mean',
    'eps_raw_ot':     'err_ours_ot_mean_raw',
    'eps_malc_ot':    'err_ours_ot_mean',
}


def _agg(a):
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return np.nan, np.nan, 0
    return float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0), int(a.size)


def _fmt(m, s):
    if not np.isfinite(m):
        return '      —      '
    return f'{m:6.2f} ± {s:5.2f}'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, help='Directory of *_r*.npz shards')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results, '*_r*.npz')))
    if not files:
        print(f'[fatal] no shards in {args.results}'); return 2

    bucket = {d: {k: [] for k in FIELDS} for d in DATASETS}
    for path in files:
        with np.load(path, allow_pickle=True) as f:
            try:
                dname = str(f['dataset'])
            except Exception:
                continue
            if dname not in bucket:
                continue
            for k, key in FIELDS.items():
                if key in f.files:
                    bucket[dname][k].append(float(f[key]))

    header = f'{"Dataset":10s} {"Variant":6s}    {"sqrt(PEHE)":>14s}    {"eps_ATE":>14s}    {"n":>4s}'
    print()
    print('── Ours(fn=50) — Table 3 (raw vs MALC point estimates) ──────────────')
    print(header)
    print('-' * len(header))
    for d in DATASETS:
        pr_m, pr_s, npr = _agg(bucket[d]['pehe_raw'])
        pm_m, pm_s, npm = _agg(bucket[d]['pehe_malc'])
        er_m, er_s, ner = _agg(bucket[d]['eps_raw_ot'])
        em_m, em_s, nem = _agg(bucket[d]['eps_malc_ot'])
        print(f'{d:10s} {"raw":6s}    {_fmt(pr_m, pr_s):>14s}    '
              f'{_fmt(er_m, er_s):>14s}    {min(npr, ner):>4d}')
        print(f'{d:10s} {"MALC":6s}    {_fmt(pm_m, pm_s):>14s}    '
              f'{_fmt(em_m, em_s):>14s}    {min(npm, nem):>4d}')
    print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
