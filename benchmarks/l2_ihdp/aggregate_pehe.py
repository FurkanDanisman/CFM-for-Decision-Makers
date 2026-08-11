"""Aggregate per-realization PEHE and eps_ATE across IHDP shards.

Reads shards produced by eval_realization.py (which now saves
{method}__pehe and {method}__eps_ate). Reports mean +/- std across
realizations, matching Table 3's IHDP convention (100 realizations,
raw Y units, one PEHE per realization computed as
sqrt(mean_over_queries((cate_hat_raw - true_cate_raw)^2)).

Usage
-----
    python R-PFN/benchmarks/l2_ihdp/aggregate_pehe.py \\
        --shards-glob "/scratch/furkanbd/rpfn_bench_kit/l2_ihdp/out.r*.npz" \\
        --override-glob "ours_fn10=/scratch/furkanbd/rpfn_bench_kit/l2_ihdp/out_maxK1.r*.npz" \\
        --override-glob "ours_fn50=/scratch/furkanbd/rpfn_bench_kit/l2_ihdp/out.r*.npz" \\
        --override-glob "ours_dopfn_bb=/scratch/furkanbd/rpfn_bench_kit/l2_ihdp/out_dopfn_bb.r*.npz"
"""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np


METHODS = ['ours_fn50', 'ours_fn10', 'ours_dopfn_bb', 'dopfn', 'uwyk_noanc']
LABELS = {
    'ours_fn50':     'Ours(fn=50)',
    'ours_fn10':     'Ours(fn=10)',
    'ours_dopfn_bb': 'Ours(DoPFN-bb)',
    'dopfn':         'Do-PFN',
    'uwyk_noanc':    'UWYK-NoAnc',
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards-glob', required=True,
                    help='default per-realization NPZ shard glob')
    ap.add_argument('--override-glob', action='append', default=[],
                    metavar='METHOD=GLOB',
                    help='pull a specific method\'s shards from a different glob')
    ap.add_argument('--out-csv', default='',
                    help='optional CSV of the summary table')
    args = ap.parse_args()

    overrides: dict[str, str] = {}
    for entry in args.override_glob:
        if '=' not in entry:
            print(f'[fatal] bad --override-glob {entry!r}'); return 2
        m, g = entry.split('=', 1)
        overrides[m.strip()] = g.strip()

    per_method_shards = {m: sorted(glob.glob(overrides.get(m, args.shards_glob)))
                         for m in METHODS}

    print(f'{"method":18s} {"n":>4s} {"PEHE mean":>10s} {"PEHE std":>10s}   '
          f'{"eps_ATE mean":>12s} {"eps_ATE std":>12s}')
    print('-' * 74)
    rows = []
    for m in METHODS:
        pehes = []
        eps_ates = []
        for path in per_method_shards[m]:
            with np.load(path) as f:
                pehe_key = f'{m}__pehe'
                eps_key  = f'{m}__eps_ate'
                if pehe_key not in f.files:
                    continue
                pehes.append(float(f[pehe_key]))
                eps_ates.append(float(f[eps_key]))
        if not pehes:
            print(f'{LABELS[m]:18s} {"-":>4s}')
            continue
        pehes = np.array(pehes)
        eps_ates = np.array(eps_ates)
        print(f'{LABELS[m]:18s} {pehes.size:>4d} '
              f'{pehes.mean():>10.3f} {pehes.std(ddof=1):>10.3f}   '
              f'{eps_ates.mean():>12.4f} {eps_ates.std(ddof=1):>12.4f}')
        rows.append((LABELS[m], pehes.size, pehes.mean(), pehes.std(ddof=1),
                     eps_ates.mean(), eps_ates.std(ddof=1)))

    if args.out_csv:
        import csv
        with open(args.out_csv, 'w', newline='') as fp:
            w = csv.writer(fp)
            w.writerow(['method', 'n', 'PEHE_mean', 'PEHE_std',
                        'eps_ATE_mean', 'eps_ATE_std'])
            for r in rows:
                w.writerow([r[0], r[1],
                            f'{r[2]:.6f}', f'{r[3]:.6f}',
                            f'{r[4]:.6f}', f'{r[5]:.6f}'])
        print(f'\n[csv] {args.out_csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
