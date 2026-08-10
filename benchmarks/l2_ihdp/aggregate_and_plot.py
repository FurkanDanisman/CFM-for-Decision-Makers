"""Aggregate IHDP L2 shards and produce boxplots.

Reads $OUT.r*.npz shards produced by eval_realization.py and produces one
figure with 4 subplots (p_y0, p_y1, p_tau, p_ate), each showing a boxplot
per method with mean/median markers.

Per method, per subplot:
    p_y0, p_y1, p_tau  -> one boxplot point per (realization, query)
                           => ~100 * 75 = ~7500 points
    p_ate              -> one point per realization  => 100 points

Usage
-----
    python R-PFN/benchmarks/l2_ihdp/aggregate_and_plot.py \\
        --shards-glob $OUT.r*.npz --out-fig ihdp_l2_boxplot.png
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


METHOD_ORDER = ['uwyk_noanc', 'dopfn', 'ours_fn10', 'ours_fn50']
METHOD_LABELS = {
    'uwyk_noanc': 'UWYK-NoAnc',
    'dopfn':      'Do-PFN',
    'ours_fn10':  'Ours(fn=10)-2DMALC',
    'ours_fn50':  'Ours(fn=50)-2DMALC',
}
METHOD_COLORS = {
    'uwyk_noanc': '#B84A2A',
    'dopfn':      '#F5A623',
    'ours_fn10':  '#2E4A6F',
    'ours_fn50':  '#0F8A3C',
}
DENSITIES = [
    ('p_y0',  r'$L_2$ error, $p(Y_{do0}\mid X)$', 'per-query'),
    ('p_y1',  r'$L_2$ error, $p(Y_{do1}\mid X)$', 'per-query'),
    ('p_tau', r'$L_2$ error, $p(\tau\mid X)$',     'per-query'),
    ('p_ate', r'$L_2$ error, $p_{\mathrm{ATE}}$',  'per-realization'),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards-glob', required=True,
                    help='glob for per-realization NPZ shards')
    ap.add_argument('--out-fig', required=True, help='output figure path')
    ap.add_argument('--out-csv', default='', help='optional CSV of summary stats')
    args = ap.parse_args()

    shards = sorted(glob.glob(args.shards_glob))
    if not shards:
        print(f'[fatal] no shards match {args.shards_glob}'); return 2
    print(f'[load] {len(shards)} shards')

    # method -> density_key -> list of arrays (per shard)
    collect: dict[str, dict[str, list[np.ndarray]]] = {
        m: {k: [] for k, _, _ in DENSITIES} for m in METHOD_ORDER}
    n_realizations = 0
    for path in shards:
        with np.load(path, allow_pickle=False) as f:
            for m in METHOD_ORDER:
                for dk, _, _ in DENSITIES:
                    key = f'{m}__l2_{dk[2:]}' if dk != 'p_ate' else f'{m}__l2_ate'
                    if key not in f:
                        continue
                    arr = np.atleast_1d(np.asarray(f[key]))
                    collect[m][dk].append(arr)
        n_realizations += 1

    # Flatten per (method, density) into one 1D array
    flat: dict[str, dict[str, np.ndarray]] = {
        m: {dk: (np.concatenate(collect[m][dk]) if collect[m][dk] else np.array([]))
            for dk, _, _ in DENSITIES}
        for m in METHOD_ORDER}

    _print_summary(flat, n_realizations)

    if args.out_csv:
        _write_csv(flat, args.out_csv)

    _plot(flat, args.out_fig)
    return 0


def _print_summary(flat, n_realizations):
    print()
    print(f'{"density":10s} {"method":22s} {"n":>6s} {"mean":>8s} {"median":>8s} {"std":>8s}')
    print('-' * 72)
    for dk, _, _ in DENSITIES:
        for m in METHOD_ORDER:
            v = flat[m][dk]
            if v.size == 0:
                print(f'{dk:10s} {METHOD_LABELS[m]:22s} {"-":>6s}')
                continue
            print(f'{dk:10s} {METHOD_LABELS[m]:22s} {v.size:>6d} '
                  f'{v.mean():>8.4f} {np.median(v):>8.4f} {v.std(ddof=1):>8.4f}')
    print(f'\n[from {n_realizations} realizations]')


def _write_csv(flat, path):
    import csv
    with open(path, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['density', 'method', 'n', 'mean', 'median', 'std'])
        for dk, _, _ in DENSITIES:
            for m in METHOD_ORDER:
                v = flat[m][dk]
                if v.size == 0:
                    continue
                w.writerow([dk, METHOD_LABELS[m], v.size,
                            f'{v.mean():.6f}', f'{np.median(v):.6f}',
                            f'{v.std(ddof=1):.6f}'])
    print(f'[csv] {path}')


def _plot(flat, out_fig):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(4.4 * 4, 4.4), sharey=False)
    for ax, (dk, title, granularity) in zip(axes, DENSITIES):
        data = [flat[m][dk] for m in METHOD_ORDER]
        labels = [METHOD_LABELS[m] for m in METHOD_ORDER]
        colors = [METHOD_COLORS[m] for m in METHOD_ORDER]

        bp = ax.boxplot(
            data, tick_labels=labels, patch_artist=True,
            showfliers=False, widths=0.55,
            medianprops={'color': 'black', 'linewidth': 1.5},
            boxprops={'linewidth': 1.0},
        )
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.55)

        for i, arr in enumerate(data):
            if arr.size == 0:
                continue
            ax.plot([i + 1], [arr.mean()], marker='D', ms=8,
                    markerfacecolor='white', markeredgecolor='black',
                    markeredgewidth=1.2, zorder=5, clip_on=False)

        ax.set_title(f'{title}\n({granularity})', fontsize=11)
        ax.set_ylabel(r'$L_2$ distance to true density')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=25)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color='black', linewidth=1.5, label='median'),
        Line2D([], [], marker='D', markerfacecolor='white',
               markeredgecolor='black', markersize=8, linestyle='none',
               label='mean'),
    ]
    fig.legend(handles=handles, loc='upper right', frameon=False)
    fig.suptitle('IHDP: L2 distance to analytical DGP densities',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=160, bbox_inches='tight')
    print(f'[fig] {out_fig}')


if __name__ == '__main__':
    sys.exit(main())
