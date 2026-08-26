"""Sample-size-requirement plot: N = 1250·d line vs real benchmarks.

x-axis: feature dimension d ∈ [2, 60]
y-axis: training context size N (log scale — 2500 to 75000 is a wide range)

Line: N = 1250 · d (labelled "Required Sufficient N").
Dots: (d, N) for each benchmark (IHDP, ACIC, CPS, PSID). Label next to each dot.

Adjust the `DATASETS` dict below if the exact (d, N) numbers for your
benchmarks differ from the defaults.

Usage:
    python n_vs_d/plot_n_vs_d.py                    # writes ./n_vs_d/n_vs_d.png
    python n_vs_d/plot_n_vs_d.py --out foo.png
"""
from __future__ import annotations
import argparse
import numpy as np


DATASETS = {
    # name:  (d, N, color, marker, label_pos: 'auto' or (dx, dy, ha, va))
    'IHDP':  dict(d=25, N=672,   color='#1f77b4', marker='o'),
    'ACIC':  dict(d=56, N=4802,  color='#d62728', marker='s',
                    label_pos=(-8, -12, 'right', 'top')),   # bottom-left of marker
    'CPS':   dict(d=8,  N=15992, color='#2ca02c', marker='^'),
    'PSID':  dict(d=8,  N=2490,  color='#ff7f0e', marker='D'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='n_vs_d/n_vs_d.png')
    ap.add_argument('--slope', type=float, default=1250.0,
                     help='N = slope · d for the requirement line (default 1250)')
    ap.add_argument('--d-min', type=int, default=2)
    ap.add_argument('--d-max', type=int, default=60)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = np.arange(args.d_min, args.d_max + 1)
    N_line = args.slope * d

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Requirement line
    ax.plot(d, N_line, color='k', lw=2.2, label='Required Sufficient N')
    ax.fill_between(d, N_line, N_line[-1] * 5, color='green', alpha=0.05)
    ax.fill_between(d, 0.1, N_line, color='red', alpha=0.05)

    # Dataset dots
    for name, m in DATASETS.items():
        ax.scatter([m['d']], [m['N']],
                    s=140, color=m['color'], marker=m['marker'],
                    edgecolor='k', linewidth=0.9, zorder=5)
        # Position label — use explicit label_pos if given, else auto
        if 'label_pos' in m:
            dx, dy, ha, va = m['label_pos']
        else:
            y_below = m['N'] < args.slope * m['d']
            dx = 8
            dy = -12 if y_below else 12
            ha = 'left'
            va = 'top' if y_below else 'bottom'
        ax.annotate(f'{name}  (d={m["d"]}, N={m["N"]})',
                     (m['d'], m['N']),
                     textcoords='offset points',
                     xytext=(dx, dy), ha=ha, va=va, fontsize=10,
                     fontweight='bold', color=m['color'])

    ax.set_xlabel('Feature dimension  d', fontsize=12)
    ax.set_ylabel('Training context size  N', fontsize=12)
    ax.set_yscale('log')
    ax.set_xlim(args.d_min - 1, args.d_max + 2)
    ax.set_ylim(2e2, 1e5)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.set_title('Sample-size requirement vs feature dimension', fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches='tight')
    print(f'[saved] {args.out}')


if __name__ == '__main__':
    main()
