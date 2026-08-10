"""Inspect actual CATE density shapes: are fn=10's outputs Gaussian-like or artefactual?

Loads a single realization shard, picks a handful of queries, and dumps a
side-by-side plot of the true CATE density vs each method's predicted
density. Also prints per-density diagnostics (peak location, peak value,
support-integrated mass, tail mass beyond ±2 sigma of the truth).

If fn=10's densities are Gaussian-shaped but off-centre/narrow, the L2
result reflects real calibration. If they are spiky/bimodal/boundary-heavy,
MALC hyperparameters for fn=10 need attention.

Usage
-----
    python R-PFN/benchmarks/l2_ihdp/diagnose_shapes.py \\
        --shard /scratch/furkanbd/rpfn_bench_kit/l2_ihdp/out.r000.npz \\
        --out-dir /scratch/furkanbd/rpfn_bench_kit/l2_ihdp/shape_diag \\
        --n-queries 6
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


TAU_EDGES = np.linspace(-3.0, 3.0, 601)
TAU_CENTERS = 0.5 * (TAU_EDGES[:-1] + TAU_EDGES[1:])
TAU_BIN = float(TAU_CENTERS[1] - TAU_CENTERS[0])

Y_EDGES = np.linspace(-1.5, 1.5, 101)
Y_CENTERS = 0.5 * (Y_EDGES[:-1] + Y_EDGES[1:])
Y_BIN = float(Y_CENTERS[1] - Y_CENTERS[0])

METHODS = ['ours_fn50', 'ours_fn10', 'dopfn', 'uwyk_noanc']
COLORS  = {'ours_fn50': '#0F8A3C', 'ours_fn10': '#2E4A6F',
           'dopfn': '#F5A623', 'uwyk_noanc': '#B84A2A'}
LABELS  = {'ours_fn50': 'Ours(fn=50)', 'ours_fn10': 'Ours(fn=10)',
           'dopfn': 'Do-PFN', 'uwyk_noanc': 'UWYK-NoAnc'}


def _mv(f):
    m = float((TAU_CENTERS * f).sum() * TAU_BIN)
    v = float(((TAU_CENTERS - m) ** 2 * f).sum() * TAU_BIN)
    return m, v


def _report_row(name, f, mu_true, sig_true, tau_true_bin):
    m, v = _mv(f)
    sig = float(np.sqrt(max(v, 0)))
    integ = float(f.sum() * TAU_BIN)
    peak_idx = int(np.argmax(f))
    peak_tau = float(TAU_CENTERS[peak_idx])
    peak_val = float(f[peak_idx])
    # tail mass beyond ± (mu_true + 2*sig_true)
    lo = mu_true - 2 * sig_true
    hi = mu_true + 2 * sig_true
    tail = float(f[(TAU_CENTERS < lo) | (TAU_CENTERS > hi)].sum() * TAU_BIN)
    # bimodality proxy: number of local maxima above 25% of peak
    thresh = 0.25 * peak_val
    above = f > thresh
    modes = 0
    for i in range(1, len(f) - 1):
        if above[i] and f[i] >= f[i-1] and f[i] >= f[i+1] and f[i] > thresh:
            modes += 1
    print(f'  {name:12s}  integ={integ:.3f}  peak_tau={peak_tau:+.3f} '
          f'(true={tau_true_bin:+.3f})  peak_val={peak_val:5.2f}  '
          f'mean={m:+.3f}  std={sig:.3f}  (true_std={sig_true:.3f})  '
          f'tail_mass_±2σ={tail:.3f}  #modes>25%={modes}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--n-queries', type=int, default=6)
    args = ap.parse_args()

    with np.load(args.shard) as f:
        p_tau_true = np.asarray(f['p_tau_true'])
        data = {m: np.asarray(f[f'{m}__p_tau']) for m in METHODS}

    n_q = p_tau_true.shape[0]
    # pick queries: the 3 with lowest ours_fn10 L2 to true, and 3 with highest
    diffs = np.array([float(((data['ours_fn10'][q] - p_tau_true[q]) ** 2).sum() * TAU_BIN)
                      for q in range(n_q)])
    order = np.argsort(diffs)
    picks = list(order[:args.n_queries // 2]) + list(order[-(args.n_queries - args.n_queries // 2):])
    print(f'[picks] n_queries={n_q}   picked {picks}   L2² range: '
          f'{diffs[picks].min():.3f}..{diffs[picks].max():.3f}')

    # Per-query numeric report
    for q in picks:
        m_true, v_true = _mv(p_tau_true[q])
        sig_true = float(np.sqrt(max(v_true, 0)))
        tau_true_bin = float(TAU_CENTERS[int(np.argmax(p_tau_true[q]))])
        print(f'\nquery {q}   true: mean={m_true:+.3f}  std={sig_true:.3f}  '
              f'peak_tau={tau_true_bin:+.3f}')
        for m in METHODS:
            _report_row(LABELS[m], data[m][q], m_true, sig_true, tau_true_bin)

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(args.out_dir, exist_ok=True)
    n_cols = 3
    n_rows = (len(picks) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.0 * n_cols, 3.4 * n_rows),
                             squeeze=False)
    for k, q in enumerate(picks):
        ax = axes[k // n_cols][k % n_cols]
        ax.fill_between(TAU_CENTERS, p_tau_true[q], color='red',
                        alpha=0.20, label='true')
        ax.plot(TAU_CENTERS, p_tau_true[q], color='red', lw=2.0)
        for m in METHODS:
            ax.plot(TAU_CENTERS, data[m][q], color=COLORS[m], lw=1.6,
                    alpha=0.95, label=LABELS[m])
        ax.set_xlim(-1.5, 1.5)
        ax.set_title(f'query {q}   ours_fn10 L2²={diffs[q]:.3f}')
        ax.set_xlabel(r'$\tau$ (scaled)')
        ax.set_ylabel('density')
        ax.grid(alpha=0.25)
        if k == 0:
            ax.legend(fontsize=8, loc='upper right')
    for k in range(len(picks), n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_visible(False)
    fig.tight_layout()
    out = os.path.join(args.out_dir, 'cate_shapes.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[fig] {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
