"""Box plots of per-query L2 distributions — one PNG per (dataset × method-group × metric).

Consumes the .npz dumps written by benchmarks/l2_ihdp/l2_per_bin_prob.py
with --save-l2-npz. For each dataset (IHDP, ACIC) writes 8 PNGs:
    <out>_<dataset>_ours_{y0,y1,tau,ate}.png
    <out>_<dataset>_baselines_{y0,y1,tau,ate}.png

Total: 16 PNGs.

x-axis: method names. y-axis: distribution of per-query L2 as a boxplot
(box = IQR, median line, whiskers = 1.5·IQR, outliers as points).

Usage:
    python benchmarks/plots/plot_l2_boxes.py \\
      --ihdp $DEPLOY_ROOT/ihdp_l2_master.npz \\
      --acic $DEPLOY_ROOT/acic_l2_master.npz \\
      --out  $DEPLOY_ROOT/l2_box
"""
from __future__ import annotations
import argparse, os
import numpy as np


OURS_METHODS = [
    ('dopfn',           'Do-PFN'),
    ('bb_2dmarg_b1000', 'Do-PFN-bb'),
]

BASELINE_METHODS = [
    ('fn50_2d_b1000', 'fn=50'),
    ('uwyk_noanc',    'UWYK-NoAnc'),
    ('uwyk_anc',      'UWYK-FullAnc'),
]

METRICS = [
    ('y0',  '$p(Y_{do(0)})$'),
    ('y1',  '$p(Y_{do(1)})$'),
    ('tau', r'$p(\tau \mid x)$ — CATE'),
    ('ate', r'$p(\tau_{ATE})$ — ATE'),
]


def _load(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _draw_box(ax, methods, ds, metric, color):
    labels = [lab for _, lab in methods]
    values = []
    counts = []
    for key, _ in methods:
        arr = ds.get(f'{key}__{metric}', np.array([]))
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        values.append(arr if arr.size else np.array([np.nan]))
        counts.append(int(arr.size))

    positions = np.arange(len(methods)) + 1
    bp = ax.boxplot(values, positions=positions, widths=0.55, patch_artist=True,
                     showmeans=True, meanline=True,
                     meanprops=dict(color='red', lw=1.5),
                     medianprops=dict(color='k', lw=1.2),
                     flierprops=dict(marker='o', markersize=3, alpha=0.4))
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('k')
        patch.set_linewidth(0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel(r'per-query $L^2$')
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    # sample-size annotation above each box
    for i, (arr, n) in enumerate(zip(values, counts)):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0: continue
        y = float(np.nanmax(finite))
        ax.text(positions[i], y, f'n={n}', ha='center', va='bottom',
                 fontsize=7, color='dimgray')


def _make(methods, ds, ds_name, group_name, color, out_prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for mkey, mlabel in METRICS:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        _draw_box(ax, methods, ds, mkey, color)
        ax.set_title(f'{ds_name.upper()} — {group_name} — {mlabel}', fontsize=11)
        fig.tight_layout()
        outpng = f'{out_prefix}_{ds_name}_{group_name.lower()}_{mkey}.png'
        fig.savefig(outpng, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {outpng}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ihdp', required=True)
    ap.add_argument('--acic', required=True)
    ap.add_argument('--out',  required=True, help='output PNG path prefix')
    args = ap.parse_args()

    ihdp = _load(args.ihdp)
    acic = _load(args.acic)
    print(f'[ihdp] n_expected={int(ihdp["n_expected"])}')
    print(f'[acic] n_expected={int(acic["n_expected"])}')

    # 4 IHDP-ours + 4 IHDP-baselines + 4 ACIC-ours + 4 ACIC-baselines = 16
    _make(OURS_METHODS,     ihdp, 'ihdp', 'Ours',      '#1f77b4', args.out)
    _make(BASELINE_METHODS, ihdp, 'ihdp', 'Baselines', '#1f77b4', args.out)
    _make(OURS_METHODS,     acic, 'acic', 'Ours',      '#ff7f0e', args.out)
    _make(BASELINE_METHODS, acic, 'acic', 'Baselines', '#ff7f0e', args.out)


if __name__ == '__main__':
    main()
