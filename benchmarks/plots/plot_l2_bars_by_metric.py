"""Bar plots — x-axis is METRIC (y0, y1, τ, ATE), one grouped bar per method.

Produces 4 PNGs:
  <out>_ihdp_ours.png         2 bars per metric (Do-PFN, Do-PFN-bb)
  <out>_ihdp_baselines.png    3 bars per metric (fn=50, UWYK-NoAnc, UWYK-FullAnc)
  <out>_acic_ours.png
  <out>_acic_baselines.png

Bar height = mean per-bin L2, error bar = SEM. n=… coverage annotated.

Usage:
    python benchmarks/plots/plot_l2_bars_by_metric.py \\
      --ihdp $DEPLOY_ROOT/ihdp_l2_master.npz \\
      --acic $DEPLOY_ROOT/acic_l2_master.npz \\
      --out  $DEPLOY_ROOT/l2_bar_by_metric
"""
from __future__ import annotations
import argparse
import numpy as np


OURS_METHODS = [
    ('dopfn',           'Do-PFN',       '#1f77b4'),
    ('bb_2dmarg_b1000', 'Do-PFN-bb',    '#d62728'),
]

BASELINE_METHODS = [
    ('fn50_2d_b1000', 'fn=50',        '#2ca02c'),
    ('uwyk_noanc',    'UWYK-NoAnc',   '#ff7f0e'),
    ('uwyk_anc',      'UWYK-FullAnc', '#9467bd'),
]

METRICS = [
    ('y0',  r'$p(Y_{do(0)})$'),
    ('y1',  r'$p(Y_{do(1)})$'),
    ('tau', r'$p(\tau|x)$'),
    ('ate', r'$p(\tau_{ATE})$'),
]


def _mean_sem_n(vals):
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (np.nan, np.nan, 0)
    m = float(a.mean())
    sem = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return (m, sem, int(a.size))


def _load(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _plot(ds, ds_name, methods, group_name, outpng):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_metrics = len(METRICS)
    n_meth    = len(methods)
    x = np.arange(n_metrics)
    total_w = 0.72
    w = total_w / n_meth

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (mkey, mlabel, color) in enumerate(methods):
        means, sems, ns = [], [], []
        for metric_key, _ in METRICS:
            m, s, n = _mean_sem_n(ds.get(f'{mkey}__{metric_key}', np.array([])))
            means.append(m); sems.append(s); ns.append(n)
        # offset each method's bar within the metric group
        offset = (i - (n_meth - 1) / 2) * w
        bars = ax.bar(x + offset, means, w, yerr=sems, label=mlabel,
                       color=color, edgecolor='k', linewidth=0.5, capsize=3)
        for xi, (mm, ss, nn) in enumerate(zip(means, sems, ns)):
            if np.isfinite(mm):
                ax.text(x[xi] + offset, mm + ss + 0.005, f'n={nn}',
                         ha='center', va='bottom', fontsize=6, color='dimgray')

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in METRICS])
    ax.set_ylabel(r'per-bin $L^2$ (mean ± SEM)')
    ax.set_title(f'{ds_name.upper()} — {group_name}', fontsize=12)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.legend(frameon=False, loc='upper right', ncol=1, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpng, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {outpng}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ihdp', required=True)
    ap.add_argument('--acic', required=True)
    ap.add_argument('--out',  required=True)
    args = ap.parse_args()

    ihdp = _load(args.ihdp)
    acic = _load(args.acic)
    print(f'[ihdp] n_expected={int(ihdp["n_expected"])}')
    print(f'[acic] n_expected={int(acic["n_expected"])}')

    _plot(ihdp, 'ihdp', OURS_METHODS,     'Do-PFN vs Do-PFN-bb',           f'{args.out}_ihdp_ours.png')
    _plot(ihdp, 'ihdp', BASELINE_METHODS, 'fn=50 vs UWYK-NoAnc vs UWYK-FullAnc', f'{args.out}_ihdp_baselines.png')
    _plot(acic, 'acic', OURS_METHODS,     'Do-PFN vs Do-PFN-bb',           f'{args.out}_acic_ours.png')
    _plot(acic, 'acic', BASELINE_METHODS, 'fn=50 vs UWYK-NoAnc vs UWYK-FullAnc', f'{args.out}_acic_baselines.png')


if __name__ == '__main__':
    main()
