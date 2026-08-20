"""Bar plots for the master L2 tables — IHDP + ACIC side-by-side.

Consumes the .npz dumps written by benchmarks/l2_ihdp/l2_per_bin_prob.py
with --save-l2-npz, and produces two figures:

  <out>_ours.png       — Do-PFN vs Do-PFN-bb (2D-τ 2D-marg B=1000, J=10 grid)
  <out>_baselines.png  — fn=50 (2D-marg B=1000) vs UWYK-NoAnc vs UWYK-FullAnc

Each figure has 4 panels (y0, y1, τ, ATE). Each panel groups two bars per
method — IHDP (left, blue) and ACIC (right, orange) — with SEM error bars.

Usage:
    python benchmarks/plots/plot_l2_bars.py \\
      --ihdp $DEPLOY_ROOT/ihdp_l2_master.npz \\
      --acic $DEPLOY_ROOT/acic_l2_master.npz \\
      --out $DEPLOY_ROOT/l2_bars
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


def _mean_sem(vals):
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


def _draw(ax, methods, ihdp, acic, metric):
    labels = [lab for _, lab in methods]
    n = len(labels)
    x = np.arange(n)
    w = 0.36

    ihdp_m = [_mean_sem(ihdp.get(f'{k}__{metric}', np.array([])))
              for k, _ in methods]
    acic_m = [_mean_sem(acic.get(f'{k}__{metric}', np.array([])))
              for k, _ in methods]

    ax.bar(x - w/2, [m for m, _, _ in ihdp_m], w,
           yerr=[s for _, s, _ in ihdp_m], color='#1f77b4', label='IHDP',
           capsize=3, edgecolor='k', linewidth=0.5)
    ax.bar(x + w/2, [m for m, _, _ in acic_m], w,
           yerr=[s for _, s, _ in acic_m], color='#ff7f0e', label='ACIC',
           capsize=3, edgecolor='k', linewidth=0.5)

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel(r'per-bin $L^2$')
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    # coverage annotations
    for i, (mm, ss, n_ihdp) in enumerate(ihdp_m):
        if np.isfinite(mm):
            ax.text(i - w/2, mm + ss, f'n={n_ihdp}',
                    ha='center', va='bottom', fontsize=7, color='#1f77b4')
    for i, (mm, ss, n_acic) in enumerate(acic_m):
        if np.isfinite(mm):
            ax.text(i + w/2, mm + ss, f'n={n_acic}',
                    ha='center', va='bottom', fontsize=7, color='#ff7f0e')


def _make_fig(methods, ihdp, acic, group_name, out_prefix):
    """One PNG per metric — 4 files per method group (8 total across groups)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for mkey, mlabel in METRICS:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        _draw(ax, methods, ihdp, acic, mkey)
        ax.set_title(f'{group_name} — {mlabel}  (per-bin $L^2$, mean ± SEM)',
                     fontsize=11)
        ax.legend(loc='upper right', frameon=False)
        fig.tight_layout()
        outpng = f'{out_prefix}_{mkey}.png'
        fig.savefig(outpng, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {outpng}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ihdp', required=True, help='IHDP l2-npz from l2_per_bin_prob.py')
    ap.add_argument('--acic', required=True, help='ACIC l2-npz')
    ap.add_argument('--out',  required=True, help='output PNG path prefix')
    args = ap.parse_args()

    ihdp = _load(args.ihdp)
    acic = _load(args.acic)
    print(f'[ihdp] dataset={ihdp["dataset"]}  n_expected={int(ihdp["n_expected"])}')
    print(f'[acic] dataset={acic["dataset"]}  n_expected={int(acic["n_expected"])}')

    _make_fig(OURS_METHODS,      ihdp, acic, 'Our methods',
              f'{args.out}_ours')
    _make_fig(BASELINE_METHODS,  ihdp, acic, 'Baselines',
              f'{args.out}_baselines')


if __name__ == '__main__':
    main()
