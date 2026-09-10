"""Per-realization PEHE / L1-ATE boxplots for the SCM case-study sweep,
grouped by 1D↔2D model pairs.

One figure per (dataset, context size): a 6-case × 2-metric (PEHE, L1)
grid. Each panel is a grouped boxplot over the 100 realizations, with
the three architecture pairs side by side:
    (cpfn1d, cpfn2d)                — CausalPFN 1D vs 2D joint
    (uwyk_noanc, graph2d_noanc)     — graph-conditioned 1D vs 2D
    (dopfn_native, dopfn_bb)        — DoPFN 1D vs 2D (BB)
1D = light bar, 2D = dark bar.

Reuses aggregate_scm_ctx_sweep._cell_pehe_l1 to pull the per-realization
arrays (handles the 3 npz formats + dopfn_bb summary).

Usage:
    python realcause_eval/plot_scm_boxplots.py \\
        --sweep $DEPLOY_ROOT/results_scm_sweep_shift2 \\
        --name shift2 --out-dir $DEPLOY_ROOT/results_scm_sweep_shift2/boxplots
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


CONTEXTS = [50, 100, 250, 500, 1000]
CASES = ['Observed_Confounder', 'Observed_Mediator',
         'Observed_Mediator_and_Confounder', 'Unobserved_Confounder',
         'Frontdoor_Criterion', 'Backdoor_Criterion']

# (1D name, 2D name) — matched by MODEL_SPECS display names.
# graph-conditioned comparison split into noanc and v3a variants.
PAIRS = [
    ('cpfn1d',       'cpfn2d'),
    ('uwyk_noanc',   'graph2d_noanc'),
    ('uwyk_v3a',     'graph2d_v3a'),
    ('dopfn_native', 'dopfn_bb'),
]


def _make_fig(sweep, ctx, name, spec_by_name, cell_fn, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    c1d, c2d = '#7fb0d9', '#1f6fb2'
    fig, axes = plt.subplots(len(CASES), 2, figsize=(11, 3.0 * len(CASES)))

    for ri, case in enumerate(CASES):
        for ci, (metric, idx) in enumerate((('PEHE', 0), ('L1-ATE', 1))):
            ax = axes[ri, ci]
            positions, box_data, colors, xticks, xlabels = [], [], [], [], []
            pos = 1.0
            for (n1, n2) in PAIRS:
                for j, nm in enumerate((n1, n2)):
                    spec = spec_by_name.get(nm)
                    vals = []
                    if spec is not None:
                        pehe, l1, _ = cell_fn(sweep, ctx, spec, case, float('inf'))
                        arr = pehe if idx == 0 else l1
                        if arr is not None:
                            vals = [v for v in arr if np.isfinite(v)]
                    positions.append(pos)
                    box_data.append(vals if vals else [np.nan])
                    colors.append(c1d if j == 0 else c2d)
                    pos += 0.9
                xticks.append(pos - 0.9 - 0.45)
                xlabels.append(f'{n1}\nvs\n{n2}'.replace('_noanc', '').replace('_native', ''))
                pos += 0.8

            # Box spans Q1–Q3 (IQR), line = median. whis=(25,75) collapses the
            # whiskers onto the box edges and showfliers=False drops the outlier
            # dots — so nothing extends past the IQR and the axis auto-fits the
            # boxes (no outlier stretching). Median/IQR still use all 100 reals.
            bp = ax.boxplot(box_data, positions=positions, widths=0.7,
                            patch_artist=True, showfliers=False,
                            whis=(25, 75), showcaps=False,
                            medianprops=dict(color='black', lw=1.4))
            for patch, cc in zip(bp['boxes'], colors):
                patch.set_facecolor(cc); patch.set_alpha(0.8)
            ax.set_title(f'{case}  —  {metric}', fontsize=9)
            ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=7)
            ax.grid(axis='y', ls=':', alpha=0.5)

    handles = [Patch(facecolor=c1d, alpha=0.8, label='1D method'),
               Patch(facecolor=c2d, alpha=0.8, label='2D method')]
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(f'{name}  —  context size = {ctx}  (per-realization, n≤100)',
                 y=1.02, fontsize=13)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'boxplots_{name}_ctx{ctx}.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}', flush=True)


def _make_bar_fig(sweep, ctx, name, spec_by_name, cell_fn, out_dir):
    """Same grid, but median bars with IQR (Q1–Q3) error whiskers."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    c1d, c2d = '#7fb0d9', '#1f6fb2'
    fig, axes = plt.subplots(len(CASES), 2, figsize=(11, 3.0 * len(CASES)))

    for ri, case in enumerate(CASES):
        for ci, (metric, idx) in enumerate((('PEHE', 0), ('L1-ATE', 1))):
            ax = axes[ri, ci]
            positions, meds, lo_err, hi_err, colors, xticks, xlabels = [], [], [], [], [], [], []
            pos = 1.0
            for (n1, n2) in PAIRS:
                for j, nm in enumerate((n1, n2)):
                    spec = spec_by_name.get(nm)
                    med = q1 = q3 = np.nan
                    if spec is not None:
                        pehe, l1, _ = cell_fn(sweep, ctx, spec, case, float('inf'))
                        arr = pehe if idx == 0 else l1
                        if arr is not None:
                            v = np.array([x for x in arr if np.isfinite(x)])
                            if v.size:
                                med = float(np.median(v)); q1, q3 = np.percentile(v, [25, 75])
                    positions.append(pos); meds.append(med)
                    lo_err.append(med - q1 if np.isfinite(q1) else 0.0)
                    hi_err.append(q3 - med if np.isfinite(q3) else 0.0)
                    colors.append(c1d if j == 0 else c2d)
                    pos += 0.9
                xticks.append(pos - 0.9 - 0.45)
                xlabels.append(f'{n1}\nvs\n{n2}'.replace('_noanc', '').replace('_native', ''))
                pos += 0.8

            ax.bar(positions, meds, width=0.7, color=colors, alpha=0.8,
                   yerr=[lo_err, hi_err], capsize=2.5,
                   error_kw=dict(elinewidth=1, alpha=0.7))
            ax.set_title(f'{case}  —  {metric}', fontsize=9)
            ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=7)
            ax.grid(axis='y', ls=':', alpha=0.5)

    handles = [Patch(facecolor=c1d, alpha=0.8, label='1D method'),
               Patch(facecolor=c2d, alpha=0.8, label='2D method')]
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(f'{name}  —  context size = {ctx}  (median bar, IQR whiskers)',
                 y=1.02, fontsize=13)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'barplots_{name}_ctx{ctx}.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', required=True)
    ap.add_argument('--name', required=True, help='dataset label for titles/filenames')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--repo', default=None)
    ap.add_argument('--kind', default='both', choices=['box', 'bar', 'both'])
    args = ap.parse_args()

    repo = args.repo or os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(repo, 'realcause_eval'))
    from aggregate_scm_ctx_sweep import _cell_pehe_l1, MODEL_SPECS

    spec_by_name = {s[0]: s for s in MODEL_SPECS}
    out_dir = args.out_dir or os.path.join(args.sweep, 'plots')
    for ctx in CONTEXTS:
        if args.kind in ('box', 'both'):
            _make_fig(args.sweep, ctx, args.name, spec_by_name, _cell_pehe_l1, out_dir)
        if args.kind in ('bar', 'both'):
            _make_bar_fig(args.sweep, ctx, args.name, spec_by_name, _cell_pehe_l1, out_dir)


if __name__ == '__main__':
    main()
