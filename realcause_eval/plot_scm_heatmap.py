"""Compact median-performance heatmaps for the SCM case-study sweep.

Encoding value as COLOR (median) instead of bar height sidesteps the
outlier-stretched-axis problem: the median is outlier-robust, and the
colour scale is clipped so a few huge realizations don't wash out the
differences. Cells are annotated with the median value.

Two colouring modes:
  --color value : shared colour scale = median metric (clipped at p95),
                  low=green(good) high=red(bad). Absolute comparison.
  --color rank  : colour = rank within each case (1=best=green). Pure
                  ranking, fully scale-invariant across cases.

One figure per metric (PEHE, L1). Subplots = 5 context sizes in a row;
each subplot is a models(rows) × cases(cols) grid.

Usage:
    python realcause_eval/plot_scm_heatmap.py \\
        --sweep $DEPLOY_ROOT/results_scm_sweep_shift2 --name shift2 \\
        --color value --out-dir $DEPLOY_ROOT/results_scm_sweep_shift2/plots
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
CASE_SHORT = ['ObsConf', 'ObsMed', 'Med+Conf', 'UnobsConf', 'FrontDoor', 'BackDoor']


def _median_grid(sweep, ctx, metric_idx, specs, cell_fn):
    """Return (n_models, n_cases) array of median metric (NaN if missing)."""
    G = np.full((len(specs), len(CASES)), np.nan)
    for mi, spec in enumerate(specs):
        for ci, case in enumerate(CASES):
            pehe, l1, _ = cell_fn(sweep, ctx, spec, case, float('inf'))
            arr = pehe if metric_idx == 0 else l1
            if arr is not None and len(arr):
                G[mi, ci] = float(np.median([v for v in arr if np.isfinite(v)]))
    return G


def _make_heatmap(sweep, name, metric, midx, specs, cell_fn, color, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model_names = [s[0] for s in specs]
    grids = [_median_grid(sweep, ctx, midx, specs, cell_fn) for ctx in CONTEXTS]

    if color == 'value':
        allv = np.concatenate([g[np.isfinite(g)] for g in grids])
        vmax = float(np.percentile(allv, 95)) if allv.size else 1.0
        vmin = 0.0

    fig, axes = plt.subplots(1, len(CONTEXTS), figsize=(4.2 * len(CONTEXTS), 5.2),
                             sharey=True)
    for k, (ctx, G) in enumerate(zip(CONTEXTS, grids)):
        ax = axes[k]
        if color == 'rank':
            # rank within each column (case): 1=best(lowest). NaN stays NaN.
            R = np.full_like(G, np.nan)
            for ci in range(G.shape[1]):
                col = G[:, ci]; ok = np.isfinite(col)
                order = np.argsort(np.argsort(col[ok]))    # 0=best
                R[np.where(ok)[0], ci] = order + 1
            disp, cmap, vmn, vmx = R, 'RdYlGn_r', 1, len(specs)
        else:
            disp, cmap, vmn, vmx = G, 'RdYlGn_r', vmin, vmax
        im = ax.imshow(disp, aspect='auto', cmap=cmap, vmin=vmn, vmax=vmx)
        ax.set_title(f'ctx={ctx}', fontsize=10)
        ax.set_xticks(range(len(CASES)))
        ax.set_xticklabels(CASE_SHORT, rotation=45, ha='right', fontsize=7)
        if k == 0:
            ax.set_yticks(range(len(specs)))
            ax.set_yticklabels(model_names, fontsize=8)
        # annotate with the median VALUE (always, even in rank mode)
        for mi in range(G.shape[0]):
            for ci in range(G.shape[1]):
                if np.isfinite(G[mi, ci]):
                    ax.text(ci, mi, f'{G[mi, ci]:.2f}', ha='center', va='center',
                            fontsize=6.5, color='black')

    label = ('median (clipped p95)' if color == 'value'
             else 'rank within case (1=best)')
    fig.suptitle(f'{name} — median {metric} per (model, case, context) — '
                 f'colour = {label}', y=1.02, fontsize=12)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'heatmap_{name}_{metric}_{color}.png')
    fig.savefig(out, dpi=145, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--color', default='value', choices=['value', 'rank'])
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--repo', default=None)
    args = ap.parse_args()

    repo = args.repo or os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(repo, 'realcause_eval'))
    from aggregate_scm_ctx_sweep import _cell_pehe_l1, MODEL_SPECS

    out_dir = args.out_dir or os.path.join(args.sweep, 'plots')
    for metric, midx in (('PEHE', 0), ('L1', 1)):
        _make_heatmap(args.sweep, args.name, metric, midx, MODEL_SPECS,
                      _cell_pehe_l1, args.color, out_dir)


if __name__ == '__main__':
    main()
