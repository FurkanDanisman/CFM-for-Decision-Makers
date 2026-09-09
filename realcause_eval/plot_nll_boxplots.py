"""Per-realization NLL boxplots for MALC-smoothed p(τ|x) across the 5
RealCause datasets. Methods grouped by matched 1D↔2D pairs so the
1D-vs-2D uplift is visible per pair.

Grouping:
  (cpfn1d, cpfn2d)     — CausalPFN 1D marginal vs 2D joint
  (uwyk1d, graph2d)    — UWYK 1D vs graph-conditioned 2D
  (dopfn,  dopfnbb)    — DoPFN 1D vs DoPFN-BB 2D

Data source: reads ate/malc dumps + reuses eval_density_metrics's
per-realization truth setup + NLL computation. Same defaults as the
markdown tables (T=4001 tight τ grid, --joint-1d-source malc,
--joint-2d-source malc, tag=B500).

Output: one PNG with 5 subplots (one per dataset), each containing 6
boxes arranged as 3 pairs.

Usage:
    python realcause_eval/plot_nll_boxplots.py \\
        --root-1d $OUT_1D --root-2d $OUT_2D \\
        --causalpfn $CAUSALPFN --repo $DEPLOY_ROOT/R-PFN \\
        --malc-tag-1d B500 --malc-tag B500 \\
        --out plot_nll_boxplots.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


PAIRS = [
    ('cpfn1d', 'cpfn2d'),
    ('uwyk1d', 'graph2d'),
    ('dopfn',  'dopfnbb'),
]
DATASETS = ['IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal']
_ANALYTIC_GAUSSIAN = {'IHDP', 'ACIC'}


def _collect_per_realization_nll(dataset, causalpfn_dir, root_1d, root_2d,
                                    malc_tag_1d, malc_tag, repo, T):
    """Returns {method: [nll_mean_r0, nll_mean_r1, ...]}."""
    from eval_density_metrics import evaluate_realization

    if dataset == 'IHDP':
        from benchmarks import IHDPDataset; ds_obj = IHDPDataset(); n_default = 100
    elif dataset == 'ACIC':
        from benchmarks import ACIC2016Dataset; ds_obj = ACIC2016Dataset(); n_default = 10
    elif dataset == 'CPS':
        from benchmarks import RealCauseLalondeCPSDataset
        ds_obj = RealCauseLalondeCPSDataset(); n_default = 100
    else:
        from benchmarks import RealCauseLalondePSIDDataset
        ds_obj = RealCauseLalondePSIDDataset(); n_default = 100

    methods_1d = ['cpfn1d', 'dopfn', 'uwyk1d']
    methods_2d = ['cpfn2d', 'graph2d', 'dopfnbb']
    all_methods = methods_1d + methods_2d
    per_method = {m: [] for m in all_methods}

    t0 = time.time()
    for r in range(n_default):
        try:
            res = evaluate_realization(r, dataset, causalpfn_dir, ds_obj,
                                        root_1d, root_2d,
                                        methods_1d, methods_2d, T=T,
                                        joint_1d_source='malc',
                                        joint_2d_source='malc',
                                        malc_tag_1d=malc_tag_1d,
                                        malc_tag=malc_tag)
        except Exception as e:
            print(f'  [warn] {dataset} r={r:03d}: {e}', file=sys.stderr); continue
        for m in all_methods:
            info = res.get(m)
            if info and np.isfinite(info.get('nll_mean', np.nan)):
                per_method[m].append(float(info['nll_mean']))
        if (r + 1) % 10 == 0 or r == n_default - 1:
            print(f'  [{time.strftime("%H:%M:%S")}] {dataset} r={r:03d}/{n_default}  '
                  f'({time.time() - t0:.1f}s)', flush=True)
    return per_method


def _draw_boxplots(all_data, out_png):
    """all_data: {dataset: {method: [nll values]}}"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_ds = len(DATASETS)
    fig, axes = plt.subplots(1, n_ds, figsize=(4 * n_ds, 5.5))
    if n_ds == 1: axes = [axes]

    # Color by 1D vs 2D. Within a pair, 1D lighter, 2D darker.
    color_1d = '#7fb0d9'   # light blue
    color_2d = '#1f6fb2'   # dark blue

    for ax, d in zip(axes, DATASETS):
        cell = all_data.get(d, {})
        # Position: 3 pairs, gap between pairs
        positions = []; box_data = []; xticks = []; xtick_labels = []
        colors = []
        pos = 1.0
        for (m1, m2) in PAIRS:
            for j, m in enumerate((m1, m2)):
                positions.append(pos)
                vals = cell.get(m, [])
                # Handle empty / NaN
                clean = [v for v in vals if np.isfinite(v)]
                box_data.append(clean if clean else [np.nan])
                colors.append(color_1d if j == 0 else color_2d)
                pos += 0.9
            xticks.append(pos - 0.9 - 0.45)
            xtick_labels.append(f'{m1}\nvs\n{m2}')
            pos += 0.8       # extra gap between pairs

        bp = ax.boxplot(box_data, positions=positions, widths=0.7,
                         patch_artist=True, showfliers=True,
                         medianprops=dict(color='black', linewidth=1.5),
                         flierprops=dict(marker='o', markersize=3, alpha=0.5))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.75)

        n_r = max((len([v for v in vs if np.isfinite(v)]) for vs in cell.values()), default=0)
        ax.set_title(f'{d}  (n={n_r})')
        ax.set_ylabel('per-realization NLL' if d == DATASETS[0] else '')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, fontsize=9)
        ax.grid(axis='y', linestyle=':', alpha=0.5)

    # Global legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=color_1d, alpha=0.75, label='1D method (marginal-based)'),
               Patch(facecolor=color_2d, alpha=0.75, label='2D method (joint-based)')]
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Per-realization NLL — MALC-smoothed p(τ|x), K=1, B=500',
                  y=1.06, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'wrote {out_png}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root-1d', required=True)
    ap.add_argument('--root-2d', required=True)
    ap.add_argument('--causalpfn', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--malc-tag-1d', default='B500')
    ap.add_argument('--malc-tag',    default='B500')
    ap.add_argument('--T', type=int, default=4001)
    ap.add_argument('--out', default='plot_nll_boxplots.png')
    args = ap.parse_args()

    # Path setup identical to eval_density_metrics.
    for p in (args.repo, os.path.join(args.repo, 'realcause_eval'),
              os.path.join(args.repo, 'benchmarks', 'l2_ihdp'),
              os.path.join(args.repo, 'benchmarks', 'l2_acic')):
        if p not in sys.path: sys.path.insert(0, p)
    sys.path.insert(0, args.causalpfn)
    sys.path.insert(0, os.path.join(args.causalpfn, 'src'))
    try:
        import faiss  # noqa
    except ImportError:
        import types as _t
        sys.modules['faiss'] = _t.ModuleType('faiss')

    all_data = {}
    for d in DATASETS:
        print(f'\n=== {d} ===', flush=True)
        try:
            all_data[d] = _collect_per_realization_nll(
                d, args.causalpfn, args.root_1d, args.root_2d,
                args.malc_tag_1d, args.malc_tag, args.repo, args.T)
        except Exception as e:
            print(f'  [warn] {d} skipped: {e}', file=sys.stderr)
            all_data[d] = {}

    _draw_boxplots(all_data, args.out)


if __name__ == '__main__':
    main()
