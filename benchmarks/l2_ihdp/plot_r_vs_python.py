"""Load a JSON produced by dump_for_plot.py + add_r_malc_densities.R and
plot 4-panel density comparison: truth, Do-PFN, DoPFN-bb (Python MALC),
DoPFN-bb (R MALC).

Usage:
    python plot_r_vs_python.py --json <path> --out <png>
"""
from __future__ import annotations
import argparse, json
import numpy as np


def _to_array(x):
    return np.asarray(x, dtype=np.float64) if x is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--out',  required=True)
    args = ap.parse_args()

    with open(args.json) as f:
        b = json.load(f)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    Y = _to_array(b['Y_CENTERS']); TAU = _to_array(b['TAU_CENTERS'])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    panels = [
        (axes[0,0], Y, 'p(Y|do(0))',
            b['p_y0_true'], b['dopfn_p_y0'], b['py_bb_p_y0'], b.get('py_bb_madj_p_y0'), b.get('r_bb_p_y0')),
        (axes[0,1], Y, 'p(Y|do(1))',
            b['p_y1_true'], b['dopfn_p_y1'], b['py_bb_p_y1'], b.get('py_bb_madj_p_y1'), b.get('r_bb_p_y1')),
        (axes[1,0], TAU, 'p(τ)',
            b['p_tau_true'], b['dopfn_p_tau'], b['py_bb_p_tau'], None, b.get('r_bb_p_tau')),
        (axes[1,1], TAU, 'p(ATE) — same as p(τ) at single query',
            b['p_tau_true'], b['dopfn_p_tau'], b['py_bb_p_tau'], None, b.get('r_bb_p_tau')),
    ]

    r_label = f"DoPFN-bb R ({b.get('r_bb_method', 'unknown')})"
    for ax, grid, title, truth, dpf, py, madj, r in panels:
        ax.plot(grid, _to_array(truth), 'k--', lw=2, label='truth', alpha=0.85)
        if dpf is not None:
            ax.plot(grid, _to_array(dpf), color='#8A4FBE', lw=1.6, label='Do-PFN')
        if py is not None:
            ax.plot(grid, _to_array(py), color='#0F8A3C', lw=1.6, label='DoPFN-bb Py MALC-LOGLIN')
        if madj is not None and not np.all(np.isnan(_to_array(madj))):
            ax.plot(grid, _to_array(madj), color='#2E6DBF', lw=1.6, ls='-.', label='DoPFN-bb Py MALC-MADJ')
        if r is not None and not np.all(np.isnan(_to_array(r))):
            ax.plot(grid, _to_array(r), color='#B84A2A', lw=1.6, label=r_label)
        ax.set_title(title)
        ax.set_xlabel('value'); ax.set_ylabel('density')
        ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=8)

    case = b.get('case', '?')
    r = b.get('realization', '?')
    q = b.get('query', 0)
    seed = b.get('seed')
    extra = f' seed={seed}' if seed is not None else ''
    subtitle = f'{case.upper()} realization r={r} q={q}{extra}  (J={b["J"]}, MALC B={b["malc_B"]} K={b["malc_max_K"]})'
    fig.suptitle(f'Density diagnostic — {subtitle}', y=1.005, fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches='tight')
    print(f'[save] {args.out}')


if __name__ == '__main__':
    main()
