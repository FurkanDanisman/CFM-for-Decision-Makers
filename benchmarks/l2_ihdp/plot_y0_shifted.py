"""Plot p(Y|do(0)) for r=14 q=0 with an extra "shifted MALC-LOGLIN" line
where the DoPFN-bb density is shifted left by a fixed amount.

Usage:
    python plot_y0_shifted.py --json <path> --shift -0.09994635 --out <png>
"""
from __future__ import annotations
import argparse, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--shift', type=float, default=-0.09994635,
                     help='Shift amount for the MALC-LOGLIN density (negative = left).')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.json) as f:
        b = json.load(f)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    Y = np.array(b['Y_CENTERS'])
    Y_BIN = Y[1] - Y[0]

    p_true = np.array(b['p_y0_true'])
    p_dopfn = np.array(b['dopfn_p_y0'])
    p_bb = np.array(b['py_bb_p_y0'])
    p_madj = np.array(b.get('py_bb_madj_p_y0') or [np.nan]*len(Y))
    p_r = np.array(b.get('r_bb_p_y0') or [np.nan]*len(Y))

    # Shift the LOGLIN density by args.shift.
    # "Shift left by X" (X > 0) means: at each Y, use the density value that
    # WAS at Y+X. Since we want shift = -0.10 (left by 0.10), we look up at Y+0.10.
    shift = args.shift
    # interp at (Y - shift) — if shift < 0 (left), we look up at Y + |shift|
    p_bb_shifted = np.interp(Y - shift, Y, p_bb, left=0.0, right=0.0)
    # renormalise
    s = p_bb_shifted.sum() * Y_BIN
    if s > 0: p_bb_shifted = p_bb_shifted / s

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.plot(Y, p_true,  'k--', lw=2, label='truth', alpha=0.85)
    ax.plot(Y, p_dopfn, color='#8A4FBE', lw=1.6, label='Do-PFN')
    ax.plot(Y, p_bb,    color='#0F8A3C', lw=1.6, label='DoPFN-bb Py MALC-LOGLIN')
    ax.plot(Y, p_bb_shifted, color='#F5A623', lw=1.8, ls='--',
            label=f'DoPFN-bb LOGLIN shifted by {shift:+.5f}')
    if not np.all(np.isnan(p_madj)):
        ax.plot(Y, p_madj, color='#2E6DBF', lw=1.3, ls='-.', alpha=0.7,
                label='DoPFN-bb Py MALC-MADJ')
    if not np.all(np.isnan(p_r)):
        ax.plot(Y, p_r, color='#B84A2A', lw=1.3, alpha=0.7,
                label=b.get('r_bb_method', 'R MALC'))
    ax.set_title(f'p(Y|do(0)) — IHDP r={b.get("realization", "?")} q={b.get("query", 0)}  '
                  f'shifted by {shift:+.5f}')
    ax.set_xlabel('value'); ax.set_ylabel('density')
    ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[save] {args.out}')

    # Print means for sanity
    mu_true  = np.sum(Y * p_true)  * Y_BIN
    mu_bb    = np.sum(Y * p_bb)    * Y_BIN
    mu_shift = np.sum(Y * p_bb_shifted) * Y_BIN
    mu_dpfn  = np.sum(Y * p_dopfn) * Y_BIN
    print(f'  truth mean       = {mu_true:.4f}')
    print(f'  Do-PFN mean      = {mu_dpfn:.4f}')
    print(f'  LOGLIN mean      = {mu_bb:.4f}')
    print(f'  LOGLIN+shift ({shift:+.5f}) mean = {mu_shift:.4f}  (target ≈ {mu_bb + shift:.4f})')


if __name__ == '__main__':
    main()
