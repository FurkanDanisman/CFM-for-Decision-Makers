"""Simple 4-line plot for IHDP density diagnostic:
    - Truth
    - Do-PFN
    - DoPFN-bb (Python MALC-LOGLIN)
    - DoPFN-bb (RAW density — no MALC, just p_marg / bin_width resampled to Y_CENTERS)

For y0 and y1 panels, all four lines are shown.
For τ and ATE, DoPFN-bb-raw isn't well-defined (τ comes from 2D MALC),
so those panels show truth + Do-PFN + DoPFN-bb (MALC-LOGLIN) only.

Usage:
    python plot_ihdp_4method.py --json <path> --out <png>
"""
from __future__ import annotations
import argparse, json
import numpy as np


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

    Y = np.array(b['Y_CENTERS'])
    TAU = np.array(b['TAU_CENTERS'])
    Y_BIN = Y[1] - Y[0]
    edges = np.array(b['edges_scaled'])
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_w = edges[1] - edges[0]

    def raw_density_on_ycenters(p_marg):
        """Take J-bin discrete probs → per-bin density → resample onto Y_CENTERS."""
        d_native = np.array(p_marg) / max(bin_w, 1e-12)
        # linear-in-prob interp
        p = np.interp(Y, centers, d_native, left=0.0, right=0.0)
        s = p.sum() * Y_BIN
        return p / s if s > 0 else p

    # Compute raw densities from the J=10 marginals stored in the JSON
    p_y0_raw = raw_density_on_ycenters(b['p_marg_y0_raw'])
    p_y1_raw = raw_density_on_ycenters(b['p_marg_y1_raw'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # y0 panel — 4 lines
    ax = axes[0, 0]
    ax.plot(Y, b['p_y0_true'],  'k--', lw=2, label='truth', alpha=0.85)
    ax.plot(Y, b['dopfn_p_y0'], color='#8A4FBE', lw=1.6, label='Do-PFN')
    ax.plot(Y, b['py_bb_p_y0'], color='#0F8A3C', lw=1.6, label='DoPFN-bb Py MALC-LOGLIN')
    ax.plot(Y, p_y0_raw,        color='#B84A2A', lw=1.6, ls='--', label='DoPFN-bb RAW (no MALC)')
    ax.set_title('p(Y|do(0))'); ax.set_xlabel('value'); ax.set_ylabel('density')
    ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=9)

    # y1 panel — 4 lines
    ax = axes[0, 1]
    ax.plot(Y, b['p_y1_true'],  'k--', lw=2, label='truth', alpha=0.85)
    ax.plot(Y, b['dopfn_p_y1'], color='#8A4FBE', lw=1.6, label='Do-PFN')
    ax.plot(Y, b['py_bb_p_y1'], color='#0F8A3C', lw=1.6, label='DoPFN-bb Py MALC-LOGLIN')
    ax.plot(Y, p_y1_raw,        color='#B84A2A', lw=1.6, ls='--', label='DoPFN-bb RAW (no MALC)')
    ax.set_title('p(Y|do(1))'); ax.set_xlabel('value'); ax.set_ylabel('density')
    ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=9)

    # τ panel — 3 lines (no natural "raw τ" — τ comes from 2D MALC only)
    ax = axes[1, 0]
    ax.plot(TAU, b['p_tau_true'],  'k--', lw=2, label='truth', alpha=0.85)
    ax.plot(TAU, b['dopfn_p_tau'], color='#8A4FBE', lw=1.6, label='Do-PFN')
    ax.plot(TAU, b['py_bb_p_tau'], color='#0F8A3C', lw=1.6, label='DoPFN-bb Py MALC-LOGLIN')
    ax.set_title('p(τ)'); ax.set_xlabel('value'); ax.set_ylabel('density')
    ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=9)

    # ATE panel — same as τ at single query
    ax = axes[1, 1]
    ax.plot(TAU, b['p_tau_true'],  'k--', lw=2, label='truth', alpha=0.85)
    ax.plot(TAU, b['dopfn_p_tau'], color='#8A4FBE', lw=1.6, label='Do-PFN')
    ax.plot(TAU, b['py_bb_p_tau'], color='#0F8A3C', lw=1.6, label='DoPFN-bb Py MALC-LOGLIN')
    ax.set_title('p(ATE) — same as p(τ) at single query')
    ax.set_xlabel('value'); ax.set_ylabel('density')
    ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=9)

    case = b.get('case', '?')
    r = b.get('realization', '?')
    q = b.get('query', 0)
    fig.suptitle(f'{case.upper()} r={r} q={q}  —  Truth / Do-PFN / DoPFN-bb (MALC-LOGLIN) / DoPFN-bb (RAW)',
                 y=1.005, fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[save] {args.out}')

    # Print per-line means for numeric sanity
    def _m(y, p): return np.sum(y * np.array(p)) * (y[1] - y[0])
    print(f'  Y0 means:  truth={_m(Y, b["p_y0_true"]):.4f}  '
          f'Do-PFN={_m(Y, b["dopfn_p_y0"]):.4f}  '
          f'LOGLIN={_m(Y, b["py_bb_p_y0"]):.4f}  '
          f'RAW={_m(Y, p_y0_raw):.4f}')
    print(f'  Y1 means:  truth={_m(Y, b["p_y1_true"]):.4f}  '
          f'Do-PFN={_m(Y, b["dopfn_p_y1"]):.4f}  '
          f'LOGLIN={_m(Y, b["py_bb_p_y1"]):.4f}  '
          f'RAW={_m(Y, p_y1_raw):.4f}')


if __name__ == '__main__':
    main()
