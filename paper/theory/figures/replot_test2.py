"""Regenerate Test 2 plot with two y-axis variants.

Reads the npz produced by marginal_nll_test.py and outputs:
  marginal_nll_test.png       — shared y-axis (using LEFT panel's tight range)
  marginal_nll_test_-10_10.png — shared y-axis [-10, 10] for both panels

Changes from the original plotter:
  - no fig.suptitle
  - legend text simplified to "Empirical mean = N.NNN"
"""
import numpy as np
import matplotlib.pyplot as plt

npz_path = '/Users/furkandanisman/R-PFN/paper/theory/figures/marginal_nll_test.png.npz'
d = np.load(npz_path)
arr = {k: d[k] for k in d.files}

RHO_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)


def _panel(ax, num_key, den_key, title):
    gaps = arr[num_key] - arr[den_key]
    for rho in RHO_GRID:
        mask = np.isclose(arr['rho'], rho)
        ax.scatter(arr['rho'][mask], gaps[mask], color='#2E4A6F',
                    alpha=0.35, s=32, zorder=3)
        m = float(gaps[mask].mean()); s = float(gaps[mask].std())
        ax.errorbar(rho, m, yerr=s, fmt='o', color='#0F8A3C',
                     markersize=8, capsize=4, zorder=4)
    theory = float(np.mean([gaps[np.isclose(arr['rho'], r)].mean() for r in RHO_GRID]))
    ax.axhline(theory, color='k', ls='--', lw=1.2,
                label=f'Empirical mean = {theory:+.3f}')
    ax.axhline(0.0, color='r', ls=':', lw=1, alpha=0.6)
    ax.set_xlabel('true DGP ρ')
    ax.set_ylabel('marginal NLL gap  (marg - joint)')
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)


def make(y_lim, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    _panel(axes[0], 'nll_uwyk',  'nll_ours50',
            'Marginal NLL gap (UWYK - R-PFN fn=50) vs true ρ')
    _panel(axes[1], 'nll_dopfn', 'nll_ours10',
            'Marginal NLL gap (Do-PFN - R-PFN fn=10) vs true ρ')
    for ax in axes:
        ax.set_ylim(*y_lim)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[save] {out_path}')


# Version 1: use LEFT panel's tight range for both.
# Compute LEFT's tight range from the data
gaps_left = arr['nll_uwyk'] - arr['nll_ours50']
y_lo = min(gaps_left.min(), 0) * 1.05
y_hi = gaps_left.max() * 1.05
make((y_lo, y_hi),
     '/Users/furkandanisman/R-PFN/paper/theory/figures/marginal_nll_test.png')

# Version 2: fixed [-10, 10] for both.
make((-10, 10),
     '/Users/furkandanisman/R-PFN/paper/theory/figures/marginal_nll_test_pm10.png')
