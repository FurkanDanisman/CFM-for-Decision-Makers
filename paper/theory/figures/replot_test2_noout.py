"""Test 2 plot with outliers removed (Tukey 1.5·IQR filter per ρ bucket),
y-axis fixed to [-2, 2] for both panels.
"""
import numpy as np
import matplotlib.pyplot as plt

npz_path = '/Users/furkandanisman/R-PFN/paper/theory/figures/marginal_nll_test.png.npz'
d = np.load(npz_path)
arr = {k: d[k] for k in d.files}

RHO_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)


def _iqr_mask(values):
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (values >= lo) & (values <= hi)


def _panel(ax, num_key, den_key, title):
    gaps_all = arr[num_key] - arr[den_key]
    rho_all  = arr['rho']

    per_rho_means = []
    for rho in RHO_GRID:
        m = np.isclose(rho_all, rho)
        vals = gaps_all[m]
        km = _iqr_mask(vals); vals_f = vals[km]
        ax.scatter(np.full(vals_f.size, rho), vals_f,
                    color='#2E4A6F', alpha=0.35, s=32, zorder=3)
        mean_v = float(vals_f.mean()); std_v = float(vals_f.std())
        ax.errorbar(rho, mean_v, yerr=std_v, fmt='o', color='#0F8A3C',
                     markersize=8, capsize=4, zorder=4)
        per_rho_means.append(mean_v)

    theory = float(np.mean(per_rho_means))
    ax.axhline(theory, color='k', ls='--', lw=1.2,
                label=f'Empirical mean = {theory:+.3f}')
    ax.axhline(0.0, color='r', ls=':', lw=1, alpha=0.6)
    ax.set_xlabel('true DGP ρ')
    ax.set_ylabel('marginal NLL gap  (marg - joint)')
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
_panel(axes[0], 'nll_uwyk',  'nll_ours50',
        'Marginal NLL gap (UWYK - R-PFN fn=50) vs true ρ')
_panel(axes[1], 'nll_dopfn', 'nll_ours10',
        'Marginal NLL gap (Do-PFN - R-PFN fn=10) vs true ρ')

for ax in axes:
    ax.set_ylim(-0.5, 2)

fig.tight_layout()
out = '/Users/furkandanisman/R-PFN/paper/theory/figures/marginal_nll_test_noout.png'
fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
print(f'[save] {out}')
