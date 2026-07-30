"""Plot the N-sweep √PEHE ratios vs theoretical √2 line.

Reads sweep numbers hard-coded from
context_sweep aggregate table (poly source, 500 SCMs per N):
  UWYK-NoAnc, Do-PFN, Ours(fn=50) mean, Ours(fn=10) mean.
"""
import matplotlib.pyplot as plt
import numpy as np

N = np.array([50, 250, 500, 1000, 5000, 10000])
uwyk_noanc  = np.array([1.48, 1.42, 1.38, 1.35, 1.27, 1.25])
dopfn       = np.array([1.36, 1.30, 1.29, 1.27, 1.29, 1.28])
ours_fn50   = np.array([1.25, 1.11, 1.09, 1.06, 0.92, 0.87])
ours_fn10   = np.array([1.30, 1.26, 1.26, 1.23, 1.14, 1.11])

ratio_50_vs_uwyk  = uwyk_noanc / ours_fn50   # 1.184, 1.279, 1.266, 1.274, 1.380, 1.437
ratio_10_vs_dopfn = dopfn      / ours_fn10   # 1.046, 1.032, 1.024, 1.033, 1.132, 1.153

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)

for ax, ratio, label, color in [
    (axes[0], ratio_50_vs_uwyk,  r'$\sqrt{\mathrm{PEHE}}_{\mathrm{UWYK\text{-}NoAnc}} / \sqrt{\mathrm{PEHE}}_{\mathrm{Ours(fn{=}50)}}$', '#0F8A3C'),
    (axes[1], ratio_10_vs_dopfn, r'$\sqrt{\mathrm{PEHE}}_{\mathrm{Do\text{-}PFN}} / \sqrt{\mathrm{PEHE}}_{\mathrm{Ours(fn{=}10)}}$',      '#B84A2A'),
]:
    ax.axhline(np.sqrt(2), color='k', ls='--', lw=1.6, label=r'Theorem 3.2: $\sqrt{2}$')
    ax.axhline(1.0,        color='r', ls=':',  lw=1.0, alpha=0.6, label='no improvement')
    ax.plot(N, ratio, 'o-', color=color, lw=2.2, markersize=9, label='empirical')
    for xi, yi in zip(N, ratio):
        ax.annotate(f'{yi:.3f}', xy=(xi, yi), xytext=(0, 8),
                     textcoords='offset points', ha='center', fontsize=8.5)
    ax.set_xscale('log')
    ax.set_xlabel(r'Context size $N$', fontsize=11)
    ax.set_ylim(0.9, 1.55)
    ax.grid(alpha=0.3, which='both')
    ax.legend(loc='lower right', fontsize=9.5)
    ax.set_title(label, fontsize=10.5)

axes[0].set_ylabel(r'$\sqrt{\mathrm{PEHE}}$ ratio', fontsize=11)

fig.suptitle(r'CausalPFN Polynomial DGP context sweep: convergence to $\sqrt{2}$ CR limit',
              fontsize=12)
fig.tight_layout()
fig.savefig('/Users/furkandanisman/R-PFN/paper/theory/figures/n_sweep_ratio.png',
             dpi=160, bbox_inches='tight')
plt.close(fig)
print('[save] figures/n_sweep_ratio.png')
