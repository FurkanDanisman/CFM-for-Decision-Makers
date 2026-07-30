"""Plot d-sweep √PEHE ratio and stability vs theoretical CR limit."""
import matplotlib.pyplot as plt
import numpy as np

d = np.array([5, 10, 20, 30, 50])
uwyk_mean = np.array([1.488, 1.431, 1.347, 1.418, 1.450])
ours_mean = np.array([1.200, 1.326, 1.331, 1.393, 1.463])
uwyk_std  = np.array([0.58,  0.27,  0.22,  0.16,  0.13])
ours_std  = np.array([0.44,  0.29,  0.23,  0.19,  0.13])

sqrt_pehe_ratio  = uwyk_mean / ours_mean
mse_ratio        = sqrt_pehe_ratio ** 2
stability_ratio  = ours_std / uwyk_std

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

ax = axes[0]
ax.axhline(np.sqrt(2), color='k', ls='--', lw=1.6, label=r'Theorem 3.2: $\sqrt{2}$')
ax.axhline(1.0,        color='r', ls=':',  lw=1.0, alpha=0.6, label='no improvement')
ax.plot(d, sqrt_pehe_ratio, 'o-', color='#0F8A3C', lw=2.2, markersize=10)
for xi, yi in zip(d, sqrt_pehe_ratio):
    ax.annotate(f'{yi:.3f}', xy=(xi, yi), xytext=(0, 8),
                 textcoords='offset points', ha='center', fontsize=9)
ax.set_xscale('log')
ax.set_xlabel(r'Covariate dimension $d$', fontsize=11)
ax.set_ylabel(r'$\sqrt{\mathrm{PEHE}}$ ratio', fontsize=11)
ax.set_ylim(0.9, 1.55)
ax.grid(alpha=0.3, which='both')
ax.legend(loc='upper right', fontsize=10)
ax.set_title(r'$\sqrt{\mathrm{PEHE}}_{\mathrm{UWYK\text{-}NoAnc}} / \sqrt{\mathrm{PEHE}}_{\mathrm{Ours(fn{=}50)}}$', fontsize=10.5)

ax = axes[1]
ax.axhline(1/np.sqrt(2), color='k', ls='--', lw=1.6, label=r'Cor 3.3: $1/\sqrt{2}$')
ax.axhline(1.0,          color='r', ls=':',  lw=1.0, alpha=0.6, label='equal spread')
ax.plot(d, stability_ratio, 's-', color='#B84A2A', lw=2.2, markersize=10)
for xi, yi in zip(d, stability_ratio):
    ax.annotate(f'{yi:.3f}', xy=(xi, yi), xytext=(0, 8),
                 textcoords='offset points', ha='center', fontsize=9)
ax.set_xscale('log')
ax.set_xlabel(r'Covariate dimension $d$', fontsize=11)
ax.set_ylabel('stability ratio', fontsize=11)
ax.set_ylim(0.5, 1.35)
ax.grid(alpha=0.3, which='both')
ax.legend(loc='upper left', fontsize=10)
ax.set_title(r'$\mathrm{Std}_{\mathrm{Ours}} / \mathrm{Std}_{\mathrm{UWYK\text{-}NoAnc}}$ across SCMs', fontsize=10.5)

fig.suptitle(r'Controlled linear SCM d-sweep ($\rho=0$, $N=200$, $K=15$/d)', fontsize=12)
fig.tight_layout()
fig.savefig('/Users/furkandanisman/R-PFN/paper/theory/figures/d_sweep_ratio.png',
             dpi=160, bbox_inches='tight')
plt.close(fig)
print('[save] figures/d_sweep_ratio.png')
