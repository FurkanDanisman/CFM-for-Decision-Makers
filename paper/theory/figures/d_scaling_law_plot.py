"""Single-panel plot: √PEHE ratio at N = 1250·d for d = 2..11 with
√2 reference line. K = 15 SCMs per cell on the controlled linear SCM.
"""
import numpy as np, matplotlib.pyplot as plt

data = [
    (2,  2500,  1.712),
    (3,  3750,  1.564),
    (4,  5000,  1.547),
    (5,  6250,  1.417),
    (6,  7500,  1.430),
    (7,  8750,  1.514),
    (8, 10000,  1.317),
    (9, 11250,  1.321),
    (10, 12500, 1.335),
    (11, 13750, 1.283),
]
d = np.array([x[0] for x in data]); r = np.array([x[2] for x in data])

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.axhline(np.sqrt(2), color='k', ls='--', lw=1.6, label=r'Theorem 3.2: $\sqrt{2}$')
ax.axhline(1.0, color='r', ls=':', lw=1.0, alpha=0.6, label='no improvement')
ax.plot(d, r, 'o-', color='#0F8A3C', lw=2, markersize=10)
for xi, yi in zip(d, r):
    ax.annotate(f'{yi:.3f}', xy=(xi, yi), xytext=(0, 8),
                 textcoords='offset points', ha='center', fontsize=8.5)

ax.set_xlabel(r'Covariate dimension $d$   (with $N = 1250 \cdot d$)', fontsize=11)
ax.set_ylabel(r'$\sqrt{\mathrm{PEHE}}$ ratio', fontsize=11)
ax.set_ylim(0.9, 2.0)
ax.set_xticks(range(2, 12))
ax.grid(alpha=0.3)
ax.legend(loc='upper right', fontsize=10)

fig.tight_layout()
out = '/Users/furkandanisman/R-PFN/paper/theory/figures/d_scaling_law.png'
fig.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig)
print(f'[save] {out}')
