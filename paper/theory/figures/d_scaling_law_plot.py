"""Single-panel plot: √PEHE ratio at N = 1250·d for d = 2..11 with
√2 reference line and ±5%, ±10% tolerance bands.
K = 15 SCMs per cell on the controlled linear SCM.
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

sqrt2 = np.sqrt(2)
d_lo, d_hi = 1.5, 11.5

fig, ax = plt.subplots(figsize=(8.5, 5.0))

# ── Tolerance bands ──
ax.axhspan(sqrt2 * 0.90, sqrt2 * 1.10, color='#0F8A3C', alpha=0.10,
            label=r'$\pm 10\%$ of $\sqrt{2}$', zorder=1)
ax.axhspan(sqrt2 * 0.95, sqrt2 * 1.05, color='#0F8A3C', alpha=0.18,
            label=r'$\pm 5\%$ of $\sqrt{2}$', zorder=2)

# ── Reference lines ──
ax.axhline(sqrt2, color='k', ls='--', lw=1.6,
            label=r'Theorem 3.2: $\sqrt{2} \approx 1.414$', zorder=3)
ax.axhline(1.0, color='r', ls=':', lw=1.0, alpha=0.6,
            label='no improvement', zorder=3)

# ── Data ──
ax.plot(d, r, 'o-', color='#0F8A3C', lw=2, markersize=10,
         zorder=5, markeredgecolor='k', markeredgewidth=0.6)

ax.set_xlabel(r'Covariate dimension $d$   (with $N = 1250 \cdot d$)',
                fontsize=11)
ax.set_ylabel(r'$\sqrt{\mathrm{PEHE}}$ ratio', fontsize=11)
ax.set_xlim(d_lo, d_hi)
ax.set_ylim(0.9, 2.0)
ax.set_xticks(range(2, 12))
ax.grid(alpha=0.25)
ax.legend(loc='upper right', fontsize=9.5)

fig.tight_layout()
out = '/Users/furkandanisman/R-PFN/paper/theory/figures/d_scaling_law.png'
fig.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig)
print(f'[save] {out}')
