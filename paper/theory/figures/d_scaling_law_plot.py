"""Plot: √PEHE ratio at N = 1250·d for d = 2..11, with √2 line, and
the fitted N*(d) power-law curve with extrapolation to Table 3 dims.
"""
import numpy as np, matplotlib.pyplot as plt

# K=15 sweep at N = 1250·d
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
d = np.array([x[0] for x in data]); N = np.array([x[1] for x in data]); r = np.array([x[2] for x in data])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

# ── Left: √PEHE ratio vs d at N=1250d ──
ax = axes[0]
ax.axhline(np.sqrt(2), color='k', ls='--', lw=1.6, label=r'Theorem 3.2: $\sqrt{2}$')
ax.axhline(1.0, color='r', ls=':', lw=1.0, alpha=0.6, label='no improvement')
ax.plot(d, r, 'o-', color='#0F8A3C', lw=2, markersize=10)
for xi, yi in zip(d, r):
    ax.annotate(f'{yi:.3f}', xy=(xi, yi), xytext=(0, 8),
                 textcoords='offset points', ha='center', fontsize=8.5)
ax.set_xlabel(r'Covariate dimension $d$   (with $N = 1250 \cdot d$)', fontsize=11)
ax.set_ylabel(r'$\sqrt{\mathrm{PEHE}}$ ratio', fontsize=11)
ax.set_title(r'√PEHE ratio at $N = 1250 \cdot d$ (K=15 SCMs / cell)', fontsize=11)
ax.set_ylim(0.9, 2.0)
ax.grid(alpha=0.3)
ax.legend(loc='upper right', fontsize=10)

# ── Right: fitted N*(d) power law with extrapolation ──
ax = axes[1]

# Two fits
Nstar_sqrt = N * (np.sqrt(2)/r)**2
Nstar_lin  = N * (np.sqrt(2)/r)

for label, y, color in [
    (r'sqrt-rate fit: $N^*=697\cdot d^{1.32}$  ($R^2 = 0.99$)',   Nstar_sqrt, '#0F8A3C'),
    (r'linear-rate fit: $N^*=933\cdot d^{1.16}$  ($R^2 = 0.997$)', Nstar_lin,  '#B84A2A'),
]:
    logd, logN = np.log(d), np.log(y)
    A_mat = np.vstack([logd, np.ones_like(logd)]).T
    (p, logA), *_ = np.linalg.lstsq(A_mat, logN, rcond=None)
    A = np.exp(logA)
    dgrid = np.logspace(np.log10(2), np.log10(100), 100)
    ax.plot(dgrid, A * dgrid**p, '-', color=color, lw=2, alpha=0.9, label=label)
    ax.scatter(d, y, color=color, s=45, zorder=4, edgecolor='k', linewidth=0.5)

# Table 3 dim markers
table3 = {'CPS (d=8, N=14559)': (8, 14559), 'IHDP (d=25, N=672)': (25, 672),
          'ACIC (d=58, N=4321)': (58, 4321)}
for name, (dv, nv) in table3.items():
    ax.scatter(dv, nv, marker='*', s=140, color='k', zorder=5)
    ax.annotate(name, xy=(dv, nv), xytext=(4, -12),
                 textcoords='offset points', fontsize=8.5)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'Covariate dimension $d$', fontsize=11)
ax.set_ylabel(r'$N^*(d)$ required to reach $\sqrt{2}$', fontsize=11)
ax.set_title(r'Fitted $N^*(d) = O(d^p)$ scaling law + extrapolation', fontsize=11)
ax.grid(alpha=0.3, which='both')
ax.legend(loc='upper left', fontsize=9)

fig.tight_layout()
out = '/Users/furkandanisman/R-PFN/paper/theory/figures/d_scaling_law.png'
fig.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig)
print(f'[save] {out}')
