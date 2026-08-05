"""√PEHE ratio at N ≈ 1250·d for d = 2..11 with 95% CI error bars
around each point and Theorem 3.2's √2 as a reference line.

Values below are the CI-report from the K=15+ aggregate:
  mean ± SE  →  95% CI = mean ± 1.96·SE
"""
import numpy as np, matplotlib.pyplot as plt

# (d, mean, SE)  from aggregate table
data = [
    ( 2, 1.755, 0.080),
    ( 3, 1.572, 0.079),
    ( 4, 1.565, 0.070),
    ( 5, 1.424, 0.056),
    ( 6, 1.429, 0.037),
    ( 7, 1.543, 0.068),
    ( 8, 1.347, 0.063),
    ( 9, 1.336, 0.049),
    (10, 1.353, 0.058),
    (11, 1.280, 0.047),
]
d = np.array([x[0] for x in data])
mu = np.array([x[1] for x in data])
se = np.array([x[2] for x in data])
ci = 1.96 * se        # 95% half-width

sqrt2 = np.sqrt(2)

fig, ax = plt.subplots(figsize=(8.5, 5.0))

# Reference lines
ax.axhline(sqrt2, color='k', ls='--', lw=1.6,
            label=r'Theorem 3.2: $\sqrt{2} \approx 1.414$', zorder=3)
ax.axhline(1.0, color='r', ls=':', lw=1.0, alpha=0.6,
            label='no improvement', zorder=3)

# 95% CI band along the mean curve
ax.fill_between(d, mu - ci, mu + ci, color='#0F8A3C', alpha=0.20,
                  label='95% CI', zorder=2)
# Mean curve
ax.plot(d, mu, 'o-', color='#0F8A3C', lw=2, markersize=9,
         markeredgecolor='k', markeredgewidth=0.6, zorder=5,
         label='mean $\sqrt{\mathrm{PEHE}}$ ratio (K = 15+ SCMs)')

ax.set_xlabel(r'Covariate dimension $d$   (with $N \approx 1250 \cdot d$)',
                fontsize=11)
ax.set_ylabel(r'$\sqrt{\mathrm{PEHE}}$ ratio', fontsize=11)
ax.set_xlim(1.5, 11.5)
ax.set_ylim(0.9, 2.0)
ax.set_xticks(range(2, 12))
ax.grid(alpha=0.25)
ax.legend(loc='upper right', fontsize=9.5)

fig.tight_layout()
out = '/Users/furkandanisman/R-PFN/paper/theory/figures/d_scaling_law.png'
fig.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig)
print(f'[save] {out}')
