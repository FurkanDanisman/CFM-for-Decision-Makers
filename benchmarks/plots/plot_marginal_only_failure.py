"""Paper illustration: marginal-only causal foundation models are structurally
under-identified for the treatment-effect distribution.

Argument in one figure:

  * The marginal distributions p(Y_do0) and p(Y_do1) are IDENTICAL across
    three couplings (Independent / Comonotonic / Countermonotonic).
  * The corresponding treatment-effect distributions p(τ) are wildly different.
  * A marginal-only model (e.g. UWYK, CausalPFN) only receives the marginals
    — it cannot distinguish which coupling is true, and its inferred p(τ)
    contains *phantom modes* that no real coupling actually produces.

Layout (paper-quality):

  Row 1  ─  the three ACTUAL joints (A, B, C)
  Row 2  ─  the three TRUE treatment-effect distributions (their consequences)
  Row 3  ─  shared marginals + the marginal-only model's inferred TE
            with phantom modes highlighted

No model or checkpoint involved; pure numpy + matplotlib. Run:
    python benchmarks/plots/plot_marginal_only_failure.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


# ── Setup ──────────────────────────────────────────────────────────────────
# Closer marginals (Y0 modes at ±3, Y1 modes at ±4) — so p(Y_do0) and
# p(Y_do1) look almost identical, making the "same marginals → wildly
# different τ" story much more striking. τ modes are ±1 (co-) or ±7
# (counter-monotonic).
Y0_CENTRES = np.array([-3.0, +3.0])
Y1_CENTRES = np.array([-4.0, +4.0])
SIGMA_Y    = 0.25
GRID_Y     = np.linspace(-8, 8, 801)
GRID_TAU   = np.linspace(-12, 12, 1001)

CASES = [
    ('A. Independent',
     [((-3, -4), 0.25), ((-3, +4), 0.25), ((+3, -4), 0.25), ((+3, +4), 0.25)],
     [-7, -1, +1, +7]),                                   # τ modes present
    ('B. Comonotonic',
     [((-3, -4), 0.50), ((+3, +4), 0.50)],
     [-1, +1]),
    ('C. Countermonotonic',
     [((-3, +4), 0.50), ((+3, -4), 0.50)],
     [-7, +7]),
]
ALL_TAUS = [-7, -1, +1, +7]                                # phantom-mode reference

# Per-case colors — used consistently for joint frames AND TE distributions.
CASE_COLORS = ['#4B7BB1',   # A — steely blue
                '#0F8A3C',   # B — green (positive coupling)
                '#C1420F']   # C — red   (negative coupling)


def kde_1d(centres, weights, grid, sigma=SIGMA_Y):
    d = grid[:, None] - centres[None, :]
    k = np.exp(-0.5 * (d / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    return (weights[None, :] * k).sum(axis=1)


def kde_2d(joint_probs, grid_y0, grid_y1, sigma=SIGMA_Y):
    Yg0, Yg1 = np.meshgrid(grid_y0, grid_y1, indexing='ij')
    Z = np.zeros_like(Yg0)
    two_s2 = 2 * sigma ** 2
    norm  = 1 / (2 * np.pi * sigma ** 2)
    for (y0, y1), p in joint_probs:
        Z += p * norm * np.exp(-((Yg0 - y0) ** 2 + (Yg1 - y1) ** 2) / two_s2)
    return Z


def kde_tau(joint_probs, grid_tau, sigma=SIGMA_Y * np.sqrt(2)):
    d = grid_tau[:, None] - np.array([[y1 - y0] for (y0, y1), _ in joint_probs]).T
    k = np.exp(-0.5 * (d / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    w = np.array([p for _, p in joint_probs])
    return (w[None, :] * k).sum(axis=1)


# ── Shared marginals ───────────────────────────────────────────────────────
p_y0 = kde_1d(Y0_CENTRES, np.array([0.5, 0.5]), GRID_Y)
p_y1 = kde_1d(Y1_CENTRES, np.array([0.5, 0.5]), GRID_Y)

# ── Marginal-only model's implicit TE = independence assumption (case A) ───
marg_only_p_tau = kde_tau(CASES[0][1], GRID_TAU)


# ── Figure ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
})

fig = plt.figure(figsize=(15.5, 12))
gs = fig.add_gridspec(3, 3, height_ratios=[1.4, 1.0, 1.2],
                       hspace=0.55, wspace=0.30, left=0.06, right=0.98,
                       top=0.94, bottom=0.06)

C_PHANTOM = '#C1420F'      # red — phantom / spurious modes in marginal-only inference
C_JOINT   = 'Greys'        # neutral joint cmap so the per-case colors on top pop

# Row 1: joints — frame + active dots colored per case
extent = [GRID_Y.min(), GRID_Y.max(), GRID_Y.min(), GRID_Y.max()]
for c, (label, jp, real_taus) in enumerate(CASES):
    ax = fig.add_subplot(gs[0, c])
    Z = kde_2d(jp, GRID_Y, GRID_Y)
    ax.imshow(Z.T, origin='lower', extent=extent, cmap=C_JOINT, aspect='auto')
    color = CASE_COLORS[c]
    # Highlight active joint modes with the case's color; grey out the absent
    # corners so the reader immediately sees what the coupling picks out.
    for (y0, y1) in [(-3, -4), (-3, +4), (+3, -4), (+3, +4)]:
        active = any(pp[0] == (y0, y1) for pp in jp)
        ax.plot(y0, y1, 'o',
                color=color if active else 'lightgray',
                markersize=(14 if active else 7),
                markeredgecolor='black', markeredgewidth=1.4, zorder=5)
    # τ = const diagonals for reference
    for t in ALL_TAUS:
        xs = np.linspace(-8, 8, 30)
        ax.plot(xs, xs + t, ls='--', color=color, lw=0.6, alpha=0.35)
    # Coloured frame + title so joint ↔ TE row 2 pairing is obvious
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(2.4)
    ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
    ax.set_xlabel(r'$Y_{do0}$'); ax.set_ylabel(r'$Y_{do1}$')
    ax.set_title(f'{label}   joint $p(Y_{{do0}}, Y_{{do1}})$',
                  fontsize=12, color=color, fontweight='bold')


# Row 2: true TE per case — SAME per-case color as row 1
for c, (label, jp, real_taus) in enumerate(CASES):
    ax = fig.add_subplot(gs[1, c])
    color = CASE_COLORS[c]
    p_tau = kde_tau(jp, GRID_TAU)
    ax.fill_between(GRID_TAU, p_tau, alpha=0.30, color=color)
    ax.plot(GRID_TAU, p_tau, color=color, lw=2.2)
    for t in ALL_TAUS:
        ax.axvline(t, color='gray', ls=':', lw=0.7, alpha=0.4)
    for t in real_taus:
        ax.plot(t, 0, marker='v', color=color, markersize=12,
                markeredgecolor='white', markeredgewidth=0.8, zorder=5, clip_on=False)
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(2.4)
    real_str = '{' + ', '.join(f'{t:+d}' for t in real_taus) + '}'
    ax.set_title(f'TRUE  $p(\\tau)$    modes at $\\tau \\in$ {real_str}',
                  fontsize=11, color=color, fontweight='bold')
    ax.set_xlim(-12, 12)
    ax.set_ylim(0, 0.85)
    ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$')
    if c == 0: ax.set_ylabel(r'$p(\tau)$')
    ax.grid(alpha=0.25)


# Row 3: marginals (left) + marginal-only model's inferred TE (spans 2 cols)
ax_mg = fig.add_subplot(gs[2, 0])
ax_mg.fill_between(GRID_Y, p_y0, alpha=0.35, color='steelblue')
ax_mg.plot(GRID_Y, p_y0, color='steelblue', lw=2.0, label=r'$p(Y_{do0})$')
ax_mg.fill_between(GRID_Y, p_y1, alpha=0.35, color='#7B3E9E')
ax_mg.plot(GRID_Y, p_y1, color='#7B3E9E', lw=2.0, label=r'$p(Y_{do1})$')
ax_mg.set_title('Marginals (ALL that UWYK / CausalPFN see)',
                 fontsize=11, fontweight='bold')
ax_mg.set_xlim(-15, 15)
ax_mg.set_xlabel(r'$Y$')
ax_mg.set_ylabel('density')
ax_mg.legend(loc='upper right', fontsize=10)
ax_mg.grid(alpha=0.25)

ax_bad = fig.add_subplot(gs[2, 1:])
ax_bad.fill_between(GRID_TAU, marg_only_p_tau, alpha=0.28, color=C_PHANTOM)
ax_bad.plot(GRID_TAU, marg_only_p_tau, color=C_PHANTOM, lw=2.4)
# Mark each of the 4 candidate modes, and colour-code by which world would
# call it TRUE vs which would call it PHANTOM. The reader can trace:
#   ±1 modes → truthful under A, truthful under B, PHANTOM under C
#   ±7 modes → truthful under A, PHANTOM under B, truthful under C
mode_heights = {t: float(np.interp(t, GRID_TAU, marg_only_p_tau)) for t in ALL_TAUS}
# Alternate the label heights so the closely-spaced ±1 pair doesn't collide.
label_y_offset = {-7: 0.05, -1: 0.18, +1: 0.05, +7: 0.18}
for t in ALL_TAUS:
    ax_bad.plot(t, mode_heights[t], marker='v', color=C_PHANTOM, markersize=13,
                markeredgecolor='white', markeredgewidth=0.8, zorder=5)
    # Show which cases would consider this mode true (A always; B iff |τ|=1; C iff |τ|=7)
    where_true = ['A']
    if abs(t) == 1: where_true.append('B')
    if abs(t) == 7: where_true.append('C')
    where_str = ', '.join(where_true)
    y_off = label_y_offset[t]
    ax_bad.text(t, mode_heights[t] + y_off,
                 f'$\\tau={t:+d}$\nreal in {where_str}',
                 ha='center', va='bottom', color=C_PHANTOM,
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=C_PHANTOM, lw=0.8))
    # thin leader line from mode marker to label when the offset is tall
    if y_off > 0.10:
        ax_bad.plot([t, t], [mode_heights[t] + 0.005, mode_heights[t] + y_off - 0.005],
                     color=C_PHANTOM, lw=0.6, alpha=0.5, zorder=4)
ax_bad.set_title('Marginal-only model\'s INFERRED $p(\\tau)$   '
                  '(same output for A, B, C — cannot distinguish)',
                  fontsize=12, color=C_PHANTOM, fontweight='bold')
ax_bad.text(0.02, 0.97,
             'Half of these modes are PHANTOMS in worlds B and C:\n'
             '  under B (green), the $\\pm 7$ modes correspond to no real unit.\n'
             '  under C (red),   the $\\pm 1$ modes correspond to no real unit.\n'
             'A marginal-only model has no way to know which is which.',
             transform=ax_bad.transAxes, ha='left', va='top',
             fontsize=10, color='#222',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#666', lw=1))
ax_bad.set_xlim(-12, 12)
ax_bad.set_ylim(0, 0.85)
ax_bad.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$')
ax_bad.set_ylabel(r'$p(\tau)$')
for spine in ax_bad.spines.values():
    spine.set_edgecolor(C_PHANTOM); spine.set_linewidth(2.4)
ax_bad.grid(alpha=0.25)


fig.suptitle(
    r'Same marginals $\Rightarrow$ many possible joint couplings $\Rightarrow$ '
    r'many possible treatment-effect distributions.'
    '\nA marginal-only causal foundation model has no way to identify which — '
    r'so its inferred $p(\tau)$ contains phantom modes for every coupling except independence.',
    fontsize=13, y=0.995)

_HERE   = os.path.dirname(os.path.abspath(__file__))
_OUTDIR = os.path.join(_HERE, 'TE_demonstration')
os.makedirs(_OUTDIR, exist_ok=True)
out_path = os.path.join(_OUTDIR, 'marginal_only_failure.png')
fig.savefig(out_path, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'[save] {out_path}')
