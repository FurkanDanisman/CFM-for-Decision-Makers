"""Paper illustration — minimal, arrow-based flow:

  Row 1 (3 columns)  three admissible joint couplings
                      \    |    /
                       ↘   ↓   ↙       (three arrows converging)
  Row 2 (centered)   one shared marginal — all three joints produce it
                       ↙   ↓   ↘       (three arrows diverging)
                      /    |    \
  Row 3 (3 columns)  three completely different TE distributions

No titles, no axis labels — the geometry alone carries the argument.

Run:
    python benchmarks/plots/plot_marginal_only_failure.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


# ── Distribution setup ─────────────────────────────────────────────────────
# Y_do0 modes at ±1, Y_do1 modes at ±2 → τ modes at {−3, −1, +1, +3}
# (comonotonic gives ±1, countermonotonic gives ±3, independent gives all four).
Y0_CENTRES = np.array([-1.0, +1.0])
Y1_CENTRES = np.array([-2.0, +2.0])
SIGMA_Y    = 0.15
GRID_Y     = np.linspace(-4, 4, 801)
GRID_TAU   = np.linspace(-6, 6, 1001)

CASES = [
    ('A', [((-1, -2), 0.25), ((-1, +2), 0.25), ((+1, -2), 0.25), ((+1, +2), 0.25)]),
    ('B', [((-1, -2), 0.50),                                     ((+1, +2), 0.50)]),
    ('C', [                    ((-1, +2), 0.50), ((+1, -2), 0.50)                  ]),
]
CASE_COLORS = ['#4B7BB1', '#0F8A3C', '#C1420F']   # blue / green / red


def kde_1d(centres, weights, grid, sigma=SIGMA_Y):
    d = grid[:, None] - centres[None, :]
    k = np.exp(-0.5 * (d / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    return (weights[None, :] * k).sum(axis=1)


def kde_2d(joint_probs, grid_y0, grid_y1, sigma=SIGMA_Y):
    Yg0, Yg1 = np.meshgrid(grid_y0, grid_y1, indexing='ij')
    Z = np.zeros_like(Yg0)
    two_s2 = 2 * sigma ** 2
    for (y0, y1), p in joint_probs:
        Z += p * np.exp(-((Yg0 - y0) ** 2 + (Yg1 - y1) ** 2) / two_s2)
    return Z


def kde_tau(joint_probs, grid_tau, sigma=SIGMA_Y * np.sqrt(2)):
    d = grid_tau[:, None] - np.array([[y1 - y0] for (y0, y1), _ in joint_probs]).T
    k = np.exp(-0.5 * (d / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    w = np.array([p for _, p in joint_probs])
    return (w[None, :] * k).sum(axis=1)


# ── Figure — 3 rows × 3 cols, middle row has one central panel ─────────────
fig = plt.figure(figsize=(13, 12))
gs = fig.add_gridspec(3, 3,
                       height_ratios=[1.0, 0.75, 1.0],
                       hspace=0.55, wspace=0.20,
                       left=0.06, right=0.98, top=0.98, bottom=0.05)


# Row 1: three joints
extent = [GRID_Y.min(), GRID_Y.max(), GRID_Y.min(), GRID_Y.max()]
joint_axes = []
for c, ((_, jp), color) in enumerate(zip(CASES, CASE_COLORS)):
    ax = fig.add_subplot(gs[0, c])
    joint_axes.append(ax)
    Z = kde_2d(jp, GRID_Y, GRID_Y)
    ax.imshow(Z.T, origin='lower', extent=extent, cmap='Greys', aspect='auto')
    for (y0, y1) in [(-1, -2), (-1, +2), (+1, -2), (+1, +2)]:
        active = any(pp[0] == (y0, y1) for pp in jp)
        ax.plot(y0, y1, 'o',
                color=color if active else 'lightgray',
                markersize=(15 if active else 7),
                markeredgecolor='black', markeredgewidth=1.4, zorder=5)
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(2.6)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_xticks([]); ax.set_yticks([])


# Row 2: single shared marginal in the center column, other cells empty
ax_marg = fig.add_subplot(gs[1, 1])
p_y0 = kde_1d(Y0_CENTRES, np.array([0.5, 0.5]), GRID_Y)
p_y1 = kde_1d(Y1_CENTRES, np.array([0.5, 0.5]), GRID_Y)
ax_marg.fill_between(GRID_Y, p_y0, alpha=0.35, color='steelblue')
ax_marg.plot(GRID_Y, p_y0, color='steelblue', lw=2.2)
ax_marg.fill_between(GRID_Y, p_y1, alpha=0.35, color='#7B3E9E')
ax_marg.plot(GRID_Y, p_y1, color='#7B3E9E', lw=2.2)
ax_marg.set_xlim(-4, 4)
ax_marg.set_xticks([]); ax_marg.set_yticks([])
for spine in ax_marg.spines.values():
    spine.set_edgecolor('#444'); spine.set_linewidth(2.0)


# Row 3: three TE distributions
te_axes = []
for c, ((_, jp), color) in enumerate(zip(CASES, CASE_COLORS)):
    ax = fig.add_subplot(gs[2, c])
    te_axes.append(ax)
    p_tau = kde_tau(jp, GRID_TAU)
    ax.fill_between(GRID_TAU, p_tau, alpha=0.35, color=color)
    ax.plot(GRID_TAU, p_tau, color=color, lw=2.4)
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, p_tau.max() * 1.15)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(2.6)


# ── Arrows: joints ↘↓↙ into the shared marginal, then ↙↓↘ out to the TEs
fig.canvas.draw()   # force layout so bboxes are meaningful


def _bbox_center_bottom(ax):
    bb = ax.get_position()
    return (bb.x0 + bb.width / 2, bb.y0)


def _bbox_center_top(ax):
    bb = ax.get_position()
    return (bb.x0 + bb.width / 2, bb.y0 + bb.height)


for c, ax in enumerate(joint_axes):
    src = _bbox_center_bottom(ax)
    dst = _bbox_center_top(ax_marg)
    color = CASE_COLORS[c]
    arrow = FancyArrowPatch(src, dst,
                             transform=fig.transFigure,
                             arrowstyle='-|>', mutation_scale=22,
                             color=color, lw=2.2,
                             shrinkA=6, shrinkB=8)
    fig.add_artist(arrow)

for c, ax in enumerate(te_axes):
    src = _bbox_center_bottom(ax_marg)
    dst = _bbox_center_top(ax)
    color = CASE_COLORS[c]
    arrow = FancyArrowPatch(src, dst,
                             transform=fig.transFigure,
                             arrowstyle='-|>', mutation_scale=22,
                             color=color, lw=2.2,
                             shrinkA=8, shrinkB=6)
    fig.add_artist(arrow)


_HERE   = os.path.dirname(os.path.abspath(__file__))
_OUTDIR = os.path.join(_HERE, 'TE_demonstration')
os.makedirs(_OUTDIR, exist_ok=True)
out_path = os.path.join(_OUTDIR, 'marginal_only_failure.png')
fig.savefig(out_path, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'[save] {out_path}')
