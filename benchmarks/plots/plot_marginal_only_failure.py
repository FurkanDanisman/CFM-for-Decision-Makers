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
    ('A. Independent',       [((-1, -2), 0.25), ((-1, +2), 0.25), ((+1, -2), 0.25), ((+1, +2), 0.25)],
     [-3, -1, +1, +3]),
    ('B. Comonotonic',       [((-1, -2), 0.50),                                     ((+1, +2), 0.50)],
     [-1, +1]),
    ('C. Countermonotonic',  [                    ((-1, +2), 0.50), ((+1, -2), 0.50)                  ],
     [-3, +3]),
]
ALL_TAUS = [-3, -1, +1, +3]                              # union of the three cases
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


def _bbox_center_bottom(ax):
    bb = ax.get_position()
    return (bb.x0 + bb.width / 2, bb.y0)


def _bbox_center_top(ax):
    bb = ax.get_position()
    return (bb.x0 + bb.width / 2, bb.y0 + bb.height)


def build_figure(with_text: bool):
    """Build the 3-row figure. When `with_text=True` renders panel titles,
    axis labels, and the marginals legend; when False renders none of them
    (minimalist arrows-only layout)."""
    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(3, 3,
                           height_ratios=[1.0, 0.75, 1.0],
                           hspace=0.55, wspace=0.20,
                           left=0.06, right=0.98, top=0.98, bottom=0.05)

    # Row 1: three joints
    extent = [GRID_Y.min(), GRID_Y.max(), GRID_Y.min(), GRID_Y.max()]
    joint_axes = []
    for c, ((label, jp, _real_taus), color) in enumerate(zip(CASES, CASE_COLORS)):
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
        if with_text:
            ax.set_xlabel(r'$Y_{do0}$')
            ax.set_ylabel(r'$Y_{do1}$')
            ax.set_title(f'{label}   joint $p(Y_{{do0}}, Y_{{do1}})$',
                          color=color, fontsize=11, fontweight='bold')
        else:
            ax.set_xticks([]); ax.set_yticks([])

    # Row 2: shared marginal in the center column
    ax_marg = fig.add_subplot(gs[1, 1])
    p_y0 = kde_1d(Y0_CENTRES, np.array([0.5, 0.5]), GRID_Y)
    p_y1 = kde_1d(Y1_CENTRES, np.array([0.5, 0.5]), GRID_Y)
    ax_marg.fill_between(GRID_Y, p_y0, alpha=0.35, color='steelblue')
    ax_marg.plot(GRID_Y, p_y0, color='steelblue', lw=2.2, label=r'$p(Y_{do0})$')
    ax_marg.fill_between(GRID_Y, p_y1, alpha=0.35, color='#7B3E9E')
    ax_marg.plot(GRID_Y, p_y1, color='#7B3E9E', lw=2.2, label=r'$p(Y_{do1})$')
    ax_marg.set_xlim(-4, 4)
    if with_text:
        ax_marg.set_xlabel(r'$Y$')
        ax_marg.set_ylabel('density')
        ax_marg.set_title(r'Shared marginals — identical across A, B, C',
                            fontsize=11, fontweight='bold')
        ax_marg.legend(fontsize=10, loc='upper right')
    else:
        ax_marg.set_xticks([]); ax_marg.set_yticks([])
    for spine in ax_marg.spines.values():
        spine.set_edgecolor('#444'); spine.set_linewidth(2.0)

    # Row 3: three TE distributions
    te_axes = []
    for c, ((label, jp, real_taus), color) in enumerate(zip(CASES, CASE_COLORS)):
        ax = fig.add_subplot(gs[2, c])
        te_axes.append(ax)
        p_tau = kde_tau(jp, GRID_TAU)
        ax.fill_between(GRID_TAU, p_tau, alpha=0.35, color=color)
        ax.plot(GRID_TAU, p_tau, color=color, lw=2.4)
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, p_tau.max() * 1.15)
        if with_text:
            ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$')
            if c == 0: ax.set_ylabel(r'$p(\tau)$')
            modes_str = ', '.join(f'{t:+d}' for t in real_taus)
            ax.set_title(f'{label}: modes at $\\tau \\in$ {{{modes_str}}}',
                          color=color, fontsize=11, fontweight='bold')
        else:
            ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2.6)

    # Arrows joints ↘↓↙ → marginal, marginal ↙↓↘ → TEs
    fig.canvas.draw()
    _TITLE_PAD = 0.025 if with_text else 0.0
    _LABEL_PAD = 0.03 if with_text else 0.0

    for c, ax in enumerate(joint_axes):
        src_x, src_y = _bbox_center_bottom(ax)
        dst_x, dst_y = _bbox_center_top(ax_marg)
        src = (src_x, src_y - _LABEL_PAD)
        dst = (dst_x, dst_y + _TITLE_PAD)
        color = CASE_COLORS[c]
        fig.add_artist(FancyArrowPatch(src, dst,
                                        transform=fig.transFigure,
                                        arrowstyle='-|>', mutation_scale=22,
                                        color=color, lw=2.2,
                                        shrinkA=2, shrinkB=2))

    for c, ax in enumerate(te_axes):
        src_x, src_y = _bbox_center_bottom(ax_marg)
        dst_x, dst_y = _bbox_center_top(ax)
        src = (src_x, src_y - _LABEL_PAD)
        dst = (dst_x, dst_y + _TITLE_PAD)
        color = CASE_COLORS[c]
        fig.add_artist(FancyArrowPatch(src, dst,
                                        transform=fig.transFigure,
                                        arrowstyle='-|>', mutation_scale=22,
                                        color=color, lw=2.2,
                                        shrinkA=2, shrinkB=2))
    return fig


def build_phantoms_v2_figure(with_arrows: bool = False):
    """Phantoms-style variant:
      Row 1  three joints  (colored frames + bold per-case titles)
      Row 2  a single shared marginal panel (gray palette, neutral frame)
      Row 3  three TRUE p(τ)  (colored frames + A/B/C prefixed titles)

    with_arrows=True adds per-case colored arrows from each joint down to the
    shared marginal and from the marginal down to each TRUE p(τ), reinforcing
    the "joint → marginal → TE" flow visually.

    No suptitle. τ modes at {−3, −1, +1, +3}.
    """
    # Neutral gray palette for the shared marginals so they read as "input"
    # rather than being confused with any of the coloured cases.
    GRAY_DARK   = '#4A4A4A'
    GRAY_MEDIUM = '#8C8C8C'

    fig = plt.figure(figsize=(14, 11.5))
    # Tighter spacing + slightly narrower row 2 so the layout reads as one
    # compact block instead of three separated bands.
    gs = fig.add_gridspec(3, 3, height_ratios=[1.35, 0.85, 1.1],
                           hspace=0.32, wspace=0.20,
                           left=0.06, right=0.98, top=0.97, bottom=0.055)

    joint_axes, te_axes = [], []

    # Row 1: joints
    extent = [GRID_Y.min(), GRID_Y.max(), GRID_Y.min(), GRID_Y.max()]
    for c, ((label, jp, real_taus), color) in enumerate(zip(CASES, CASE_COLORS)):
        ax = fig.add_subplot(gs[0, c])
        joint_axes.append(ax)
        Z = kde_2d(jp, GRID_Y, GRID_Y)
        ax.imshow(Z.T, origin='lower', extent=extent, cmap='Greys',
                    aspect='auto', vmin=0, vmax=Z.max() * 1.15)
        for (y0, y1) in [(-1, -2), (-1, +2), (+1, -2), (+1, +2)]:
            active = any(pp[0] == (y0, y1) for pp in jp)
            ax.plot(y0, y1, 'o',
                    color=color if active else 'lightgray',
                    markersize=(16 if active else 8),
                    markeredgecolor='black', markeredgewidth=1.4, zorder=5)
        for t in ALL_TAUS:
            xs = np.linspace(-4, 4, 30)
            ax.plot(xs, xs + t, ls='--', color=color, lw=0.6, alpha=0.30)
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2.4)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_xlabel(r'$Y_{do0}$'); ax.set_ylabel(r'$Y_{do1}$')
        ax.set_title(f'{label}   joint $p(Y_{{do0}}, Y_{{do1}})$',
                      fontsize=12, color=color, fontweight='bold', pad=8)
        ax.tick_params(axis='both', which='both', length=3, labelsize=9)

    # Row 2: shared marginal — GRAY palette so it doesn't clash with the
    # per-case colors around it.
    ax_marg = fig.add_subplot(gs[1, 1])
    p_y0 = kde_1d(Y0_CENTRES, np.array([0.5, 0.5]), GRID_Y)
    p_y1 = kde_1d(Y1_CENTRES, np.array([0.5, 0.5]), GRID_Y)
    ax_marg.fill_between(GRID_Y, p_y0, alpha=0.55, color=GRAY_DARK,
                          linewidth=0)
    ax_marg.plot(GRID_Y, p_y0, color=GRAY_DARK, lw=2.0,
                  label=r'$p(Y_{do0})$')
    ax_marg.fill_between(GRID_Y, p_y1, alpha=0.35, color=GRAY_MEDIUM,
                          linewidth=0, hatch='///', edgecolor=GRAY_DARK)
    ax_marg.plot(GRID_Y, p_y1, color=GRAY_MEDIUM, lw=2.0, ls='--',
                  label=r'$p(Y_{do1})$')
    ax_marg.set_xlim(-4, 4)
    ax_marg.set_ylim(0, 1.55)
    ax_marg.set_xlabel(r'$Y$')
    ax_marg.set_ylabel('density')
    ax_marg.set_title(r'Shared marginals — identical across A, B, C',
                        fontsize=12, fontweight='bold', pad=8, color=GRAY_DARK)
    ax_marg.legend(fontsize=10, loc='upper right', frameon=True, framealpha=0.95)
    for spine in ax_marg.spines.values():
        spine.set_edgecolor(GRAY_DARK); spine.set_linewidth(1.6)
    ax_marg.tick_params(axis='both', which='both', length=3, labelsize=9)
    ax_marg.grid(alpha=0.20)

    # Row 3: TRUE p(τ) per case
    for c, ((label, jp, real_taus), color) in enumerate(zip(CASES, CASE_COLORS)):
        ax = fig.add_subplot(gs[2, c])
        te_axes.append(ax)
        p_tau = kde_tau(jp, GRID_TAU)
        ax.fill_between(GRID_TAU, p_tau, alpha=0.30, color=color, linewidth=0)
        ax.plot(GRID_TAU, p_tau, color=color, lw=2.4)
        for t in ALL_TAUS:
            ax.axvline(t, color='gray', ls=':', lw=0.7, alpha=0.4)
        for t in real_taus:
            ax.plot(t, 0, marker='v', color=color, markersize=12,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=5,
                    clip_on=False)
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2.4)
        real_str = '{' + ', '.join(f'{t:+d}' for t in real_taus) + '}'
        # E[τ] under this p(τ) — the estimation a mean-based model would give.
        E_tau = float((GRID_TAU * p_tau).sum() * (GRID_TAU[1] - GRID_TAU[0]))
        ax.axvline(E_tau, color=color, ls='--', lw=1.8, alpha=0.95, zorder=6,
                    label=fr'$\mathbb{{E}}[\tau]={E_tau:+.2f}$')
        # Prefix with the case letter (A / B / C) so row 3 reads as clearly
        # paired with row 1 without having to glance at the colour alone.
        case_letter = label.split('.')[0].strip()   # 'A', 'B', or 'C'
        ax.set_title(f'{case_letter}.  $p(\\tau)$    modes at $\\tau \\in$ {real_str}',
                      fontsize=11, color=color, fontweight='bold', pad=8)
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$')
        if c == 0: ax.set_ylabel(r'$p(\tau)$')
        ax.grid(alpha=0.20)
        ax.tick_params(axis='both', which='both', length=3, labelsize=9)
        ax.legend(fontsize=9, loc='upper right', frameon=True, framealpha=0.95)

    # Arrows: joints ↘↓↙ marginal, then marginal ↙↓↘ TE. Colored per case.
    if with_arrows:
        fig.canvas.draw()

        def _bbox_center_bottom(ax):
            bb = ax.get_position()
            return (bb.x0 + bb.width / 2, bb.y0)

        def _bbox_center_top(ax):
            bb = ax.get_position()
            return (bb.x0 + bb.width / 2, bb.y0 + bb.height)

        _TITLE_PAD = 0.025
        _LABEL_PAD = 0.028
        ARROW_COLOR = '#555555'   # single neutral color for every arrow

        for ax in joint_axes:
            src_x, src_y = _bbox_center_bottom(ax)
            dst_x, dst_y = _bbox_center_top(ax_marg)
            src = (src_x, src_y - _LABEL_PAD)
            dst = (dst_x, dst_y + _TITLE_PAD)
            fig.add_artist(FancyArrowPatch(src, dst,
                                            transform=fig.transFigure,
                                            arrowstyle='-|>', mutation_scale=22,
                                            color=ARROW_COLOR, lw=2.2,
                                            shrinkA=2, shrinkB=2))

        for ax in te_axes:
            src_x, src_y = _bbox_center_bottom(ax_marg)
            dst_x, dst_y = _bbox_center_top(ax)
            src = (src_x, src_y - _LABEL_PAD)
            dst = (dst_x, dst_y + _TITLE_PAD)
            fig.add_artist(FancyArrowPatch(src, dst,
                                            transform=fig.transFigure,
                                            arrowstyle='-|>', mutation_scale=22,
                                            color=ARROW_COLOR, lw=2.2,
                                            shrinkA=2, shrinkB=2))
    return fig


_HERE   = os.path.dirname(os.path.abspath(__file__))
_OUTDIR = os.path.join(_HERE, 'TE_demonstration')
os.makedirs(_OUTDIR, exist_ok=True)

# Save all three variants
for suffix, with_text in [('minimal', False), ('labelled', True)]:
    fig = build_figure(with_text=with_text)
    out_path = os.path.join(_OUTDIR, f'marginal_only_failure_{suffix}.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[save] {out_path}')

for suffix, with_arrows in [('phantoms_v2', False), ('phantoms_v2_arrows', True)]:
    fig = build_phantoms_v2_figure(with_arrows=with_arrows)
    out_path = os.path.join(_OUTDIR, f'marginal_only_failure_{suffix}.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[save] {out_path}')


# ── Individual panels — each subplot as its own PNG ───────────────────────
_PANEL_DIR = os.path.join(_OUTDIR, 'panels')
os.makedirs(_PANEL_DIR, exist_ok=True)


def _save_panel_joint(label, jp, color, name, dashed_frame=False):
    """Render one joint distribution as a stand-alone square figure.

    dashed_frame=True renders the coloured border as a dashed rectangle
    overlay (spine.set_linestyle('--') looks broken on short edges), used
    for the Independent case A to mark it as the marginal-only model's
    implicit assumption rather than a claim about the true coupling.
    """
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    Z = kde_2d(jp, GRID_Y, GRID_Y)
    ax.imshow(Z.T, origin='lower',
                extent=[GRID_Y.min(), GRID_Y.max(), GRID_Y.min(), GRID_Y.max()],
                cmap='Greys', aspect='auto', vmin=0, vmax=Z.max() * 1.15)
    for (y0, y1) in [(-1, -2), (-1, +2), (+1, -2), (+1, +2)]:
        active = any(pp[0] == (y0, y1) for pp in jp)
        ax.plot(y0, y1, 'o',
                color=color if active else 'lightgray',
                markersize=(16 if active else 8),
                markeredgecolor='black', markeredgewidth=1.4, zorder=5)
    for t in ALL_TAUS:
        xs = np.linspace(-4, 4, 30)
        ax.plot(xs, xs + t, ls='--', color=color, lw=0.6, alpha=0.30)
    if dashed_frame:
        # Hide the built-in solid spines and overlay a dashed rectangle border
        for spine in ax.spines.values():
            spine.set_visible(False)
        from matplotlib.patches import Rectangle
        border = Rectangle((-4, -4), 8, 8, fill=False,
                            edgecolor=color, linewidth=2.4,
                            linestyle=(0, (6, 4)),   # long-dash pattern
                            zorder=10, clip_on=False)
        ax.add_patch(border)
    else:
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2.4)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_xlabel(r'$Y_{do0}$'); ax.set_ylabel(r'$Y_{do1}$')
    ax.set_title(f'{label}   joint $p(Y_{{do0}}, Y_{{do1}})$',
                  fontsize=12, color=color, fontweight='bold', pad=8)
    ax.tick_params(axis='both', which='both', length=3, labelsize=9)
    fig.tight_layout()
    path = os.path.join(_PANEL_DIR, f'{name}.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[save] {path}')


def _save_panel_marginals(name):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    p_y0 = kde_1d(Y0_CENTRES, np.array([0.5, 0.5]), GRID_Y)
    p_y1 = kde_1d(Y1_CENTRES, np.array([0.5, 0.5]), GRID_Y)
    GRAY_DARK, GRAY_MEDIUM = '#4A4A4A', '#8C8C8C'
    ax.fill_between(GRID_Y, p_y0, alpha=0.55, color=GRAY_DARK, linewidth=0)
    ax.plot(GRID_Y, p_y0, color=GRAY_DARK, lw=2.0, label=r'$p(Y_{do0})$')
    ax.fill_between(GRID_Y, p_y1, alpha=0.35, color=GRAY_MEDIUM,
                     linewidth=0, hatch='///', edgecolor=GRAY_DARK)
    ax.plot(GRID_Y, p_y1, color=GRAY_MEDIUM, lw=2.0, ls='--',
             label=r'$p(Y_{do1})$')
    ax.set_xlim(-4, 4); ax.set_ylim(0, 1.55)
    ax.set_xlabel(r'$Y$'); ax.set_ylabel('density')
    ax.set_title(r'Shared marginals — identical across A, B, C',
                  fontsize=12, fontweight='bold', pad=8, color=GRAY_DARK)
    ax.legend(fontsize=10, loc='upper right', frameon=True, framealpha=0.95)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRAY_DARK); spine.set_linewidth(1.6)
    ax.tick_params(axis='both', which='both', length=3, labelsize=9)
    ax.grid(alpha=0.20)
    fig.tight_layout()
    path = os.path.join(_PANEL_DIR, f'{name}.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[save] {path}')


def _save_panel_te(label, jp, real_taus, color, name):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    p_tau = kde_tau(jp, GRID_TAU)
    ax.fill_between(GRID_TAU, p_tau, alpha=0.30, color=color, linewidth=0)
    ax.plot(GRID_TAU, p_tau, color=color, lw=2.4)
    for t in ALL_TAUS:
        ax.axvline(t, color='gray', ls=':', lw=0.7, alpha=0.4)
    for t in real_taus:
        ax.plot(t, 0, marker='v', color=color, markersize=12,
                markeredgecolor='white', markeredgewidth=0.8, zorder=5, clip_on=False)
    # E[τ] estimation line — dashed vertical at the mean of the distribution.
    E_tau = float((GRID_TAU * p_tau).sum() * (GRID_TAU[1] - GRID_TAU[0]))
    ax.axvline(E_tau, color=color, ls='--', lw=1.8, alpha=0.95, zorder=6,
                label=fr'$\mathbb{{E}}[\tau]={E_tau:+.2f}$')
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(2.4)
    real_str = '{' + ', '.join(f'{t:+d}' for t in real_taus) + '}'
    case_letter = label.split('.')[0].strip()
    ax.set_title(f'{case_letter}.  $p(\\tau)$    modes at $\\tau \\in$ {real_str}',
                  fontsize=11, color=color, fontweight='bold', pad=8)
    ax.set_xlim(-6, 6); ax.set_ylim(0, 1.05)
    ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$')
    ax.set_ylabel(r'$p(\tau)$')
    ax.grid(alpha=0.20)
    ax.tick_params(axis='both', which='both', length=3, labelsize=9)
    ax.legend(fontsize=9, loc='upper right', frameon=True, framealpha=0.95)
    fig.tight_layout()
    path = os.path.join(_PANEL_DIR, f'{name}.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[save] {path}')


# joints A, B, C — A gets a dashed frame (marginal-only implicit assumption)
for (label, jp, _real), color, tag in zip(CASES, CASE_COLORS, ['A', 'B', 'C']):
    _save_panel_joint(label, jp, color, f'panel_joint_{tag}',
                       dashed_frame=(tag == 'A'))

# shared marginals
_save_panel_marginals('panel_marginals')

# TRUE p(τ) A, B, C
for (label, jp, real_taus), color, tag in zip(CASES, CASE_COLORS, ['A', 'B', 'C']):
    _save_panel_te(label, jp, real_taus, color, f'panel_te_{tag}')
