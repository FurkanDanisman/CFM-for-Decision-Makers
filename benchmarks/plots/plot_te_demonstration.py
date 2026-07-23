"""Same marginals, three different joints → three completely different τ shapes.

Setup:
    Y_do0 ~ 0.5 · δ(-5) + 0.5 · δ(+5)   (small Gaussian smoothing σ_y)
    Y_do1 ~ 0.5 · δ(-8) + 0.5 · δ(+8)

These MARGINALS are fixed. The JOINT p(Y_do0, Y_do1) is not — any coupling
consistent with the two marginals is admissible. We visualise three:

    A. Independent    → 4 corners in the joint, each with mass 1/4
                        p(τ) has FOUR modes at {-13, -3, +3, +13} —
                        symmetric around 0, mean τ = 0
    B. Comonotonic    → 2 corners on the ↘ diagonal (+5, +8) and (-5, -8)
                        p(τ) has TWO modes at {-3, +3}
    C. Countermonotonic → 2 corners on the ↙ anti-diagonal (+5, -8) and (-5, +8)
                          p(τ) has TWO modes at {-13, +13}

A marginal-only model (UWYK-style) sees ONLY the marginals — it cannot tell
these three joints apart. Predicting τ = E[Y_do1] − E[Y_do0] = 0 is correct
on average for all three, but it hides everything about which SCM you're
actually in. A joint model (Ours) captures the coupling directly and produces
the right τ shape per case.

No model or checkpoint involved — pure numpy/matplotlib. Just run:
    python benchmarks/plots/plot_te_demonstration.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt


Y0_CENTRES = np.array([-5.0, +5.0])     # p(Y_do0) support
Y1_CENTRES = np.array([-8.0, +8.0])     # p(Y_do1) support
SIGMA_Y    = 0.4                          # visual smoothing for readability
GRID_Y     = np.linspace(-15, 15, 601)
GRID_TAU   = np.linspace(-20, 20, 801)


def kde_1d(centres, weights, grid, sigma=SIGMA_Y):
    """Mixture of 1D Gaussians on `grid`."""
    d = grid[:, None] - centres[None, :]
    k = np.exp(-0.5 * (d / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    return (weights[None, :] * k).sum(axis=1)


def kde_2d(joint_probs, grid_y0, grid_y1, sigma=SIGMA_Y):
    """`joint_probs` is a list of ((y0, y1), p). Renders 2D density on a grid."""
    Yg0, Yg1 = np.meshgrid(grid_y0, grid_y1, indexing='ij')
    Z = np.zeros_like(Yg0)
    two_s2 = 2 * sigma ** 2
    norm  = 1 / (2 * np.pi * sigma ** 2)
    for (y0, y1), p in joint_probs:
        Z += p * norm * np.exp(-((Yg0 - y0) ** 2 + (Yg1 - y1) ** 2) / two_s2)
    return Z


def kde_tau(joint_probs, grid_tau, sigma=SIGMA_Y * np.sqrt(2)):
    """p(τ) implied by the discrete joint. σ for τ = σ√2 because τ = Y_do1 − Y_do0."""
    d = grid_tau[:, None] - np.array([[y1 - y0] for (y0, y1), _ in joint_probs]).T
    k = np.exp(-0.5 * (d / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    w = np.array([p for _, p in joint_probs])
    return (w[None, :] * k).sum(axis=1)


# ── The three joints ────────────────────────────────────────────────────────
CASES = [
    (
        'A.  Independent coupling',
        [((-5, -8), 0.25), ((-5, +8), 0.25), ((+5, -8), 0.25), ((+5, +8), 0.25)],
        'four modes at τ ∈ {−13, −3, +3, +13}',
    ),
    (
        'B.  Comonotonic  (positive coupling)',
        [((-5, -8), 0.50),                       ((+5, +8), 0.50)],
        'two modes at τ ∈ {−3, +3}',
    ),
    (
        'C.  Countermonotonic  (negative coupling)',
        [                    ((-5, +8), 0.50), ((+5, -8), 0.50)                    ],
        'two modes at τ ∈ {−13, +13}',
    ),
]


# ── Shared marginals (same in ALL cases) ────────────────────────────────────
p_y0 = kde_1d(Y0_CENTRES, np.array([0.5, 0.5]), GRID_Y)
p_y1 = kde_1d(Y1_CENTRES, np.array([0.5, 0.5]), GRID_Y)


# ── Build figure: 4 rows × 3 cols  (row 1 spans all cols) ───────────────────
fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(3, 3, height_ratios=[0.95, 1.35, 1.0],
                      hspace=0.45, wspace=0.30)

# Row 0: shared marginals — one wide panel spanning all three columns
ax_marg = fig.add_subplot(gs[0, :])
ax_marg.fill_between(GRID_Y, p_y0, alpha=0.35, color='steelblue')
ax_marg.plot(GRID_Y, p_y0, color='steelblue',  lw=2.2, label=r'$p(Y_{do0})$ — 50/50 at ±5')
ax_marg.fill_between(GRID_Y, p_y1, alpha=0.35, color='darkorange')
ax_marg.plot(GRID_Y, p_y1, color='darkorange', lw=2.2, label=r'$p(Y_{do1})$ — 50/50 at ±8')
for c in Y0_CENTRES: ax_marg.axvline(c, color='steelblue',  ls=':', lw=1.0, alpha=0.7)
for c in Y1_CENTRES: ax_marg.axvline(c, color='darkorange', ls=':', lw=1.0, alpha=0.7)
ax_marg.set_xlim(-15, 15)
ax_marg.set_xlabel('Y (scaled)')
ax_marg.set_ylabel('density')
ax_marg.set_title('Shared marginal potential-outcome densities '
                    r'(identical across all three joint couplings below)',
                    fontsize=12)
ax_marg.legend(fontsize=11, loc='upper right')
ax_marg.grid(alpha=0.3)


# Row 1: 2D joint per case
extent = [GRID_Y.min(), GRID_Y.max(), GRID_Y.min(), GRID_Y.max()]
for c, (label, joint_probs, tau_desc) in enumerate(CASES):
    ax = fig.add_subplot(gs[1, c])
    Z = kde_2d(joint_probs, GRID_Y, GRID_Y)
    im = ax.imshow(Z.T, origin='lower', extent=extent, cmap='viridis', aspect='auto')
    # Overlay the four possible joint-corner locations for reference
    for (y0, y1) in [(-5, -8), (-5, +8), (+5, -8), (+5, +8)]:
        active = any(pp[0] == (y0, y1) for pp in joint_probs)
        ax.plot(y0, y1, 'o', color=('white' if active else 'lightgray'),
                 markersize=(11 if active else 6),
                 markeredgecolor='black', markeredgewidth=1.2, zorder=5)
    # τ contour lines: τ = Y_do1 − Y_do0 = const → straight diagonals
    for tau_val in [-13, -3, +3, +13]:
        xs = np.linspace(-15, 15, 30)
        ax.plot(xs, xs + tau_val, ls='--', color='red', lw=0.7, alpha=0.4)
    ax.set_xlim(-15, 15); ax.set_ylim(-15, 15)
    ax.set_xlabel(r'$Y_{do0}$'); ax.set_ylabel(r'$Y_{do1}$')
    ax.set_title(f'{label}\n{tau_desc}', fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)


# Row 2: induced p(τ) per case
p_tau_by_case = []
y_max_tau = 0
for label, joint_probs, _ in CASES:
    pt = kde_tau(joint_probs, GRID_TAU)
    p_tau_by_case.append(pt)
    y_max_tau = max(y_max_tau, pt.max())

for c, ((label, joint_probs, tau_desc), pt) in enumerate(zip(CASES, p_tau_by_case)):
    ax = fig.add_subplot(gs[2, c])
    ax.fill_between(GRID_TAU, pt, alpha=0.35, color='#4B7BB1')
    ax.plot(GRID_TAU, pt, color='#2E4A6F', lw=2.0)
    # Reference lines for the four τ locations
    for tau_val in [-13, -3, +3, +13]:
        ax.axvline(tau_val, color='red', ls=':', lw=1.0, alpha=0.55)
        ax.text(tau_val, y_max_tau * 1.02, f'{tau_val:+d}',
                 ha='center', fontsize=8, color='red')
    # Predicted mean of τ (identical across cases because marginals agree!)
    mean_tau = float((GRID_TAU * pt).sum() * (GRID_TAU[1] - GRID_TAU[0]))
    ax.axvline(mean_tau, color='black', ls='--', lw=1.6,
                label=f'E[τ] = {mean_tau:+.2f}')
    ax.set_xlim(-20, 20); ax.set_ylim(0, y_max_tau * 1.15)
    ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$')
    if c == 0: ax.set_ylabel(r'$p(\tau)$')
    ax.set_title(tau_desc, fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)


fig.suptitle(
    r'Same marginals $\Rightarrow$  three completely different treatment-effect distributions.'
    '\nA marginal-only model (UWYK) cannot tell A, B, C apart; a joint model (Ours) can.',
    fontsize=13, y=0.998)

_HERE   = os.path.dirname(os.path.abspath(__file__))
_OUTDIR = os.path.join(_HERE, 'TE_demonstration')
os.makedirs(_OUTDIR, exist_ok=True)
out_path = os.path.join(_OUTDIR, 'same_marginals_three_joints.png')
fig.savefig(out_path, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'[save] {out_path}')

# Also print the mean τ for each case — highlight that they all agree
print('\nMean τ (=E[Y_do1] − E[Y_do0]) is identical across all three joints:')
for (label, jp, _), pt in zip(CASES, p_tau_by_case):
    m = float((GRID_TAU * pt).sum() * (GRID_TAU[1] - GRID_TAU[0]))
    print(f'  {label:<48}  E[τ] = {m:+.3f}')
print('\nAny model predicting only the mean CATE cannot distinguish these SCMs.')
