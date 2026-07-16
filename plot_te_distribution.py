"""
Compute the treatment effect distribution p(τ) from MALC's joint density
using diagonal integration (not sampling):

    p(τ) = ∫ f(y0, y0 + τ) dy0

Change of variables (u = y0, v = y1 - y0), Jacobian 1. On the regular MALC
grid this is a sum along the diagonal offset by τ, weighted by dy0.

Plots p(τ) with the true treatment effect marked as a red point.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.environ.get('OUT_DIR', 'eval_one_point')

# ── Load MALC output + ground truth ──────────────────────────────────────────
d       = np.load(f'{OUT_DIR}/malc_density.npy')       # (Ny, Nx)  f(y1, y0)
gx      = np.load(f'{OUT_DIR}/malc_grid_x.npy')        # (Nx,)     y0 axis
gy      = np.load(f'{OUT_DIR}/malc_grid_y.npy')        # (Ny,)     y1 axis
true_Y0 = float(np.load(f'{OUT_DIR}/true_Y_do0.npy'))
true_Y1 = float(np.load(f'{OUT_DIR}/true_Y_do1.npy'))
true_TE = float(np.load(f'{OUT_DIR}/true_TE.npy'))

# Grid must be uniform (it is — from linspace)
dy0 = gx[1] - gx[0]
dy1 = gy[1] - gy[0]
assert np.allclose(dy0, dy1), "Non-square grid: TE derivation assumes matched spacing."

# ── p(τ) via diagonal integration ─────────────────────────────────────────────
# For each candidate τ:
#   1. For every y0 in the grid, find the row index in gy closest to y0 + τ.
#   2. Sum d[row, col] over valid (y0, y0+τ) pairs, * dy0.
# This is exact on the resolution of the grid; for finer τ we linearly
# interpolate the joint density along the diagonal.

tau_min = gy[0] - gx[-1]   # y1_min - y0_max
tau_max = gy[-1] - gx[0]   # y1_max - y0_min
n_tau   = 401              # τ grid resolution (2× finer than density grid)
tau     = np.linspace(tau_min, tau_max, n_tau)

p_tau = np.zeros_like(tau)
for k, t in enumerate(tau):
    # For every y0 in gx, we want f(y0, y0 + t).
    y1_target = gx + t                                  # (Nx,)
    valid = (y1_target >= gy[0]) & (y1_target <= gy[-1])
    if not np.any(valid):
        continue
    y0_v = gx[valid]
    y1_v = y1_target[valid]
    # Bilinear interpolation along y1 axis (y0 lands exactly on a column).
    col_idx = np.searchsorted(gx, y0_v) - 1
    col_idx = np.clip(col_idx, 0, len(gx) - 1)
    row_f   = (y1_v - gy[0]) / dy1
    row_lo  = np.clip(np.floor(row_f).astype(int), 0, len(gy) - 2)
    row_hi  = row_lo + 1
    w_hi    = row_f - row_lo
    w_lo    = 1.0 - w_hi
    f_diag  = w_lo * d[row_lo, col_idx] + w_hi * d[row_hi, col_idx]
    p_tau[k] = f_diag.sum() * dy0

# Numerical sanity: p(τ) should integrate to ~1 (up to the MALC density's
# own quadrature error).
dtau = tau[1] - tau[0]
mass = float(p_tau.sum() * dtau)
E_tau = float((tau * p_tau).sum() * dtau)
mode_tau = float(tau[p_tau.argmax()])

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(tau, p_tau, color='steelblue', lw=2, label=r'$p(\tau)$ from MALC')
ax.fill_between(tau, 0, p_tau, color='steelblue', alpha=0.15)

# True TE marker
p_true = float(np.interp(true_TE, tau, p_tau))
ax.plot(true_TE, p_true, 'o', color='red', markersize=10, zorder=5,
        label=f'True τ = {true_TE:+.3f}')
ax.axvline(true_TE, color='red', ls='--', lw=1, alpha=0.5)

# Predicted mean marker
ax.axvline(E_tau, color='navy', ls=':', lw=1, alpha=0.7,
           label=fr'$\mathbb{{E}}[\tau]$ = {E_tau:+.3f}')

ax.set_xlabel(r'Treatment effect  $\tau = Y_{do(1)} - Y_{do(0)}$')
ax.set_ylabel(r'Density  $p(\tau)$')
ax.set_title(
    fr'MALC-derived TE distribution   '
    fr'(mode={mode_tau:+.3f}, $\mathbb{{E}}[\tau]$={E_tau:+.3f}, ∫p={mass:.4f})'
)
ax.legend(loc='best')
ax.grid(alpha=0.3)
fig.tight_layout()

out_png = f'{OUT_DIR}/te_distribution.png'
fig.savefig(out_png, dpi=140)
print(f"Saved plot: {out_png}")

# Save the TE distribution arrays too
np.save(f'{OUT_DIR}/te_grid.npy',    tau)
np.save(f'{OUT_DIR}/te_density.npy', p_tau)
print(f"Saved arrays: {OUT_DIR}/te_grid.npy, {OUT_DIR}/te_density.npy")
print(f"\nSummary:")
print(f"  True TE      : {true_TE:+.4f}")
print(f"  Predicted E[τ]: {E_tau:+.4f}")
print(f"  Mode of p(τ) : {mode_tau:+.4f}")
print(f"  ∫ p(τ) dτ   : {mass:.4f}")
print(f"  p(τ = true)  : {p_true:.4f}")
