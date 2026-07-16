"""Derive the 1D CATE density f_i(d) from a 2D MALC_2D fit of the joint
density p_i(y_0, y_1) of the potential outcomes.

For each unit i the CATE is D = Y(1) - Y(0). After the change of variables
(Y_0, Y_1) ↦ (Y_0, D) the joint density of (Y_0, D) is p_i(y_0, y_0 + d),
and marginalising out y_0 gives

    f_i(d) = ∫ p_i(y_0, y_0 + d) dy_0.

Numerically: pick a fine 1D grid in y_0, evaluate `dmalc_2d` at the points
(y_0, y_0 + d) for each d on the output d-grid, sum with the trapezoid
rule.

The output d-grid is a *common* grid across all units (a requirement of the
OT pipeline downstream — quantile averaging needs the same support).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow imports of MALC_2D Python port without installing it
_MALC2D_DIR = Path(__file__).resolve().parent.parent / "MALC_2D" / "python"
if str(_MALC2D_DIR) not in sys.path:
    sys.path.insert(0, str(_MALC2D_DIR))

from malc_2d import MALC2DFit, dmalc_2d  # noqa: E402


def cate_density_from_malc2d(
    fit: MALC2DFit,
    d_grid: np.ndarray,
    n_y0: int = 200,
) -> np.ndarray:
    """Compute f_i(d) = ∫ p_i(y_0, y_0 + d) dy_0 on the given d_grid.

    The y_0 integration range spans the MALC_2D fit's `grid_x` range. The
    returned density is normalised to integrate to 1 on `d_grid`.

    Parameters
    ----------
    fit : MALC2DFit
        Fitted 2D joint from `MALC_2D`. Convention: x-axis is Y(0), y-axis
        is Y(1).
    d_grid : (M,) array
        1D grid of d = y_1 − y_0 values at which to evaluate f_i. Should be
        chosen wide enough to cover the support of `Y(1) - Y(0)` for every
        unit (common across units).
    n_y0 : int
        Number of integration points along the y_0 axis.

    Returns
    -------
    f_i : (M,) array
        Density on `d_grid`, summing to 1 / dd (so ∫ f_i dd = 1).
    """
    y0_min = float(fit.grid_x.min())
    y0_max = float(fit.grid_x.max())
    y0 = np.linspace(y0_min, y0_max, n_y0)
    dy0 = y0[1] - y0[0]

    # Build the (n_y0 × M) grid of points (y_0, y_0 + d) and evaluate p_i.
    M = len(d_grid)
    # broadcast: rows = d, cols = y_0
    D = np.broadcast_to(d_grid[:, None], (M, n_y0))
    Y0 = np.broadcast_to(y0[None, :], (M, n_y0))
    Y1 = Y0 + D
    pts = np.column_stack([Y0.ravel(), Y1.ravel()])
    dens = dmalc_2d(fit, pts).reshape(M, n_y0)

    # Integrate over y_0 by trapezoid rule
    f_d = 0.5 * (dens[:, 0] + dens[:, -1]) * dy0 + dens[:, 1:-1].sum(axis=1) * dy0
    f_d = np.maximum(f_d, 0.0)

    # Normalise on d_grid
    dd = d_grid[1] - d_grid[0]
    total = float(f_d.sum() * dd)
    if total > 0:
        f_d = f_d / total
    return f_d


def cate_pmat_from_density(f_d: np.ndarray, d_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a continuous CATE density `f_d` on `d_grid` (cell-centred or
    edge-aligned) into the (p_vec, grid) input format expected by `MALC_BM`.

    `d_grid` must be uniformly spaced. The returned `grid` has length
    len(d_grid) + 1 (bin breakpoints), and `p_vec` has length len(d_grid)
    (bin probabilities summing to 1).
    """
    if len(d_grid) < 2:
        raise ValueError("d_grid must have at least 2 points")
    dd = d_grid[1] - d_grid[0]
    grid = np.concatenate([[d_grid[0] - dd / 2], d_grid + dd / 2])
    p_vec = f_d * dd
    s = p_vec.sum()
    if s > 0:
        p_vec = p_vec / s
    return p_vec, grid
