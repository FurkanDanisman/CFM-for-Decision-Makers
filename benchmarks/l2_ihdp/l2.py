"""L2 distance between two 1D densities on a common grid.

Trapezoid rule on the shared support:
    ||f - g||_2 = sqrt( integral (f(x) - g(x))^2 dx )

Assumes both `f` and `g` are already sampled on the same uniform `grid`.
For methods that emit densities on a different support (Do-PFN / UWYK use
their own bar-grid or empirical histogram range), interpolate first with
`resample_onto`.
"""
from __future__ import annotations

import numpy as np


def l2_distance(f: np.ndarray, g: np.ndarray, grid: np.ndarray) -> float:
    """Trapezoid L2 distance between two densities on the same 1D grid."""
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    grid = np.asarray(grid, dtype=np.float64).reshape(-1)
    if f.shape != g.shape or f.shape[0] != grid.shape[0]:
        raise ValueError(
            f'shape mismatch: f={f.shape}, g={g.shape}, grid={grid.shape}')
    diff2 = (f - g) ** 2
    return float(np.sqrt(np.trapezoid(diff2, grid)))


def resample_onto(src_grid: np.ndarray, src_density: np.ndarray,
                  dst_grid: np.ndarray) -> np.ndarray:
    """Linear-interpolate a density onto a different uniform grid.

    Zeros outside the source support. Re-normalises to integrate to 1 on the
    destination grid (matches how sibling scripts treat empirical densities
    when the src support is narrower than dst).
    """
    src_grid = np.asarray(src_grid, dtype=np.float64).reshape(-1)
    src_density = np.asarray(src_density, dtype=np.float64).reshape(-1)
    dst_grid = np.asarray(dst_grid, dtype=np.float64).reshape(-1)
    out = np.interp(dst_grid, src_grid, src_density, left=0.0, right=0.0)
    dx = float(dst_grid[1] - dst_grid[0])
    total = float(out.sum() * dx)
    if total > 0:
        out = out / total
    return out
