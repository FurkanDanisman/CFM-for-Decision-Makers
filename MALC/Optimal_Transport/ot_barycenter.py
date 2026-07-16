"""1D 2-Wasserstein barycenter via quantile averaging — closed form.

Given N densities f_1, …, f_N on a common grid with weights α_1, …, α_N
(sum to 1), the 2-Wasserstein barycenter has quantile function

    Q_bary(τ) = Σ_i α_i Q_i(τ),    τ ∈ (0, 1),

where Q_i = F_i^{-1} is the quantile function of f_i. The barycenter density
is the pushforward of Uniform(0, 1) by Q_bary; we recover it on the original
grid by inverting Q_bary and differentiating.

Also exposed: `linear_mixture` for comparison.
"""

from __future__ import annotations

import numpy as np


def _density_to_quantile(f: np.ndarray, x_grid: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Quantile function Q(τ) of a 1D density `f` on `x_grid`.

    Trapezoid-cumulative CDF, then `np.interp` inverse. F has the same length
    as x_grid (F[k] = cumulative integral of f from x_grid[0] to x_grid[k]).
    """
    dx = x_grid[1] - x_grid[0]
    F = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * dx)])
    if F[-1] <= 0:
        return np.full_like(taus, float(np.mean(x_grid)))
    F = F / F[-1]
    return np.interp(taus, F, x_grid)


def wasserstein_barycenter_1d(
    densities: np.ndarray,
    x_grid: np.ndarray,
    weights: np.ndarray | None = None,
    n_tau: int = 4001,
) -> np.ndarray:
    """1D 2-Wasserstein barycenter on a common x_grid.

    Parameters
    ----------
    densities : (N, M) array
        N densities each on the same `x_grid` of M points.
    x_grid : (M,) array
        Common 1D grid (uniformly spaced).
    weights : (N,) array or None
        Barycenter weights (sum to 1). Default uniform 1/N.
    n_tau : int
        Number of probability levels τ at which to sample the quantile
        functions. Higher → smoother density estimate.

    Returns
    -------
    bary : (M,) array
        Barycenter density on `x_grid`, normalised so ∫ bary dx = 1.
    """
    densities = np.atleast_2d(np.asarray(densities, dtype=float))
    N, M = densities.shape
    if M != len(x_grid):
        raise ValueError("densities and x_grid must have matching lengths")

    if weights is None:
        weights = np.full(N, 1.0 / N)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

    taus = (np.arange(n_tau) + 0.5) / n_tau

    # Per-density quantile functions
    Q = np.zeros((N, n_tau))
    for i in range(N):
        Q[i] = _density_to_quantile(densities[i], x_grid, taus)

    Q_bary = (weights[:, None] * Q).sum(axis=0)  # (n_tau,)

    # Recover barycenter density on x_grid by inverting Q_bary and differencing.
    # F_bary(x) ≈ τ(x)  via np.interp on (Q_bary → taus).
    # We need Q_bary monotone-increasing; it is by construction.
    F_bary = np.interp(x_grid, Q_bary, taus, left=0.0, right=1.0)
    dx = x_grid[1] - x_grid[0]
    f_bary = np.gradient(F_bary, dx)
    f_bary = np.clip(f_bary, 0.0, None)
    s = float(f_bary.sum() * dx)
    if s > 0:
        f_bary = f_bary / s
    return f_bary


def linear_mixture(
    densities: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Plain linear (mixture) average of densities for comparison."""
    densities = np.atleast_2d(np.asarray(densities, dtype=float))
    N = densities.shape[0]
    if weights is None:
        weights = np.full(N, 1.0 / N)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
    return (weights[:, None] * densities).sum(axis=0)


def l2_distance(f: np.ndarray, g: np.ndarray, x_grid: np.ndarray) -> float:
    """L²(f − g) on x_grid."""
    dx = x_grid[1] - x_grid[0]
    return float(np.sqrt(((f - g) ** 2 * dx).sum()))


def w2_distance(f: np.ndarray, g: np.ndarray, x_grid: np.ndarray, n_tau: int = 4001) -> float:
    """2-Wasserstein distance via quantile L²."""
    taus = (np.arange(n_tau) + 0.5) / n_tau
    Qf = _density_to_quantile(f, x_grid, taus)
    Qg = _density_to_quantile(g, x_grid, taus)
    return float(np.sqrt(((Qf - Qg) ** 2).mean()))
