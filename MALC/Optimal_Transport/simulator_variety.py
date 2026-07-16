"""Variety-of-distributions simulator for the OT-CATE-ATE pipeline.

Goes beyond Gaussian mixtures. Each causal world `k` is specified by a
location-shiftable 1D density family. Standard families included:

    gaussian          N(0, σ²)
    beta              shifted/scaled Beta(α, β)
    gamma             shifted Gamma(shape, scale)
    skewnorm          skew-normal (asymmetric, log-concave for moderate a)
    t                 Student-t (heavy-tailed — VIOLATES log-concavity for low df)
    laplace           Laplace (log-linear, log-concave but non-smooth)
    truncnorm         truncated Gaussian (bounded support)

Per-unit CATE:
    f_i(d)  =  Σ_k  π_{i,k} · world_k(d − (m_k + δ_{i,k}))

Population ATE:
    F_true(d) = Σ_k w_k · world_k(d − (m_k + δ̄_k))

where δ̄_k is the empirical mean shift in the sample (so F_true is the right
target for THIS particular draw), and w_k = mean_i π_{i,k}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import beta as sbeta
from scipy.stats import gamma as sgamma
from scipy.stats import laplace as slaplace
from scipy.stats import norm as snorm
from scipy.stats import skewnorm as sskewnorm
from scipy.stats import t as st_t
from scipy.stats import truncnorm as strunc

WorldPDF = Callable[[np.ndarray, float], np.ndarray]


def gaussian(sd: float = 0.6) -> WorldPDF:
    return lambda d, loc: snorm.pdf(d, loc=loc, scale=sd)


def beta_scaled(a: float, b: float, scale: float = 2.0) -> WorldPDF:
    """Beta(a, b) on [loc, loc+scale]. The 'loc' parameter is the LEFT edge,
    so the density support is [loc, loc + scale]. We re-centre so the world
    mode sits at `loc` for ease of comparison with other families:
    centre = a/(a+b)·scale; we pass `loc_arg = loc - centre`."""
    centre = a / (a + b) * scale
    return lambda d, loc: sbeta.pdf(d, a, b, loc=loc - centre, scale=scale)


def gamma_shifted(shape: float, scale: float = 1.0) -> WorldPDF:
    """Gamma(shape, scale), centred at its mode = (shape − 1)·scale.
    'loc' becomes the mode location."""
    mode = (shape - 1) * scale
    return lambda d, loc: sgamma.pdf(d, shape, loc=loc - mode, scale=scale)


def skewnormal(a: float = 5.0, sd: float = 0.6) -> WorldPDF:
    """Skew-normal with shape `a`, scale `sd`. Larger |a| → more asymmetric.
    We centre by subtracting the analytic mean of skewnorm so that 'loc'
    is the distribution's mean."""
    delta = a / np.sqrt(1 + a * a)
    mean_z = delta * np.sqrt(2 / np.pi)
    centre = sd * mean_z
    return lambda d, loc: sskewnorm.pdf(d, a, loc=loc - centre, scale=sd)


def student_t(df: float = 3.0, sd: float = 0.6) -> WorldPDF:
    """Student-t (heavy-tailed). For df ≤ 2 NOT log-concave."""
    return lambda d, loc: st_t.pdf(d, df, loc=loc, scale=sd)


def laplace(sd: float = 0.6) -> WorldPDF:
    """Laplace centred at loc. Log-concave (log-density piecewise linear)."""
    return lambda d, loc: slaplace.pdf(d, loc=loc, scale=sd)


def truncated_gaussian(sd: float = 0.6, half_width: float = 1.0) -> WorldPDF:
    """N(loc, sd²) truncated to [loc - half_width, loc + half_width]."""
    a, b = -half_width / sd, half_width / sd
    return lambda d, loc: strunc.pdf(d, a, b, loc=loc, scale=sd)


@dataclass
class DGPVariety:
    K: int
    worlds: list[WorldPDF]
    m: np.ndarray           # (K,) base locations for each world
    w: np.ndarray           # (K,) world weights
    shift_sd: float         # sd of per-unit shifts δ
    alpha_dirichlet: float = 20.0


def sample_variety(dgp: DGPVariety, N: int, d_grid: np.ndarray, seed: int = 20180621):
    """Sample N CATE densities. Returns (f_per_unit, pi_true, delta_true)."""
    rng = np.random.default_rng(seed)
    K, M = dgp.K, len(d_grid)

    alphas = np.maximum(dgp.alpha_dirichlet * dgp.w * K, 0.5)
    pi_true = rng.dirichlet(alphas, size=N)
    delta_true = rng.normal(0.0, dgp.shift_sd, size=(N, K))

    f = np.zeros((N, M))
    for i in range(N):
        for k in range(K):
            f[i] += pi_true[i, k] * dgp.worlds[k](d_grid, dgp.m[k] + delta_true[i, k])
    dd = d_grid[1] - d_grid[0]
    f = f / np.maximum(f.sum(axis=1, keepdims=True) * dd, 1e-12)
    return f, pi_true, delta_true


def true_ate(dgp: DGPVariety, d_grid: np.ndarray,
             pi_true: np.ndarray | None = None,
             delta_true: np.ndarray | None = None) -> np.ndarray:
    """F_true = Σ_k w_k · world_k(d − (m_k + δ̄_k))."""
    K = dgp.K
    if pi_true is None:
        w = dgp.w.copy()
        delta_bar = np.zeros(K)
    else:
        w = pi_true.mean(axis=0)
        w = w / w.sum()
        delta_bar = delta_true.mean(axis=0) if delta_true is not None else np.zeros(K)

    f = np.zeros_like(d_grid, dtype=float)
    for k in range(K):
        f += w[k] * dgp.worlds[k](d_grid, dgp.m[k] + delta_bar[k])
    dd = d_grid[1] - d_grid[0]
    f = f / max(float(f.sum() * dd), 1e-12)
    return f
