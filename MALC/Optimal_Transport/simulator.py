"""Ground-truth data-generating process for the OT-CATE-ATE pipeline.

K causal worlds, each a canonical Gaussian shape `g_k = N(m_k, s_k²)`.
For each unit i:
    - draw mixing weights π_{i,k} from a Dirichlet so they sum to 1
    - draw small per-unit shifts δ_{i,k} ~ N(0, shift_sd²) for each world
    - per-unit CATE density:  f_i(d) = Σ_k π_{i,k} · N(m_k + δ_{i,k}, s_k²)(d)

The "true" ATE distribution we want our pipeline to recover is

    F_true(d) = Σ_k w_k · N(m_k + δ̄_k, s_k²)(d)
    where w_k = E[π_{i,k}], δ̄_k = E[δ_{i,k}].

In expectation (over the DGP) this is the OT-barycenter answer; the linear
mixture F_lin = (1/N) Σ_i f_i is wider because it convolves in the shifts
δ_{i,k}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class TrueDGP:
    K: int
    m: np.ndarray         # (K,) world centres
    s: np.ndarray         # (K,) world sds
    w: np.ndarray         # (K,) world weights (population means of π)
    shift_sd: float       # sd of per-unit shifts δ_{i,k}
    alpha_dirichlet: float  # concentration of Dirichlet(α·w · K) on π_i


def make_dgp_K2(asym: float = 0.6, sd: float = 0.6, shift_sd: float = 0.3) -> TrueDGP:
    """Two-world example. `asym` is the world-2 weight (so K-1 weights are
    `(1 - asym, asym)`)."""
    return TrueDGP(
        K=2,
        m=np.array([-2.0, 2.0]),
        s=np.array([sd, sd]),
        w=np.array([1 - asym, asym]),
        shift_sd=shift_sd,
        alpha_dirichlet=20.0,
    )


def make_dgp_K3(sd: float = 0.6, shift_sd: float = 0.3) -> TrueDGP:
    """Three-world example, roughly equal weights."""
    return TrueDGP(
        K=3,
        m=np.array([-3.0, 0.0, 3.0]),
        s=np.array([sd, sd, sd]),
        w=np.array([0.3, 0.4, 0.3]),
        shift_sd=shift_sd,
        alpha_dirichlet=20.0,
    )


def sample_dgp(
    dgp: TrueDGP,
    N: int,
    d_grid: np.ndarray,
    seed: int = 20180621,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw N units. Returns:
        f_per_unit : (N, M)  unit CATE densities on `d_grid`
        pi_true    : (N, K)  per-unit mixing weights
        delta_true : (N, K)  per-unit shifts
    """
    rng = np.random.default_rng(seed)
    K = dgp.K
    M = len(d_grid)

    # Per-unit mixing weights via Dirichlet centred at dgp.w
    alphas = dgp.alpha_dirichlet * dgp.w * K  # so sum(alphas) controls concentration
    alphas = np.maximum(alphas, 0.5)  # avoid degenerate alphas
    pi_true = rng.dirichlet(alphas, size=N)

    # Per-unit shifts
    delta_true = rng.normal(0.0, dgp.shift_sd, size=(N, K))

    f_per_unit = np.zeros((N, M))
    for i in range(N):
        for k in range(K):
            f_per_unit[i] += pi_true[i, k] * norm.pdf(d_grid, dgp.m[k] + delta_true[i, k], dgp.s[k])

    # Normalise each to integrate to 1 on the grid (correct tiny numerical drift)
    dd = d_grid[1] - d_grid[0]
    f_per_unit = f_per_unit / (f_per_unit.sum(axis=1, keepdims=True) * dd)

    return f_per_unit, pi_true, delta_true


def true_ate_distribution(
    dgp: TrueDGP,
    d_grid: np.ndarray,
    pi_true: np.ndarray | None = None,
    delta_true: np.ndarray | None = None,
) -> np.ndarray:
    """Return F_true on d_grid.

    If `pi_true` and `delta_true` are provided (i.e. the empirical realisation
    in the sampled units), use the *empirical* world weights `w_k = mean(π_{i,k})`
    and the empirical mean shift `δ̄_k = mean(δ_{i,k})`. Otherwise use the
    population means (`dgp.w`, zero shift).
    """
    K = dgp.K
    if pi_true is not None:
        w = pi_true.mean(axis=0)
    else:
        w = dgp.w.copy()
    w = w / w.sum()

    if delta_true is not None:
        delta_bar = delta_true.mean(axis=0)
    else:
        delta_bar = np.zeros(K)

    f = np.zeros_like(d_grid)
    for k in range(K):
        f += w[k] * norm.pdf(d_grid, dgp.m[k] + delta_bar[k], dgp.s[k])
    dd = d_grid[1] - d_grid[0]
    f = f / (f.sum() * dd)
    return f
