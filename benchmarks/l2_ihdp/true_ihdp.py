"""Analytical true densities for IHDP (Hill 2011).

Under the shipped IHDP DGP the per-arm potential outcomes are
    Y_do0 | x ~ N(mu0(x), sigma^2)
    Y_do1 | x ~ N(mu1(x), sigma^2)
with sigma shared across arms and noise draws independent across arms.
The induced per-unit CATE is therefore
    tau | x ~ N(mu1(x) - mu0(x), 2 sigma^2).

mu0, mu1 per unit are shipped in `ihdp_npci_1-100.{train,test}.npz`. sigma is
not shipped; we estimate it per realization from training-side factual
residuals std(y_train - mu_factual_train). See
benchmarks/plots/plot_ihdp_n10_true.py for the same construction.

The population ATE density is defined as the 2-Wasserstein barycenter of the
per-query true CATE densities with uniform weights, matching the aggregation
Section 2.2.2 of the draft.

Everything is expressed on the model's rescaled Y axis (Y ↦ 2 (Y − y_min) /
y_rng − 1) so it can be compared to method outputs directly.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np


# ── Common grids used by the sibling scripts ─────────────────────────────
Y_EDGES = np.linspace(-1.5, 1.5, 101)
Y_CENTERS = 0.5 * (Y_EDGES[:-1] + Y_EDGES[1:])
Y_BIN = float(Y_CENTERS[1] - Y_CENTERS[0])

TAU_EDGES = np.linspace(-3.0, 3.0, 601)
TAU_CENTERS = 0.5 * (TAU_EDGES[:-1] + TAU_EDGES[1:])
TAU_BIN = float(TAU_CENTERS[1] - TAU_CENTERS[0])


@dataclass
class IHDPRealizationTruth:
    """Everything derived from the raw NPZ + training residuals for one r."""

    r: int
    mu0_test_scaled: np.ndarray            # (n_test,)
    mu1_test_scaled: np.ndarray            # (n_test,)
    sigma_scaled: float
    y_min: float
    y_rng: float


def _gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def load_ihdp_truth(r: int, causalpfn_dir: str, y_train_full: np.ndarray) -> IHDPRealizationTruth:
    """Load μ0, μ1, σ for realization r on the rescaled Y axis.

    `y_train_full` is the full training-Y for realization r (raw units); it
    fixes the rescaling y ↦ 2 (y − y_min) / y_rng − 1 that the model uses.
    """
    ihdp_dir = os.path.join(causalpfn_dir, 'benchmarks', 'IHDP')
    if not os.path.isdir(ihdp_dir):
        ihdp_dir = os.path.join(causalpfn_dir, 'IHDP')
    if not os.path.isdir(ihdp_dir):
        raise FileNotFoundError(
            f'IHDP NPZ directory not found at {causalpfn_dir}/benchmarks/IHDP '
            f'nor {causalpfn_dir}/IHDP')

    train_npz = np.load(os.path.join(ihdp_dir, 'ihdp_npci_1-100.train.npz'))
    test_npz = np.load(os.path.join(ihdp_dir, 'ihdp_npci_1-100.test.npz'))

    mu0_test = test_npz['mu0'][..., r].astype(np.float32).reshape(-1)
    mu1_test = test_npz['mu1'][..., r].astype(np.float32).reshape(-1)

    yf_train = train_npz['yf'][..., r].astype(np.float32).reshape(-1)
    t_train = train_npz['t'][..., r].astype(np.float32).reshape(-1)
    mu0_train = train_npz['mu0'][..., r].astype(np.float32).reshape(-1)
    mu1_train = train_npz['mu1'][..., r].astype(np.float32).reshape(-1)
    mu_factual = np.where(t_train > 0.5, mu1_train, mu0_train)
    sigma_orig = float(np.std(yf_train - mu_factual, ddof=1))

    y_min = float(y_train_full.min())
    y_max = float(y_train_full.max())
    y_rng = max(y_max - y_min, 1e-6)

    mu0_scaled = (mu0_test - y_min) / y_rng * 2.0 - 1.0
    mu1_scaled = (mu1_test - y_min) / y_rng * 2.0 - 1.0
    sigma_scaled = sigma_orig * (2.0 / y_rng)

    return IHDPRealizationTruth(
        r=r,
        mu0_test_scaled=mu0_scaled,
        mu1_test_scaled=mu1_scaled,
        sigma_scaled=sigma_scaled,
        y_min=y_min,
        y_rng=y_rng,
    )


def true_marginals_per_query(truth: IHDPRealizationTruth) -> tuple[np.ndarray, np.ndarray]:
    """Returns (p_y0_true, p_y1_true), each (n_test, len(Y_CENTERS))."""
    n = truth.mu0_test_scaled.shape[0]
    p_y0 = np.stack([_gauss(Y_CENTERS, truth.mu0_test_scaled[i], truth.sigma_scaled)
                     for i in range(n)])
    p_y1 = np.stack([_gauss(Y_CENTERS, truth.mu1_test_scaled[i], truth.sigma_scaled)
                     for i in range(n)])
    return p_y0, p_y1


def true_cate_per_query(truth: IHDPRealizationTruth) -> np.ndarray:
    """Returns p_tau_true, shape (n_test, len(TAU_CENTERS))."""
    tau_mean = truth.mu1_test_scaled - truth.mu0_test_scaled       # (n_test,)
    tau_sigma = np.sqrt(2.0) * truth.sigma_scaled
    return np.stack([_gauss(TAU_CENTERS, tau_mean[i], tau_sigma)
                     for i in range(tau_mean.shape[0])])


def true_ate_barycenter(p_taus_true: np.ndarray, bary_fn) -> np.ndarray:
    """Population ATE = 2-Wasserstein barycenter of per-query true CATEs.

    `bary_fn` is the wasserstein_barycenter_1d imported from
    MALC/Optimal_Transport/ot_barycenter.py — passed in to keep this module
    dependency-light. Returns a density on TAU_CENTERS normalised to
    integrate to 1.
    """
    bary = bary_fn(p_taus_true, TAU_CENTERS)
    s = bary.sum() * TAU_BIN
    if s > 0:
        bary = bary / s
    return bary
