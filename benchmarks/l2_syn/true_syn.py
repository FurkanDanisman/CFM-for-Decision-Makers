"""Closed-form Gaussian truth for the linear-Gaussian SCM in syn_dgp.py.

Same 100-bin Y grid and 600-bin τ grid as l2_ihdp — so methods_densities.py
works unchanged. Y-axis rescaling matches the model's convention: rescale
raw Y by y_min / y_rng estimated from the training Y range of one realization.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from syn_dgp import SynTruth


Y_EDGES = np.linspace(-1.5, 1.5, 101)
Y_CENTERS = 0.5 * (Y_EDGES[:-1] + Y_EDGES[1:])
Y_BIN = float(Y_CENTERS[1] - Y_CENTERS[0])

TAU_EDGES = np.linspace(-3.0, 3.0, 601)
TAU_CENTERS = 0.5 * (TAU_EDGES[:-1] + TAU_EDGES[1:])
TAU_BIN = float(TAU_CENTERS[1] - TAU_CENTERS[0])


@dataclass
class SynRealizationTruth:
    seed: int
    mu0_test_scaled: np.ndarray
    mu1_test_scaled: np.ndarray
    sigma_scaled: float
    y_min: float
    y_rng: float


def _gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def build_syn_truth(truth_raw: SynTruth, y_train_full: np.ndarray) -> SynRealizationTruth:
    """Compute per-query mu0, mu1, sigma on the scaled Y axis."""
    X_test = truth_raw.X_test
    mu0 = truth_raw.alpha + X_test @ truth_raw.gamma            # (n_test,)
    mu1 = truth_raw.alpha + truth_raw.beta + X_test @ truth_raw.gamma \
                          + truth_raw.delta * X_test[:, 0]
    sigma_orig = truth_raw.sigma_y

    y_min = float(y_train_full.min())
    y_max = float(y_train_full.max())
    y_rng = max(y_max - y_min, 1e-6)

    mu0_s = (mu0 - y_min) / y_rng * 2.0 - 1.0
    mu1_s = (mu1 - y_min) / y_rng * 2.0 - 1.0
    sigma_s = sigma_orig * (2.0 / y_rng)

    return SynRealizationTruth(
        seed=truth_raw.seed,
        mu0_test_scaled=mu0_s.astype(np.float32),
        mu1_test_scaled=mu1_s.astype(np.float32),
        sigma_scaled=float(sigma_s),
        y_min=y_min, y_rng=y_rng,
    )


def true_marginals_per_query(truth: SynRealizationTruth):
    n = truth.mu0_test_scaled.shape[0]
    p_y0 = np.stack([_gauss(Y_CENTERS, truth.mu0_test_scaled[i], truth.sigma_scaled)
                     for i in range(n)])
    p_y1 = np.stack([_gauss(Y_CENTERS, truth.mu1_test_scaled[i], truth.sigma_scaled)
                     for i in range(n)])
    return p_y0, p_y1


def true_cate_per_query(truth: SynRealizationTruth) -> np.ndarray:
    tau_mean = truth.mu1_test_scaled - truth.mu0_test_scaled
    tau_sigma = np.sqrt(2.0) * truth.sigma_scaled
    return np.stack([_gauss(TAU_CENTERS, tau_mean[i], tau_sigma)
                     for i in range(tau_mean.shape[0])])


def true_ate_barycenter(p_taus_true: np.ndarray, bary_fn) -> np.ndarray:
    bary = bary_fn(p_taus_true, TAU_CENTERS)
    s = bary.sum() * TAU_BIN
    if s > 0:
        bary = bary / s
    return bary
