"""Analytical (Gaussian-assumption) true densities for ACIC 2016.

ACIC 2016 ships per-unit (mu0, mu1, y0, y1) for the whole population and 10
realizations. Under a Gaussian noise assumption analogous to Hill 2011 for IHDP:

    Y_do0 | x ~ N(mu0(x), sigma^2)
    Y_do1 | x ~ N(mu1(x), sigma^2)
    tau  | x ~ N(mu1(x) - mu0(x), 2 sigma^2)

with sigma estimated per realization from combined arm residuals std(y_t - mu_t)
across all training units.

Rescales onto the model's scaled Y axis (Y ↦ 2 (Y − y_min) / y_rng − 1) via the
training-Y range, matching l2_ihdp/true_ihdp.py.

Truth is reconstructed by re-reading the raw ACIC CSVs (URLs identical to
CausalPFN's ACIC2016Dataset). The train/test split uses the same seed pattern
(seed + realization idx) so mu0_test / mu1_test line up with cd.true_cate from
ACIC2016Dataset()[realization].
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── Common grids (same as l2_ihdp) ────────────────────────────────────────
Y_EDGES = np.linspace(-1.5, 1.5, 101)
Y_CENTERS = 0.5 * (Y_EDGES[:-1] + Y_EDGES[1:])
Y_BIN = float(Y_CENTERS[1] - Y_CENTERS[0])

TAU_EDGES = np.linspace(-3.0, 3.0, 601)
TAU_CENTERS = 0.5 * (TAU_EDGES[:-1] + TAU_EDGES[1:])
TAU_BIN = float(TAU_CENTERS[1] - TAU_CENTERS[0])


X_CSV_URL = ('https://raw.githubusercontent.com/BiomedSciAI/causallib/master/'
             'causallib/datasets/data/acic_challenge_2016/x.csv')
ZY_CSV_URL = (lambda i: f'https://raw.githubusercontent.com/BiomedSciAI/causallib/'
                        f'master/causallib/datasets/data/acic_challenge_2016/zymu_{i}.csv')


@dataclass
class ACICRealizationTruth:
    r: int
    mu0_test_scaled: np.ndarray
    mu1_test_scaled: np.ndarray
    sigma_scaled: float
    y_min: float
    y_rng: float


def _gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def _load_x_frame(cache_dir: str | None = None) -> pd.DataFrame:
    """Load ACIC x.csv from a local cache (or download once)."""
    if cache_dir:
        p = os.path.join(cache_dir, 'x.csv')
        if os.path.isfile(p):
            return pd.read_csv(p)
    return pd.read_csv(X_CSV_URL)


def _load_zy_frame(idx: int, cache_dir: str | None = None) -> pd.DataFrame:
    if cache_dir:
        p = os.path.join(cache_dir, f'zymu_{idx + 1}.csv')
        if os.path.isfile(p):
            return pd.read_csv(p)
    return pd.read_csv(ZY_CSV_URL(idx + 1))


def load_acic_truth(r: int,
                    y_train_full: np.ndarray,
                    seed: int = 42,
                    test_ratio: float = 0.1,
                    cache_dir: str | None = None) -> ACICRealizationTruth:
    """Load mu0_test, mu1_test, sigma for realization r (0..9) on scaled Y axis.

    `y_train_full` fixes the y_min / y_rng rescaling used by the model.
    seed / test_ratio must match ACIC2016Dataset defaults (seed=42, 0.1).
    """
    x_frame = _load_x_frame(cache_dir)
    sim = _load_zy_frame(r, cache_dir)
    sim.columns = ['z', 'y0', 'y1', 'mu0', 'mu1']

    z  = sim['z'].values.astype(np.float32)
    y0 = sim['y0'].values.astype(np.float32)
    y1 = sim['y1'].values.astype(np.float32)
    mu0 = sim['mu0'].values.astype(np.float32)
    mu1 = sim['mu1'].values.astype(np.float32)
    y_factual = np.where(z == 1, y1, y0)

    n = x_frame.shape[0]
    rng = np.random.default_rng(seed + r)
    perm = rng.permutation(n)
    split_idx = int(n * (1 - test_ratio))
    train_idx = perm[:split_idx]
    test_idx  = perm[split_idx:]

    mu_factual_train = np.where(z[train_idx] == 1, mu1[train_idx], mu0[train_idx])
    residuals = y_factual[train_idx] - mu_factual_train
    sigma_orig = float(np.std(residuals, ddof=1))

    mu0_test = mu0[test_idx]
    mu1_test = mu1[test_idx]

    y_min = float(y_train_full.min())
    y_max = float(y_train_full.max())
    y_rng = max(y_max - y_min, 1e-6)

    mu0_scaled = (mu0_test - y_min) / y_rng * 2.0 - 1.0
    mu1_scaled = (mu1_test - y_min) / y_rng * 2.0 - 1.0
    sigma_scaled = sigma_orig * (2.0 / y_rng)

    return ACICRealizationTruth(
        r=r,
        mu0_test_scaled=mu0_scaled,
        mu1_test_scaled=mu1_scaled,
        sigma_scaled=sigma_scaled,
        y_min=y_min,
        y_rng=y_rng,
    )


def true_marginals_per_query(truth: ACICRealizationTruth):
    n = truth.mu0_test_scaled.shape[0]
    p_y0 = np.stack([_gauss(Y_CENTERS, truth.mu0_test_scaled[i], truth.sigma_scaled)
                     for i in range(n)])
    p_y1 = np.stack([_gauss(Y_CENTERS, truth.mu1_test_scaled[i], truth.sigma_scaled)
                     for i in range(n)])
    return p_y0, p_y1


def true_cate_per_query(truth: ACICRealizationTruth) -> np.ndarray:
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
