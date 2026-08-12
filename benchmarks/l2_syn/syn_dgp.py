"""Linear-Gaussian synthetic SCM with closed-form true per-query densities.

DGP:
    X   ∼ N(0, I_d)
    T   ∼ Bernoulli(σ(w · X))
    Y   = α + β·T + γ·X + δ·(T · X_1) + ε,   ε ∼ N(0, σ_y²)

Per query x:
    Y_do(t) | x ∼ N(μ_t(x), σ_y²)     with μ_t(x) = α + β·t + γ·x + δ·t·x_1
    τ | x     ∼ N(β + δ·x_1, 2 σ_y²)   (independent noise across arms)

Population marginals (integrating over X ∼ N(0, I_d)):
    Y_do(t) ∼ N(α + β·t, γᵀγ + σ_y²)  when t is fixed and δ=0
    (with δ ≠ 0 the marginal is still Gaussian:
     Y_do(t) ∼ N(α + β·t + δ·t·0, ‖γ + t δ e_1‖² + σ_y²))

Everything closed-form; no Monte Carlo. Reasonable defaults chosen so the
signal-to-noise ratio matches IHDP-ish scale.

Returns a `SynDataset` object with attributes matching CausalPFN's
CATE_Dataset (X_train, t_train, y_train, X_test, true_cate) so downstream
methods_densities.py works unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_D = 5
DEFAULT_N_TRAIN = 500
DEFAULT_N_TEST = 200

# Fixed DGP coefficients — same across seeds so truth is a function of (seed).
# Change here if you want a different SNR / effect-size regime.
ALPHA = 0.0
BETA  = 3.0                              # main treatment effect (constant term)
DELTA = 2.0                              # T×X_1 interaction (drives heterogeneity)
SIGMA_Y = 1.0                            # outcome noise std


def _rng_gamma(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(1_000 + seed)
    # γ scaled so ‖γ‖ ≈ 1 to keep noise comparable to signal.
    g = rng.normal(size=d).astype(np.float32)
    return g / max(np.linalg.norm(g), 1e-8)


def _rng_w(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(2_000 + seed)
    w = rng.normal(size=d).astype(np.float32)
    return w * 0.5                                     # moderate confounding


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class SynTruth:
    seed: int
    gamma: np.ndarray               # (d,)
    beta: float
    delta: float
    alpha: float
    sigma_y: float
    X_test: np.ndarray              # (n_test, d) — raw feature values


class SynDataset:
    """Duck-typed CATE_Dataset."""
    def __init__(self, X_train, t_train, y_train, X_test, true_cate):
        self.X_train = X_train
        self.t_train = t_train
        self.y_train = y_train
        self.X_test = X_test
        self.true_cate = true_cate


def sample_realization(seed: int,
                        d: int = DEFAULT_D,
                        n_train: int = DEFAULT_N_TRAIN,
                        n_test: int = DEFAULT_N_TEST) -> tuple[SynDataset, SynTruth]:
    rng = np.random.default_rng(seed)
    gamma = _rng_gamma(d, seed)
    w = _rng_w(d, seed)

    X = rng.normal(size=(n_train + n_test, d)).astype(np.float32)
    pi = _sigmoid(X @ w)
    T = (rng.uniform(size=n_train + n_test) < pi).astype(np.float32)

    mu = ALPHA + BETA * T + X @ gamma + DELTA * T * X[:, 0]
    Y = mu + rng.normal(scale=SIGMA_Y, size=n_train + n_test).astype(np.float32)

    X_train = X[:n_train]; T_train = T[:n_train]; Y_train = Y[:n_train]
    X_test  = X[n_train:]
    cate_test = BETA + DELTA * X_test[:, 0]              # τ(x) = β + δ·x₁

    cd = SynDataset(
        X_train=X_train,
        t_train=T_train,
        y_train=Y_train,
        X_test=X_test,
        true_cate=cate_test.astype(np.float32),
    )
    truth = SynTruth(
        seed=seed, gamma=gamma, beta=float(BETA), delta=float(DELTA),
        alpha=float(ALPHA), sigma_y=float(SIGMA_Y),
        X_test=X_test,
    )
    return cd, truth
