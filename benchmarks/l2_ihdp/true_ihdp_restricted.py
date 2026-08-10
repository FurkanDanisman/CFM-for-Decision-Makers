"""IHDP truth restricted to the first `d_restrict` covariates.

Motivation
----------
The shipped IHDP DGP (Hill 2011) computes mu_0(X), mu_1(X) from all 25 features
with independent per-arm noise. When a model (fn=10) only conditions on the
first 10 features, its correct target is not p(Y | X_25) — it is the marginal
p(Y | X_10) obtained by integrating out X[10:25]. That marginal induces
correlation between Y_do(0) and Y_do(1) even though the noise is arm-independent,
because both means depend on the hidden features.

This module estimates that marginal nonparametrically. For each test unit i:
  1. Find the K nearest neighbours of X_i[:d_restrict] in X_train[:, :d_restrict]
     (Euclidean distance in standardised space).
  2. Their published (mu_0, mu_1) pairs form a Monte-Carlo mixture approximating
     p(mu_0, mu_1 | X_i[:d_restrict]).
  3. The truth marginal per arm is a mixture of Gaussians:
        f_true_0(y_0 | X_i) = (1/K) sum_j N(y_0; mu_0(X_j), sigma^2)
     (analogously for y_1).
  4. The truth CATE is also a mixture of Gaussians. Conditionally on X_j the
     two arms are independent Gaussians, so tau_j ~ N(mu_1(X_j) - mu_0(X_j),
     2*sigma^2). Marginalising over neighbours:
        f_true_tau(t | X_i) = (1/K) sum_j N(t; mu_1(X_j) - mu_0(X_j), 2*sigma^2).
  5. The induced Pearson correlation of the marginal joint equals the correlation
     of (mu_0(X_j), mu_1(X_j)) across neighbours.

Everything is on the scaled Y axis (Y ↦ 2 (Y − y_min) / y_rng − 1).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


Y_EDGES = np.linspace(-1.5, 1.5, 101)
Y_CENTERS = 0.5 * (Y_EDGES[:-1] + Y_EDGES[1:])
Y_BIN = float(Y_CENTERS[1] - Y_CENTERS[0])

TAU_EDGES = np.linspace(-3.0, 3.0, 601)
TAU_CENTERS = 0.5 * (TAU_EDGES[:-1] + TAU_EDGES[1:])
TAU_BIN = float(TAU_CENTERS[1] - TAU_CENTERS[0])


@dataclass
class RestrictedTruth:
    r: int
    d_restrict: int
    K: int
    sigma_scaled: float
    y_min: float
    y_rng: float
    # per-test-unit mixture components (K nearest neighbours in restricted space)
    mu0_neighbours_scaled: np.ndarray   # (n_test, K)
    mu1_neighbours_scaled: np.ndarray   # (n_test, K)


def _standardize(A: np.ndarray) -> np.ndarray:
    mu = A.mean(axis=0, keepdims=True)
    sd = A.std(axis=0, keepdims=True) + 1e-9
    return (A - mu) / sd


def load_ihdp_restricted_truth(r: int, causalpfn_dir: str,
                                X_train_full: np.ndarray, X_test_full: np.ndarray,
                                y_train_full: np.ndarray,
                                d_restrict: int = 10, K: int = 20) -> RestrictedTruth:
    """Build restricted truth for realisation `r`.

    X_train_full: (N_train, d_full) covariate matrix of the training set
        (real features; before any padding/truncation).
    X_test_full: (N_test, d_full) covariate matrix of the test set.
    y_train_full: (N_train,) training-side outcomes (raw units).
    """
    ihdp_dir = os.path.join(causalpfn_dir, 'benchmarks', 'IHDP')
    if not os.path.isdir(ihdp_dir):
        ihdp_dir = os.path.join(causalpfn_dir, 'IHDP')

    train_npz = np.load(os.path.join(ihdp_dir, 'ihdp_npci_1-100.train.npz'))
    test_npz  = np.load(os.path.join(ihdp_dir, 'ihdp_npci_1-100.test.npz'))
    mu0_test = test_npz['mu0'][..., r].astype(np.float32).reshape(-1)
    mu1_test = test_npz['mu1'][..., r].astype(np.float32).reshape(-1)          # noqa: F841

    mu0_train = train_npz['mu0'][..., r].astype(np.float32).reshape(-1)
    mu1_train = train_npz['mu1'][..., r].astype(np.float32).reshape(-1)
    yf_train  = train_npz['yf'][..., r].astype(np.float32).reshape(-1)
    t_train   = train_npz['t' ][..., r].astype(np.float32).reshape(-1)
    mu_factual = np.where(t_train > 0.5, mu1_train, mu0_train)
    sigma_orig = float(np.std(yf_train - mu_factual, ddof=1))

    y_min = float(y_train_full.min())
    y_max = float(y_train_full.max())
    y_rng = max(y_max - y_min, 1e-6)
    sigma_scaled = sigma_orig * (2.0 / y_rng)

    mu0_train_s = (mu0_train - y_min) / y_rng * 2.0 - 1.0
    mu1_train_s = (mu1_train - y_min) / y_rng * 2.0 - 1.0

    # k-NN in the standardised first-d_restrict feature subspace.
    d_avail = X_train_full.shape[1]
    d = min(d_restrict, d_avail)
    X_ref = np.concatenate([X_train_full[:, :d], X_test_full[:, :d]], axis=0)
    X_std = _standardize(X_ref.astype(np.float32))
    Xt_std = X_std[X_train_full.shape[0]:]         # test rows in standardised space
    Xtr_std = X_std[:X_train_full.shape[0]]        # train rows

    K_use = min(K, Xtr_std.shape[0])
    n_test = Xt_std.shape[0]
    mu0_neigh = np.zeros((n_test, K_use), dtype=np.float32)
    mu1_neigh = np.zeros((n_test, K_use), dtype=np.float32)
    for i in range(n_test):
        d2 = ((Xtr_std - Xt_std[i]) ** 2).sum(axis=1)
        nn = np.argpartition(d2, K_use - 1)[:K_use]
        mu0_neigh[i] = mu0_train_s[nn]
        mu1_neigh[i] = mu1_train_s[nn]

    return RestrictedTruth(
        r=r, d_restrict=d, K=K_use,
        sigma_scaled=sigma_scaled, y_min=y_min, y_rng=y_rng,
        mu0_neighbours_scaled=mu0_neigh,
        mu1_neighbours_scaled=mu1_neigh,
    )


def _gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def _mixture_density(x: np.ndarray, means: np.ndarray, sigma: float) -> np.ndarray:
    """Mixture-of-Gaussians density on x, uniform weights over `means`.

    means: (K,)   x: (M,)   returns (M,)
    """
    # Vectorised: (M, K) matrix of component densities -> mean over K
    comp = _gauss(x[:, None], means[None, :], sigma)
    return comp.mean(axis=1)


def restricted_marginals_per_query(truth: RestrictedTruth):
    """Returns (p_y0_true, p_y1_true), each (n_test, len(Y_CENTERS))."""
    n = truth.mu0_neighbours_scaled.shape[0]
    p_y0 = np.stack([_mixture_density(Y_CENTERS, truth.mu0_neighbours_scaled[i],
                                       truth.sigma_scaled) for i in range(n)])
    p_y1 = np.stack([_mixture_density(Y_CENTERS, truth.mu1_neighbours_scaled[i],
                                       truth.sigma_scaled) for i in range(n)])
    return p_y0, p_y1


def restricted_cate_per_query(truth: RestrictedTruth) -> np.ndarray:
    """Returns p_tau_true, shape (n_test, len(TAU_CENTERS)).

    Per-neighbour CATE component: tau_j ~ N(mu_1(X_j) - mu_0(X_j), 2*sigma^2).
    Mixture over neighbours.
    """
    n = truth.mu0_neighbours_scaled.shape[0]
    sig_tau = np.sqrt(2.0) * truth.sigma_scaled
    out = np.stack([_mixture_density(
        TAU_CENTERS,
        truth.mu1_neighbours_scaled[i] - truth.mu0_neighbours_scaled[i],
        sig_tau,
    ) for i in range(n)])
    return out


def restricted_ate_barycenter(p_taus_true: np.ndarray, bary_fn) -> np.ndarray:
    bary = bary_fn(p_taus_true, TAU_CENTERS)
    s = bary.sum() * TAU_BIN
    if s > 0:
        bary = bary / s
    return bary


def induced_correlation(truth: RestrictedTruth) -> np.ndarray:
    """Pearson rho of the marginal joint p(Y_0, Y_1 | X_i[:d_restrict]).

    Under the mixture:
        E[Y_0 Y_1 | i] = (1/K) sum_j mu_0(X_j) * mu_1(X_j)    [indep per component]
        E[Y_0 | i]     = (1/K) sum_j mu_0(X_j)
        Var(Y_0 | i)   = sigma^2 + (1/K) sum_j (mu_0(X_j) - E[Y_0|i])^2
        Cov            = (1/K) sum_j (mu_0(X_j) - E[Y_0|i]) (mu_1(X_j) - E[Y_1|i])
    """
    m0 = truth.mu0_neighbours_scaled
    m1 = truth.mu1_neighbours_scaled
    K = m0.shape[1]
    Em0 = m0.mean(axis=1)
    Em1 = m1.mean(axis=1)
    var0 = truth.sigma_scaled ** 2 + ((m0 - Em0[:, None]) ** 2).mean(axis=1)
    var1 = truth.sigma_scaled ** 2 + ((m1 - Em1[:, None]) ** 2).mean(axis=1)
    cov  = ((m0 - Em0[:, None]) * (m1 - Em1[:, None])).mean(axis=1)
    return cov / np.sqrt(np.maximum(var0 * var1, 1e-16))
