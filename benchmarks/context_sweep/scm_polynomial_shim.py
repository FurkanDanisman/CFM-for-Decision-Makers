"""Minimal polynomial SCM sampler — drop-in replacement for CausalPFN's
`benchmarks.PolynomialDataset` when the CausalPFN source isn't installed.

Design mirrors the original signature: `PolynomialDataset(n_tables, n_samples,
test_ratio, seed)[i]` yields `(cate_dataset, ate_dataset)` where `cate_dataset`
carries `X_train, t_train, y_train, X_test, true_cate` matching what OURS and
UWYK pipelines expect.

The SCM per realization:
  1. x_dim ∈ [5, 10] uniform.
  2. Covariates X ~ Uniform(-2, 2, shape=(n_samples, x_dim)).
  3. Treatment logits = sum of monomials of degree ∈ [2,4] over random subsets
     of X, with weights ~ Uniform(-5, 5), plus Laplace noise.
     T = 1 if logit > median.
  4. Outcome mechanism: another polynomial of X, then add a treatment-effect
     term = beta_T · x1_scaled·T, plus outcome noise.
  5. Y_do0 and Y_do1 are computed by the same mechanism with T fixed → CATE.

Distribution is CLOSE to CausalPFN's but not bit-for-bit identical (their
implementation picks slightly different monomials + noise). For the purpose
of a within-shim comparison (OURS vs UWYK-NoAnc on identical SCMs), this is
sufficient because every method sees the same draw.
"""
from __future__ import annotations
import numpy as np
import torch


class _CATE:
    __slots__ = ('X_train', 't_train', 'y_train', 'X_test', 'true_cate')


class _ATE:
    __slots__ = ('true_ate',)


def _polynomial_response(X: np.ndarray, rng: np.random.Generator,
                          n_terms: int = 5,
                          weight_range: tuple = (-5, 5),
                          degree_range: tuple = (2, 4)) -> np.ndarray:
    """Build a scalar polynomial response y(X) = Σ w_k · Π x_{s_k}^{d_k}."""
    n, d = X.shape
    y = np.zeros(n, dtype=np.float64)
    for _ in range(n_terms):
        deg = rng.integers(degree_range[0], degree_range[1] + 1)
        cols = rng.integers(0, d, size=deg)
        w = rng.uniform(*weight_range)
        y += w * np.prod(X[:, cols], axis=1)
    return y


def _sample_one_scm(seed: int, n_samples: int, test_ratio: float) -> tuple[_CATE, _ATE]:
    rng = np.random.default_rng(seed)
    x_dim = int(rng.integers(5, 11))                     # [5, 10]

    X = rng.uniform(-2.0, 2.0, size=(n_samples, x_dim)).astype(np.float32)

    # Treatment: polynomial logit + Laplace noise → binarize by median
    t_logit = _polynomial_response(X.astype(np.float64), rng,
                                    n_terms=4, weight_range=(-2, 2), degree_range=(2, 3))
    t_logit += rng.laplace(0.0, 1.0, size=n_samples)
    T = (t_logit > np.median(t_logit)).astype(np.float32)

    # Outcome mechanism: covariate polynomial + treatment effect + Gaussian noise
    y_cov = _polynomial_response(X.astype(np.float64), rng,
                                  n_terms=5, weight_range=(-5, 5), degree_range=(2, 4))
    # Heterogeneous treatment effect proportional to a random combination of X features.
    tau_coef = rng.uniform(-3.0, 3.0, size=x_dim)
    tau = X.astype(np.float64) @ tau_coef                # (n,)
    noise = rng.normal(0.0, 1.0, size=n_samples)

    Y_obs  = y_cov + tau * T + noise
    Y_do0  = y_cov + noise                               # same noise realization
    Y_do1  = y_cov + tau + noise
    true_cate_full = Y_do1 - Y_do0                        # = tau (noise cancels)

    # Split into train/test
    n_test = max(1, int(round(n_samples * test_ratio)))
    idx    = rng.permutation(n_samples)
    train_idx = idx[:n_samples - n_test]
    test_idx  = idx[n_samples - n_test:]

    cd = _CATE()
    cd.X_train  = torch.from_numpy(X[train_idx])
    cd.t_train  = torch.from_numpy(T[train_idx])
    cd.y_train  = torch.from_numpy(Y_obs[train_idx].astype(np.float32))
    cd.X_test   = torch.from_numpy(X[test_idx])
    cd.true_cate = torch.from_numpy(true_cate_full[test_idx].astype(np.float32))

    ad = _ATE()
    ad.true_ate = float(true_cate_full.mean())
    return cd, ad


class PolynomialDataset:
    """Drop-in replacement for causalpfn.benchmarks.PolynomialDataset."""

    def __init__(self, n_tables: int, n_samples: int, test_ratio: float, seed: int = 42):
        self.n_tables   = int(n_tables)
        self.n_samples  = int(n_samples)
        self.test_ratio = float(test_ratio)
        self.seed       = int(seed)

    def __getitem__(self, idx: int) -> tuple[_CATE, _ATE]:
        if idx >= self.n_tables:
            raise IndexError(f"table index {idx} >= n_tables {self.n_tables}")
        return _sample_one_scm(self.seed + idx, self.n_samples, self.test_ratio)
