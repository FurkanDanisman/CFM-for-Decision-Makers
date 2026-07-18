"""SCM sampler from CausalPFN's PolynomialDataset (Option B).

Polynomial mechanisms with configurable noise — held out from our training
prior, so this doubles as an out-of-distribution check.
"""
from __future__ import annotations
import numpy as np
import torch


def sample_as_cate_dataset(scm_seed: int, n_context: int, n_test: int = 50,
                            x_dim: int | None = None):
    """Return a CATE_Dataset-like namespace for one polynomial SCM.

    Notes
    -----
    - PolynomialDataset splits its `n_samples` into train/test by `test_ratio`.
      We size `n_samples` = n_context + n_test and set `test_ratio` accordingly.
    - `scm_seed` seeds the specific realization index.
    """
    from benchmarks import PolynomialDataset

    n_samples = n_context + n_test
    test_ratio = n_test / n_samples
    ds = PolynomialDataset(
        n_tables=max(scm_seed + 1, 1),
        n_samples=n_samples,
        test_ratio=test_ratio,
        seed=42 + scm_seed,
    )
    cd, _ad = ds[scm_seed]
    return cd
