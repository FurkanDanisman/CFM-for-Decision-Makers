"""SCM sampler from CausalPFN's PolynomialDataset (Option B).

Polynomial mechanisms with configurable noise — held out from our training
prior, so this doubles as an out-of-distribution check.
"""
from __future__ import annotations
import numpy as np
import torch


def _load_polynomial_dataset_class(causalpfn_root: str):
    """Load causalpfn's `PolynomialDataset` by file path so we don't collide
    with our own R-PFN benchmarks/ namespace on sys.path.

    Falls back to `scm_polynomial_shim.PolynomialDataset` if CausalPFN isn't
    available (e.g. on clusters where its `faiss-cpu` dep won't build).
    """
    import importlib.util
    import os
    pkg_init = os.path.join(causalpfn_root or '', 'benchmarks', '__init__.py')
    if causalpfn_root and os.path.isfile(pkg_init):
        pkg_dir = os.path.join(causalpfn_root, 'benchmarks')
        spec = importlib.util.spec_from_file_location(
            'causalpfn_benchmarks', pkg_init,
            submodule_search_locations=[pkg_dir],
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.PolynomialDataset
    # Fallback: minimal shim in the same folder
    from scm_polynomial_shim import PolynomialDataset
    return PolynomialDataset


def sample_as_cate_dataset(scm_seed: int, n_context: int, n_test: int = 50,
                            x_dim: int | None = None,
                            causalpfn_root: str | None = None):
    """Return a CATE_Dataset-like namespace for one polynomial SCM.

    Notes
    -----
    - PolynomialDataset splits its `n_samples` into train/test by `test_ratio`.
      We size `n_samples` = n_context + n_test and set `test_ratio` accordingly.
    - `scm_seed` seeds the specific realization index.
    """
    import os
    if causalpfn_root is None:
        causalpfn_root = os.environ.get('CAUSALPFN_ROOT', '/tmp/causalpfn_full')
    PolynomialDataset = _load_polynomial_dataset_class(causalpfn_root)

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
