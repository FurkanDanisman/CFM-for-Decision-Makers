"""SCM-aware `density_truth`, shadowing the original for case studies only.

`benchmarks/eval_graph2d/density_truth.py` ends with

    raise ValueError(f'Density truth supports IHDP and ACIC, got {dataset!r}')

so the tauC runner cannot score the six SCM case studies. This module sits in
`case_study/density_eval/`, which the copied `eval_density_tauC.py` puts FIRST
on sys.path, so `from density_truth import ...` resolves here. IHDP and ACIC
delegate to the original, unmodified, loaded by path from `_ORIG`.

TAU IS DETERMINISTIC IN THE CASE-STUDY DGP. `generation.py` draws the outcome
noise ONCE per realization and reuses it for do(T=0) and do(T=1), so it cancels:
verified max|Y1 - Y0 - (mu1 - mu0)| ~ 1e-16. Therefore

    tau_star == cate == mu_1 - mu_0     exactly, and    sigma_tau == 0.

That is why `sigma_raw` is reported as 0 here. Coverage / length / Winkler /
CRPS need only the POINT truth, so they are unaffected. Anything that builds a
truth DENSITY (`truth_tau_density`, KL, L2) is ill-defined against a point mass
and will raise rather than silently use a fabricated sigma.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
_ORIG = os.path.join(_REPO, 'benchmarks', 'eval_graph2d')

_spec = importlib.util.spec_from_file_location(
    '_orig_density_truth', os.path.join(_ORIG, 'density_truth.py'))
_orig = importlib.util.module_from_spec(_spec)
# Must be registered BEFORE exec: @dataclass resolves sys.modules[cls.__module__]
# while processing the class body, and gets None if the module is not there yet.
sys.modules[_spec.name] = _orig
_spec.loader.exec_module(_orig)

DensityTruth = _orig.DensityTruth
harness_y_affine = _orig.harness_y_affine

SCM_CASES = ('Observed_Confounder', 'Backdoor_Criterion', 'Observed_Mediator',
             'Observed_Mediator_and_Confounder', 'Unobserved_Confounder',
             'Frontdoor_Criterion')


def _scm_npz(dataset: str, r: int) -> str:
    root = os.environ.get('CASE_STUDY_DATA_ROOT')
    n = os.environ.get('CASE_STUDY_N')
    if not root or not n:
        raise RuntimeError('CASE_STUDY_DATA_ROOT and CASE_STUDY_N must be set '
                           f'to score {dataset!r}')
    cell = os.path.join(root, dataset, f'N{int(n)}')
    path = os.path.join(cell, f'{dataset}_{r}.npz')
    if not os.path.isfile(path):
        have = len(glob.glob(os.path.join(cell, f'{dataset}_*.npz')))
        raise FileNotFoundError(f'{path} not found ({have} realizations in {cell})')
    return path


def load_density_truth(dataset: str, r: int, *, y_shift: float, y_scale: float,
                       causalpfn_dir: str, acic_cache_dir: str):
    if dataset not in SCM_CASES:
        return _orig.load_density_truth(
            dataset, r, y_shift=y_shift, y_scale=y_scale,
            causalpfn_dir=causalpfn_dir, acic_cache_dir=acic_cache_dir)

    z = np.load(_scm_npz(dataset, r), allow_pickle=True)
    mu0 = np.asarray(z['mu_0'], dtype=np.float64).reshape(-1)
    mu1 = np.asarray(z['mu_1'], dtype=np.float64).reshape(-1)
    cate = np.asarray(z['cate'], dtype=np.float64).reshape(-1)

    # The identity the whole design rests on — assert it, do not assume it.
    gap = float(np.max(np.abs((mu1 - mu0) - cate)))
    if gap > 1e-5:
        raise AssertionError(f'{dataset} r={r}: cate != mu_1 - mu_0 '
                             f'(max |diff| = {gap:.3g}); tau is not deterministic '
                             'and the point-truth assumption is invalid')

    q = os.environ.get('SCM_N_QUERY')
    if q:
        k = min(int(q), cate.size)
        mu0, mu1, cate = mu0[:k], mu1[:k], cate[:k]

    return DensityTruth(
        mu0_scaled=(mu0 - y_shift) / y_scale,
        mu1_scaled=(mu1 - y_shift) / y_scale,
        tau_star_scaled=cate / y_scale,     # shift cancels in a difference
        sigma_raw=0.0, sigma_scaled=0.0,
        sigma_residual_raw=0.0, sigma_residual_scaled=0.0,
    )


def scm_true_cate(dataset: str, r: int) -> np.ndarray:
    """Raw-unit true CATE for one realization (what the metrics score against)."""
    z = np.load(_scm_npz(dataset, r), allow_pickle=True)
    cate = np.asarray(z['cate'], dtype=np.float64).reshape(-1)
    q = os.environ.get('SCM_N_QUERY')
    return cate[:min(int(q), cate.size)] if q else cate
