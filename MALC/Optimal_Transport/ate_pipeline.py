"""End-to-end ATE distribution pipeline:

    f_1, …, f_N  (1D CATE densities on a common grid)
            │
            │ MALC_BM(K=K)  per unit  →  (π_{i,k}, g_{i,k}) for k=1..K
            ▼
    per-world OT barycenter g_k_bary on common grid
            │  weights α_{i,k} = π_{i,k} / Σ_j π_{j,k}
            ▼
    F_ATE = Σ_k w_k g_k_bary    w_k = (1/N) Σ_i π_{i,k}

Also reports the naive linear-mixture baseline F_lin = (1/N) Σ_i f_i.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Make the MALC_BM Python port importable
_MALCBM_DIR = Path(__file__).resolve().parent.parent / "MALC_BM" / "python"
if str(_MALCBM_DIR) not in sys.path:
    sys.path.insert(0, str(_MALCBM_DIR))

from log_concave_1d import dlcd_1d_smoothed  # noqa: E402
from malc_bm import (  # noqa: E402
    MALC_BM_fit,
    dmalc_bm,
    sort_components_by_mode,
)

from ot_barycenter import (  # noqa: E402
    linear_mixture,
    wasserstein_barycenter_1d,
)


@dataclass
class ATEDecomposition:
    """Result of the per-world OT pipeline."""
    K: int
    d_grid: np.ndarray             # (M,) common grid for d
    # per-unit decomposition
    pi_all: np.ndarray             # (N, K) mixing weights
    g_all: np.ndarray              # (N, K, M) within-world densities on d_grid
    # per-world aggregation
    w_world: np.ndarray            # (K,) world weights = mean over units of pi
    g_world_bary: np.ndarray       # (K, M) per-world OT barycenter density
    g_world_linear: np.ndarray     # (K, M) per-world linear mixture (for compare)
    # combined
    F_ATE_ot: np.ndarray           # (M,) combined ATE via OT
    F_ATE_lin: np.ndarray          # (M,) naive linear-mixture ATE


def _component_density_on_grid(component_fit, d_grid):
    """Evaluate one MALC_BM component's smoothed log-concave density on d_grid."""
    d_grid = np.asarray(d_grid, dtype=float)
    dens = dlcd_1d_smoothed(d_grid, component_fit.fhatn)
    dens = np.maximum(dens, 0.0)
    dd = d_grid[1] - d_grid[0]
    s = dens.sum() * dd
    if s > 0:
        dens = dens / s
    return dens


def _decompose_unit_worker(args):
    """ProcessPoolExecutor worker: fit MALC_BM on one unit and return aligned
    π and component densities on d_grid."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    p_vec, grid_edges, d_grid, K, B, seed = args
    fit = MALC_BM_fit(p_vec, grid_edges, K=K, B=B, seed=seed, verbose=False)
    fit = sort_components_by_mode(fit)
    M = len(d_grid)
    g = np.zeros((K, M))
    for k in range(K):
        if fit.fits[k] is not None:
            g[k] = _component_density_on_grid(fit.fits[k], d_grid)
    return np.asarray(fit.pi), g


def decompose_units(
    f_per_unit: np.ndarray,
    d_grid: np.ndarray,
    K: int,
    B: int = 10000,
    seed: int = 20180621,
    verbose: bool = False,
    parallel: bool = True,
    n_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run MALC_BM on each unit's CATE density and return aligned components.

    Parallelised across units via ProcessPoolExecutor (each unit's MALC_BM
    is independent — perfect parallelism).

    Returns
    -------
    pi_all : (N, K)
    g_all : (N, K, M)
    """
    f_per_unit = np.atleast_2d(np.asarray(f_per_unit, dtype=float))
    N, M = f_per_unit.shape
    if M != len(d_grid):
        raise ValueError("f_per_unit columns must match d_grid length")

    dd = d_grid[1] - d_grid[0]
    grid_edges = np.concatenate([[d_grid[0] - dd / 2], d_grid + dd / 2])

    pi_all = np.zeros((N, K))
    g_all = np.zeros((N, K, M))

    worker_args = []
    for i in range(N):
        p_vec = f_per_unit[i] * dd
        s = p_vec.sum()
        if s <= 0:
            raise ValueError(f"unit {i} has zero total mass")
        p_vec = p_vec / s
        worker_args.append((p_vec, grid_edges, d_grid, K, B, seed + i))

    if parallel and N > 1:
        nw = n_workers if n_workers is not None else min(N, mp.cpu_count())
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=nw, mp_context=ctx) as ex:
            for i, (pi_i, g_i) in enumerate(ex.map(_decompose_unit_worker, worker_args)):
                pi_all[i] = pi_i
                g_all[i] = g_i
                if verbose:
                    print(f"  unit {i+1}/{N}: pi={np.round(pi_i, 3)}", flush=True)
    else:
        for i, args in enumerate(worker_args):
            pi_i, g_i = _decompose_unit_worker(args)
            pi_all[i] = pi_i
            g_all[i] = g_i
            if verbose:
                print(f"  unit {i+1}/{N}: pi={np.round(pi_i, 3)}", flush=True)

    return pi_all, g_all


def aggregate_ate(
    pi_all: np.ndarray,
    g_all: np.ndarray,
    d_grid: np.ndarray,
) -> ATEDecomposition:
    """Per-world OT barycenter + combination into F_ATE.

    Inputs come from `decompose_units`. Output is the full `ATEDecomposition`.
    """
    pi_all = np.asarray(pi_all, dtype=float)
    g_all = np.asarray(g_all, dtype=float)
    N, K, M = g_all.shape
    if M != len(d_grid):
        raise ValueError("g_all third axis must match d_grid")

    # World weights
    w_world = pi_all.mean(axis=0)
    w_world = w_world / w_world.sum()

    # Per-world barycenter and linear mixture
    g_world_bary = np.zeros((K, M))
    g_world_linear = np.zeros((K, M))
    for k in range(K):
        col_pi = pi_all[:, k]
        s = col_pi.sum()
        if s <= 0:
            # No mass in world k for any unit — fall back to uniform
            g_world_bary[k] = 1.0 / (d_grid[-1] - d_grid[0])
            g_world_linear[k] = g_world_bary[k]
            continue
        alpha = col_pi / s
        g_world_bary[k] = wasserstein_barycenter_1d(g_all[:, k, :], d_grid, weights=alpha)
        g_world_linear[k] = linear_mixture(g_all[:, k, :], weights=alpha)

    F_ATE_ot = (w_world[:, None] * g_world_bary).sum(axis=0)
    F_ATE_lin = (w_world[:, None] * g_world_linear).sum(axis=0)

    # Normalise the combined densities (against tiny numerical drift)
    dd = d_grid[1] - d_grid[0]
    s_ot = F_ATE_ot.sum() * dd
    if s_ot > 0:
        F_ATE_ot = F_ATE_ot / s_ot
    s_lin = F_ATE_lin.sum() * dd
    if s_lin > 0:
        F_ATE_lin = F_ATE_lin / s_lin

    return ATEDecomposition(
        K=K,
        d_grid=d_grid,
        pi_all=pi_all,
        g_all=g_all,
        w_world=w_world,
        g_world_bary=g_world_bary,
        g_world_linear=g_world_linear,
        F_ATE_ot=F_ATE_ot,
        F_ATE_lin=F_ATE_lin,
    )


def ate_pipeline(
    f_per_unit: np.ndarray,
    d_grid: np.ndarray,
    K: int,
    B: int = 10000,
    seed: int = 20180621,
    verbose: bool = False,
    parallel: bool = True,
    n_workers: int | None = None,
) -> ATEDecomposition:
    """One-shot: decompose every unit with MALC_BM at fixed K and aggregate."""
    pi_all, g_all = decompose_units(
        f_per_unit, d_grid, K, B=B, seed=seed, verbose=verbose,
        parallel=parallel, n_workers=n_workers,
    )
    return aggregate_ate(pi_all, g_all, d_grid)


def naive_linear_mixture(f_per_unit: np.ndarray, d_grid: np.ndarray) -> np.ndarray:
    """Direct (no decomposition) baseline:  F_lin(d) = (1/N) Σ_i f_i(d)."""
    return linear_mixture(np.atleast_2d(np.asarray(f_per_unit, dtype=float)))
