"""Validation suite for the OT-CATE-ATE pipeline.

Two studies:

  1. Ground-truth comparison at fixed N: simulate from a known DGP, run the
     pipeline, plot F_ATE_ot vs F_lin vs F_true. Report L² and W² distances.

  2. Sample-size ablation: vary N ∈ {N_min … N_max}, report the L²(F_ATE_ot,
     F_true) and L²(F_lin, F_true) as a function of N. Repeat with multiple
     seeds for a confidence band.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from ate_pipeline import ate_pipeline
from ot_barycenter import l2_distance, w2_distance
from simulator import TrueDGP, sample_dgp, true_ate_distribution


# ---------- single-run ground-truth comparison ---------- #


def run_ground_truth_check(
    dgp: TrueDGP,
    d_grid: np.ndarray,
    N: int = 50,
    B: int = 10000,
    seed: int = 20180621,
    title: str = "",
    savepath: str | None = None,
):
    """Simulate one dataset of N units, run the pipeline, plot the result."""
    f_per_unit, pi_true, delta_true = sample_dgp(dgp, N=N, d_grid=d_grid, seed=seed)
    F_true = true_ate_distribution(dgp, d_grid, pi_true=pi_true, delta_true=delta_true)

    print(f"  Decomposing {N} units with K={dgp.K} (B={B}) …", flush=True)
    res = ate_pipeline(f_per_unit, d_grid, K=dgp.K, B=B, seed=seed, verbose=False)

    l2_ot = l2_distance(res.F_ATE_ot, F_true, d_grid)
    l2_lin = l2_distance(res.F_ATE_lin, F_true, d_grid)
    w2_ot = w2_distance(res.F_ATE_ot, F_true, d_grid)
    w2_lin = w2_distance(res.F_ATE_lin, F_true, d_grid)
    print(f"    L²(F_ATE_ot,  F_true) = {l2_ot:.4f}")
    print(f"    L²(F_ATE_lin, F_true) = {l2_lin:.4f}")
    print(f"    W²(F_ATE_ot,  F_true) = {w2_ot:.4f}")
    print(f"    W²(F_ATE_lin, F_true) = {w2_lin:.4f}")
    print(f"    world weights estimated = {np.round(res.w_world, 3)}   "
          f"true population = {np.round(dgp.w, 3)}   "
          f"true empirical (this sample) = {np.round(pi_true.mean(axis=0), 3)}")

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(d_grid, F_true, "r-", lw=2.5, label="F_true (oracle)")
    ax.plot(d_grid, res.F_ATE_ot, "b-", lw=2, label=f"F_ATE (OT)   L²={l2_ot:.3f}")
    ax.plot(d_grid, res.F_ATE_lin, "k--", lw=1.5, label=f"F_lin (mixture)   L²={l2_lin:.3f}")
    # Show the cloud of unit CATE densities lightly
    for i in range(N):
        ax.plot(d_grid, f_per_unit[i], color="grey", alpha=0.08, lw=0.7)
    ax.set_xlabel("d  (treatment effect)")
    ax.set_ylabel("density")
    ax.set_title(title or f"N={N}, K={dgp.K}")
    ax.legend(loc="best")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120)
        print(f"  saved {savepath}")
    plt.close(fig)
    return res, F_true, (l2_ot, l2_lin, w2_ot, w2_lin)


# ---------- sample-size ablation ---------- #


@dataclass
class AblationPoint:
    N: int
    seed: int
    l2_ot: float
    l2_lin: float
    w2_ot: float
    w2_lin: float


def run_sample_size_ablation(
    dgp: TrueDGP,
    d_grid: np.ndarray,
    Ns: list[int],
    n_reps: int = 3,
    B: int = 10000,
    base_seed: int = 0,
    savepath: str | None = None,
):
    """Vary N, run the pipeline `n_reps` times per N with different seeds.

    Plot mean ± std of L²(F_ATE_*, F_true) and W²(F_ATE_*, F_true).
    """
    rows: list[AblationPoint] = []
    for N in Ns:
        for r in range(n_reps):
            seed = base_seed + 1000 * N + r
            f_per_unit, pi_true, delta_true = sample_dgp(dgp, N=N, d_grid=d_grid, seed=seed)
            F_true = true_ate_distribution(dgp, d_grid, pi_true=pi_true, delta_true=delta_true)
            res = ate_pipeline(f_per_unit, d_grid, K=dgp.K, B=B, seed=seed, verbose=False)
            l2_ot = l2_distance(res.F_ATE_ot, F_true, d_grid)
            l2_lin = l2_distance(res.F_ATE_lin, F_true, d_grid)
            w2_ot = w2_distance(res.F_ATE_ot, F_true, d_grid)
            w2_lin = w2_distance(res.F_ATE_lin, F_true, d_grid)
            rows.append(AblationPoint(N, seed, l2_ot, l2_lin, w2_ot, w2_lin))
            print(f"  N={N:4d}  rep={r+1}/{n_reps}   L²(OT)={l2_ot:.4f}   L²(lin)={l2_lin:.4f}", flush=True)

    Ns_arr = np.array(Ns)

    def _agg(metric):
        out_mean, out_std = [], []
        for N in Ns:
            vals = [getattr(r, metric) for r in rows if r.N == N]
            out_mean.append(np.mean(vals))
            out_std.append(np.std(vals))
        return np.array(out_mean), np.array(out_std)

    l2_ot_mean, l2_ot_std = _agg("l2_ot")
    l2_lin_mean, l2_lin_std = _agg("l2_lin")
    w2_ot_mean, w2_ot_std = _agg("w2_ot")
    w2_lin_mean, w2_lin_std = _agg("w2_lin")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.errorbar(Ns_arr, l2_ot_mean, yerr=l2_ot_std, fmt="-o", color="C0",
                label="OT", capsize=3)
    ax.errorbar(Ns_arr, l2_lin_mean, yerr=l2_lin_std, fmt="--s", color="C3",
                label="linear mixture", capsize=3)
    ax.set_xlabel("N (units)")
    ax.set_ylabel("L²(F̂_ATE, F_true)")
    ax.set_title("Sample-size ablation — L²")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    ax = axes[1]
    ax.errorbar(Ns_arr, w2_ot_mean, yerr=w2_ot_std, fmt="-o", color="C0",
                label="OT", capsize=3)
    ax.errorbar(Ns_arr, w2_lin_mean, yerr=w2_lin_std, fmt="--s", color="C3",
                label="linear mixture", capsize=3)
    ax.set_xlabel("N (units)")
    ax.set_ylabel("W²(F̂_ATE, F_true)")
    ax.set_title("Sample-size ablation — W²")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120)
        print(f"  saved {savepath}")
    plt.close(fig)
    return rows
