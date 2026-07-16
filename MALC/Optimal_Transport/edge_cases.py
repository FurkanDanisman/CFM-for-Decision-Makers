"""Edge-case + variety study for the OT-CATE-ATE pipeline.

Now goes BEYOND Gaussian mixtures: includes Beta, Gamma, skew-normal,
Student-t (heavy-tail; violates log-concavity), Laplace, and mixed-family
worlds. Eight scenarios spanning:

  Favorable for OT:
    E1 — Gaussian baseline (large shift, the dramatic OT-wins case)
    E2 — Gamma mixture (right-skewed, positive support)
    E3 — Beta mixture (bounded support, asymmetric shapes)
    E4 — Mixed-family: world 1 Gaussian, world 2 Gamma (heterogeneous worlds)

  Edge cases / failure modes:
    E5 — t(df=3) mixture (NOT log-concave → MALC_BM mis-specified)
    E6 — K UNDER-specified (true K=3 Gaussians, fit K=2)
    E7 — Skew-normal with strong asymmetry (tests within-world shape recovery)
    E8 — Laplace mixture (log-concave but non-smooth at the mode)

Produces a 2×4 multi-panel figure + a printed summary table with the
linear-vs-OT L² and W² ratios.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from ate_pipeline import ate_pipeline
from ot_barycenter import l2_distance, w2_distance
from simulator_variety import (
    DGPVariety,
    beta_scaled,
    gamma_shifted,
    gaussian,
    laplace,
    sample_variety,
    skewnormal,
    student_t,
    true_ate,
)

SCENARIOS = [
    # ---------- Favorable ----------
    dict(
        label="E1 ★ Gaussian + LARGE shift",
        note="K=2 N(±2, 0.4²), σ_shift=1.2  — linear blurs, OT recovers",
        dgp=DGPVariety(
            K=2, worlds=[gaussian(0.4), gaussian(0.4)],
            m=np.array([-2.0, 2.0]), w=np.array([0.5, 0.5]),
            shift_sd=1.2,
        ),
        K_fit=2, N=50,
    ),
    dict(
        label="E2  Gamma mixture",
        note="K=2 Gamma(5) + Gamma(10), σ_shift=0.4 (right-skewed worlds)",
        dgp=DGPVariety(
            K=2, worlds=[gamma_shifted(shape=5.0, scale=1.0),
                          gamma_shifted(shape=10.0, scale=1.0)],
            m=np.array([-1.5, 1.5]), w=np.array([0.5, 0.5]),
            shift_sd=0.4,
        ),
        K_fit=2, N=50,
    ),
    dict(
        label="E3  Beta mixture",
        note="K=2 Beta(8,2) (right-skewed) + Beta(2,8) (left-skewed) on bounded support",
        dgp=DGPVariety(
            K=2, worlds=[beta_scaled(8, 2, scale=2.0),
                          beta_scaled(2, 8, scale=2.0)],
            m=np.array([-2.0, 2.0]), w=np.array([0.5, 0.5]),
            shift_sd=0.3,
        ),
        K_fit=2, N=50,
    ),
    dict(
        label="E4  Mixed-family worlds",
        note="World 1 Gaussian N(-2, 0.5²), World 2 Gamma(shape=6)",
        dgp=DGPVariety(
            K=2, worlds=[gaussian(0.5), gamma_shifted(shape=6.0, scale=0.5)],
            m=np.array([-2.0, 2.0]), w=np.array([0.5, 0.5]),
            shift_sd=0.4,
        ),
        K_fit=2, N=50,
    ),
    # ---------- Hard / edge cases ----------
    dict(
        label="E5 ✗ t(df=3) heavy tails (non-log-concave)",
        note="K=2 t(df=3) at ±2 — violates MALC_BM's log-concave assumption",
        dgp=DGPVariety(
            K=2, worlds=[student_t(df=3.0, sd=0.7), student_t(df=3.0, sd=0.7)],
            m=np.array([-2.0, 2.0]), w=np.array([0.5, 0.5]),
            shift_sd=0.3,
        ),
        K_fit=2, N=50,
    ),
    dict(
        label="E6  Gaussian trimodal (K=3 correctly fit)",
        note="true K=3 Gaussians at (-3, 0, 3), σ_shift=0.3, fit K=3 — all modes recovered",
        dgp=DGPVariety(
            K=3, worlds=[gaussian(0.5), gaussian(0.5), gaussian(0.5)],
            m=np.array([-3.0, 0.0, 3.0]), w=np.array([1/3, 1/3, 1/3]),
            shift_sd=0.3,
        ),
        K_fit=3, N=50,
    ),
    dict(
        label="E7  Skew-normal (strong asymmetry)",
        note="K=2 skew-normal a=±5 at ±2 — tests asymmetric within-world recovery",
        dgp=DGPVariety(
            K=2, worlds=[skewnormal(a=5.0, sd=0.6), skewnormal(a=-5.0, sd=0.6)],
            m=np.array([-2.0, 2.0]), w=np.array([0.5, 0.5]),
            shift_sd=0.3,
        ),
        K_fit=2, N=50,
    ),
    dict(
        label="E8  Laplace mixture (non-smooth peak)",
        note="K=2 Laplace at ±2, scale=0.5 — log-concave but kinked",
        dgp=DGPVariety(
            K=2, worlds=[laplace(0.5), laplace(0.5)],
            m=np.array([-2.0, 2.0]), w=np.array([0.5, 0.5]),
            shift_sd=0.4,
        ),
        K_fit=2, N=50,
    ),
]


@dataclass
class Result:
    label: str
    note: str
    f_per_unit: np.ndarray
    F_true: np.ndarray
    F_ATE_ot: np.ndarray
    F_ATE_lin: np.ndarray
    l2_ot: float
    l2_lin: float
    w2_ot: float
    w2_lin: float
    time_s: float


def main():
    d_grid = np.linspace(-7, 7, 161)
    seed = 20180621

    results: list[Result] = []
    for sc in SCENARIOS:
        print(f"\n=== {sc['label']}  —  {sc['note']} ===", flush=True)
        t0 = time.time()
        f, pi_true, delta_true = sample_variety(sc["dgp"], N=sc["N"], d_grid=d_grid, seed=seed)
        F_true = true_ate(sc["dgp"], d_grid, pi_true=pi_true, delta_true=delta_true)
        res = ate_pipeline(f, d_grid, K=sc["K_fit"], B=10000, seed=seed, verbose=False)
        el = time.time() - t0
        l2_ot = l2_distance(res.F_ATE_ot, F_true, d_grid)
        l2_lin = l2_distance(res.F_ATE_lin, F_true, d_grid)
        w2_ot = w2_distance(res.F_ATE_ot, F_true, d_grid)
        w2_lin = w2_distance(res.F_ATE_lin, F_true, d_grid)
        results.append(Result(
            label=sc["label"], note=sc["note"],
            f_per_unit=f, F_true=F_true,
            F_ATE_ot=res.F_ATE_ot, F_ATE_lin=res.F_ATE_lin,
            l2_ot=l2_ot, l2_lin=l2_lin, w2_ot=w2_ot, w2_lin=w2_lin,
            time_s=el,
        ))
        ratio = l2_lin / l2_ot if l2_ot > 1e-9 else float("inf")
        winner = "OT" if l2_ot < l2_lin else "LIN"
        print(f"  t={el:.0f}s  L²(OT)={l2_ot:.4f}  L²(lin)={l2_lin:.4f}  ratio={ratio:.2f}×  winner={winner}",
              flush=True)

    # ---------- Multi-panel figure ----------
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    for ax, r in zip(axes.ravel(), results):
        for j in range(len(r.f_per_unit)):
            ax.plot(d_grid, r.f_per_unit[j], color="grey", alpha=0.10, lw=0.6)
        ax.plot(d_grid, r.F_true, "r-", lw=2.5, label="F_true")
        ax.plot(d_grid, r.F_ATE_ot, "b-", lw=2, label=f"OT  L²={r.l2_ot:.3f}")
        ax.plot(d_grid, r.F_ATE_lin, "k--", lw=1.5, label=f"linear  L²={r.l2_lin:.3f}")
        ratio = r.l2_lin / r.l2_ot if r.l2_ot > 1e-9 else float("inf")
        winner = "OT" if r.l2_ot < r.l2_lin else "linear"
        ax.set_title(f"{r.label}\n{r.note}", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("d")
        ax.set_ylabel("density")
        ax.text(0.02, 0.97, f"winner: {winner} ({ratio:.1f}×)",
                transform=ax.transAxes, fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="lightblue" if winner == "OT" else "lightyellow",
                          ec="gray"))
    fig.suptitle("OT-CATE-ATE: edge cases across distribution families",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig("edge_cases_panel.png", dpi=120)
    plt.close(fig)
    print("\nsaved edge_cases_panel.png")

    # ---------- Summary table ----------
    print("\n" + "=" * 115)
    print(f"{'Scenario':<45} {'L²(OT)':>9} {'L²(lin)':>9} {'L² ratio':>10} "
          f"{'W²(OT)':>9} {'W²(lin)':>9} {'W² ratio':>10}  winner")
    print("-" * 115)
    for r in results:
        l2r = r.l2_lin / r.l2_ot if r.l2_ot > 1e-9 else float("inf")
        w2r = r.w2_lin / r.w2_ot if r.w2_ot > 1e-9 else float("inf")
        winner = "OT" if r.l2_ot < r.l2_lin else "linear"
        print(f"{r.label:<45} {r.l2_ot:>9.4f} {r.l2_lin:>9.4f} {l2r:>9.2f}× "
              f"{r.w2_ot:>9.4f} {r.w2_lin:>9.4f} {w2r:>9.2f}×  {winner}")


if __name__ == "__main__":
    main()
