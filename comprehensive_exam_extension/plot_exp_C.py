"""Plot Experiment C: rho(lambda) = median error at lambda / median error in-prior, one panel
per g (columns) and case study (rows), one line per target equation. One figure per metric.
Paired bootstrap 95% intervals (datasets resampled jointly at lambda and in-prior).

    python plot_exp_C.py --res results/exp_C
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

p = argparse.ArgumentParser()
p.add_argument("--res", default="results/exp_C")
p.add_argument("--boot", type=int, default=2000)
a = p.parse_args()
df = pd.read_csv(os.path.join(a.res, "per_dataset.csv"))
rng = np.random.default_rng(0)

CASES = ["Observed_Confounder", "Unobserved_Confounder"]
G_FUNCS = {"sin": "g = sin (non-monotone)", "sign": "g = sign (a step)", "cubic": "g = cubic (fast growth)"}
TARGETS = {"outcome": "#2a78d6", "treatment": "#eb6834", "both": "#1baf7a"}
METRICS = {"mse_cate": "MSE (CATE)", "rel_ate": "Rel. ATE error"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def ratio_ci(num, den):
    m = np.isfinite(num) & np.isfinite(den)
    num, den = num[m], den[m]
    idx = rng.integers(0, len(num), (a.boot, len(num)))
    b = np.median(num[idx], axis=1) / np.median(den[idx], axis=1)
    return np.median(num) / np.median(den), *np.quantile(b, [0.025, 0.975])


summary = []
for metric, mlabel in METRICS.items():
    fig, axes = plt.subplots(len(CASES), len(G_FUNCS), figsize=(13, 7), sharey="row")
    for i, case in enumerate(CASES):
        base = df[(df.case == case) & (df.dial == "in_prior")].sort_values("r")[metric].values
        for j, (g, title) in enumerate(G_FUNCS.items()):
            ax = axes[i, j]
            for target, color in TARGETS.items():
                xs, ys, los, his = [0.0], [1.0], [1.0], [1.0]
                for lam in sorted(df[(df.dial == g)].level.unique()):
                    cur = df[(df.case == case) & (df.dial == g) & (df.level == lam)
                             & (df.target == target)].sort_values("r")[metric].values
                    if len(cur) != len(base):
                        continue
                    v, lo, hi = ratio_ci(cur, base)
                    summary.append(dict(metric=metric, case=case, g=g, target=target, lam=lam,
                                        rho=v, lo=lo, hi=hi))
                    xs.append(lam); ys.append(v); los.append(lo); his.append(hi)
                ax.fill_between(xs, los, his, color=color, alpha=0.10, lw=0)
                ax.plot(xs, ys, "-", color=color, lw=2, marker="o", ms=5, mec="white", mew=1)
            ax.axhline(1, color=MUTED, lw=0.8)
            ax.set_yscale("log")
            ax.grid(axis="y", color=GRID, lw=0.8, which="both")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=9)
            ax.set_xlabel("λ  (0 = in-prior function, 1 = g)", fontsize=9, color=INK)
            if i == 0:
                ax.set_title(title, fontsize=10, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel(f"{case.replace('_', ' ')}\nρ(λ), {mlabel}", fontsize=10, color=INK)
    handles = [Line2D([], [], color=c, lw=2, label=f"{t} equation" if t != "both" else "both equations")
               for t, c in TARGETS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(f"Experiment C: Do-PFN (v1) {mlabel} relative to in-prior, N=200, median over 100 datasets "
                 "(95% paired bootstrap)", fontsize=11, color=INK, x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(a.res, f"exp_C_rho_{metric}.png")
    fig.savefig(out, dpi=200, facecolor="white")
    print(out)
pd.DataFrame(summary).to_csv(os.path.join(a.res, "rho_summary.csv"), index=False)
