"""Plot Experiment B: degradation ratio rho(s) = median error at level s / median error in-prior,
one panel per dial, one line per case study, solid = noise on all nodes, dashed = outcome only.
Paired bootstrap 95% intervals (datasets resampled jointly at level s and in-prior).

    python plot_exp_B.py --res results/exp_B
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--res", default="results/exp_B")
p.add_argument("--boot", type=int, default=2000)
a = p.parse_args()
df = pd.read_csv(os.path.join(a.res, "per_dataset.csv"))
rng = np.random.default_rng(0)

CASES = {"Observed_Confounder": "#2a78d6", "Unobserved_Confounder": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
# x positions: in-prior level first, then increasing departure (B3 has levels on both sides of 1).
DIALS = {
    "B1_tails": ("B1 Tails: Student-t ν", [("Gauss", None), ("30", 30), ("10", 10), ("5", 5), ("3", 3)]),
    "B2_skew": ("B2 Skew: Gamma shape k", [("Gauss", None), ("16", 16), ("4", 4), ("1", 1)]),
    "B3_scale": ("B3 Scale: noise s.d. × c", [("0.5", 0.5), ("1", None), ("2", 2), ("4", 4), ("8", 8)]),
    "B4_hetero": ("B4 Heteroscedasticity γ", [("0", None), ("0.5", 0.5), ("1", 1), ("2", 2)]),
}
METRICS = {"mse_cate": "MSE (CATE)", "rel_ate": "Rel. ATE error"}


def ratio_ci(num, den):
    m = np.isfinite(num) & np.isfinite(den)
    num, den = num[m], den[m]
    idx = rng.integers(0, len(num), (a.boot, len(num)))
    b = np.median(num[idx], axis=1) / np.median(den[idx], axis=1)
    return np.median(num) / np.median(den), *np.quantile(b, [0.025, 0.975])


summary = []
fig, axes = plt.subplots(len(METRICS), len(DIALS), figsize=(15, 7), sharey="row")
for i, (metric, mlabel) in enumerate(METRICS.items()):
    for j, (dial, (title, levels)) in enumerate(DIALS.items()):
        ax = axes[i, j]
        for case, color in CASES.items():
            base = df[(df.case == case) & (df.dial == "in_prior")].sort_values("r")[metric].values
            for target, ls in (("all", "-"), ("outcome", "--")):
                xs, ys, los, his = [], [], [], []
                for x, (lab, lev) in enumerate(levels):
                    if lev is None:
                        v, lo, hi = 1.0, 1.0, 1.0
                    else:
                        cur = df[(df.case == case) & (df.dial == dial) & (df.level == lev)
                                 & (df.target == target)].sort_values("r")[metric].values
                        if len(cur) != len(base):
                            continue
                        v, lo, hi = ratio_ci(cur, base)
                        summary.append(dict(metric=metric, dial=dial, level=lev, case=case,
                                            target=target, rho=v, lo=lo, hi=hi))
                    xs.append(x); ys.append(v); los.append(lo); his.append(hi)
                ax.fill_between(xs, los, his, color=color, alpha=0.10 if ls == "-" else 0.06, lw=0)
                ax.plot(xs, ys, ls, color=color, lw=2, marker="o", ms=5, mec="white", mew=1)
        ax.axhline(1, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(levels)), [l for l, _ in levels])
        ax.set_yscale("log")
        ax.grid(axis="y", color=GRID, lw=0.8, which="both")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
        if i == 0:
            ax.set_title(title, fontsize=10, color=INK, loc="left")
        if j == 0:
            ax.set_ylabel(f"ρ(s), {mlabel}", fontsize=10, color=INK)

from matplotlib.lines import Line2D  # noqa: E402
handles = [Line2D([], [], color=c, lw=2, label=k.replace("_", " ")) for k, c in CASES.items()]
handles += [Line2D([], [], color=MUTED, lw=2, ls="-", label="noise on all nodes"),
            Line2D([], [], color=MUTED, lw=2, ls="--", label="noise on outcome only")]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.95))
fig.suptitle("Experiment B: Do-PFN (v1) error relative to in-prior, N=200, median over 100 datasets "
             "(95% paired bootstrap)", fontsize=11, color=INK, x=0.01, ha="left", y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.90))
out = os.path.join(a.res, "exp_B_rho.png")
fig.savefig(out, dpi=200, facecolor="white")
pd.DataFrame(summary).to_csv(os.path.join(a.res, "rho_summary.csv"), index=False)
print(out)
