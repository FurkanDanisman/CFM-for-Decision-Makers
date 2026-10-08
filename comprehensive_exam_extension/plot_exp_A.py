"""Plot Experiment A: our Do-PFN v1 normalized CID MSE medians (bootstrap 95% CI) next to the paper's Figure 3.

    python plot_exp_A.py --res results/exp_A
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Do-PFN (v1) bars of Robertson et al. (2025), Figure 3: median and 95% CI,
# read off the published figure by pixel measurement (approximate).
# Only row 1 (normalized CID MSE, App. D.2) has a definition in the paper.
PAPER = {
    "Observed_Confounder": (0.00104, 0.0007, 0.0016),
    "Unobserved_Confounder": (0.0277, 0.0244, 0.0308),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
PAPER_C, PLAIN_C, NORM_C = "#52514e", "#2a78d6", "#eb6834"

p = argparse.ArgumentParser()
p.add_argument("--res", default="results/exp_A")
p.add_argument("--boot", type=int, default=2000)
a = p.parse_args()
df = pd.read_csv(os.path.join(a.res, "per_dataset.csv"))
rng = np.random.default_rng(0)


def med_ci(x):
    x = np.asarray(x[np.isfinite(x)])
    b = np.median(rng.choice(x, (a.boot, len(x))), axis=1)
    return np.median(x), *np.quantile(b, [0.025, 0.975])


cases = ["Observed_Confounder", "Unobserved_Confounder"]
ns = sorted(df.n.unique())
fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
for ax, case in zip(axes, cases):
    d = df[df.case == case]
    rows = [("Paper, Fig. 3", PAPER[case], PAPER_C, "D")]
    rows += [(f"Ours, N={n}", med_ci(d[d.n == n].nmse_cid.values), PLAIN_C, "o") for n in ns]
    y = np.arange(len(rows))[::-1]
    for yy, (lab, (m, lo, hi), c, mk) in zip(y, rows):
        ax.plot([lo, hi], [yy, yy], color=c, lw=2, solid_capstyle="round")
        ax.plot(m, yy, mk, color=c, ms=8, mec="white", mew=1.5, zorder=3)
        ax.annotate(f"{m:.2g}", (hi, yy), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=MUTED)
    pm = PAPER[case]
    ax.axvspan(pm[1], pm[2], color=PAPER_C, alpha=0.08, lw=0)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8.5, color=INK)
    ax.set_ylim(y.min() - 0.7, y.max() + 0.7)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.15)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.set_xlabel("Normalized MSE (CID)", fontsize=9, color=INK)
    ax.set_title(case.replace("_", " "), fontsize=10, color=INK, loc="left")
fig.suptitle("Do-PFN (v1), in-prior: paper Fig. 3 (row 1) vs. ours — median, 95% CI over 100 datasets",
             fontsize=11, color=INK, x=0.01, ha="left")
fig.tight_layout()
out = os.path.join(a.res, "exp_A_vs_paper.png")
fig.savefig(out, dpi=200, facecolor="white")
print(out)
