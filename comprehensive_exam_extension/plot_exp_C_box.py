"""Experiment C as box plots in the format of the Experiment B box plots (paper Fig. 4 style):
per-data-set MSE (CATE) at each lambda on a log y axis, IOD (in-prior function, lambda = 0) as the red box,
red dashed line at the IOD median, no outlier points, whiskers at 1.5 IQR, one shared y range
per figure, one legend in the top-left panel. One figure per (case study, equation changed):
6 figures, each with one panel per new function g.

Merges realizations 0-99 (results/exp_C), any local extension (results/exp_C_ext/*) and the
cluster run (results/exp_C_1000_cluster/<g>/, copied back from $SCRATCH/exam_ext/exp_C_1000)
into results/exp_C_1000/per_dataset.csv first.

    python plot_exp_C_box.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

KEY = ["case", "dial", "level", "target", "r"]
parts = ["results/exp_C/per_dataset.csv"] + sorted(glob.glob("results/exp_C_ext/*/per_dataset.csv")) \
    + sorted(glob.glob("results/exp_C_1000_cluster/*/per_dataset.csv"))   # cluster run (r = 0..999)
df_all = pd.concat([pd.read_csv(p) for p in parts if os.path.exists(p)], ignore_index=True)
df_all = df_all.drop_duplicates(KEY).sort_values(KEY)
os.makedirs("results/exp_C_1000", exist_ok=True)
df_all.to_csv("results/exp_C_1000/per_dataset.csv", index=False)
n = df_all.groupby(["case", "dial", "level", "target"]).size()
print(f"{len(df_all)} rows; conditions with 1000 data sets: {(n == 1000).sum()} / {len(n)}")

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "axes.unicode_minus": False})
BOX, EDGE, RED = "#2e78b0", "#3c3c3c", "#e34948"
CASES = {"Observed_Confounder": "OC", "Unobserved_Confounder": "UC"}
TARGETS = ("outcome", "treatment", "both")
G_FUNCS = {"sin": "sin(x)", "sign": "sign(x)", "cubic": "x³", "expsq": "exp(−x²/2)", "sin3": "sin(3x)"}
LAMBDAS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)

out_dir = "results/exp_C_1000/box"
os.makedirs(out_dir, exist_ok=True)
for case, tag in CASES.items():
    df = df_all[df_all.case == case]
    iod = df[df.dial == "in_prior"].mse_cate.values
    for target in TARGETS:
        fig, grid = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
        axes = grid.ravel()[:5]
        grid.ravel()[5].axis("off")
        top, bot = 0.0, np.inf
        for ax, (g, title) in zip(axes, G_FUNCS.items()):
            data = [iod] + [df[(df.dial == g) & (df.level == lam) & (df.target == target)].mse_cate.values
                            for lam in LAMBDAS]
            if any(len(d) == 0 for d in data):
                ax.set_title(f"OOD - {title} (running)", fontsize=14)
                continue
            bp = ax.boxplot(data, widths=0.7, patch_artist=True, showfliers=False, whis=1.5,
                            medianprops=dict(color=EDGE, lw=1.3), whiskerprops=dict(color=EDGE, lw=1.3),
                            capprops=dict(color=EDGE, lw=1.3), boxprops=dict(edgecolor=EDGE, lw=1.3))
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(RED if i == 0 else BOX)
            ax.axhline(np.median(iod), color=RED, ls="--", lw=1.4, zorder=0)
            top = max(top, max(w.get_ydata().max() for w in bp["whiskers"]))
            bot = min(bot, min(w.get_ydata().min() for w in bp["whiskers"]))
            ax.set_xticks(range(1, len(LAMBDAS) + 2), ["IOD"] + [rf"$\lambda$={l:g}" for l in LAMBDAS],
                          fontsize=11)
            ax.tick_params(direction="in", top=True, right=True, labelsize=11, labelleft=True)
            ax.set_title(f"OOD - {title}", fontsize=14)
            ax.set_xlabel(rf"$\lambda$ (0 = in-prior function, 1 = {title})", fontsize=12)
        # log y (errors span ~3 orders of magnitude across g); shared, with headroom for the legend
        for ax in axes:
            ax.set_yscale("log")
            ax.set_ylim(max(bot, 1e-6) / 2, top * 8)
        for ax in grid[:, 0]:
            ax.set_ylabel("MSE (CATE)", fontsize=12)
        axes[0].legend(handles=[Patch(facecolor=RED, edgecolor=EDGE, label="IOD (in-prior function)"),
                                Patch(facecolor=BOX, edgecolor=EDGE, label="OOD function"),
                                plt.Line2D([], [], color=RED, ls="--", lw=1.4, label="IOD median")],
                       loc="upper left", fontsize=11, frameon=True, facecolor="white", edgecolor="none",
                       framealpha=1)
        fig.tight_layout()
        out = os.path.join(out_dir, f"exp_C_box_{tag}_{target}.png")
        fig.savefig(out, dpi=180, facecolor="white")
        plt.close(fig)
        print(out)
