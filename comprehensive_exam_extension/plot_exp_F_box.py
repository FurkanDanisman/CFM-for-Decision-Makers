"""Experiment F as box plots in the format of the Experiment C box plots (paper Fig. 4 style):
per-data-set MSE (CATE) per level on a log y axis, in-prior as the red box, red dashed line at
its median, no outlier points, whiskers at 1.5 IQR, one shared y range per figure, one legend
in the left panel. One figure per dial, panels Observed / Unobserved Confounder.

Reads the cluster run copied back from $SCRATCH/exam_ext/exp_F_1000 into results/exp_F_1000/<dial>/.

    python plot_exp_F_box.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

parts = sorted(glob.glob("results/exp_F_1000/*/per_dataset.csv"))
df_all = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
n = df_all.groupby(["case", "dial", "level"]).size()
print(f"{len(df_all)} rows; conditions with 1000 data sets: {(n == 1000).sum()} / {len(n)}")

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "axes.unicode_minus": False})
BOX, EDGE, RED = "#2e78b0", "#3c3c3c", "#e34948"
CASES = {"Observed_Confounder": "Observed Confounder", "Unobserved_Confounder": "Unobserved Confounder"}
# dial: (file tag, x label, levels in plot order with the in-prior as None, tick label per level)
DIALS = {
    "treat": ("treatment_balance", "treated fraction",
              (0.1, 0.2, 0.3, 0.4, None, 0.6, 0.7, 0.8, 0.9), lambda l: f"{100 * (0.5 if l is None else l):.0f}%"),
    "covk": ("covariate_type", "number of categories of the confounder",
             (None, 20, 10, 5, 3, 2), lambda l: "continuous" if l is None else f"k={l:g}"),
    "irrel": ("irrelevant_covariates", "number of irrelevant covariates",
              (None, 1, 2, 5, 10, 20), lambda l: "m=0" if l is None else f"m={l:g}"),
    "outk": ("outcome_type", "number of categories of the outcome",
             (None, 20, 10, 5, 3, 2), lambda l: "continuous" if l is None else f"k={l:g}"),
}

out_dir = "results/exp_F_1000/box"
os.makedirs(out_dir, exist_ok=True)
for dial, (tag, xlabel, levels, tick) in DIALS.items():
    if not (df_all.dial == dial).any():
        print(f"{dial}: no results yet")
        continue
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    top, bot = 0.0, np.inf
    for ax, (case, title) in zip(axes, CASES.items()):
        df = df_all[df_all.case == case]
        iod = df[df.dial == "in_prior"].mse_cate.values
        data = [iod if l is None else df[(df.dial == dial) & np.isclose(df.level, l)].mse_cate.values for l in levels]
        keep = [i for i, d in enumerate(data) if len(d)]
        bp = ax.boxplot([data[i] for i in keep], positions=[i + 1 for i in keep], widths=0.7, patch_artist=True,
                        showfliers=False, whis=1.5,
                        medianprops=dict(color=EDGE, lw=1.3), whiskerprops=dict(color=EDGE, lw=1.3),
                        capprops=dict(color=EDGE, lw=1.3), boxprops=dict(edgecolor=EDGE, lw=1.3))
        for i, patch in zip(keep, bp["boxes"]):
            patch.set_facecolor(RED if levels[i] is None else BOX)
        ax.axhline(np.median(iod), color=RED, ls="--", lw=1.4, zorder=0)
        top = max(top, max(w.get_ydata().max() for w in bp["whiskers"]))
        bot = min(bot, min(w.get_ydata().min() for w in bp["whiskers"]))
        ax.set_xticks(range(1, len(levels) + 1), [tick(l) for l in levels], fontsize=11)
        ax.tick_params(direction="in", top=True, right=True, labelsize=11, labelleft=True)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(max(bot, 1e-6) / 2, top * 8)
    axes[0].set_ylabel("MSE (CATE)", fontsize=12)
    axes[0].legend(handles=[Patch(facecolor=RED, edgecolor=EDGE, label="IOD (in-prior)"),
                            Patch(facecolor=BOX, edgecolor=EDGE, label="OOD"),
                            plt.Line2D([], [], color=RED, ls="--", lw=1.4, label="IOD median")],
                   loc="upper left", fontsize=11, frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    fig.tight_layout()
    out = os.path.join(out_dir, f"exp_F_box_{tag}.png")
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)
    print(out)
