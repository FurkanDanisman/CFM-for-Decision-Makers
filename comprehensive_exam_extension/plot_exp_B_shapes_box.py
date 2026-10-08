"""Experiment B shape paths as box plots in the format of the Do-PFN paper's Figure 4:
per-data-set MSE (CATE) at each level, IOD (Gaussian) first, no outlier points, whiskers
at 1.5 IQR, IOD box in red,
dashed line at the IOD median. One figure per (scale, case study, noise
target): 8 figures, each with one panel per noise family. 1000 data sets per box.

    python plot_exp_B_shapes_box.py --res results/exp_B_shapes_1000
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.special import gamma as G

p = argparse.ArgumentParser()
p.add_argument("--res", default="results/exp_B_shapes_1000")
a = p.parse_args()

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "axes.unicode_minus": False})
BOX, EDGE, RED = "#2e78b0", "#3c3c3c", "#e34948"
CASES = {"Observed_Confounder": ("Observed Confounder", "OC"),
         "Unobserved_Confounder": ("Unobserved Confounder", "UC")}
TARGETS = {"all": ("noise on all nodes", "all"), "outcome": ("noise on outcome only", "outcome")}


def lognorm_var(s):
    w = np.exp(s ** 2)
    return (w - 1) * w


# family -> (panel title, parameter symbol, levels (scaled order), variance in textbook form)
FAMILIES = {
    "gamma": ("Gamma", "k", (16, 8, 4, 2, 1), lambda k: k),
    "bimodal": ("Bimodal", "a", (1, 1.5, 2, 2.5, 3), lambda m: 1 + m ** 2),
    "contam": ("Contaminated", "c", (2, 3, 5, 7, 10), lambda c: 0.9 + 0.1 * c ** 2),
    "lognorm": ("Log-normal", "s", (0.25, 0.5, 0.75, 1.0, 1.25), lognorm_var),
    "gennorm": ("Gen. normal", r"$\beta$", (1.5, 1.25, 1.0, 0.75, 0.6), lambda b: G(3 / b) / G(1 / b)),
}

out_dir = os.path.join(a.res, "box")
os.makedirs(out_dir, exist_ok=True)
for scale in ("scaled", "unscaled"):
    df_all = pd.read_csv(os.path.join(a.res, scale, "per_dataset.csv"))
    for case, (case_lab, case_tag) in CASES.items():
        df = df_all[df_all.case == case]
        iod = df[df.dial == "in_prior"].mse_cate.values
        for target, (t_lab, t_tag) in TARGETS.items():
            fig, grid = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
            axes, note_ax = grid.ravel()[:5], grid.ravel()[5]
            top = 0.0
            for ax, (fam, (title, sym, levels, var)) in zip(axes, FAMILIES.items()):
                lv = tuple(sorted(levels, key=var)) if scale == "unscaled" else levels
                data = [iod] + [df[(df.dial == fam) & (df.level == float(l)) & (df.target == target)]
                                .mse_cate.values for l in lv]
                labels = ["IOD"] + [f"{sym}={l:g}" + (f"\nvar {var(l):.2g}" if scale == "unscaled" else "")
                                    for l in lv]
                bp = ax.boxplot(data, widths=0.7, patch_artist=True, showfliers=False, whis=1.5,
                                medianprops=dict(color=EDGE, lw=1.3),
                                whiskerprops=dict(color=EDGE, lw=1.3), capprops=dict(color=EDGE, lw=1.3),
                                boxprops=dict(edgecolor=EDGE, lw=1.3))
                for i, patch in enumerate(bp["boxes"]):
                    patch.set_facecolor(RED if i == 0 else BOX)   # first box: IOD (Gaussian) baseline
                ax.axhline(np.median(iod), color=RED, ls="--", lw=1.4, zorder=0)
                ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=11)
                top = max(top, max(w.get_ydata().max() for w in bp["whiskers"]))
                ax.tick_params(direction="in", top=True, right=True, labelsize=11)
                ax.set_title(f"OOD - {title}", fontsize=14)
                ax.set_xlabel(f"{title} parameter",
                              fontsize=12)
            # one y range for the whole figure: up to the longest whisker, starting below 0
            # so the boxes near zero are not pressed against the axis
            for ax in axes:
                ax.set_ylim(-0.06 * top, 1.30 * top)   # headroom for the legend
                ax.tick_params(labelleft=True)
            axes[0].legend(handles=[Patch(facecolor=RED, edgecolor=EDGE, label="IOD (Gaussian noise)"),
                                    Patch(facecolor=BOX, edgecolor=EDGE, label="OOD noise"),
                                    plt.Line2D([], [], color=RED, ls="--", lw=1.4, label="IOD median")],
                           loc="upper left", fontsize=11, frameon=True, facecolor="white", edgecolor="none", framealpha=1)
            for ax in grid[:, 0]:
                ax.set_ylabel("MSE (CATE)", fontsize=12)
            note_ax.axis("off")
            fig.tight_layout()
            out = os.path.join(out_dir, f"exp_B_box_{scale}_{case_tag}_{t_tag}.png")
            fig.savefig(out, dpi=180, facecolor="white")
            plt.close(fig)
            print(out)
