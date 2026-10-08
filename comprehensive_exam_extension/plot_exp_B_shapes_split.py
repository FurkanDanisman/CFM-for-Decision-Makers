"""Experiment B shape paths, one figure per (scale, case study, noise target): 8 figures.
Each figure has one panel per noise family; rho = median MSE_CATE at a level / median at
the Gaussian baseline, 95% paired bootstrap, values labelled.

    python plot_exp_B_shapes_split.py --res results/exp_B_shapes_1000
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator
from scipy.special import gamma as G

p = argparse.ArgumentParser()
p.add_argument("--res", default="results/exp_B_shapes_1000")
p.add_argument("--boot", type=int, default=2000)
a = p.parse_args()
rng = np.random.default_rng(0)
INK, MUTED, GRID, RED = "#0b0b0b", "#52514e", "#e6e5e1", "#e34948"
CASES = {"Observed_Confounder": ("#2a78d6", "Observed Confounder", "OC"),
         "Unobserved_Confounder": ("#eb6834", "Unobserved Confounder", "UC")}
TARGETS = {"all": ("noise on all nodes", "all"), "outcome": ("noise on outcome only", "outcome")}


def lognorm_stats(s):
    w = np.exp(s ** 2)
    return (w + 2) * np.sqrt(w - 1), (w - 1) * w


FAMILIES = {
    "gamma": ("Gamma, centred", "k", (16, 8, 4, 2, 1), lambda k: f"skew {2 / np.sqrt(k):.2f}", lambda k: k),
    "bimodal": ("Bimodal ½N(±a, 1)", "a", (1, 1.5, 2, 2.5, 3),
                lambda m: f"kurt {(m**4 + 6*m**2 + 3) / (1 + m**2)**2 - 3:+.2f}", lambda m: 1 + m ** 2),
    "contam": ("Contaminated 0.9N(0,1)+0.1N(0,c²)", "c", (2, 3, 5, 7, 10),
               lambda c: f"kurt {3 * (0.9 + 0.1*c**4) / (0.9 + 0.1*c**2)**2 - 3:+.1f}",
               lambda c: 0.9 + 0.1 * c ** 2),
    "lognorm": ("Log-normal, centred", "s", (0.25, 0.5, 0.75, 1.0, 1.25),
                lambda s: f"skew {lognorm_stats(s)[0]:.1f}", lambda s: lognorm_stats(s)[1]),
    "gennorm": ("Generalized normal (β=1: Laplace)", "β", (1.5, 1.25, 1.0, 0.75, 0.6),
                lambda b: f"kurt {G(5/b) * G(1/b) / G(3/b)**2 - 3:+.1f}", lambda b: G(3 / b) / G(1 / b)),
}


def ratio_ci(num, den):
    idx = rng.integers(0, len(num), (a.boot, len(num)))
    b = np.median(num[idx], axis=1) / np.median(den[idx], axis=1)
    return np.median(num) / np.median(den), *np.quantile(b, [0.025, 0.975])


out_dir = os.path.join(a.res, "split")
os.makedirs(out_dir, exist_ok=True)
for scale in ("scaled", "unscaled"):
    df_all = pd.read_csv(os.path.join(a.res, scale, "per_dataset.csv"))
    for case, (color, case_lab, case_tag) in CASES.items():
        df = df_all[df_all.case == case]
        base = df[df.dial == "in_prior"].sort_values("r").mse_cate.values
        for target, (t_lab, t_tag) in TARGETS.items():
            # compute every point first, so the figure can share one y range
            series = {}
            for fam, (title, sym, levels, stat, var) in FAMILIES.items():
                lv = tuple(sorted(levels, key=var)) if scale == "unscaled" else levels
                pts = [(0, 1.0, 1.0, 1.0)]
                for x, lev in enumerate(lv, start=1):
                    cur = df[(df.dial == fam) & (df.level == float(lev)) & (df.target == target)] \
                        .sort_values("r").mse_cate.values
                    if len(cur) == len(base):
                        pts.append((x, *ratio_ci(cur, base)))
                series[fam] = (lv, pts)
            lo_all = min(min(q[2] for q in pts) for _, pts in series.values()) * 0.95
            hi_all = max(max(q[3] for q in pts) for _, pts in series.values()) * 1.08
            fig, grid = plt.subplots(2, 3, figsize=(15, 8.4))
            axes, note_ax = grid.ravel()[:5], grid.ravel()[5]
            for ax, (fam, (title, sym, levels, stat, var)) in zip(axes, FAMILIES.items()):
                lv, pts = series[fam]
                xs, ys, los, his = map(list, zip(*pts))
                ax.fill_between(xs, los, his, color=color, alpha=0.15, lw=0)
                ax.plot(xs, ys, "-o", color=color, lw=2.2, ms=6, mec="white", mew=1.2)
                for x, y in zip(xs[1:], ys[1:]):
                    ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 9), textcoords="offset points",
                                ha="center", fontsize=9, color=color, fontweight="bold")
                ax.axhline(1, color=RED, lw=1.6)
                if scale == "unscaled":
                    ax.set_yscale("log")
                    cand = ((0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15) if hi_all > 3 else
                            (0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2, 2.5, 3))
                else:
                    cand = (0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5)
                ax.set_ylim(lo_all, hi_all)
                yt = [t for t in cand if lo_all <= t <= hi_all]
                ax.yaxis.set_major_locator(FixedLocator(yt))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.set_yticklabels([f"{t:g}" for t in yt])
                ax.set_xticks(range(len(lv) + 1),
                              ["Gaussian"] + [f"{sym}={lev:g}\n" + (stat(lev) if scale == "scaled"
                                                                     else f"var {var(lev):.3g}") for lev in lv],
                              fontsize=9)
                ax.grid(axis="y", color=GRID, lw=0.8)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                ax.tick_params(colors=MUTED, labelsize=9.5)
                ax.set_title(title, loc="left", fontsize=11, color=INK)
            for ax in grid[:, 0]:
                ax.set_ylabel("ρ = MSE (CATE) relative to Gaussian", fontsize=10.5, color=INK)
            note_ax.axis("off")
            note_ax.text(0.05, 0.6, f"{case_lab}\n{t_lab}\n\n"
                         + ("scaled: variance fixed at the in-prior level\n(only the shape changes)"
                            if scale == "scaled" else
                            "unscaled: textbook form, variance changes\n(levels ordered by variance)")
                         + "\n\n1000 data sets per level, 95% bootstrap\nred line: Gaussian baseline (ρ = 1)",
                         transform=note_ax.transAxes, fontsize=11.5, color=INK, va="center")
            fig.tight_layout()
            out = os.path.join(out_dir, f"exp_B_shapes_{scale}_{case_tag}_{t_tag}.png")
            fig.savefig(out, dpi=180, facecolor="white")
            plt.close(fig)
            print(out)
