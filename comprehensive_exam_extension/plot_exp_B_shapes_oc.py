"""Observed Confounder only: Experiment B shape paths with 1000 data sets per level.
Rows: scaled (variance fixed) / unscaled (variance grows); columns: noise families.
rho = median MSE_CATE at a level / median at the Gaussian baseline; 95% paired bootstrap.

    python plot_exp_B_shapes_oc.py --families gamma bimodal
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
p.add_argument("--families", nargs="+", default=["gamma", "bimodal"])
p.add_argument("--boot", type=int, default=2000)
a = p.parse_args()
rng = np.random.default_rng(0)
CASE = "Observed_Confounder"
TARGETS = {"all": ("#2a78d6", "noise on all nodes"), "outcome": ("#eb6834", "noise on outcome only")}
INK, MUTED, GRID, RED = "#0b0b0b", "#52514e", "#e6e5e1", "#e34948"


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


fams = a.families
fig, axes = plt.subplots(2, len(fams), figsize=(6.2 * len(fams), 9), squeeze=False)
for i, scale in enumerate(("scaled", "unscaled")):
    df = pd.read_csv(os.path.join(a.res, scale, "per_dataset.csv"))
    df = df[df.case == CASE]
    base = df[df.dial == "in_prior"].sort_values("r").mse_cate.values
    for j, fam in enumerate(fams):
        title, sym, levels, stat, var = FAMILIES[fam]
        if scale == "unscaled":
            levels = tuple(sorted(levels, key=var))
        ax = axes[i, j]
        allv, pts = [1.0], {}
        for target, (color, lab) in TARGETS.items():
            xs, ys, los, his = [0], [1.0], [1.0], [1.0]
            for x, lev in enumerate(levels, start=1):
                cur = df[(df.dial == fam) & (df.level == float(lev)) & (df.target == target)] \
                    .sort_values("r").mse_cate.values
                if len(cur) != len(base):
                    continue
                v, lo, hi = ratio_ci(cur, base)
                xs.append(x); ys.append(v); los.append(lo); his.append(hi)
            allv += los + his
            ax.fill_between(xs, los, his, color=color, alpha=0.15, lw=0)
            ax.plot(xs, ys, "-o", color=color, lw=2.2, ms=6, mec="white", mew=1.2, label=lab)
            pts[target] = dict(zip(xs[1:], ys[1:]))
        # value labels: the higher of the two lines is labelled above, the lower below
        for x in set(pts.get("all", {})) | set(pts.get("outcome", {})):
            vals = {t: pts[t][x] for t in pts if x in pts[t]}
            top = max(vals, key=vals.get)
            for t, y in vals.items():
                ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 9 if t == top else -15),
                            textcoords="offset points", ha="center", fontsize=9, color=TARGETS[t][0],
                            fontweight="bold")
        ax.axhline(1, color=RED, lw=1.6, label="Gaussian baseline (ρ = 1)")
        if scale == "unscaled":
            ax.set_yscale("log")
            ticks = [t for t in (0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20)
                     if min(allv) * 0.9 <= t <= max(allv) * 1.1]
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.set_xticks(range(len(levels) + 1),
                      ["Gaussian"] + [f"{sym}={lev:g}\n" + (stat(lev) if scale == "scaled" else f"var {var(lev):.3g}")
                                      for lev in levels], fontsize=9.5)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.set_title(f"{title}: {'scaled (variance = 1)' if scale == 'scaled' else 'unscaled (variance grows)'}",
                     loc="left", fontsize=11.5, color=INK)
        if j == 0:
            ax.set_ylabel("ρ = MSE (CATE) relative to Gaussian", fontsize=11, color=INK)
        ax.legend(fontsize=9.5, frameon=False, loc="upper left")
fig.suptitle("Observed Confounder: Do-PFN error vs noise shape, 1000 data sets per level (95% bootstrap)",
             fontsize=12.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = os.path.join(a.res, f"exp_B_shapes_OC_{'_'.join(fams)}.png")
fig.savefig(out, dpi=180, facecolor="white")
print(out)
