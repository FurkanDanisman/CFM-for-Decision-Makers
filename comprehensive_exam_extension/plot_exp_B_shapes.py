"""Plot the Experiment B shape paths: rho = median MSE_CATE at a level / median in-prior
(Gaussian), against each family's shape parameter (Gaussian first). One file per scale,
one panel per family; lines per case study, solid = noise on all nodes, dashed = outcome
only; paired bootstrap 95% intervals. Also writes rho_summary.csv (both metrics).

    python plot_exp_B_shapes.py --res results/exp_B_shapes
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator
from scipy.special import gamma as G

p = argparse.ArgumentParser()
p.add_argument("--res", default="results/exp_B_shapes")
p.add_argument("--boot", type=int, default=2000)
a = p.parse_args()
rng = np.random.default_rng(0)

CASES = {"Observed_Confounder": "#2a78d6", "Unobserved_Confounder": "#eb6834"}
INK, MUTED, GRID, RED = "#0b0b0b", "#52514e", "#e6e5e1", "#e34948"


def lognorm_stats(s):
    w = np.exp(s ** 2)
    return (w + 2) * np.sqrt(w - 1), (w - 1) * w


# family -> (title, parameter symbol, levels, shape statistic text, unscaled variance)
FAMILIES = {
    "gamma": ("Gamma, centred", "k", (16, 8, 4, 2, 1),
              lambda k: f"skew {2 / np.sqrt(k):.2f}", lambda k: k),
    "bimodal": ("Bimodal ½N(±a, 1)", "a", (1, 1.5, 2, 2.5, 3),
                lambda m: f"kurt {(m**4 + 6*m**2 + 3) / (1 + m**2)**2 - 3:+.2f}", lambda m: 1 + m ** 2),
    "contam": ("Contaminated 0.9N(0,1)+0.1N(0,c²)", "c", (2, 3, 5, 7, 10),
               lambda c: f"kurt {3 * (0.9 + 0.1*c**4) / (0.9 + 0.1*c**2)**2 - 3:+.1f}", lambda c: 0.9 + 0.1 * c ** 2),
    "lognorm": ("Log-normal, centred", "s", (0.25, 0.5, 0.75, 1.0, 1.25),
                lambda s: f"skew {lognorm_stats(s)[0]:.1f}", lambda s: lognorm_stats(s)[1]),
    "gennorm": ("Generalized normal (β=1: Laplace)", "β", (1.5, 1.25, 1.0, 0.75, 0.6),
                lambda b: f"kurt {G(5/b) * G(1/b) / G(3/b)**2 - 3:+.1f}", lambda b: G(3 / b) / G(1 / b)),
}


def ratio_ci(num, den):
    m = np.isfinite(num) & np.isfinite(den)
    num, den = num[m], den[m]
    idx = rng.integers(0, len(num), (a.boot, len(num)))
    b = np.median(num[idx], axis=1) / np.median(den[idx], axis=1)
    return np.median(num) / np.median(den), *np.quantile(b, [0.025, 0.975])


summary = []
for scale in ("scaled", "unscaled"):
    f = os.path.join(a.res, scale, "per_dataset.csv")
    if not os.path.exists(f):
        continue
    df = pd.read_csv(f)
    for metric in ("mse_cate", "rel_ate"):
        for fam, (title, sym, levels, stat, var) in FAMILIES.items():
            for case in CASES:
                base = df[(df.case == case) & (df.dial == "in_prior")].sort_values("r")[metric].values
                for target in ("all", "outcome"):
                    for lev in levels:
                        cur = df[(df.case == case) & (df.dial == fam) & (df.level == float(lev))
                                 & (df.target == target)].sort_values("r")[metric].values
                        if len(cur) == len(base) and len(cur):
                            v, lo, hi = ratio_ci(cur, base)
                            summary.append(dict(scale=scale, metric=metric, family=fam, level=lev, case=case,
                                                target=target, rho=v, lo=lo, hi=hi))
summ = pd.DataFrame(summary)
summ.to_csv(os.path.join(a.res, "rho_summary.csv"), index=False)

for scale in ("scaled", "unscaled"):
    s0 = summ[(summ.scale == scale) & (summ.metric == "mse_cate")] if len(summ) else summ
    if s0.empty:
        continue
    fig, grid = plt.subplots(2, 3, figsize=(15, 8.4), sharey=True)
    axes, legend_ax = grid.ravel()[:5], grid.ravel()[5]
    for ax, (fam, (title, sym, levels, stat, var)) in zip(axes, FAMILIES.items()):
        if scale == "unscaled":   # Gaussian (variance 1) first, then increasing variance
            levels = tuple(sorted(levels, key=var))
        for case, color in CASES.items():
            for target, ls in (("all", "-"), ("outcome", "--")):
                s = s0[(s0.family == fam) & (s0.case == case) & (s0.target == target)]
                xs, ys, los, his = [0], [1.0], [1.0], [1.0]
                for i, lev in enumerate(levels, start=1):
                    r = s[s.level == lev]
                    if len(r):
                        xs.append(i); ys.append(r.rho.iloc[0]); los.append(r.lo.iloc[0]); his.append(r.hi.iloc[0])
                ax.fill_between(xs, los, his, color=color, alpha=0.10 if ls == "-" else 0.06, lw=0)
                ax.plot(xs, ys, ls, color=color, lw=2, marker="o", ms=5, mec="white", mew=1)
        ax.axhline(1, color=RED, lw=1.5)
        ticks = ["Gaussian"] + [f"{sym}={lev:g}\n" + (stat(lev) if scale == "scaled" else f"var {var(lev):.3g}")
                                for lev in levels]
        ax.set_xticks(range(len(ticks)), ticks, fontsize=8)
        ax.set_yscale("log")
        ax.grid(axis="y", color=GRID, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("shape parameter (further from Gaussian →)" if scale == "scaled"
                      else "shape parameter, ordered by noise variance (increasing →)", fontsize=9, color=INK)
    # plain-number y ticks (shared y across panels)
    lo_all = min(s0.lo.min(), 1.0) * 0.95
    hi_all = max(s0.hi.max(), 1.0) * 1.05
    yt = [t for t in (0.5, 0.6, 0.75, 0.9, 1, 1.1, 1.2, 1.3, 1.5, 2, 3, 4, 5, 6, 8, 10, 15)
          if lo_all <= t <= hi_all]
    if scale == "unscaled":
        yt = [t for t in yt if t not in (0.6, 0.9, 1.1, 1.2, 1.3)]
    for ax in axes:
        ax.set_ylim(lo_all, hi_all)
        ax.yaxis.set_major_locator(FixedLocator(yt))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels([f"{t:g}" for t in yt])
    for ax in grid[:, 0]:
        ax.set_ylabel("ρ = median MSE (CATE) / Gaussian", fontsize=10, color=INK)
    handles = [Line2D([], [], color=c, lw=2, label=k.replace("_", " ")) for k, c in CASES.items()]
    handles += [Line2D([], [], color=MUTED, lw=2, ls="-", label="noise on all nodes"),
                Line2D([], [], color=MUTED, lw=2, ls="--", label="noise on outcome only"),
                Line2D([], [], color=RED, lw=1.5, label="Gaussian baseline (ρ = 1)")]
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="center", frameon=False, fontsize=10.5,
                     title=f"{scale}: {'variance fixed at the in-prior level' if scale == 'scaled' else 'textbook form, variance changes'}",
                     title_fontsize=10)
    fig.tight_layout()
    out = os.path.join(a.res, f"exp_B_shapes_{scale}.png")
    fig.savefig(out, dpi=200, facecolor="white")
    print(out)
