"""Jittered Twins, d = 10, MSE: 2D (conditioned on the known twin) against 1D conditioned through a Gaussian copula at
every rho of the analyst's range [0, 1] (step 0.1), and against the 1D average over that range; averaged over
w = 0.1, 0.25, 0.5. One panel per family: the 1D MSE over the assumed rho (+- SE band), its average (dashed) and the 2D
model (+- SE band); the paper's colours (1D blue, green at matched resolution; 2D orange). The significance of 2D
against the 1D average is printed, not drawn.
python plot_twins_rho_grid.py [--is] [--no1] [--range] [--full | --mr | --ml] OUT_PREFIX CSV [CSV ...]   (the score_twins_deq CSVs: the 1D grid runs and the 2D
runs). By default every 1D baseline found is drawn: full resolution (blue), matched resolution (green), matched label
(purple); --full / --mr / --ml: that one only)

Averages: over w, a model's mean is the average of its per-w means and its SE is bounded by the average of its per-w
SEs; over the rho grid, the same (the average of the 11 rho means, SE bounded by the average of their SEs). Both bounds
hold under any dependence across w and rho. Significance (the paper's test, Appendix F): 2D against the 1D average,
two-sided z-test with SE = sqrt(SE_2D^2 + SE_1D^2), one test per family (d = 10 only, so no correction is needed);
* at 0.1, ** at 0.05.
"""
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import matplotlib.ticker                                         # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper"))
import paper_style as PS                                         # noqa: E402

FAM = [("Do-PFN", "Do-PFN 2D", "Do-PFN"), ("UWYK No-Anc", "UWYK 2D No-Anc", "UWYK No-Anc"),
       ("UWYK Anc", "UWYK 2D Anc", "UWYK Anc"), ("CausalPFN-C", "CausalPFN-C 2D", "CausalPFN-C")]
RHOS = [k / 10 for k in range(11)]
WS = ("0.1", "0.25", "0.5")
D = "d=10"
METRIC = "mse"                                                   # --is: the interval score instead
mean = lambda x: sum(x) / len(x)


def load(paths):
    rows = {}
    for path in paths:
        for r in csv.DictReader(open(path)):
            if r["variant"] == "twins":
                rows[(r["w"], r["setting"], r["model"])] = r
    return rows


def over_w(rows, model):
    """(mean, SE bound) of the MSE averaged over w; None if a w is missing"""
    c = [rows.get((w, D, model)) for w in WS]
    if any(r is None for r in c):
        return None
    return mean([float(r[METRIC]) for r in c]), mean([float(r[METRIC + "_se"]) for r in c])


SERIES = [("", PS.COL_1D), (" (matched resolution)", PS.COL_RES), (" (matched label)", PS.COL_LAB)]


def main(out, paths, series):
    """series: the 1D baselines drawn, as (label suffix, colour); a baseline missing from the CSVs is skipped"""
    rows = load(paths)
    res = []
    for one, two, lab in FAM:
        t = over_w(rows, two)
        curves = []
        for suffix, col in series:
            g = [over_w(rows, f"{one}{suffix} copula ρ={r}") for r in RHOS]
            if t is None or any(x is None for x in g):
                continue
            a = (mean([m for m, _ in g]), mean([s for _, s in g]))
            p = math.erfc(abs(t[0] - a[0]) / math.hypot(t[1], a[1]) / math.sqrt(2))
            stars = 2 if p < 0.05 else 1 if p < 0.1 else 0
            curves.append((g, a, col))
            print(f"{one + suffix:34s} 2D {t[0]:.3f} ± {t[1]:.3f}   1D avg {a[0]:.3f} ± {a[1]:.3f}   p {p:.2g}   "
                  + (("2D lower " if t[0] < a[0] else "2D higher ") + "*" * stars if stars else "not significant"))
        if curves:
            res.append((lab, curves, t))
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 12.5,
                         "ytick.labelsize": 12.5, "legend.fontsize": 13.5})
    fig, axes = plt.subplots(1, len(res), figsize=(3.2 * len(res), 3.3), squeeze=False)
    for ax, (lab, curves, t) in zip(axes[0], res):
        for g, a, col in curves:
            m1, s1 = [m for m, _ in g], [s for _, s in g]
            ax.plot(RHOS, m1, "-o", ms=6, lw=2.4, color=col)
            ax.fill_between(RHOS, [x - e for x, e in zip(m1, s1)], [x + e for x, e in zip(m1, s1)], color=col, alpha=0.2, lw=0)
            ax.axhline(a[0], color=col, ls="--", lw=2.2)
        ax.axhline(t[0], color=PS.COL_2D, lw=3.0)
        ax.axhspan(t[0] - t[1], t[0] + t[1], color=PS.COL_2D, alpha=0.15, lw=0)
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_title(lab)
        ax.set_xlabel("ρ")
    axes[0][0].set_ylabel("MSE" if METRIC == "mse" else "Interval score")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {out}.pdf/.png")


RANGES = {"0.1": (-4, 8), "0.25": (-7, 9), "0.5": (-9, 10)}     # = score_twins_deq.RANGES (tenths)


def main_range(out, paths, series):
    """the observed-data range of rho (score_twins_deq ... range): the 1D curve over the rho common to every w,
    its average MSE over each w's own range (dashed),
    each averaged over w; 2D (orange)"""
    rows = load(paths)
    common = [k / 10 for k in range(max(r[0] for r in RANGES.values()), min(r[1] for r in RANGES.values()) + 1)]
    res = []
    for one, two, lab in FAM:
        t = over_w(rows, two)
        curves = []
        for suffix, col in series:
            name = f"{one}{suffix}"
            g = [over_w(rows, f"{name} copula ρ={r}") for r in common]
            per_w = []
            for w in WS:
                lo, hi = RANGES[w]
                v = [rows.get((w, D, f"{name} copula ρ={k / 10}")) for k in range(lo, hi + 1)]
                per_w.append(None if any(x is None for x in v) else mean([float(x[METRIC]) for x in v]))
            ap = over_w(rows, f"{name} copula ρ=avg")
            if t is None or any(x is None for x in g) or None in per_w or ap is None:
                continue
            curves.append((g, mean(per_w), ap[0], col))
        if curves:
            res.append((lab, curves, t))
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 12.5,
                         "ytick.labelsize": 12.5})
    fig, axes = plt.subplots(1, len(res), figsize=(3.2 * len(res), 3.3), squeeze=False)
    for ax, (lab, curves, t) in zip(axes[0], res):
        for g, avg_s, avg_p, col in curves:
            m1, s1 = [m for m, _ in g], [s for _, s in g]
            ax.plot(common, m1, "-o", ms=6, lw=2.4, color=col)
            ax.fill_between(common, [x - e for x, e in zip(m1, s1)], [x + e for x, e in zip(m1, s1)], color=col, alpha=0.2, lw=0)
            ax.axhline(avg_s, color=col, ls="--", lw=2.2)
        ax.axhline(t[0], color=PS.COL_2D, lw=3.0)
        ax.axhspan(t[0] - t[1], t[0] + t[1], color=PS.COL_2D, alpha=0.15, lw=0)
        ax.set_xticks([-0.4, 0, 0.4, 0.8]); ax.set_xticklabels(["-0.4", "0", "0.4", "0.8"])
        ax.set_title(lab)
        ax.set_xlabel("ρ")
    axes[0][0].set_ylabel("MSE")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {out}.pdf/.png")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[0] == "--is":
        METRIC = "IS"; argv = argv[1:]
    if argv[0] == "--no1":                                       # the grid without rho = 1: rho in [0, 0.9]
        RHOS = RHOS[:-1]; argv = argv[1:]
    rng_mode = argv[0] == "--range"                               # the observed-data range of rho per w
    if rng_mode:
        argv = argv[1:]
    abl = argv[0][2:] if argv[0] in ("--full", "--mr", "--ml") else None   # one baseline only; default: all
    a = argv[1:] if abl else argv
    (main_range if rng_mode else main)(a[0], a[1:], {"full": SERIES[0:1], "mr": SERIES[1:2], "ml": SERIES[2:3]}.get(abl, SERIES))
