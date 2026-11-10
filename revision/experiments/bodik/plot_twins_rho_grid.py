"""Jittered Twins, d = 10, MSE: 2D (conditioned on the known twin) against 1D conditioned through a Gaussian copula at
every rho of the analyst's range [0, 1] (step 0.1), and against the 1D average over that range; averaged over
w = 0.1, 0.25, 0.5. In the paper's figure style (revision/paper/paper_style.py): one row per family, horizontal bars
(1D blue, or green at matched resolution; 2D orange; mean +- SE), the average in a grey panel with its value printed,
the lower one in bold and the stars on the 1D bar.
python plot_twins_rho_grid.py [--mr] OUT_PREFIX CSV [CSV ...]   (the score_twins_deq CSVs: the 1D grid runs and the 2D
runs; --mr: the 1D baselines at matched resolution)

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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper"))
import paper_style as PS                                         # noqa: E402

FAM = [("Do-PFN", "Do-PFN 2D", "Do-PFN"), ("UWYK No-Anc", "UWYK 2D No-Anc", "UWYK\nNo-Anc"),
       ("UWYK Anc", "UWYK 2D Anc", "UWYK\nAnc"), ("CausalPFN-C", "CausalPFN-C 2D", "Causal-\nPFN-C")]
RHOS = [k / 10 for k in range(11)]
WS = ("0.1", "0.25", "0.5")
D = "d=10"
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
    return mean([float(r["mse"]) for r in c]), mean([float(r["mse_se"]) for r in c])


def main(out, paths, mr):
    PS.apply()
    rows = load(paths)
    col1 = PS.COL_RES if mr else PS.COL_1D
    res = []
    for one, two, lab in FAM:
        one = f"{one} (matched resolution)" if mr else one
        t = over_w(rows, two)
        g = [over_w(rows, f"{one} copula ρ={r}") for r in RHOS]
        if t is None or any(x is None for x in g):
            print("missing", one); continue
        a = (mean([m for m, _ in g]), mean([s for _, s in g]))
        p = math.erfc(abs(t[0] - a[0]) / math.hypot(t[1], a[1]) / math.sqrt(2))
        stars = 2 if p < 0.05 else 1 if p < 0.1 else 0
        res.append((lab, g, a, t, p, stars))
        print(f"{one:34s} 2D {t[0]:.3f} ± {t[1]:.3f}   1D avg {a[0]:.3f} ± {a[1]:.3f}   p {p:.2g}   "
              + (("2D lower " if t[0] < a[0] else "2D higher ") + "*" * stars if stars else "not significant"))
    nser = len(RHOS) + 1
    row_h = 0.125 * nser + 0.36
    fig = plt.figure(figsize=(PS.TEXTWIDTH * 0.7, row_h * len(res) + 0.55))
    gs = fig.add_gridspec(len(res), 2, wspace=0.3, hspace=0.5, width_ratios=[1, 1.6])
    ylab = [r"$\rho=0$" if r == 0 else r"$\rho=0.5$" if r == 0.5 else r"$\rho=1$" if r == 1 else "" for r in RHOS] + ["2D"]
    for i, (lab, g, a, t, p, stars) in enumerate(res):
        ax = fig.add_subplot(gs[i, 0])
        y = PS.hbars(ax, [m for m, _ in g] + [t[0]], [s for _, s in g] + [t[1]], [col1] * len(RHOS) + [PS.COL_2D],
                     height=0.8, compact=True)
        ax.set_yticks(y)
        ax.set_yticklabels(ylab)
        ax.tick_params(axis="x", labelsize=PS.TICK - 0.5)
        ax.set_ylabel(lab, fontweight="bold", labelpad=3, linespacing=1.0)
        if i == 0:
            ax.set_title(r"assumed $\rho$ of 1D, and 2D", pad=3)
        ax2 = fig.add_subplot(gs[i, 1])
        PS.avg_panel(ax2, [a[0], t[0]], [a[1], t[1]], [col1, PS.COL_2D], stars=[stars, 0], bold_min=True)
        if i == 0:
            ax2.set_title(r"$\mathbf{Average}$ over $\rho\in[0,1]$", pad=3)
    fig.supxlabel("Mean squared error", fontsize=PS.LABEL, y=0.005)
    fig.subplots_adjust(left=0.2, right=0.985, top=1 - 0.35 / fig.get_figheight(), bottom=0.45 / fig.get_figheight())
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {out}.pdf/.png")


if __name__ == "__main__":
    mr = sys.argv[1] == "--mr"
    a = sys.argv[2:] if mr else sys.argv[1:]
    main(a[0], a[1:], mr)
