"""Jittered Twins, d = 10, MSE: 2D (conditioned on the known twin) against 1D conditioned through a Gaussian copula at
every rho of the analyst's range [0, 1] (step 0.1), and against the 1D average over that range; averaged over
w = 0.1, 0.25, 0.5. One panel per family: the 1D MSE over the assumed rho (+- SE band), its average (dashed) and the 2D
model (+- SE band); the paper's colours (1D blue, green at matched resolution; 2D orange). The significance of 2D
against the 1D average is printed, not drawn.
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
import matplotlib.ticker                                         # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper"))
import paper_style as PS                                         # noqa: E402

FAM = [("Do-PFN", "Do-PFN 2D", "Do-PFN"), ("UWYK No-Anc", "UWYK 2D No-Anc", "UWYK No-Anc"),
       ("UWYK Anc", "UWYK 2D Anc", "UWYK Anc"), ("CausalPFN-C", "CausalPFN-C 2D", "CausalPFN-C")]
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
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 12.5,
                         "ytick.labelsize": 12.5, "legend.fontsize": 13.5})
    fig, axes = plt.subplots(1, len(res), figsize=(3.2 * len(res), 3.3), squeeze=False)
    for ax, (lab, g, a, t, p, stars) in zip(axes[0], res):
        m1, s1 = [m for m, _ in g], [s for _, s in g]
        ax.plot(RHOS, m1, "-o", ms=6, lw=2.4, color=col1, label="1D, assumed ρ")
        ax.fill_between(RHOS, [x - e for x, e in zip(m1, s1)], [x + e for x, e in zip(m1, s1)], color=col1, alpha=0.2, lw=0)
        ax.axhline(a[0], color=col1, ls="--", lw=2.2, label="1D, average over ρ ∈ [0, 1]")
        ax.axhline(t[0], color=PS.COL_2D, lw=3.0, label="2D")
        ax.axhspan(t[0] - t[1], t[0] + t[1], color=PS.COL_2D, alpha=0.15, lw=0)
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_title(lab)
        ax.set_xlabel("ρ")
    axes[0][0].set_ylabel("MSE")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {out}.pdf/.png")


if __name__ == "__main__":
    mr = sys.argv[1] == "--mr"
    a = sys.argv[2:] if mr else sys.argv[1:]
    main(a[0], a[1:], mr)
