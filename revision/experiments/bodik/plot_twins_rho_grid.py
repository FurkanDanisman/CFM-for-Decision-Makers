"""Jittered Twins, d = 10, MSE: 2D (conditioned on the known twin) against 1D conditioned through a Gaussian copula at
every rho of the analyst's range [0, 1] (step 0.1), and against the 1D average over that range; averaged over
w = 0.1, 0.25, 0.5. One panel per family: the 1D MSE over the assumed rho (+- SE band), its average (dashed) and the 2D
model (+- SE band); the paper's figure style (revision/paper/paper_style.py: STIX fonts, text width, 1D blue or green at
matched resolution, 2D orange, dotted top/right frame), the stars on the 1D average.
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
    fig, axes = plt.subplots(1, len(res), figsize=(PS.TEXTWIDTH, 1.95), gridspec_kw=dict(wspace=0.42))
    for ax, (lab, g, a, t, p, stars) in zip(axes, res):
        m1, s1 = [m for m, _ in g], [s for _, s in g]
        ax.fill_between(RHOS, [x - e for x, e in zip(m1, s1)], [x + e for x, e in zip(m1, s1)], color=col1, alpha=0.25, lw=0)
        ax.plot(RHOS, m1, "-o", color=col1, lw=1.2, ms=2.5, label=r"1D, assumed $\rho$")
        ax.axhline(a[0], color=col1, lw=1.0, ls=(0, (3, 1.5)), label=r"1D, average over $\rho\in[0,1]$")
        ax.axhspan(t[0] - t[1], t[0] + t[1], color=PS.COL_2D, alpha=0.25, lw=0)
        ax.axhline(t[0], color=PS.COL_2D, lw=1.6, label="2D")
        if stars:                                                # on the 1D average, as the paper's stars on the 1D bar
            ax.text(1.0, a[0], "$^{" + "*" * stars + "}$", ha="right", va="bottom", fontsize=PS.LABEL)
        ax.set_title(lab.replace("\n", " ").replace("Causal- PFN-C", "CausalPFN-C"), pad=3)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(PS.fmt))
        PS.frame(ax)
        ax.tick_params(axis="y", length=2, width=0.5, pad=1.5)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, steps=[1, 2, 5]))   # few round ticks
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(PS.fmt))
    axes[0].set_ylabel("Mean squared error")
    fig.supxlabel(r"assumed $\rho$ of 1D", fontsize=PS.LABEL, y=0.08)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.13), handlelength=1.8,
               columnspacing=1.2)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.31)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {out}.pdf/.png")


if __name__ == "__main__":
    mr = sys.argv[1] == "--mr"
    a = sys.argv[2:] if mr else sys.argv[1:]
    main(a[0], a[1:], mr)
