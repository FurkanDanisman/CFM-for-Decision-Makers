"""Jittered Twins: 2D (conditioned on the known twin) against 1D conditioned through a Gaussian copula at every rho of
the analyst's range [0, 1] (step 0.1), and against the 1D average over that range; MSE and IS, averaged over
w = 0.1, 0.25, 0.5.
python plot_twins_rho_grid.py [--mr] OUT_PREFIX CSV [CSV ...]   (the score_twins_deq CSVs: the 1D grid runs and the 2D
runs; --mr: the 1D baselines at matched resolution)

Averages: over w, a model's mean is the average of its per-w means and its SE is bounded by the average of its per-w
SEs; over the rho grid, the same (the average of the 11 rho means, SE bounded by the average of their SEs). Both bounds
hold under any dependence across w and rho. Significance (the paper's test, Appendix F): 2D against the 1D average,
two-sided z-test with SE = sqrt(SE_2D^2 + SE_1D^2), Hochberg within each model and metric over d = 10 and d = 1;
* at 0.1, ** at 0.05.
"""
import csv
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

PAIRS = [("Do-PFN", "Do-PFN 2D"), ("CausalPFN-C", "CausalPFN-C 2D"),
         ("UWYK No-Anc", "UWYK 2D No-Anc"), ("UWYK Anc", "UWYK 2D Anc")]
RHOS = [k / 10 for k in range(11)]
WS = ("0.1", "0.25", "0.5")
DS = ("d=10", "d=1")
METRICS = [("mse", "MSE"), ("IS", "Interval score")]
mean = lambda x: sum(x) / len(x)


def hochberg(p):
    M = len(p)
    out = [0] * M
    order = sorted(range(M), key=lambda i: p[i])
    for a in (0.1, 0.05):
        ks = [k for k in range(1, M + 1) if p[order[k - 1]] <= a / (M - k + 1)]
        for i in (order[:max(ks)] if ks else []):
            out[i] += 1
    return out


def load(paths):
    rows = {}
    for path in paths:
        for r in csv.DictReader(open(path)):
            if r["variant"] == "twins":
                rows[(r["w"], r["setting"], r["model"])] = r
    return rows


def over_w(rows, d, model, metric):
    """(mean, SE bound) averaged over w; None if a w is missing"""
    c = [rows.get((w, d, model)) for w in WS]
    if any(r is None for r in c):
        return None
    return mean([float(r[metric]) for r in c]), mean([float(r[metric + "_se"]) for r in c])


def main(out, paths):
    rows = load(paths)
    res = {}
    for one, two in PAIRS:
        for d in DS:
            for metric, _ in METRICS:
                two_d = over_w(rows, d, two, metric)
                grid = [over_w(rows, d, f"{one} copula ρ={r}", metric) for r in RHOS]
                if two_d is None or any(g is None for g in grid):
                    continue
                avg = (mean([m for m, _ in grid]), mean([s for _, s in grid]))
                z = (two_d[0] - avg[0]) / math.sqrt(two_d[1] ** 2 + avg[1] ** 2)
                res[(one, d, metric)] = dict(two=two_d, grid=grid, avg=avg, p=math.erfc(abs(z) / math.sqrt(2)))
    for one, _ in PAIRS:
        for metric, _ in METRICS:
            keys = [(one, d, metric) for d in DS if (one, d, metric) in res]
            for k, n in zip(keys, hochberg([res[k]["p"] for k in keys])):
                res[k]["stars"] = n
    print(f"{'model':12s} {'d':>4s} {'metric':>6s} {'2D':>14s} {'1D avg ρ∈[0,1]':>16s}  p        2D vs 1D avg")
    for (one, d, metric), r in res.items():
        verdict = ("2D lower" if r["two"][0] < r["avg"][0] else "2D higher") + " " + "*" * r["stars"] if r["stars"] \
            else "not significant"
        print(f"{one:12s} {d[2:]:>4s} {metric:>6s} {r['two'][0]:7.3f} ± {r['two'][1]:.3f} {r['avg'][0]:8.3f} ± "
              f"{r['avg'][1]:.3f}  {r['p']:.1e}  {verdict}")
    for d in DS:
        fig, axes = plt.subplots(2, len(PAIRS), figsize=(3.2 * len(PAIRS), 5.4), squeeze=False)
        for j, (one, _) in enumerate(PAIRS):
            for i, (metric, mlabel) in enumerate(METRICS):
                ax = axes[i][j]
                r = res.get((one, d, metric))
                if r is None:
                    ax.set_axis_off(); continue
                m1 = [m for m, _ in r["grid"]]; s1 = [s for _, s in r["grid"]]
                ax.plot(RHOS, m1, "-o", ms=3, color="C0", label="1D, assumed ρ")
                ax.fill_between(RHOS, [a - b for a, b in zip(m1, s1)], [a + b for a, b in zip(m1, s1)],
                                color="C0", alpha=0.2, lw=0)
                ax.axhline(r["avg"][0], color="C0", ls="--", lw=1.2, label="1D, average over ρ ∈ [0, 1]")
                ax.axhline(r["two"][0], color="C3", lw=1.6, label="2D")
                ax.axhspan(r["two"][0] - r["two"][1], r["two"][0] + r["two"][1], color="C3", alpha=0.15, lw=0)
                if r["stars"]:
                    ax.text(0.97, 0.95, "*" * r["stars"], transform=ax.transAxes, ha="right", va="top", fontsize=14)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                if i == 0:
                    ax.set_title(one.replace(" (matched resolution)", "\n(1D at matched resolution)"))
                if j == 0:
                    ax.set_ylabel(mlabel)
                if i == 1:
                    ax.set_xlabel("assumed ρ (1D)")
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"Twins (jittered, averaged over w = 0.1, 0.25, 0.5), {d.replace('=', ' = ')}", y=0.99)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        for ext in ("png", "pdf"):
            fig.savefig(f"{out}_{d.replace('=', '')}.{ext}", dpi=200, bbox_inches="tight")
        print(f"wrote {out}_{d.replace('=', '')}.png/.pdf")


if __name__ == "__main__":
    if sys.argv[1] == "--mr":
        PAIRS[:] = [(f"{one} (matched resolution)", two) for one, two in PAIRS]
        main(sys.argv[2], sys.argv[3:])
    else:
        main(sys.argv[1], sys.argv[2:])
