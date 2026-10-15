#!/usr/bin/env python
"""Point-estimation figures in the style of plot_point_cs_cm.py (horizontal bars, mean +- SE, the 1D / 2D
colours of plot_fig3_dotgrid.py), one figure per metric.

  rho   : controlled-correlation benchmarks (revision/results/d262t1/rho_ctrl.csv). Rows = the three rho bins,
          columns = the four pre-treatment case studies and ComplexMech (n = 5); bars = the baseline / 2D pairs
          (plus CausalPFN-C at matched resolution once it is in the csv).
  ablation : matched-label / matched-resolution ablations (revision/results/d262t1/ablation_point.csv). Rows =
          RealCause, case studies, ComplexMech; bars per family: matched label, matched resolution, 2D.

    python benchmarks/plots/plot_point_rho_ablation.py rho      --csv revision/results/d262t1/rho_ctrl.csv --metric pehe --out X.png
    python benchmarks/plots/plot_point_rho_ablation.py ablation --csv revision/results/d262t1/ablation_point.csv --metric eps --out Y.png
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../case_study/cluster"))
from plot_fig3_dotgrid import C_1D, C_2D, EDGE  # noqa: E402

XLAB = {"pehe": "PEHE", "eps": "Relative ATE error"}


def ypos(groups):
    """Top-to-bottom y positions: bars of one family adjacent, a gap between families."""
    yp, y = [], 0.0
    for g in groups:
        for _ in range(g):
            yp.append(y); y += 1.0
        y += 0.7
    yp = np.asarray(yp)
    return yp.max() - yp


def draw(ax, vals, colors, hatches, yp):
    for i, (mu, se) in enumerate(vals):
        if not np.isfinite(mu):
            continue
        ax.barh(yp[i], mu, xerr=se, height=0.85, color=colors[i], hatch=hatches[i], edgecolor=EDGE, linewidth=0.7,
                ecolor=EDGE, capsize=2, error_kw=dict(lw=0.9), zorder=3)
    ax.set_yticks(yp)
    ax.set_ylim(yp.min() - 0.8, yp.max() + 0.8)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.grid(axis="x", ls=":", alpha=0.5, zorder=0)


def rho_fig(path, metric, out):
    r = list(csv.DictReader(open(path)))
    have = {x["model"] for x in r}
    fams = [["Do-PFN", "Do-PFN 2D"], ["UWYK No-Anc", "UWYK 2D No-Anc"], ["UWYK Anc", "UWYK 2D Anc"],
            ["CausalPFN-C"] + (["CausalPFN-C (matched resolution)"] if "CausalPFN-C (matched resolution)" in have else [])
            + ["CausalPFN-C 2D"]]
    show = {"UWYK 2D No-Anc": "UWYK No-Anc 2D", "UWYK 2D Anc": "UWYK Anc 2D",
            "CausalPFN-C (matched resolution)": "CausalPFN-C (matched res.)"}
    models = [m for f in fams for m in f]
    colors = [C_2D if "2D" in m else C_1D for m in models]
    hatches = ["//" if "matched" in m else None for m in models]
    yp = ypos([len(f) for f in fams])
    benches = [("Observed Confounder", "Observed\nConfounder"), ("Backdoor Criterion", "Back-Door\nCriterion"),
               ("Frontdoor Criterion", "Front-Door\nCriterion"), ("Unobserved Confounder", "Unobserved\nConfounder"),
               ("ComplexMech n=5", "ComplexMech\n(nodes = 5)")]
    bins = [("[-0.5, 0)", r"$\rho\in[-0.5,0)$"), ("[0, 0.5)", r"$\rho\in[0,0.5)$"), ("[0.5, 1]", r"$\rho\in[0.5,1]$")]
    col, se = ("sqrt_pehe", "sqrt_pehe_se") if metric == "pehe" else ("eps_ate", "eps_ate_se")
    fig, axes = plt.subplots(3, 5, figsize=(14.5, 7.4), sharey=True)
    for i, (bk, blab) in enumerate(bins):
        for j, (bench, title) in enumerate(benches):
            ax = axes[i][j]
            vals = []
            for m in models:
                x = [y for y in r if y["benchmark"] == bench and y["model"] == m and y["bin"] == bk]
                vals.append((float(x[0][col]), float(x[0][se])) if x else (np.nan, np.nan))
            draw(ax, vals, colors, hatches, yp)
            if i == 0:
                ax.set_title(title, fontsize=9)
            if i == 2:
                ax.set_xlabel(XLAB[metric], fontsize=8)
        axes[i][0].set_yticklabels([show.get(m, m) for m in models], fontsize=7)
        axes[i][0].set_ylabel(blab, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


def ablation_fig(path, metric, out):
    r = list(csv.DictReader(open(path)))
    fams = ["Do-PFN", "UWYK No-Anc", "UWYK Anc", "CausalPFN-C"]
    variants = [("matched label", "(matched label)"), ("matched resolution", "(matched res.)"), ("2D", "2D")]
    labels = [f"{f} {v[1]}" for f in fams for v in variants]
    colors = [C_2D if v[0] == "2D" else C_1D for f in fams for v in variants]
    hatches = ["//" if v[0] == "matched resolution" else None for f in fams for v in variants]
    yp = ypos([3] * len(fams))
    rows = [("RealCause", ["IHDP", "ACIC", "CPS", "PSID", "PSID_bal"], ["IHDP", "ACIC", "CPS", "PSID", "PSID (balanced)"]),
            ("CaseStudy", ["Observed_Confounder", "Observed_Mediator", "Observed_Mediator_and_Confounder",
                           "Unobserved_Confounder", "Frontdoor_Criterion", "Backdoor_Criterion"],
             ["Observed\nConfounder", "Observed\nMediator", "Confounder +\nMediator", "Unobserved\nConfounder",
              "Front-Door\nCriterion", "Back-Door\nCriterion"]),
            ("ComplexMech", ["5", "10", "20", "30", "40", "50"], [f"nodes = {n}" for n in (5, 10, 20, 30, 40, 50)])]
    col, se = ("pehe", "pehe_se") if metric == "pehe" else ("eps_ate", "eps_se")
    fig, axes = plt.subplots(3, 6, figsize=(16.2, 9.0), sharey=True)
    for i, (bench, settings, titles) in enumerate(rows):
        for j in range(6):
            ax = axes[i][j]
            if j >= len(settings):
                ax.axis("off")
                continue
            vals = []
            for f in fams:
                for v, _ in variants:
                    x = [y for y in r if y["benchmark"] == bench and y["setting"] == settings[j]
                         and y["model"] == f and y["variant"] == v]
                    vals.append((float(x[0][col]), float(x[0][se])) if x else (np.nan, np.nan))
            draw(ax, vals, colors, hatches, yp)
            ax.set_title(titles[j], fontsize=9)
            ax.set_xlabel(XLAB[metric], fontsize=8)
        axes[i][0].set_yticklabels(labels, fontsize=7)
        axes[i][0].set_ylabel({"CaseStudy": "Case studies"}.get(bench, bench), fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["rho", "ablation"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--metric", choices=["pehe", "eps"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    (rho_fig if a.which == "rho" else ablation_fig)(a.csv, a.metric, a.out)
