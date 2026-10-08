#!/usr/bin/env python
"""PEHE or relative ATE error in one 2 x 6 figure: case studies on the top row, ComplexMech on the bottom.

Top row    = the six case studies, pooled over d and shift: the PEHE / epsilon_ATE cells of Table
             tab:cs-point, read from the paper source so the figure shows the reported numbers.
Bottom row = ComplexMech (all rho) per node count, from cm_point_allrho.csv (the same
             numbers as the ComplexMech point table).
Bars, colours and model order are those of plot_fig3_dotgrid.py.

    python benchmarks/plots/plot_point_cs_cm.py --tex revision/paper/iclr2027/iclr2027_conference.tex \
        --cm-csv revision/results/d262/cm_point_allrho.csv --metric pehe --out pehe_cs_cm.png
"""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../case_study/cluster"))
from plot_fig3_dotgrid import _CASES, C_1D, C_2D, EDGE, _bar_ypos  # noqa: E402

# label in the paper tables, ComplexMech csv key
MODELS = [("Do-PFN", "dopfn_native"), ("Do-PFN 2D", "dopfn_repro_joint2d_262k"),
          ("UWYK No-Anc", "uwyk1d-noanc"), ("UWYK No-Anc 2D", "graph2d-noanc"),
          ("UWYK Anc", "uwyk1d-v3a"), ("UWYK Anc 2D", "graph2d-v3a"),
          ("CausalPFN-C", "cpfn1d_j1024"), ("CausalPFN-C 2D", "cpfn2d_eta0")]
NODES = [5, 10, 20, 30, 40, 50]
YP = _bar_ypos(len(MODELS) // 2)
# metric: (offset of its cell within each case pair in tab:cs-point, csv mean, csv SE, axis label)
METRICS = {"pehe": (0, "pehe", "pehe_sem", "PEHE"), "eps": (1, "eps_ate", "eps_sem", "Relative ATE error")}


def cs_from_tex(path, off):
    """{model label: [(value, se) per case]} from the tab:cs-point rows (PEHE then epsilon_ATE per case)."""
    lines = open(path).read().split("\n")
    a = next(i for i, l in enumerate(lines) if l.startswith("\\label{tab:cs-point}"))
    b = next(i for i in range(a, len(lines)) if "\\end{tabular}" in lines[i])
    out = {}
    for l in lines[a:b]:
        c = [x.strip() for x in l.rstrip("\\ ").split("&")]
        if c[0] in dict(MODELS):
            cells = [re.search(r"([\d.]+)\}?\\,\{\\scriptsize\$\\pm\$([\d.]+)", x) for x in c[1:]]
            out[c[0]] = [(float(m.group(1)), float(m.group(2))) for m in cells[off:12:2]]
    missing = [m for m, _ in MODELS if m not in out]
    if missing:
        raise SystemExit(f"tab:cs-point has no row for {missing}")
    return out


def draw(ax, vals):
    for i, (mu, se) in enumerate(vals):
        ax.barh(YP[i], mu, xerr=se, height=0.85, color=(C_2D if i % 2 else C_1D),
                edgecolor=EDGE, linewidth=0.7, ecolor=EDGE, capsize=2,
                error_kw=dict(lw=0.9), zorder=3)
    ax.set_yticks(YP)
    ax.set_ylim(YP.min() - 0.8, YP.max() + 0.8)
    ax.grid(axis="x", ls=":", alpha=0.5, zorder=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", required=True)
    ap.add_argument("--cm-csv", required=True)
    ap.add_argument("--metric", choices=list(METRICS), default="pehe")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    off, col, sem, xlab = METRICS[a.metric]
    cs = cs_from_tex(a.tex, off)
    cm = pd.read_csv(a.cm_csv)
    cm = cm[cm["context"] == 1000].set_index(["model", "nodes"])

    fig, axes = plt.subplots(2, 6, figsize=(16.2, 4.6), sharey=True)
    for j, (_, clab) in enumerate(_CASES):
        ax = axes[0][j]
        draw(ax, [cs[m][j] for m, _ in MODELS])
        ax.set_title(clab, fontsize=9)
        ax.set_xlabel(xlab, fontsize=8)
    for j, n in enumerate(NODES):
        ax = axes[1][j]
        draw(ax, [(float(cm.at[(k, n), col]), float(cm.at[(k, n), sem])) for _, k in MODELS])
        ax.set_title(f"nodes = {n}", fontsize=9)
        ax.set_xlabel(xlab, fontsize=8)
    for r, name in enumerate(("Case studies", "ComplexMech")):
        axes[r][0].set_yticklabels([m for m, _ in MODELS], fontsize=7)
        axes[r][0].set_ylabel(name, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
