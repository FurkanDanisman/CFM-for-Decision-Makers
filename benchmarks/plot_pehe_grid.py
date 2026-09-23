#!/usr/bin/env python
"""The two PEHE grids: ComplexMech (nodes x context) and case study (case x d).

    cmech_pehe_by_d.png       rows = node count (6), cols = context N (5)
    fig3_cen3_pehe_by_d.png   rows = case (6),       cols = d (6)

Each cell is a horizontal grouped bar chart over models, x = the metric, with SEM
error bars. Horizontal because model names are long: vertical bars would need
rotated labels, which are harder to read than a left-aligned list.

One metric per figure, never two on a shared axis -- PEHE and ATE error are
different measures on different scales. Pass --metric twice to get two files.

Rows share an x axis within a row by default so the trend across the column
variable is readable; --free-x lets each cell autoscale instead.

Input is the tidy CSV from cmech_point_total.py --csv / cs_point_total.py --csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 1D light / 2D dark, as in the existing 1D-vs-2D figures. Validated there:
# dE 16.2 deutan / 14.6 tritan, normal-vision 18.8, both >= 3:1 contrast.
C_1D, C_2D = "#D97706", "#8C2F39"
_2D = ("cpfn2d", "graph2d", "joint2d", "dopfn_bb")

# The reported figure: four architecture pairs, 1D above its 2D twin, pairs
# separated by a gap so the comparison the figure exists to make is the adjacent
# one. Keys are the model names the point scorers emit.
PAIRS = [
    ("Do-PFN",         "dopfn_native",        "Do-PFN 2D",         "dopfn_repro_joint2d"),
    ("UWYK No-Anc",    "uwyk1d-noanc",        "UWYK No-Anc 2D",    "graph2d-noanc"),
    ("UWYK Anc",       "uwyk1d-v3a",          "UWYK Anc 2D",       "graph2d-v3a"),
    ("CausalPFN-C",    "cpfn1d",              "CausalPFN-C 2D",    "cpfn2d_eta0"),
]

# Short column headings; the raw case names are too long to sit above a panel.
CASE_LABEL = {
    "Observed_Confounder": "Observed\nConfounder",
    "Observed_Mediator": "Observed\nMediator",
    "Observed_Mediator_and_Confounder": "Confounder +\nMediator",
    "Unobserved_Confounder": "Unobserved\nConfounder",
    "Frontdoor_Criterion": "Front-Door\nCriterion",
    "Backdoor_Criterion": "Back-Door\nCriterion",
}
CASE_ORDER = ["Observed_Confounder", "Observed_Mediator",
              "Observed_Mediator_and_Confounder", "Unobserved_Confounder",
              "Frontdoor_Criterion", "Backdoor_Criterion"]


def is_2d(model):
    m = model.lower()
    return any(k in m for k in _2D)


def load(path, row_key, col_key):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append(r)
            except Exception:
                continue
    if not rows:
        sys.exit(f"FATAL: no rows in {path}")
    for k in (row_key, col_key, "model"):
        if k not in rows[0]:
            sys.exit(f"FATAL: {path} has no column {k!r}; has {list(rows[0])}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--kind", required=True, choices=["cmech", "case"])
    ap.add_argument("--metric", nargs="+", default=["pehe"],
                    choices=["pehe", "l1_ate", "eps_ate"],
                    help="one figure PER metric -- they are different measures "
                         "on different scales and never share an axis")
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict and order the bars; default the four pairs")
    ap.add_argument("--all-models", action="store_true",
                    help="every model in the CSV instead of the four pairs")
    ap.add_argument("--shared-x", action="store_true",
                    help="share the x axis along each row (default: per cell, "
                         "which is what the reported figure does)")
    ap.add_argument("--out", required=True,
                    help="png path for a single metric, or a prefix when several "
                         "are given (writes <prefix>_<metric>.png)")
    a = ap.parse_args()

    # Case study: rows = d, cols = case (the reported layout).
    # ComplexMech: rows = node count, cols = context N.
    row_key, col_key = ("nodes", "context") if a.kind == "cmech" else ("d", "case")
    rows = load(a.csv, row_key, col_key)
    rc = 0
    for _metric in a.metric:
        rc |= draw(a, rows, row_key, col_key, _metric)
    return rc


def draw(a, rows, row_key, col_key, metric):
    sem_key = {"pehe": "pehe_sem", "l1_ate": "l1_sem",
               "eps_ate": "eps_sem"}[metric]

    def _ord(v):
        try:
            return (0, float(v))
        except ValueError:
            return (1, str(v))

    rvals = sorted({r[row_key] for r in rows}, key=_ord)
    if col_key == "case":
        present = {r[col_key] for r in rows}
        cvals = [c for c in CASE_ORDER if c in present] + \
                sorted(present - set(CASE_ORDER))
    else:
        cvals = sorted({r[col_key] for r in rows}, key=_ord)

    # Bars: four 1D/2D pairs by default, with a gap between pairs.
    if a.all_models or a.models:
        names = a.models or sorted({r["model"] for r in rows})
        bars = [(m, m, is_2d(m)) for m in names]
        ypos = list(range(len(bars)))
    else:
        bars, ypos, y = [], [], 0.0
        for lab1, k1, lab2, k2 in PAIRS:
            bars += [(lab1, k1, False), (lab2, k2, True)]
            ypos += [y, y + 1.0]
            y += 2.6                      # 0.6 of blank between pairs
    labels = [b[0] for b in bars]

    cell = defaultdict(dict)
    for r in rows:
        try:
            cell[(r[row_key], r[col_key])][r["model"]] = (
                float(r[metric]), float(r[sem_key]))
        except (ValueError, KeyError):
            continue

    nr, nc = len(rvals), len(cvals)
    fig, axes = plt.subplots(nr, nc, figsize=(2.55 * nc, 0.30 * len(bars) * nr),
                             squeeze=False,
                             sharex=("row" if a.shared_x else "none"))
    yp = np.asarray(ypos, dtype=float)
    missing = 0
    for i, rv in enumerate(rvals):
        for j, cv in enumerate(cvals):
            ax = axes[i][j]
            d = cell.get((rv, cv), {})
            vals = [d.get(k, (np.nan, np.nan))[0] for _, k, _ in bars]
            errs = [d.get(k, (np.nan, np.nan))[1] for _, k, _ in bars]
            cols = [C_2D if two else C_1D for _, _, two in bars]
            ax.barh(yp, vals, xerr=errs, color=cols, height=0.92,
                    edgecolor="black", linewidth=0.4,
                    error_kw=dict(lw=0.7, capsize=1.6, ecolor="#303030"))
            ax.set_yticks(yp)
            ax.set_yticklabels(labels if j == 0 else [], fontsize=5.6)
            ax.set_ylim(yp.max() + 1.0, yp.min() - 1.0)   # inverted, padded
            ax.tick_params(axis="x", labelsize=5.6)
            ax.grid(axis="x", lw=0.4, alpha=0.3, ls=":")
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(CASE_LABEL.get(cv, str(cv)) if col_key == "case"
                             else f"N = {cv}", fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{'d' if col_key == 'case' else 'nodes'} = {rv}",
                              fontsize=7.5)
            if i == nr - 1:
                ax.set_xlabel({"pehe": "PEHE (raw)",
                               "l1_ate": "L1 ATE error",
                               "eps_ate": "relative ATE error"}[metric],
                              fontsize=6.5)
            if not any(np.isfinite(v) for v in vals):
                missing += 1
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=6, color="#999")

    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=C_1D, ec="black", lw=0.4),
                        plt.Rectangle((0, 0), 1, 1, fc=C_2D, ec="black", lw=0.4)],
               labels=["1D head", "2D head"], loc="upper right",
               fontsize=7, frameon=False, ncol=2,
               bbox_to_anchor=(0.995, 1.004))
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = a.out
    if len(a.metric) > 1 or not out.endswith(".png"):
        out = f"{out[:-4] if out.endswith('.png') else out}_{metric}.png"
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({nr}x{nc} grid, {len(bars)} bars)")
    if missing:
        print(f"  NOTE: {missing} of {nr*nc} cells had no data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
