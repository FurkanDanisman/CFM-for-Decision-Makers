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
C_1D, C_2D = "#D97706", "#B91C1C"
_2D = ("cpfn2d", "graph2d", "joint2d", "dopfn_bb")


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
    ap.add_argument("--metric", default="pehe",
                    choices=["pehe", "l1_ate", "eps_ate"])
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict and order the bars; default all, sorted")
    ap.add_argument("--free-x", action="store_true")
    ap.add_argument("--out", required=True, help="output png")
    a = ap.parse_args()

    row_key, col_key = ("nodes", "context") if a.kind == "cmech" else ("case", "d")
    rows = load(a.csv, row_key, col_key)
    sem_key = {"pehe": "pehe_sem", "l1_ate": "l1_sem",
               "eps_ate": "eps_sem"}[a.metric]

    def _ord(v):
        try:
            return (0, float(v))
        except ValueError:
            return (1, str(v))
    rvals = sorted({r[row_key] for r in rows}, key=_ord)
    cvals = sorted({r[col_key] for r in rows}, key=_ord)
    models = a.models or sorted({r["model"] for r in rows})

    cell = defaultdict(dict)
    for r in rows:
        try:
            cell[(r[row_key], r[col_key])][r["model"]] = (
                float(r[a.metric]), float(r[sem_key]))
        except (ValueError, KeyError):
            continue

    nr, nc = len(rvals), len(cvals)
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 0.42 * len(models) * nr),
                             squeeze=False,
                             sharex=("none" if a.free_x else "row"))
    y = np.arange(len(models))
    for i, rv in enumerate(rvals):
        for j, cv in enumerate(cvals):
            ax = axes[i][j]
            d = cell.get((rv, cv), {})
            vals = [d.get(m, (np.nan, np.nan))[0] for m in models]
            errs = [d.get(m, (np.nan, np.nan))[1] for m in models]
            cols = [C_2D if is_2d(m) else C_1D for m in models]
            ax.barh(y, vals, xerr=errs, color=cols, height=0.72,
                    error_kw=dict(lw=0.8, capsize=2, ecolor="#404040"))
            ax.set_yticks(y)
            ax.set_yticklabels(models if j == 0 else [], fontsize=6)
            ax.invert_yaxis()
            ax.tick_params(axis="x", labelsize=6)
            ax.grid(axis="x", lw=0.4, alpha=0.35)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(f"{col_key}={cv}", fontsize=8)
            if j == nc - 1:
                ax.text(1.02, 0.5, f"{row_key}={rv}", transform=ax.transAxes,
                        rotation=-90, va="center", ha="left", fontsize=8)
            if not d:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color="#888")
    lab = {"pehe": "PEHE", "l1_ate": "L1 ATE error",
           "eps_ate": "relative ATE error"}[a.metric]
    for j in range(nc):
        axes[-1][j].set_xlabel(lab, fontsize=7)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_1D),
                        plt.Rectangle((0, 0), 1, 1, color=C_2D)],
               labels=["1D head", "2D head"], loc="upper right",
               fontsize=7, frameon=False, ncol=2)
    fig.suptitle(f"{'ComplexMech' if a.kind == 'cmech' else 'Case study'} — {lab}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=180, bbox_inches="tight")
    print(f"wrote {a.out}  ({nr}x{nc} grid, {len(models)} models)")
    empty = sum(1 for rv in rvals for cv in cvals if not cell.get((rv, cv)))
    if empty:
        print(f"  NOTE: {empty} of {nr*nc} cells had no data (drawn as 'no data')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
