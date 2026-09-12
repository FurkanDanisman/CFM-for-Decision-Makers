"""Figure-3-style dot grid from the d_variation CSV (dsweep_report.py).

Layout mirrors Do-PFN's Figure 3:
    columns = the 6 case studies,
    rows    = PEHE and L1-ATE,
    each cell = one dot per model at its metric value, with a ±SEM x-error bar
                (Cleveland/forest style; models stacked on the y-axis).

Two products:
  --mode per-d    one figure PER d value (default: all d in the CSV), each a
                  2×6 grid (PEHE row, L1-ATE row). Files: <out>_d<K>.png
  --mode combined ONE 2×6 figure where every cell overlays all d values,
                  colour-coded by d (legend), so you can read how each model's
                  PEHE / L1 moves as covariate count grows. File: <out>_combined.png

Reads the tidy CSV written by dsweep_report.py (any shift label, per-case rows):
    shift,d,N,case,model, pehe_raw,pehe_raw_sem,..., l1_raw,l1_raw_sem,..., n

Examples:
    # per-d PEHE/L1 dot grids at N=1000, pooled 0/-2/+2:
    python plot_fig3_dotgrid.py --csv $RES/combined_cen3.csv --shift cen3 --n 1000 \
        --mode per-d --out fig3_cen3
    # single combined figure over all d:
    python plot_fig3_dotgrid.py --csv $RES/combined_cen3.csv --shift cen3 --n 1000 \
        --mode combined --out fig3_cen3
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Model row order + display labels (top row = first entry).
_MODELS = [
    ("dopfn_native", "Do-PFN"),
    ("dopfn_bb", "Do-PFN 2D"),
    ("cpfn1d_perarm", "CausalPFN 1D"),
    ("cpfn2d_pooled", "CausalPFN 2D"),
    ("graph2d_noanc", "Graph2D (no-anc)"),
    ("graph2d_v3a", "Graph2D (v3a)"),
    ("graph2d_v3b", "Graph2D (v3b)"),
    ("uwyk_noanc", "UWYK (no-anc)"),
    ("uwyk_v3a", "UWYK (v3a)"),
    ("uwyk", "UWYK"),
]
_CASES = [
    ("Observed_Confounder", "Observed\nConfounder"),
    ("Observed_Mediator", "Observed\nMediator"),
    ("Observed_Mediator_and_Confounder", "Confounder +\nMediator"),
    ("Unobserved_Confounder", "Unobserved\nConfounder"),
    ("Frontdoor_Criterion", "Front-Door\nCriterion"),
    ("Backdoor_Criterion", "Back-Door\nCriterion"),
]
_METRICS = [("PEHE", "pehe"), ("L1-ATE", "l1")]


def _present(df, key, order):
    have = set(df[key])
    return [(k, lab) for k, lab in order if k in have]


def _panel(ax, sub, models, metric, logx):
    """One cell: dot per model at mean, horizontal ±SEM bar. sub is a
    per-(case,d) slice indexed by model."""
    ys = np.arange(len(models))[::-1]                       # first model on top
    xs, es = [], []
    for mkey, _ in models:
        if mkey in sub.index:
            xs.append(float(sub.at[mkey, metric]))
            es.append(float(sub.at[mkey, metric + "_sem"]))
        else:
            xs.append(np.nan); es.append(np.nan)
    ax.errorbar(xs, ys, xerr=es, fmt="o", ms=5, capsize=2.5, lw=1.1,
                color="#1f77b4", ecolor="#888", mfc="#1f77b4", mec="k", mew=0.4)
    ax.set_yticks(ys)
    if logx:
        ax.set_xscale("log")
    ax.grid(axis="x", ls=":", alpha=0.5)
    return models


def _fig_per_d(df, d, models, cases, logx, title):
    nC = len(cases)
    fig, axes = plt.subplots(len(_METRICS), nC, figsize=(2.55 * nC, 5.4),
                             squeeze=False, sharey=True)
    for ri, (mlabel, mkey) in enumerate(_METRICS):
        col = f"{mkey}_{ARGS.readout}"
        for ci, (ckey, clab) in enumerate(cases):
            ax = axes[ri][ci]
            sub = df[(df["case"] == ckey) & (df["d"] == d)].set_index("model")
            _panel(ax, sub, models, col, logx)
            if ri == 0:
                ax.set_title(clab, fontsize=9)
            if ci == 0:
                ax.set_yticklabels([lab for _, lab in models], fontsize=8)
                ax.set_ylabel(f"{mlabel} ({ARGS.readout})", fontsize=10)
            if ri == len(_METRICS) - 1:
                ax.set_xlabel(mlabel, fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _fig_combined(df, dvals, models, cases, logx, title):
    nC = len(cases)
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(dvals)))
    fig, axes = plt.subplots(len(_METRICS), nC, figsize=(2.55 * nC, 5.8),
                             squeeze=False, sharey=True)
    off = np.linspace(-0.28, 0.28, len(dvals))              # vertical jitter per d
    for ri, (mlabel, mkey) in enumerate(_METRICS):
        col = f"{mkey}_{ARGS.readout}"
        for ci, (ckey, clab) in enumerate(cases):
            ax = axes[ri][ci]
            base = np.arange(len(models))[::-1]
            for di, d in enumerate(dvals):
                sub = df[(df["case"] == ckey) & (df["d"] == d)].set_index("model")
                xs = [float(sub.at[m, col]) if m in sub.index else np.nan
                      for m, _ in models]
                es = [float(sub.at[m, col + "_sem"]) if m in sub.index else np.nan
                      for m, _ in models]
                ax.errorbar(xs, base + off[di], xerr=es, fmt="o", ms=3.5,
                            capsize=1.5, lw=0.8, color=cmap[di], ecolor=cmap[di],
                            mec="k", mew=0.25, label=f"d={d}" if (ri == 0 and ci == 0) else None)
            ax.set_yticks(base)
            if logx:
                ax.set_xscale("log")
            ax.grid(axis="x", ls=":", alpha=0.5)
            if ri == 0:
                ax.set_title(clab, fontsize=9)
            if ci == 0:
                ax.set_yticklabels([lab for _, lab in models], fontsize=8)
                ax.set_ylabel(f"{mlabel} ({ARGS.readout})", fontsize=10)
            if ri == len(_METRICS) - 1:
                ax.set_xlabel(mlabel, fontsize=8)
    fig.legend(loc="lower center", ncol=len(dvals), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return fig


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--shift", default="cen3")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--readout", choices=["raw", "em"], default="raw")
    ap.add_argument("--mode", choices=["per-d", "combined", "both"], default="both")
    ap.add_argument("--d-values", nargs="*", type=int, default=None)
    ap.add_argument("--logx", action="store_true", help="log-scale the metric axis (like the paper).")
    ap.add_argument("--out", default="fig3", help="output path prefix (no extension).")
    ARGS = ap.parse_args()

    df = pd.read_csv(ARGS.csv)
    df = df[(df["shift"] == ARGS.shift) & (df["N"] == ARGS.n) & (df["case"] != "ALL")]
    if df.empty:
        raise SystemExit(f"no rows for shift={ARGS.shift} N={ARGS.n} in {ARGS.csv}")
    if ARGS.d_values:
        df = df[df["d"].isin(ARGS.d_values)]
    dvals = sorted(int(x) for x in df["d"].unique())
    models = _present(df, "model", _MODELS)
    cases = _present(df, "case", _CASES)
    os.makedirs(os.path.dirname(os.path.abspath(ARGS.out)) or ".", exist_ok=True)

    made = []
    if ARGS.mode in ("per-d", "both"):
        for d in dvals:
            f = _fig_per_d(df, d, models, cases, ARGS.logx,
                           f"{ARGS.shift}  N={ARGS.n}  d={d}  ({ARGS.readout})")
            p = f"{ARGS.out}_d{d}.png"; f.savefig(p, dpi=150); plt.close(f); made.append(p)
    if ARGS.mode in ("combined", "both"):
        f = _fig_combined(df, dvals, models, cases, ARGS.logx,
                          f"{ARGS.shift}  N={ARGS.n}  d={{{','.join(map(str, dvals))}}}  ({ARGS.readout})")
        p = f"{ARGS.out}_combined.png"; f.savefig(p, dpi=150); plt.close(f); made.append(p)

    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()
