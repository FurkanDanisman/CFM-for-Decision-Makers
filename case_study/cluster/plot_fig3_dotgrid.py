"""Figure-3-style GROUPED BAR grid from the d_variation CSV (dsweep_report.py).

Layout mirrors Do-PFN's Figure 3:
    columns = the 6 case studies,
    rows    = PEHE and L1-ATE,
    each cell = horizontal bars, one per method, x-axis = metric value, ±SEM bars.

Methods are grouped as 1D-vs-2D pairs (four pairs, eight bars), coloured
    1D = orange, 2D = red,
top-to-bottom in this order:
    Do-PFN            (dopfn_native)   | Do-PFN 2D        (dopfn_bb)
    UWYK No-Anc       (uwyk_noanc)     | UWYK No-Anc 2D   (graph2d_noanc)
    UWYK Anc          (uwyk_v3a)       | UWYK Anc 2D      (graph2d_v3a)
    CausalPFN-C       (cpfn1d_perarm)  | CausalPFN-C 2D   (cpfn2d_pooled)

Two products:
  --mode per-d    one figure PER d value, each a 2×6 grid.  Files: <out>_d<K>.png
  --mode combined ONE 2×6 figure with each bar POOLED across all d (exact pooled
                  mean ± SEM, n-weighted).  File: <out>_combined.png

Examples:
    python plot_fig3_dotgrid.py --csv $RES/combined_cen3.csv --shift cen3 --n 1000 \
        --mode both --out fig3_cen3
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# (display label, csv model key) — 1D then its 2D partner, four pairs.
PAIRS = [
    ("Do-PFN",         "dopfn_native"),
    ("Do-PFN 2D",      "dopfn_bb"),
    ("UWYK No-Anc",    "uwyk_noanc"),
    ("UWYK No-Anc 2D", "graph2d_noanc"),
    ("UWYK Anc",       "uwyk_v3a"),
    ("UWYK Anc 2D",    "graph2d_v3a"),
    ("CausalPFN-C",    "cpfn1d_perarm"),
    ("CausalPFN-C 2D", "cpfn2d_pooled"),
]
C_1D = "#E4A15B"     # warm ochre (1D)
C_2D = "#8C4A5F"     # muted plum-red (2D)
EDGE = "black"       # bar edges + error bars, regardless of fill
_CASES = [
    ("Observed_Confounder", "Observed\nConfounder"),
    ("Observed_Mediator", "Observed\nMediator"),
    ("Observed_Mediator_and_Confounder", "Confounder +\nMediator"),
    ("Unobserved_Confounder", "Unobserved\nConfounder"),
    ("Frontdoor_Criterion", "Front-Door\nCriterion"),
    ("Backdoor_Criterion", "Back-Door\nCriterion"),
]
_METRICS = [("PEHE", "pehe"), ("L1-ATE", "l1")]


def _bar_ypos(n_pairs):
    """Top-to-bottom y positions, with a gap between each 1D/2D pair."""
    yp, y = [], 0.0
    for _ in range(n_pairs):
        yp.append(y); y += 1.0
        yp.append(y); y += 1.0
        y += 0.7                                   # gap after the pair
    yp = np.asarray(yp)
    return yp.max() - yp                           # invert: first entry on top


_YP = _bar_ypos(len(PAIRS) // 2)


def _draw(ax, getter, logx):
    for i, (_, key) in enumerate(PAIRS):
        mu, se = getter(key)
        ax.barh(_YP[i], mu, xerr=(se if np.isfinite(se) else None),
                height=0.85, color=(C_2D if i % 2 else C_1D),
                edgecolor=EDGE, linewidth=0.7,
                ecolor=EDGE, capsize=2, error_kw=dict(lw=0.9), zorder=3)
    ax.set_yticks(_YP)
    ax.set_ylim(_YP.min() - 0.8, _YP.max() + 0.8)
    if logx:
        ax.set_xscale("log")
    ax.grid(axis="x", ls=":", alpha=0.5, zorder=0)


def _pool_over_d(sub, col):
    """Exact pooled (mean, SEM) over all d rows for one (case, model), from each
    row's (mean, SEM, n). Equivalent to pooling every realization: n-weighted
    grand mean; total variance = within + between (ANOVA identity)."""
    m = sub[col].to_numpy(float)
    sem = sub[col + "_sem"].to_numpy(float)
    n = sub["n"].to_numpy(float)
    ok = np.isfinite(m) & np.isfinite(n) & (n > 0)
    m, sem, n = m[ok], sem[ok], n[ok]
    if n.sum() == 0:
        return np.nan, np.nan
    N = n.sum()
    grand = float((n * m).sum() / N)
    sd = np.where(n > 1, sem * np.sqrt(n), 0.0)
    ss = float((sd ** 2 * np.maximum(n - 1, 0)).sum()) + float((n * (m - grand) ** 2).sum())
    return grand, (0.0 if N <= 1 else float(np.sqrt(ss / (N - 1) / N)))


def _grid(getter_for, cases, logx, title):
    nC = len(cases)
    fig, axes = plt.subplots(len(_METRICS), nC, figsize=(2.7 * nC, 6.2),
                             squeeze=False, sharey=True)
    for ri, (mlabel, mkey) in enumerate(_METRICS):
        col = f"{mkey}_{ARGS.readout}"
        for ci, (ckey, clab) in enumerate(cases):
            ax = axes[ri][ci]
            _draw(ax, getter_for(ckey, col), logx)
            if ri == 0:
                ax.set_title(clab, fontsize=9)
            if ci == 0:
                ax.set_yticklabels([lab for lab, _ in PAIRS], fontsize=8)
            ax.set_xlabel(f"{mlabel} ({ARGS.readout})", fontsize=8)   # every row labelled
    fig.tight_layout()
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
    ap.add_argument("--logx", action="store_true", help="log-scale the metric axis.")
    ap.add_argument("--out", default="fig3", help="output path prefix (no extension).")
    ARGS = ap.parse_args()

    df = pd.read_csv(ARGS.csv)
    df = df[(df["shift"] == ARGS.shift) & (df["N"] == ARGS.n) & (df["case"] != "ALL")]
    if df.empty:
        raise SystemExit(f"no rows for shift={ARGS.shift} N={ARGS.n} in {ARGS.csv}")
    if ARGS.d_values:
        df = df[df["d"].isin(ARGS.d_values)]
    dvals = sorted(int(x) for x in df["d"].unique())
    cases = [(k, lab) for k, lab in _CASES if k in set(df["case"])]
    os.makedirs(os.path.dirname(os.path.abspath(ARGS.out)) or ".", exist_ok=True)

    made = []
    if ARGS.mode in ("per-d", "both"):
        for d in dvals:
            def gf(ckey, col, _d=d):
                sub = df[(df["case"] == ckey) & (df["d"] == _d)].set_index("model")
                return lambda key: ((float(sub.at[key, col]), float(sub.at[key, col + "_sem"]))
                                    if key in sub.index else (np.nan, np.nan))
            f = _grid(gf, cases, ARGS.logx, f"{ARGS.shift}  N={ARGS.n}  d={d}  ({ARGS.readout})")
            p = f"{ARGS.out}_d{d}.png"; f.savefig(p, dpi=150); plt.close(f); made.append(p)
    if ARGS.mode in ("combined", "both"):
        def gf(ckey, col):
            sub = df[df["case"] == ckey]
            return lambda key: _pool_over_d(sub[sub["model"] == key], col)
        f = _grid(gf, cases, ARGS.logx,
                  f"{ARGS.shift}  N={ARGS.n}  ({ARGS.readout})  [pooled over d={{{','.join(map(str, dvals))}}}]")
        p = f"{ARGS.out}_combined.png"; f.savefig(p, dpi=150); plt.close(f); made.append(p)

    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()
