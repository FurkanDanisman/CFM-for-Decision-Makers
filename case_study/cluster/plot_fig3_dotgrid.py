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

# (display label, candidate csv model keys) — 1D then its 2D partner.
#
# Several keys per slot because the two CSV producers name models differently:
# dsweep_report writes uwyk_noanc / cpfn1d_perarm, while cs_point_total writes
# uwyk1d-noanc / cpfn1d. The first key present in the CSV wins, so one PAIRS table
# serves both instead of the figure silently losing rows on the other's output.
#
# Do-PFN 2D is dopfn_repro_joint2d, NOT dopfn_bb: the reported comparison is
# native-vs-joint2d, and dopfn_bb is a different checkpoint.
PAIRS = [
    ("Do-PFN",         ("dopfn_native",)),
    ("Do-PFN 2D",      ("dopfn_repro_joint2d", "dopfn_bb")),
    ("UWYK No-Anc",    ("uwyk1d-noanc", "uwyk_noanc")),
    ("UWYK No-Anc 2D", ("graph2d-noanc", "graph2d_noanc")),
    ("UWYK Anc",       ("uwyk1d-v3a", "uwyk_v3a")),
    ("UWYK Anc 2D",    ("graph2d-v3a", "graph2d_v3a")),
    ("CausalPFN-C",    ("cpfn1d_j1024", "cpfn1d", "cpfn1d_perarm")),
    ("CausalPFN-C 2D", ("cpfn2d_eta0", "cpfn2d_pooled", "cpfn2d")),
]
# Drop pairs that no row supplies, so a case-study CSV keeps its 8 bars and a
# cmech CSV gets 10, without either carrying empty slots.
_PAIRS_ALL = list(PAIRS)
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
# label, key. eps is the RELATIVE ATE error |ATE_hat - ATE_true| / |ATE_true|, which
# is scale-free and therefore the one that pools across cases and d; l1 is the absolute
# error in the outcome's units. Selectable with --metrics.
_ALL_METRICS = [("PEHE", "pehe"), ("L1-ATE", "l1"), ("Relative ATE error", "eps")]
_METRICS = list(_ALL_METRICS[:2])


def _metric_col(df, mkey):
    """The metric column, with or without the readout suffix.

    dsweep_report writes pehe_raw / pehe_em; cs_point_total writes a bare pehe. Trying
    the suffixed name first and falling back keeps one script working on both, instead
    of a KeyError that looks like a missing metric.
    """
    # The two producers disagree on the metric names too: dsweep_report writes
    # l1_raw / eps_raw, cs_point_total writes l1_ate / eps_ate. Try the suffixed form,
    # the bare form, then the known aliases, before giving up.
    alias = {"l1": ("l1_ate",), "eps": ("eps_ate",), "pehe": ()}
    for cand in (f"{mkey}_{ARGS.readout}", mkey, *alias.get(mkey, ())):
        if cand in df.columns:
            return cand
    raise SystemExit(f"FATAL: {ARGS.csv} has no column for metric {mkey!r} "
                     f"(tried {mkey}_{ARGS.readout}, {mkey}"
                     + "".join(f", {a}" for a in alias.get(mkey, ()))
                     + f"); columns are {list(df.columns)}")


def _sem_col(df, col):
    # l1_ate's SEM column is l1_sem, not l1_ate_sem -- the producer drops the _ate.
    cands = [f"{col}_sem",
             col.replace("_raw", "").replace("_em", "") + "_sem",
             col.replace("_ate", "") + "_sem"]
    for c in cands:
        if c in df.columns:
            return c
    return None


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
    sem = sub[_sem_col(sub, col) or (col + "_sem")].to_numpy(float)
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
    # DF, not a parameter: _grid's signature is fixed by its callers and the metric
    # column now has to be looked up against the CSV's actual headers. Set in main()
    # alongside ARGS, which this module already threads the same way.
    df = DF
    nC = len(cases)
    fig, axes = plt.subplots(len(_METRICS), nC, figsize=(2.7 * nC, 6.2),
                             squeeze=False, sharey=True)
    for ri, (mlabel, mkey) in enumerate(_METRICS):
        col = _metric_col(df, mkey)
        _SEM = _sem_col(df, col)
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


def _grid_by_d(df, cases, dvals, mkey, mlabel, logx,
               row_col="d", col_col="case", row_fmt="d = {}"):
    """One figure for ONE metric: rows = `row_col`, columns = `col_col`.

    The per-d and combined modes put the two metrics on separate ROWS of the
    same figure, which leaves no room for a d axis. For an appendix that has to
    show the d-split, it is cleaner to give each metric its own figure and
    spend the rows on d.

    The two axes are parameters because ComplexMech has no case dimension: it is
    rows = node count, columns = context N. Everything else -- the pairs, the bar
    geometry, the per-panel scaling -- is the same figure, so it would be wrong to
    fork a second script for it.
    """
    col = _metric_col(df, mkey)
    _SEM = _sem_col(df, col)
    nC, nD = len(cases), len(dvals)
    fig, axes = plt.subplots(nD, nC, figsize=(2.7 * nC, 1.55 * nD + 0.9),
                             squeeze=False, sharey=True)
    for ri, d in enumerate(dvals):
        for ci, (ckey, clab) in enumerate(cases):
            ax = axes[ri][ci]
            sub = df[(df[col_col] == ckey) & (df[row_col] == d)].set_index("model")
            getter = (lambda key, _s=sub: (
                (float(_s.at[key, col]),
                 float(_s.at[key, _SEM]) if _SEM else float("nan"))
                if key in _s.index else (np.nan, np.nan)))
            _draw(ax, getter, logx)
            if ri == 0:
                ax.set_title(clab, fontsize=9)
            if ci == 0:
                ax.set_yticklabels([lab for lab, _ in PAIRS], fontsize=7)
                ax.set_ylabel(row_fmt.format(d), fontsize=9)
            # label the metric axis on the bottom row only: with many d rows the
            # per-row labels collide and add nothing.
            if ri == nD - 1:
                ax.set_xlabel(f"{mlabel} ({ARGS.readout})", fontsize=8)
    fig.tight_layout()
    return fig


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--shift", default="cen3")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--readout", choices=["raw", "em"], default="raw")
    ap.add_argument("--mode",
                    choices=["per-d", "combined", "both", "by-d"],
                    default="both",
                    help="by-d: one figure PER METRIC with d on the rows "
                         "(appendix layout).")
    ap.add_argument("--d-values", nargs="*", type=int, default=None)
    ap.add_argument("--kind", choices=["case", "cmech"], default="case",
                    help="case: rows = d, cols = the six case studies. "
                         "cmech: rows = node count, cols = context N -- ComplexMech "
                         "has no case dimension.")
    ap.add_argument("--swap-axes", action="store_true",
                    help="put the row variable on the columns and vice versa. With a "
                         "single context, cmech otherwise draws one narrow column; "
                         "swapping gives nodes across the top, which reads like the "
                         "case-study figure.")
    ap.add_argument("--metrics", nargs="+", default=["pehe", "l1"],
                    choices=[k for _, k in _ALL_METRICS],
                    help="which metrics to draw. In by-d mode each gets its own "
                         "figure; in the 2xN modes they are the rows.")
    ap.add_argument("--logx", action="store_true", help="log-scale the metric axis.")
    ap.add_argument("--out", default="fig3", help="output path prefix (no extension).")
    ARGS = ap.parse_args()
    global _METRICS
    _METRICS = [(lab, k) for lab, k in _ALL_METRICS if k in ARGS.metrics]

    global DF
    df = DF = pd.read_csv(ARGS.csv)
    # Filter only on the columns this CSV actually has. dsweep_report emits shift and
    # N columns plus an "ALL" pooled case; cs_point_total emits a CSV that is ALREADY
    # pooled over the three shifts at one N, so demanding those columns turned a
    # perfectly good input into a KeyError.
    if "shift" in df.columns:
        df = df[df["shift"] == ARGS.shift]
    if "N" in df.columns:
        df = df[df["N"] == ARGS.n]
    if "case" in df.columns:
        df = df[df["case"] != "ALL"]
    if df.empty:
        raise SystemExit(f"no rows for shift={ARGS.shift} N={ARGS.n} in {ARGS.csv}")
    # ComplexMech has no d / case columns at all, so these lookups must not run for it.
    if ARGS.kind == "cmech":
        dvals, cases = [], []
    else:
        if ARGS.d_values:
            df = df[df["d"].isin(ARGS.d_values)]
        dvals = sorted(int(x) for x in df["d"].unique())
    global PAIRS
    have = set(df["model"])
    # Resolve each slot to the first candidate key the CSV actually supplies, and drop
    # slots it supplies none for -- so a CSV from either producer keeps its eight bars
    # rather than silently rendering blanks.
    resolved = []
    for lab, keys in _PAIRS_ALL:
        k = next((k for k in keys if k in have), None)
        if k is not None:
            resolved.append((lab, k))
    PAIRS = resolved or [(lab, keys[0]) for lab, keys in _PAIRS_ALL]
    if ARGS.kind != "cmech":
        present = set(df["case"])
        cases = [(k, lab) for k, lab in _CASES if k in present]
        # Anything not in the hardcoded case-study list was previously dropped
        # SILENTLY, producing an empty figure with no error. Keep it, labelled by
        # its own key.
        extra = sorted(present - {k for k, _ in _CASES})
        cases += [(k, k.replace("_", " ")) for k in extra]
        if not cases:
            raise SystemExit(f"no usable 'case' values in {ARGS.csv}: "
                             f"{sorted(present)}")
    missing_models = sorted(set(df["model"]) - {k for _, k in PAIRS})
    if missing_models:
        print("[warn] models in the CSV but not in PAIRS (not plotted): "
              + ", ".join(missing_models))
    os.makedirs(os.path.dirname(os.path.abspath(ARGS.out)) or ".", exist_ok=True)

    made = []
    if ARGS.mode in ("per-d", "both") and ARGS.mode != "by-d":
        for d in dvals:
            def gf(ckey, col, _d=d):
                sub = df[(df["case"] == ckey) & (df["d"] == _d)].set_index("model")
                return lambda key: ((float(sub.at[key, col]),
                                     float(sub.at[key, _SEM]) if _SEM else float("nan"))
                                    if key in sub.index else (np.nan, np.nan))
            f = _grid(gf, cases, ARGS.logx, f"{ARGS.shift}  N={ARGS.n}  d={d}  ({ARGS.readout})")
            p = f"{ARGS.out}_d{d}.png"; f.savefig(p, dpi=150); plt.close(f); made.append(p)
    if ARGS.mode in ("combined", "both") and ARGS.mode != "by-d":
        def gf(ckey, col):
            sub = df[df["case"] == ckey]
            return lambda key: _pool_over_d(sub[sub["model"] == key], col)
        f = _grid(gf, cases, ARGS.logx,
                  f"{ARGS.shift}  N={ARGS.n}  ({ARGS.readout})  [pooled over d={{{','.join(map(str, dvals))}}}]")
        p = f"{ARGS.out}_combined.png"; f.savefig(p, dpi=150); plt.close(f); made.append(p)

    if ARGS.mode == "by-d":
        made = []
        if ARGS.kind == "cmech":
            # Columns are the context sizes present, ascending; rows the node counts.
            ctxs = sorted(df["context"].unique())
            cols = [(c, f"N = {c}") for c in ctxs]
            rows_ = sorted(df["nodes"].unique())
            rc, cc, rfmt = "nodes", "context", "nodes = {}"
            if ARGS.swap_axes:
                cols = [(n, f"nodes = {n}") for n in sorted(df["nodes"].unique())]
                rows_ = ctxs
                rc, cc, rfmt = "context", "nodes", "N = {}"
        else:
            cols, rows_ = cases, dvals
            rc, cc, rfmt = "d", "case", "d = {}"
            if ARGS.swap_axes:
                cols = [(d, f"d = {d}") for d in dvals]
                rows_ = [k for k, _ in cases]
                rc, cc, rfmt = "case", "d", "{}"
        for mlabel, mkey in _METRICS:
            f = _grid_by_d(df, cols, rows_, mkey, mlabel, ARGS.logx,
                           row_col=rc, col_col=cc, row_fmt=rfmt)
            p = f"{ARGS.out}_{mkey}_by_d.png"
            f.savefig(p, dpi=150, bbox_inches="tight"); plt.close(f); made.append(p)

    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()
