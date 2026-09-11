"""Readable pivots from the d_variation CSV (case_study/cluster/dsweep_report.py).

Rows = model, columns = the swept axis (--by: d or N), values = one metric,
for a chosen (shift, N-or-d, case-or-mean). Fast — reads the CSV, not the npz.

Examples:
    # PEHE-raw vs d, shift+2, N=500, averaged over the 6 cases:
    python dsweep_pivot.py --csv $RES/dsweep.csv --shift shift+2 --n 500

    # em PEHE vs d for one case:
    python dsweep_pivot.py --csv $RES/dsweep.csv --shift shift+2 --n 500 \
        --case Backdoor_Criterion --metric pehe_em

    # PEHE-raw vs N (fix d), all shifts printed separately:
    python dsweep_pivot.py --csv $RES/dsweep.csv --by N --d 3 --all-shifts
"""
from __future__ import annotations

import argparse

import pandas as pd

_MODEL_ORDER = ["dopfn_native", "dopfn_bb", "graph2d_noanc", "uwyk_noanc",
                "graph2d_v3a", "uwyk_v3a", "graph2d_v3b", "uwyk_v3b",
                "cpfn1d_perarm", "cpfn2d_pooled"]
_CASE_ORDER = ["Observed_Confounder", "Observed_Mediator",
               "Observed_Mediator_and_Confounder", "Unobserved_Confounder",
               "Frontdoor_Criterion", "Backdoor_Criterion"]


def _order_rows(piv):
    order = [m for m in _MODEL_ORDER if m in piv.index] + \
            [m for m in piv.index if m not in _MODEL_ORDER]
    return piv.reindex(order)


def _pivot(df, by, metric, case):
    sub = df if case is None else df[df["case"] == case]
    return _order_rows(sub.pivot_table(index="model", columns=by,
                                       values=metric, aggfunc="mean"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--by", choices=["d", "N"], default="d", help="Columns axis (non-panel mode).")
    ap.add_argument("--metric", default="pehe_raw",
                    choices=["pehe_raw", "pehe_em", "l1_raw", "l1_em"])
    ap.add_argument("--shift", default="shift+2")
    ap.add_argument("--n", type=int, default=500, help="Fixed N.")
    ap.add_argument("--d", type=int, default=3, help="Fixed d (when --by N).")
    ap.add_argument("--case", default=None, help="One case, or omit for mean over cases.")
    ap.add_argument("--all-shifts", action="store_true")
    ap.add_argument("--panels", choices=["none", "d", "N", "case"], default="none",
                    help="'d' -> one model x case table per d (fixed N,shift); "
                         "'N' -> one per N (fixed d,shift); "
                         "'case' -> one PEHE + one L1 table per case (rows=model, "
                         "cols=d) at fixed N,shift, cells = mean±SEM | median[IQR].")
    ap.add_argument("--d-values", nargs="*", type=int, default=None,
                    help="Restrict the d columns (e.g. 5 10 20 30 40 50).")
    ap.add_argument("--readout", choices=["raw", "em"], default="raw",
                    help="Panel mode: show PEHE + L1 for this readout.")
    ap.add_argument("--stat", choices=["mean", "median"], default="mean",
                    help="mean -> 'mean±SEM'; median -> 'median'.")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    _short = lambda c: (c.replace("Observed_", "Obs").replace("_Criterion", "")
                        .replace("_and_Confounder", "+Cf").replace("_Confounder", "Cf")
                        .replace("_Mediator", "Med"))

    # ── Panel mode 'case': one PEHE + one L1 table per case; rows=model, ──
    #    cols=d; cells = "mean±sem | median[q1,q3]"; fixed N + shift. ──
    if a.panels == "case":
        sh_df = df[df["shift"] == a.shift]
        if a.d_values:
            sh_df = sh_df[sh_df["d"].isin(a.d_values)]
        dvals = [int(x) for x in sorted(sh_df["d"].unique())]
        cases = [c for c in _CASE_ORDER if c in set(sh_df["case"])]
        metrics = [(f"PEHE ({a.readout})", f"pehe_{a.readout}"),
                   (f"L1-ATE ({a.readout})", f"l1_{a.readout}")]

        def _cell(r):
            return (f"{r[m]:.3f}±{r[m+'_sem']:.3f} | "
                    f"{r[m+'_med']:.3f}[{r[m+'_q1']:.3f},{r[m+'_q3']:.3f}]")

        print(f"\n############ {a.shift}  N={a.n}  readout={a.readout}  d={dvals}  "
              f"(cell = mean±SEM | median[Q1,Q3]; rows=model, cols=d) ############")
        for case in cases:
            sub = sh_df[(sh_df["case"] == case) & (sh_df["N"] == a.n)]
            if sub.empty:
                continue
            models = [x for x in _MODEL_ORDER if x in set(sub["model"])] + \
                     [x for x in sub["model"].unique() if x not in _MODEL_ORDER]
            print(f"\n════════ case = {case} ════════")
            for title, m in metrics:
                data = {}
                for dv in dvals:
                    dd = sub[sub["d"] == dv].set_index("model")
                    col = {mdl: _cell(dd.loc[mdl]) for mdl in models if mdl in dd.index}
                    data[f"d{dv}"] = col
                tbl = pd.DataFrame(data).reindex(index=models)[[f"d{dv}" for dv in dvals]]
                print(f"\n  -- {title} --")
                print(tbl.to_string())
        print()
        return

    # ── Panel mode 'd'/'N': one table per d (or per N); rows=model, cols=case, ──
    #    cells = "mean±sem" or "median"; PEHE and L1 tables for the readout. ──
    if a.panels != "none":
        panel_ax = a.panels
        fixed = ("N", a.n) if panel_ax == "d" else ("d", a.d)
        sh_df = df[df["shift"] == a.shift]
        vals = sorted(sh_df[panel_ax].unique())
        cols = [c for c in _CASE_ORDER if c in set(sh_df["case"])]
        metrics = [(f"PEHE ({a.readout})", f"pehe_{a.readout}"),
                   (f"L1-ATE ({a.readout})", f"l1_{a.readout}")]
        disp = "mean±SEM" if a.stat == "mean" else "median"
        print(f"\n############ {a.shift}  {fixed[0]}={fixed[1]}  readout={a.readout}"
              f"  ({disp}; rows=model, cols=case; one block per {panel_ax}) ############")
        for v in vals:
            sub = sh_df[(sh_df[panel_ax] == v) & (sh_df[fixed[0]] == fixed[1])]
            if sub.empty:
                continue
            models = [m for m in _MODEL_ORDER if m in set(sub["model"])] + \
                     [m for m in sub["model"].unique() if m not in _MODEL_ORDER]
            print(f"\n════════ {panel_ax}={v} ════════")
            for title, metric in metrics:
                data = {}
                for case in cols:
                    cc = sub[sub["case"] == case].set_index("model")
                    col = {}
                    for mdl in models:
                        if mdl not in cc.index:
                            continue
                        if a.stat == "mean":
                            col[mdl] = f"{cc.at[mdl, metric]:.3f}±{cc.at[mdl, metric+'_sem']:.3f}"
                        else:
                            col[mdl] = f"{cc.at[mdl, metric+'_med']:.3f}"
                    data[_short(case)] = col
                tbl = pd.DataFrame(data).reindex(index=models)
                print(f"\n  -- {title} ({disp}) --")
                print(tbl.to_string())
        print()
        return

    # ── Default: rows=model, cols = --by axis ──
    shifts = sorted(df["shift"].unique()) if a.all_shifts else [a.shift]
    fixed = ("N", a.n) if a.by == "d" else ("d", a.d)
    for sh in shifts:
        d = df[(df["shift"] == sh) & (df[fixed[0]] == fixed[1])]
        if d.empty:
            print(f"\n(no rows for {sh}, {fixed[0]}={fixed[1]})"); continue
        piv = _pivot(d, a.by, a.metric, a.case)
        scope = a.case or "mean over 6 cases"
        print(f"\n══ {a.metric}  vs {a.by}   {sh}  {fixed[0]}={fixed[1]}  ({scope}) ══")
        print(piv.round(3).to_string())
    print()


if __name__ == "__main__":
    main()
