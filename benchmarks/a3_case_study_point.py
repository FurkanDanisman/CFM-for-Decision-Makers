#!/usr/bin/env python
"""Case-study PEHE / eps_ATE: one table, six case columns, shifts and d pooled.

    python benchmarks/a3_case_study_point.py \
        --root $SCRATCH/aurora3/cs/dopfn_1d_botharms \
        --methods dopfn_native --out-md cs_point.md

Pooling follows final_table.py, where the two metrics are NOT pooled the same
way and swapping them is a silent error:

  PEHE       root-mean-square across cells. PEHE is itself an RMSE, so averaging
             PEHE values understates it -- only the squares add.
  eps_ATE    plain mean across cells, since it is already one error per cell.

A cell is one (shift, d) at fixed ctx for one case. Cells with no dumps are
skipped and counted, so a partial tree cannot masquerade as a complete one.

Why not run_cell() over all 18 dirs at once: it concatenates queries and would
then report a single pooled ATE over the union, not the mean of per-cell ATE
errors. The PEHE would come out right; the ATE column would not.
"""
from __future__ import annotations
import argparse, os, sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, os.path.join(_REPO, "UWYK_Fig3_4"), os.path.join(_REPO, "MALC"),
           os.path.join(_REPO, "realcause_eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import point_raw_em as P                                    # noqa: E402
from cate_density_metrics import METHODS                    # noqa: E402

CASES = ["Observed_Confounder", "Backdoor_Criterion", "Observed_Mediator",
         "Observed_Mediator_and_Confounder", "Unobserved_Confounder",
         "Frontdoor_Criterion"]
SHIFTS = ["0", "+2", "-2"]
DS = ["5", "10", "20", "30", "40", "50"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="cs/<model> dump root")
    ap.add_argument("--methods", nargs="+", required=True,
                    help="cate_density_metrics METHODS names, e.g. uwyk1d-v3a")
    ap.add_argument("--shifts", nargs="+", default=SHIFTS)
    ap.add_argument("--ds", nargs="+", default=DS)
    ap.add_argument("--ctx", default="1000")
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--mode", default="raw", choices=["raw", "em"])
    ap.add_argument("--ate-metric", default="l1", choices=["l1", "rel"])
    ap.add_argument("--out-md", default=None)
    a = ap.parse_args()

    P._ATE_METRIC = a.ate_metric
    sel = [m for m in METHODS if m[0] in a.methods]
    missing = set(a.methods) - {m[0] for m in sel}
    if missing:
        sys.exit(f"unknown method(s): {sorted(missing)}")

    rows = {}
    for name, subdir, tag in sel:
        per_case = {}
        for case in a.cases:
            pehes, epss, n_cells, n_empty, n_real = [], [], 0, 0, 0
            for sh in a.shifts:
                for d in a.ds:
                    cell = os.path.join(a.root, f"shift{sh}", f"d{d}",
                                        f"ctx{a.ctx}", subdir, case)
                    if not os.path.isdir(cell) or not P._files_in(cell):
                        n_empty += 1
                        continue
                    r = P.run_cell([cell], tag, a.mode)
                    if r is None:
                        n_empty += 1
                        continue
                    pehes.append(r["pehe"]); epss.append(r["eps"])
                    n_cells += 1; n_real += r["n"]
            if n_cells:
                per_case[case] = dict(
                    # squares add: RMS across cells, never the mean of PEHEs
                    pehe=float(np.sqrt(np.mean(np.square(pehes)))),
                    eps=float(np.mean(epss)),
                    cells=n_cells, empty=n_empty, n=n_real)
            else:
                per_case[case] = None
            print(f"  {name:16s} {case:34s} cells={n_cells:3d} "
                  f"empty={n_empty:3d} realizations={n_real}", file=sys.stderr)
        rows[name] = per_case

    short = {c: c.replace("Observed_", "Obs_").replace("_Criterion", "")
                 .replace("Unobserved_", "Unobs_") for c in a.cases}
    out = [f"<!-- ate_metric={a.ate_metric} mode={a.mode} "
           f"shifts={','.join(a.shifts)} d={','.join(a.ds)} ctx={a.ctx} -->",
           "### Case study — PEHE / eps_ATE, shifts and d pooled", "",
           "PEHE pooled as RMS across (shift, d) cells; eps_ATE as a plain mean.",
           ""]
    hdr = "| method | metric | " + " | ".join(short[c] for c in a.cases) + " |"
    out += [hdr, "|" + "---|" * (len(a.cases) + 2)]
    for name in rows:
        for label, key in (("PEHE", "pehe"), (f"eps_ATE({a.ate_metric})", "eps")):
            cells = []
            for c in a.cases:
                v = rows[name][c]
                cells.append("--" if v is None else f"{v[key]:.4f}")
            out.append(f"| {name} | {label} | " + " | ".join(cells) + " |")
    txt = "\n".join(out)
    print("\n" + txt)
    if a.out_md:
        with open(a.out_md, "w") as f:
            f.write(txt + "\n")
        print(f"\nwrote {a.out_md}")


if __name__ == "__main__":
    main()
