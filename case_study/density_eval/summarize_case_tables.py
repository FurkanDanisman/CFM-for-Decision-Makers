"""Per-case-study interval tables at one context, aggregated over d and shifts.

Reads the .json written beside each cate/ate_density_<CASE>_indep.md and
produces ONE TABLE PER CASE STUDY, rows = method.

Aggregation is WEIGHTED BY n_query, not a plain mean over cells. coverage95,
length and is05 are per-query means, so pooling them needs the query counts;
an unweighted mean over cells silently gives a d=2 cell with few queries the
same say as a d=50 cell with many.

`sd_ratio` is reported as a diagnostic, not pooled as if it were an estimate:
values far from 1 mean a density sits on the wrong axis, and seeing that per
case is the point.

    python summarize_case_tables.py $CPF --target cate
    python summarize_case_tables.py $CPF --target ate --ctx 1000
    python summarize_case_tables.py $CPF --target cate --shifts shift0
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np

CASES = ["Observed_Confounder", "Backdoor_Criterion", "Observed_Mediator",
         "Observed_Mediator_and_Confounder", "Unobserved_Confounder",
         "Frontdoor_Criterion"]
COLS = ["coverage95", "length", "is05", "sd_ratio"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="cpfn-schema tree (dvar_cpfn)")
    ap.add_argument("--target", default="cate", choices=["cate", "ate"])
    ap.add_argument("--ctx", type=int, default=1000)
    ap.add_argument("--shifts", nargs="*", default=None,
                    help="restrict to these shift dirs (default: all found)")
    ap.add_argument("--cases", nargs="*", default=CASES)
    ap.add_argument("--out", default=None, help="also write markdown here")
    a = ap.parse_args()

    # (case, method) -> list of (weight, row)
    acc = collections.defaultdict(list)
    ds_seen = collections.defaultdict(set)
    for case in a.cases:
        pat = os.path.join(a.root, "*", "d*", f"ctx{a.ctx}",
                           f"{a.target}_density_{case}_indep.json")
        for f in sorted(glob.glob(pat)):
            parts = f.split(os.sep)
            shift, dname = parts[-4], parts[-3]
            if a.shifts and shift not in a.shifts:
                continue
            try:
                rows = json.load(open(f))
            except Exception:
                continue
            for r in rows:
                w = float(r.get("n_query", 0)) or 1.0
                acc[(case, r["method"])].append((w, r))
                ds_seen[(case, r["method"])].add(dname)

    if not acc:
        raise SystemExit(f"no {a.target} json under {a.root} (ctx{a.ctx})")

    lines = [f"# {a.target.upper()} interval calibration by case study "
             f"(N={a.ctx}, pooled over d and shifts)", "",
             "Weighted by n_query. coverage95 target 0.95; read WITH length "
             "(a wide interval buys coverage). is05 prices both.", ""]
    for case in a.cases:
        methods = sorted({m for (c, m) in acc if c == case})
        if not methods:
            continue
        print(f"\n=== {case}  ({a.target}, N={a.ctx}) ===")
        hdr = (f"{'method':16s} {'cov95':>7s} {'length':>10s} {'is05':>10s} "
               f"{'sd_ratio':>9s} {'n_query':>10s} {'cells':>6s}")
        print(hdr); print("-" * len(hdr))
        lines += [f"## {case}", "",
                  "| method | coverage95 | length | is05 | sd_ratio | n_query | cells |",
                  "|---|---|---|---|---|---|---|"]
        for m in methods:
            items = acc[(case, m)]
            w = np.array([x[0] for x in items], dtype=float)
            def wm(k):
                v = np.array([float(x[1].get(k, np.nan)) for x in items])
                ok = np.isfinite(v)
                return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else float("nan")
            vals = {k: wm(k) for k in COLS}
            nq, nc = int(w.sum()), len(items)
            print(f"{m:16s} {vals['coverage95']:7.3f} {vals['length']:10.4f} "
                  f"{vals['is05']:10.4f} {vals['sd_ratio']:9.2f} {nq:10d} {nc:6d}")
            lines.append(f"| {m} | {vals['coverage95']:.3f} | {vals['length']:.4f} "
                         f"| {vals['is05']:.4f} | {vals['sd_ratio']:.2f} | {nq} | {nc} |")
        lines.append("")

    if a.out:
        with open(a.out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\n[written] {a.out}")


if __name__ == "__main__":
    main()
