#!/usr/bin/env python
"""Case-study point estimates: PEHE, L1_ATE and RELATIVE eps_ATE, one table per case.

The case studies only ever had absolute L1_ATE, because point_raw_em was invoked
with --ate-metric l1 and the two metrics cannot be converted after the fact: the
relative form needs each realization's own true ATE, which the markdown does not
carry. This rescores the existing dumps (no inference) and computes both in one
pass, so they are guaranteed consistent.

Aggregation matches final_table: one table PER CASE, pooled over shifts (0/+2/-2)
and over the reported d values. The mechanism is what is being compared, so cases
stay separate; d is a nuisance axis. PEHE pools as a root-mean-square because it is
itself an RMSE; both ATE errors pool as means.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "UWYK_Fig3_4"))
sys.path.insert(0, os.path.join(_REPO, "realcause_eval"))

from cate_density_metrics import METHODS, _resolve_dir        # noqa: E402
from point_raw_em import cate_for_file, _files_in             # noqa: E402
from final_table import ROOTS, CS_D_DEFAULT, display_name, RELEASED  # noqa: E402

CASES = ("Observed_Confounder", "Backdoor_Criterion", "Observed_Mediator",
         "Observed_Mediator_and_Confounder", "Unobserved_Confounder",
         "Frontdoor_Criterion")


def score_cell(cell, subdir, tag, case, mode="raw"):
    """Per-realization (pehe, l1, rel) for one (cell, method, case)."""
    d = _resolve_dir(cell, subdir, case)
    if not d or not os.path.isdir(d):
        return []
    out = []
    for f in _files_in(d):
        try:
            got = cate_for_file(f, tag, mode)
        except Exception:
            got = None
        if got is None:
            continue
        tau = np.asarray(got[0], float).ravel()
        tru = np.asarray(got[1], float).ravel()
        if tau.size == 0 or not np.isfinite(tau).all():
            continue
        pehe = float(np.sqrt(np.mean((tau - tru) ** 2)))
        l1 = float(abs(tau.mean() - tru.mean()))
        out.append((pehe, l1, l1 / max(abs(float(tru.mean())), 0.1)))
    return out


def _cell_d(cell):
    """The d value this cell belongs to, read from its path."""
    m = re.search(r"/d(\d+)/", cell)
    return m.group(1) if m else "?"


def _one_cell(task):
    """(label, single, cell, case, mode) -> [(key, display_name, pehe, l1, rel)].

    One (cell, case) is the work unit: independent of every other, and big enough
    (16 method entries x ~100 realizations) that process overhead is negligible.
    """
    label, single, cell, case, mode, by_d = task
    key = f"{case}|d{_cell_d(cell)}" if by_d else case
    out = []
    for meth, subdir, tag in METHODS:
        for pehe, l1, rel in score_cell(cell, subdir, tag, case, mode):
            out.append((key, display_name(label, meth, single), pehe, l1, rel))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default=os.environ.get("SCRATCH", ""))
    ap.add_argument("--cs-d", nargs="+", default=CS_D_DEFAULT)
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--mode", default="raw", choices=["raw", "em"])
    ap.add_argument("--by-d", action="store_true",
                    help="one table per (case, d) instead of pooling over d")
    ap.add_argument("--csv", default=None,
                    help="also write tidy long-form rows for plotting")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1") or 1),
                    help="parallel worker processes (default: $SLURM_CPUS_PER_TASK)")
    a = ap.parse_args()
    SC = a.scratch
    keep_d = {str(x) for x in a.cs_d}

    # (case, model) -> lists over (cell x realization)
    acc = defaultdict(lambda: defaultdict(
        lambda: {"pehe2": [], "l1": [], "rel": [], "cells": 0, "reals": 0}))
    tasks = []
    for label, rc, cs, single in ROOTS:
        root = os.path.join(SC, cs)
        if not os.path.isdir(root):
            continue
        cells = sorted(glob.glob(os.path.join(root, "shift*", "d*", "ctx*")))
        cells = [c for c in cells
                 if (m := re.search(r"/d(\d+)/", c)) and m.group(1) in keep_d]
        if not cells:
            continue
        print(f"[plan] {label}: {len(cells)} cell(s)", flush=True)
        for cell in cells:
            for case in a.cases:
                tasks.append((label, single, cell, case, a.mode, a.by_d))

    nw = max(1, int(a.workers))
    print(f"[plan] {len(tasks)} (cell, case) units on {nw} worker(s)", flush=True)

    def absorb(rows, seen):
        """One returned unit -> accumulator. `cells` counts (cell, case, model)
        units contributing, matching the previous sequential meaning."""
        for case, nm, pehe, l1, rel in rows:
            d = acc[case][nm]
            if (case, nm) not in seen:
                d["cells"] += 1
                seen.add((case, nm))
            d["reals"] += 1
            d["pehe2"].append(pehe ** 2)
            d["l1"].append(l1)
            d["rel"].append(rel)

    done = 0
    if nw == 1:
        for t in tasks:
            absorb(_one_cell(t), set())
            done += 1
            if done % 20 == 0:
                print(f"[progress] {done}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=nw) as ex:
            for rows in ex.map(_one_cell, tasks, chunksize=1):
                absorb(rows, set())
                done += 1
                if done % 20 == 0:
                    print(f"[progress] {done}/{len(tasks)}", flush=True)

    L = []
    def _split(k):
        return (k.split("|", 1) + [""])[:2] if "|" in k else (k, "")
    keys = sorted(acc, key=lambda k: (a.cases.index(_split(k)[0])
                                      if _split(k)[0] in a.cases else 99,
                                      int(_split(k)[1][1:]) if _split(k)[1] else 0))
    csv_rows = []
    for case in keys:
        if case not in acc:
            continue
        _c, _d = _split(case)
        L += ["", f"## Case study — {_c}"
                  + (f", d={_d[1:]}" if _d else "")
                  + f"   (pooled over shifts 0/+2/-2 and "
                  f"d in {{{', '.join(sorted(keep_d, key=int))}}})", "",
              "| model | cells | realizations | PEHE | L1_ATE "
              "| eps_ATE (relative) |",
              "|---|---|---|---|---|---|"]
        rows = []
        for nm, d in acc[case].items():
            if not d["pehe2"]:
                continue
            sem = lambda v: (float(np.std(v, ddof=1) / np.sqrt(len(v)))
                             if len(v) > 1 else float("nan"))
            # MEAN is primary: point_raw_em (and therefore every previously
            # published case-study number) reports the mean of per-realization
            # PEHE. RMS is kept beside it, not instead of it.
            per = np.sqrt(np.asarray(d["pehe2"]))
            rows.append((nm, d["cells"], d["reals"],
                         float(per.mean()), sem(per),
                         float(np.sqrt(np.mean(d["pehe2"]))),
                         float(np.mean(d["l1"])), sem(d["l1"]),
                         float(np.mean(d["rel"])), sem(d["rel"])))
        for (nm, nc, nr, pm, pse, prms, l1, l1e, rel,
             rele) in sorted(rows, key=lambda t: t[3]):
            tg = " *(released)*" if nm in RELEASED else ""
            L.append(f"| {nm}{tg} | {nc} | {nr} | {pm:.4f} ± {pse:.4f} | "
                     f"{l1:.4f} ± {l1e:.4f} | {rel:.4f} ± {rele:.4f} |")
            csv_rows.append([_c, _d[1:] if _d else "all", nm, nr,
                             f"{pm:.6f}", f"{pse:.6f}", f"{l1:.6f}",
                             f"{l1e:.6f}", f"{rel:.6f}", f"{rele:.6f}"])
    L += ["",
          "All three are means +- SEM over (cell, realization), matching the\nconvention point_raw_em uses elsewhere (mode=raw throughout; EM is never\ncomputed).",
          "eps_ATE = L1 / max(|true ATE|, 0.1). The 0.1 floor matches",
          "eval_dopfn_bb_raw, so a near-zero true ATE cannot turn a small absolute",
          "error into an enormous ratio -- but it also means that wherever the true",
          "ATE is below 0.1 the relative column is just 10x L1 rather than a genuine",
          "ratio. Check that before quoting it."]
    if a.csv:
        import csv as _csv
        os.makedirs(os.path.dirname(os.path.abspath(a.csv)) or ".", exist_ok=True)
        with open(a.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["case", "d", "model", "n_real", "pehe", "pehe_sem",
                        "l1_ate", "l1_sem", "eps_ate", "eps_sem"])
            w.writerows(csv_rows)
        print(f"wrote {a.csv} ({len(csv_rows)} rows)", file=sys.stderr)

    txt = "\n".join(L)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
