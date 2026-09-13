"""Completion dashboard for the density sweep: % done per model x case study.

Pure filesystem scan -- no scoring, so it is instant and safe to run repeatedly
while jobs are in flight.

A cell is (shift, d, N, model, case). It counts as:
    done     metrics.json present
    dumped   density npz present but not yet scored (job still running / died
             after the eval but before the scorer)
    missing  nothing

    python case_study/density_eval/progress.py --root $RES
    python case_study/density_eval/progress.py --root $RES --by case --n 1000
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict

import numpy as np

# A density dump carries these; a POINT-eval npz does not. Checking the key
# list (not the arrays) is cheap -- an npz is a zip and numpy reads the
# namelist without decompressing.
_DENSITY_KEYS = ("p_joint_scaled", "p_y0_scaled")


def _is_density_npz(path):
    try:
        with np.load(path, allow_pickle=True) as z:
            return any(k in z.files for k in _DENSITY_KEYS)
    except Exception:
        return False

SHIFTS = ["shift0", "shift+2", "shift-2"]
DS = [2, 3, 5, 10, 20, 30, 40, 50]
CTX = [50, 100, 250, 500, 1000]
CASES = ["Observed_Confounder", "Backdoor_Criterion", "Observed_Mediator",
         "Observed_Mediator_and_Confounder", "Unobserved_Confounder",
         "Frontdoor_Criterion"]
MODELS = ["cpfn1d", "cpfn2d", "graph2d", "uwyk", "uwyk_v3a", "uwyk_noanc",
          "dopfn_native", "dopfn_bb"]
SHORT = {"Observed_Confounder": "ObsConf", "Backdoor_Criterion": "Backdoor",
         "Observed_Mediator": "ObsMed",
         "Observed_Mediator_and_Confounder": "ObsMed+Conf",
         "Unobserved_Confounder": "UnobsConf",
         "Frontdoor_Criterion": "Frontdoor"}


def scan(root, shifts, ds, ctx, models, cases):
    """-> {(model, case): [n_done, n_dumped, n_total]}"""
    st = defaultdict(lambda: [0, 0, 0])
    for s in shifts:
        for d in ds:
            for n in ctx:
                for m in models:
                    for c in cases:
                        cell = os.path.join(root, s, f"d{d}", f"ctx{n}", m, c)
                        k = (m, c)
                        st[k][2] += 1
                        if os.path.isfile(os.path.join(cell, "metrics.json")):
                            st[k][0] += 1
                            continue
                        # Only count npz that are actually DENSITY dumps: these
                        # directories also hold point-eval npz from the earlier
                        # dsweep run, and counting those reports 100% dumped for
                        # models that have produced nothing.
                        cand = sorted(glob.glob(os.path.join(cell, "*.npz")))
                        cand = [f for f in cand
                                if "summary" not in os.path.basename(f)]
                        if cand and _is_density_npz(cand[0]):
                            st[k][1] += 1
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--shifts", nargs="*", default=SHIFTS)
    ap.add_argument("--ds", nargs="*", type=int, default=DS)
    ap.add_argument("--contexts", nargs="*", type=int, default=CTX)
    ap.add_argument("--n", type=int, default=None, help="shorthand for one context")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--cases", nargs="*", default=CASES)
    a = ap.parse_args()
    ctx = [a.n] if a.n is not None else a.contexts

    st = scan(a.root, a.shifts, a.ds, ctx, a.models, a.cases)
    per_cell = len(a.shifts) * len(a.ds) * len(ctx)
    print(f"root={a.root}")
    print(f"grid: {len(a.shifts)} shifts x {len(a.ds)} d x {len(ctx)} contexts "
          f"= {per_cell} cells per (model, case)\n")

    w = max(len(m) for m in a.models) + 1
    cols = [SHORT.get(c, c)[:11] for c in a.cases]
    print(f"{'model':{w}s}" + "".join(f"{c:>13s}" for c in cols) + f"{'TOTAL':>13s}")
    print("-" * (w + 13 * (len(cols) + 1)))
    grand = [0, 0, 0]
    for m in a.models:
        line = f"{m:{w}s}"
        tot = [0, 0, 0]
        for c in a.cases:
            done, dump, n = st[(m, c)]
            for i, v in enumerate((done, dump, n)):
                tot[i] += v
            pct = 100.0 * done / n if n else 0.0
            mark = f"{pct:5.0f}%"
            if dump:
                mark += f" +{dump}"
            line += f"{mark:>13s}"
        pct = 100.0 * tot[0] / tot[2] if tot[2] else 0.0
        line += f"{pct:12.0f}%"
        print(line)
        for i in range(3):
            grand[i] += tot[i]
    print("-" * (w + 13 * (len(cols) + 1)))
    print(f"{'ALL':{w}s}" + " " * (13 * len(cols))
          + f"{100.0 * grand[0] / grand[2] if grand[2] else 0:12.0f}%")
    print(f"\n{grand[0]} scored / {grand[2]} cells"
          + (f"   ({grand[1]} dumped but not scored)" if grand[1] else ""))
    if grand[1]:
        print("  '+N' = DENSITY npz present, metrics.json not written yet "
              "(running, or the scorer failed).")
        print("        Point-eval npz in the same directory are ignored.")


if __name__ == "__main__":
    main()
