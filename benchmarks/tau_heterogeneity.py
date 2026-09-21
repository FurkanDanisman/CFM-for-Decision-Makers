#!/usr/bin/env python
"""Measure tau's heterogeneity per realization, from the density dumps.

WHY FROM THE DUMPS. The case-study data files may store only the FACTUAL outcome,
in which case Y(0) and Y(1) are never both observed for a unit and rho is not
recoverable from them (check_arm_noise_coupling says exactly this). The dumps,
however, must carry the true tau per query in order to have been scored at all --
so tau's spread IS measurable there even where rho is not.

Reported per realization, then summarised across realizations:

    sd(tau)              absolute spread of the true effect across units
    sd(tau)/|mean(tau)|  coefficient of variation, scale-free
    frac zero-spread     realizations whose tau is constant across units

The last is the one that bears on coverage. A realization with constant tau gives
every query the same truth, so its coverage can only be 0 or 1 -- no model choice
changes that. On ComplexMech this was 13% of realizations before filtering.

Truth is read through cate_density_metrics' own loader, so "tau" here means
exactly what the scorer scored, not a re-derivation of it.

    python benchmarks/tau_heterogeneity.py --root $SCRATCH/cs_dvar_dens \
        --case-study --label case_study
    python benchmarks/tau_heterogeneity.py --root $SCRATCH/rc_dens_uni \
        --dataset IHDP --label realcause_IHDP
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "UWYK_Fig3_4"))
from cate_density_metrics import METHODS, _load_arrays_raw, _resolve_dir  # noqa: E402


def truth_of(path):
    """True tau per query for one realization npz, or None."""
    for _lbl, subdir, tag in METHODS:
        got = _load_arrays_raw(path, tag)
        if got is not None:
            return np.asarray(got[2], dtype=float).ravel()
    return None


def collect(files, zero_tol):
    rows = []
    for f in files:
        t = truth_of(f)
        if t is None or t.size < 2:
            continue
        sd = float(t.std(ddof=1))
        mu = float(t.mean())
        rows.append((sd, sd / max(abs(mu), 1e-12), abs(mu), t.size))
    return rows


def q(v, p):
    a = np.asarray(v, dtype=float)
    return float(np.quantile(a, p)) if a.size else float("nan")


def report(label, rows, zero_tol):
    if not rows:
        print(f"\n### {label}: no scorable realizations found")
        return
    sd = np.array([r[0] for r in rows])
    cv = np.array([r[1] for r in rows])
    nq = np.array([r[3] for r in rows])
    zero = sd <= zero_tol
    print(f"\n### {label}")
    print(f"  realizations       n = {len(rows)}   queries/realization "
          f"median = {int(np.median(nq))}")
    print(f"  sd(tau)            mean = {sd.mean():.4g}  sd = {sd.std(ddof=1):.4g}")
    print(f"                     p10 = {q(sd,.10):.4g}  p50 = {q(sd,.50):.4g}  "
          f"p90 = {q(sd,.90):.4g}")
    print(f"  sd(tau)/|E tau|    p10 = {q(cv,.10):.4g}  p50 = {q(cv,.50):.4g}  "
          f"p90 = {q(cv,.90):.4g}")
    print(f"  dynamic range      max/min sd(tau) = "
          f"{sd.max()/max(sd.min(),1e-300):.4g}")
    print(f"  ZERO-SPREAD tau    {int(zero.sum())}/{len(rows)} "
          f"({100.0*zero.mean():.1f}%)   [sd(tau) <= {zero_tol:g}]")
    print(f"    -> those realizations can only report coverage 0 or 1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, nargs="+")
    ap.add_argument("--dataset", default=None, help="RealCause dataset name")
    ap.add_argument("--case-study", action="store_true",
                    help="walk shift*/d*/ctx*/<subdir>/<Case>/ instead")
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--label", default="tau")
    ap.add_argument("--zero-tol", type=float, default=1e-12)
    ap.add_argument("--max-real", type=int, default=None)
    ap.add_argument("--by-cell", action="store_true",
                    help="report each (shift,d,case) cell separately as well")
    a = ap.parse_args()

    groups = {}
    for root in a.root:
        if a.case_study:
            for cell in sorted(glob.glob(os.path.join(root, "shift*", "d*", "ctx*"))):
                rel = os.path.relpath(cell, root)
                for _lbl, subdir, _tag in METHODS:
                    base = os.path.join(cell, subdir)
                    if not os.path.isdir(base):
                        continue
                    for case in sorted(os.listdir(base)):
                        d = os.path.join(base, case)
                        if not os.path.isdir(d):
                            continue
                        fs = sorted(glob.glob(os.path.join(d, "*.npz")))
                        fs = [f for f in fs if os.path.basename(f) != "summary.npz"]
                        if a.cases and case not in a.cases:
                            continue
                        if fs:
                            groups.setdefault(f"{rel}/{case}", []).extend(fs)
                    break          # one method's dumps suffice: truth is shared
        else:
            for _lbl, subdir, _tag in METHODS:
                d = _resolve_dir(root, subdir, a.dataset)
                if d and os.path.isdir(d):
                    fs = sorted(glob.glob(os.path.join(d, "*.npz")))
                    fs = [f for f in fs if os.path.basename(f) != "summary.npz"]
                    if fs:
                        groups.setdefault(a.dataset or "all", []).extend(fs)
                        break

    if not groups:
        print("no dumps found under the given root(s)")
        return 1

    pooled = []
    for name in sorted(groups):
        fs = groups[name][: a.max_real] if a.max_real else groups[name]
        rows = collect(fs, a.zero_tol)
        pooled += rows
        if a.by_cell:
            report(name, rows, a.zero_tol)
    report(f"{a.label} — ALL CELLS POOLED", pooled, a.zero_tol)
    print("\nTruth is read via cate_density_metrics' own loader, so tau here is")
    print("exactly what the scorer scored. Only ONE method's dumps are read per")
    print("cell: the true effect does not depend on which model predicted it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
