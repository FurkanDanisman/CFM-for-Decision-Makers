"""Completion percentages for an eval sweep, point and density separately.

Counts realization NPZs per cell and reports them against the expected total,
so a sweep in flight can be read at a glance instead of by eyeballing squeue.

Point-estimate and density runs are tracked separately because they are
different jobs writing different trees, and because the point dumps have no
per-query densities -- a cell can be complete for PEHE and empty for
calibration at the same time.

EXPECTED TOTALS. RealCause and the case studies are fixed by the benchmark
(realization counts per dataset; 3 shifts x 8 d x 6 cases x 100). ComplexMech's
are read off a reference root rather than hardcoded, because its realization
count per cell varies (119-141 files) with the subset split.

Layouts, which differ between the point and density paths:
    point    <root>/realcause/<DS>/                       (no model subdir)
             <root>/N<ctx>/<model>/CMECH_n<d>_<subset>/
             <root>/shift<S>/d<D>/ctx<N>/<model>/<CASE>/
    density  <root>/<model>/<DS>/
             ... the other two as above

Usage:
    python benchmarks/eta0_progress.py                       # defaults below
    python benchmarks/eta0_progress.py --model cpfn2d_pooled \\
        --pt-rc $SCRATCH/pt_eta0 --dens-rc $SCRATCH/rc_dens_eta0
"""
from __future__ import annotations

import argparse
import glob
import os

_RC = {"IHDP": 100, "ACIC": 10, "CPS": 100, "PSID": 100, "PSID_bal": 100}
_CASES = ["Observed_Confounder", "Observed_Mediator",
          "Observed_Mediator_and_Confounder", "Unobserved_Confounder",
          "Frontdoor_Criterion", "Backdoor_Criterion"]
_SHIFTS = ["0", "+2", "-2"]
_DS = [2, 3, 5, 10, 20, 30, 40, 50]
_NODES = [5, 10, 20, 30, 40, 50]
_SUBSETS = ["nonzero", "zero"]


def n_npz(pat):
    return sum(1 for f in glob.glob(pat) if not f.endswith("summary.npz"))


def bar(done, total, width=22):
    if not total:
        return "?" * width
    k = int(round(width * min(done / total, 1.0)))
    return "#" * k + "." * (width - k)


def line(label, done, total):
    pct = (100.0 * done / total) if total else float("nan")
    print(f"  {label:34s} {bar(done,total)} {done:6d}/{total:<6d} {pct:5.1f}%")
    return done, total


def rc(root, model, is_point):
    tot = don = 0
    have = bool(root) and os.path.isdir(root)
    if not have:
        print(f"  (root absent: {root})")
    for ds, n in _RC.items():
        sub = f"{root}/realcause/{ds}" if is_point else f"{root}/{model}/{ds}"
        d, t = line(ds, n_npz(f"{sub}/*.npz") if have else 0, n)
        don += d; tot += t
    return don, tot


def cmech(root, model, ref):
    """Expected counts come from `ref`, so the target is right even when the
    new root does not exist yet -- otherwise an absent root silently drops
    ComplexMech out of the TOTAL denominator and the overall percentage reads
    higher than it is."""
    tot = don = 0
    have = bool(root) and os.path.isdir(root)
    if not have:
        print(f"  (root absent: {root} -- targets from {ref})")
    for nd in _NODES:
        for ss in _SUBSETS:
            cell = f"N1000/{model}/CMECH_n{nd}_{ss}"
            exp = n_npz(f"{ref}/{cell}/*.npz") if ref else 0
            got = n_npz(f"{root}/{cell}/*.npz") if have else 0
            if have or exp:
                d, t = line(f"d={nd} {ss}", got, exp)
                don += d; tot += t
    return don, tot


def cs(root, model, ctx=1000, per_case=100):
    """Per-shift rows are printed even when the root is absent, so an
    unstarted benchmark shows as explicit 0% rows rather than one note. The
    denominator was always correct; only the display differed from cmech()."""
    tot = don = 0
    have = bool(root) and os.path.isdir(root)
    if not have:
        print(f"  (root absent: {root})")
    for sh in _SHIFTS:
        got = (sum(n_npz(f"{root}/shift{sh}/d{d}/ctx{ctx}/{model}/{c}/*.npz")
                   for d in _DS for c in _CASES) if have else 0)
        d_, t_ = line(f"shift {sh}  ({len(_DS)} d x {len(_CASES)} cases)",
                      got, len(_DS) * len(_CASES) * per_case)
        don += d_; tot += t_
    return don, tot


def main():
    S = os.environ.get("SCRATCH", "")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="cpfn2d_pooled")
    ap.add_argument("--pt-rc", default=f"{S}/pt_eta0")
    ap.add_argument("--pt-cm", default=f"{S}/ptcm_eta0")
    ap.add_argument("--pt-cs", default=f"{S}/ptcs_eta0")
    ap.add_argument("--dens-rc", default=f"{S}/rc_dens_eta0")
    ap.add_argument("--dens-cm", default=f"{S}/cmech_eta0")
    ap.add_argument("--dens-cs", default=f"{S}/cs_dvar_eta0")
    ap.add_argument("--ref-cm", default=f"{S}/cmech_dens",
                    help="reference root for ComplexMech expected counts")
    a = ap.parse_args()

    for title, kind in (("POINT ESTIMATES  (PEHE / ATE error)", "point"),
                        ("DENSITY  (coverage / length / IS)", "dens")):
        pt = kind == "point"
        print(f"\n{'='*72}\n  {title}\n{'='*72}")
        grand_d = grand_t = 0
        print(" RealCause")
        d, t = rc(a.pt_rc if pt else a.dens_rc, a.model, pt); grand_d += d; grand_t += t
        print(" ComplexMech  (N=1000)")
        d, t = cmech(a.pt_cm if pt else a.dens_cm, a.model, a.ref_cm); grand_d += d; grand_t += t
        print(" Case studies")
        d, t = cs(a.pt_cs if pt else a.dens_cs, a.model); grand_d += d; grand_t += t
        pct = (100.0 * grand_d / grand_t) if grand_t else float("nan")
        print(f"  {'TOTAL':34s} {bar(grand_d,grand_t)} {grand_d:6d}/{grand_t:<6d} {pct:5.1f}%")


if __name__ == "__main__":
    main()
