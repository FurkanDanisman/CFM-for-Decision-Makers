#!/usr/bin/env python
"""One table per benchmark, every model, every metric we have.

Merges the two places results live:

  PEHE / ATE   markdown tables written by point_raw_em.py, beside the dumps
  Cov/Len/IS   per-realization .npz written by coverage_by_realization.py

Aggregation rules, which differ by metric and are not interchangeable:

  Cov/Len/IS   pooled by CONCATENATING per-realization arrays, then averaged.
               Case studies pool shifts 0/+2/-2 and all cases, which is what the
               case-study number means here.
  PEHE         pooled as a ROOT-MEAN-SQUARE across cells. PEHE is itself an RMSE,
               so averaging PEHE values understates it; only the squares add.
  ATE error    pooled as a plain mean, since it is already an absolute (or
               relative) error per cell.

Model naming: the shared root carries eight method rows, so its models are named
by method. Every other root holds one model and is named by the root.

    python benchmarks/final_table.py --perreal $SCRATCH/perreal
    python benchmarks/final_table.py --perreal $SCRATCH/perreal --out table.md
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np

_KEYS = ("cover", "length", "is05", "crps")
RC_DS = ["IHDP", "ACIC", "CPS", "PSID", "PSID_bal"]

# Released upstream weights, not this project's models. Marked so a reader cannot
# mistake a baseline for a contribution.
RELEASED = {"dopfn_native", "dopfn_bb"}

# Case-study d values to report. d=2 and d=3 are dropped by default: they exist in
# the dumps but are not part of the reported sweep.
CS_D_DEFAULT = ["5", "10", "20", "30", "40", "50"]

# label | realcause dump root | case-study dump root | single-model name or None
ROOTS = [
    ("orig",     "rc_dens_uni",                   "cs_dvar_dens",                   None),
    ("eta0",     "rc_dens_eta0",                  "cs_dvar_eta0",                   "cpfn2d_eta0"),
    ("J10",      "dumps_all/dopfn_repro_1d_J10/rc",  "dumps_all/dopfn_repro_1d_J10/cs",  "dopfn_repro_1d_J10"),
    ("J100",     "dumps_all/dopfn_repro_1d_J100/rc", "dumps_all/dopfn_repro_1d_J100/cs", "dopfn_repro_1d_J100"),
    ("joint2d",  "dumps_all/dopfn_repro_joint2d/rc", "dumps_all/dopfn_repro_joint2d/cs", "dopfn_repro_joint2d"),
    ("j32",      "dumps_all/cpfn1d_j32/rc",       "dumps_all/cpfn1d_j32/cs",        "cpfn1d_j32"),
    ("botharms", "dumps_all/cpfn1d_botharms/rc",  "dumps_all/cpfn1d_botharms/cs",   "cpfn1d_botharms"),
    ("cpfn_v0",  "dumps_all/cpfn_v0/rc",          "dumps_all/cpfn_v0/cs",           "cpfn_v0"),
    ("uwyk_bin", "dumps_all/uwyk_bin/rc",         "dumps_all/uwyk_bin/cs",          "uwyk_bin"),
]

_ROW = re.compile(r"^\|\s*([A-Za-z0-9_\-]+)\s*\|\s*(\d+)\s*\|(.*)\|\s*$")
_NUM = re.compile(r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")


def parse_point(path):
    """-> {method: (n, pehe, ate)} from a point_raw_em markdown table."""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        m = _ROW.match(line.rstrip("\n"))
        if not m:
            continue
        method, n, rest = m.group(1), int(m.group(2)), m.group(3)
        if method.lower() in ("method",):
            continue
        cells = [c.strip() for c in rest.split("|")]
        nums = []
        for c in cells[:2]:                     # PEHE (raw) | eps_ATE (raw)
            f = _NUM.search(c)
            nums.append(float(f.group(1)) if f else float("nan"))
        if len(nums) == 2:
            out[method] = (n, nums[0], nums[1])
    return out


def load_perreal(perreal, label, stage, pattern, keep_d=None):
    """-> {method: {key: [arrays]}} for one stage, over files matching pattern.

    keep_d restricts case-study files to those d values. File names carry the
    reporting group as d<D>_<Case>, so the d is read from the name rather than
    inferred from a directory.
    """
    acc = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(perreal, label, f"{stage}__{pattern}.npz"))):
        if keep_d is not None:
            m = re.match(r"^[a-z_]+__d(\d+)_", os.path.basename(f))
            if m and m.group(1) not in keep_d:
                continue
        try:
            z = np.load(f)
        except Exception:
            continue
        for k in z.files:
            if "__" not in k:
                continue
            method, key = k.rsplit("__", 1)
            if key not in _KEYS:
                continue
            a = np.asarray(z[k], dtype=float).ravel()
            if a.size:
                acc[method][key].append(a)
    return acc


def stat(arrays):
    if not arrays:
        return None
    v = np.concatenate(arrays)
    if v.size == 0:
        return None
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    return float(v.mean()), sd, int(v.size)


def fmt(s, prec=3):
    if s is None:
        return "—"
    return f"{s[0]:.{prec}f} ± {s[1]:.{prec}f}"


def display_name(label, method, single):
    return single if single else method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perreal", required=True)
    ap.add_argument("--scratch", default=os.environ.get("SCRATCH", ""))
    ap.add_argument("--out", default=None)
    ap.add_argument("--malc-tag", default="malc")
    ap.add_argument("--cs-d", nargs="+", default=CS_D_DEFAULT,
                    help="case-study d values to report (default drops d=2 and d=3)")
    a = ap.parse_args()
    SC = a.scratch or os.path.dirname(a.perreal.rstrip("/"))
    keep_d = [str(x) for x in a.cs_d]
    L = []

    def emit(s=""):
        L.append(s)

    # ── RealCause, one table per dataset ────────────────────────────────────
    for ds in RC_DS:
        rows = []
        for label, rc, cs, single in ROOTS:
            pt = parse_point(os.path.join(SC, rc, f"point_raw_em_{ds}.md"))
            raw = load_perreal(a.perreal, label, "raw", f"{ds}__-")
            mal = load_perreal(a.perreal, label, a.malc_tag, f"{ds}__-")
            ind = load_perreal(a.perreal, label, "indep_raw", f"{ds}__-")
            # The indep stage produces a MALC pass too; without this it was computed
            # and then never reported.
            indm = load_perreal(a.perreal, label, "indep_malc", f"{ds}__-")
            methods = sorted(set(pt) | set(raw) | set(mal) | set(ind) | set(indm))
            for meth in methods:
                n, pehe, ate = pt.get(meth, (None, float("nan"), float("nan")))
                r = {k: stat(raw.get(meth, {}).get(k, [])) for k in _KEYS}
                m = {k: stat(mal.get(meth, {}).get(k, [])) for k in _KEYS}
                i = {k: stat(ind.get(meth, {}).get(k, [])) for k in _KEYS}
                im = {k: stat(indm.get(meth, {}).get(k, [])) for k in _KEYS}
                if all(v is None for v in r.values()) and n is None:
                    continue
                rows.append((display_name(label, meth, single), n, pehe, ate, r, m, i, im))
        if not rows:
            continue
        emit(f"\n## RealCause — {ds}   (eps_ATE is RELATIVE)\n")
        emit("| model | n | PEHE | eps_ATE | Cov (raw) | Len (raw) "
             "| Cov (MALC) | Len (MALC) |")
        emit("|" + "---|" * 8)
        for nm, n, pehe, ate, r, m, i, im in sorted(rows, key=lambda t: t[2]):
            tag = " *(released)*" if nm in RELEASED else ""
            emit(f"| {nm}{tag} | {n if n else '—'} | "
                 f"{pehe:.4f} | {ate:.4f} | "
                 f"{fmt(r['cover'])} | {fmt(r['length'], 4)} | "
                 f"{fmt(m['cover'])} | {fmt(m['length'], 4)} |")

    # ── Case study: pooled over shifts, d and cases ─────────────────────────
    rows = []
    for label, rc, cs, single in ROOTS:
        # PEHE is an RMSE: only its squares pool. ATE error pools as a mean.
        per_method = defaultdict(lambda: {"pehe2": [], "ate": [], "n": 0, "cells": 0})
        for f in glob.glob(os.path.join(SC, cs, "shift*", "d*", "ctx*",
                                        "point_raw_em_*.md")):
            dm = re.search(r"/d(\d+)/", f)
            if dm and dm.group(1) not in keep_d:
                continue
            for meth, (n, pehe, ate) in parse_point(f).items():
                d = per_method[meth]
                if np.isfinite(pehe):
                    d["pehe2"].append(pehe ** 2)
                if np.isfinite(ate):
                    d["ate"].append(ate)
                d["n"] += n
                d["cells"] += 1
        raw = load_perreal(a.perreal, label, "raw", "*__shift*", keep_d)
        mal = load_perreal(a.perreal, label, a.malc_tag, "*__shift*", keep_d)
        methods = sorted(set(per_method) | set(raw) | set(mal))
        for meth in methods:
            d = per_method.get(meth)
            pehe = float(np.sqrt(np.mean(d["pehe2"]))) if d and d["pehe2"] else float("nan")
            ate = float(np.mean(d["ate"])) if d and d["ate"] else float("nan")
            cells = d["cells"] if d else 0
            r = {k: stat(raw.get(meth, {}).get(k, [])) for k in _KEYS}
            m = {k: stat(mal.get(meth, {}).get(k, [])) for k in _KEYS}
            if all(v is None for v in r.values()) and not np.isfinite(pehe):
                continue
            rows.append((display_name(label, meth, single), cells, pehe, ate, r, m))
    if rows:
        emit(f"\n## Case study — pooled over shifts 0/+2/-2, all cases, "
             f"d in {{{', '.join(keep_d)}}}   (L1_ATE is ABSOLUTE)\n")
        emit("| model | cells | PEHE (rms) | L1_ATE | Cov (raw) | Len (raw) "
             "| Cov (MALC) | Len (MALC) |")
        emit("|" + "---|" * 8)
        for nm, cells, pehe, ate, r, m in sorted(rows, key=lambda t: t[2]):
            tag = " *(released)*" if nm in RELEASED else ""
            emit(f"| {nm}{tag} | {cells} | {pehe:.4f} | {ate:.4f} | "
                 f"{fmt(r['cover'])} | {fmt(r['length'], 4)} | "
                 f"{fmt(m['cover'])} | {fmt(m['length'], 4)} |")

    emit("\n---\n")
    emit("Cov/Len/IS pool per-realization arrays by concatenation, so how the work")
    emit("was split across jobs does not change the number. PEHE pools as an RMS")
    emit("across cells because PEHE is itself an RMSE; ATE error pools as a mean.")
    emit("RealCause reports RELATIVE eps_ATE, the case studies absolute L1_ATE.")
    emit("Models marked *(released)* are upstream DoPFN weights, not this project's.")
    emit("The forced-independent ablation is computed but not shown here; it is a")
    emit("RealCause-only 2D diagnostic rather than a headline result.")
    emit("A dash means that stage has not been scored yet.")

    txt = "\n".join(L)
    print(txt)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(txt + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
