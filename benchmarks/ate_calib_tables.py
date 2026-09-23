#!/usr/bin/env python
"""ATE-interval calibration for the case studies and ComplexMech.

A DIFFERENT claim from CATE coverage: one interval for the average effect per
realization, not one per query. The scorer already writes these for the case
studies (submit_full_table runs --target ate in both the raw and malc stages) and
final_table never read them -- its ATE section globs only the RealCause spelling
`calib_<ds>_raw_ate.md`, while the case studies are `calib_raw_ate_<case>.md`.

ComplexMech had no ATE stage at all in the scorer that was run, so those files
exist only after submit_ate_calib.sbatch has produced them; absent, the rows are
reported as missing rather than silently skipped.

Pooling: coverage / length / IS are averaged across cells WEIGHTED BY n_files, so
a cell that scored more realizations counts proportionally. n is the total.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from final_table import ROOTS, CS_D_DEFAULT, display_name, RELEASED, parse_calib  # noqa: E402

CASES = ("Observed_Confounder", "Backdoor_Criterion", "Observed_Mediator",
         "Observed_Mediator_and_Confounder", "Unobserved_Confounder",
         "Frontdoor_Criterion")


def _pool(entries):
    """[(n_files, cov, len, is05)] -> (n_total, cov, len, is05) weighted by n_files."""
    e = [x for x in entries if x and np.isfinite(x[1])]
    if not e:
        return None
    w = np.asarray([x[0] for x in e], float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    out = [int(w.sum())]
    for j in (1, 2, 3):
        v = np.asarray([x[j] for x in e], float)
        ok = np.isfinite(v)
        out.append(float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any()
                   else float("nan"))
    return tuple(out)


def case_study(SC, keep_d, cases, malc_tag):
    """-> {case: {model: {'raw': pooled, 'malc': pooled}}}"""
    acc: dict = {}
    for label, rc, cs, single in ROOTS:
        root = os.path.join(SC, cs)
        if not os.path.isdir(root):
            continue
        for cell in sorted(glob.glob(os.path.join(root, "shift*", "d*", "ctx*"))):
            m = re.search(r"/d(\d+)/", cell)
            if not m or m.group(1) not in keep_d:
                continue
            for case in cases:
                raw = parse_calib(os.path.join(cell, f"calib_raw_ate_{case}.md"))
                mal = {}
                for cand in sorted(glob.glob(
                        os.path.join(cell, f"calib_T_*_ate_{case}.md"))):
                    mal = parse_calib(cand)
                for meth in sorted(set(raw) | set(mal)):
                    nm = display_name(label, meth, single)
                    d = acc.setdefault(case, {}).setdefault(nm, {"raw": [], "malc": []})
                    if meth in raw:
                        d["raw"].append(raw[meth])
                    if meth in mal:
                        d["malc"].append(mal[meth])
    return {c: {nm: {k: _pool(v[k]) for k in ("raw", "malc")}
                for nm, v in models.items()} for c, models in acc.items()}


def cmech(dumps, nodes, ctx, malc_tag):
    """-> {node: {model: {'raw': pooled, 'malc': pooled}}} from calib_*ate*.md."""
    acc: dict = {}
    for md in sorted(d for d in glob.glob(f"{dumps}/*") if os.path.isdir(d)):
        name = os.path.basename(md)
        for n in nodes:
            raw, mal = {}, {}
            for p in glob.glob(f"{md}/**/calib_cmech_n{n}_raw_ate.md", recursive=True):
                raw = parse_calib(p)
            for p in sorted(glob.glob(
                    f"{md}/**/calib_cmech_n{n}_T_*_ate.md", recursive=True)):
                mal = parse_calib(p)
            for meth in sorted(set(raw) | set(mal)):
                nm = display_name(name, meth, name)
                d = acc.setdefault(n, {}).setdefault(nm, {"raw": [], "malc": []})
                if meth in raw:
                    d["raw"].append(raw[meth])
                if meth in mal:
                    d["malc"].append(mal[meth])
    return {n: {nm: {k: _pool(v[k]) for k in ("raw", "malc")}
                for nm, v in models.items()} for n, models in acc.items()}


def _emit(L, title, models):
    L += ["", f"## {title}", "",
          "| model | n | Cov (raw) | Len (raw) | IS (raw) "
          "| Cov (MALC) | Len (MALC) | IS (MALC) |", "|" + "---|" * 8]
    f = lambda v: ("—" if v is None or not np.isfinite(v) else f"{v:.4f}")
    rows = []
    for nm, d in models.items():
        r, m = d["raw"], d["malc"]
        if r is None and m is None:
            continue
        n = (r or m)[0]
        rows.append((nm, n, r, m))
    for nm, n, r, m in sorted(rows, key=lambda t: t[0]):
        tg = " *(released)*" if nm in RELEASED else ""
        L.append(f"| {nm}{tg} | {n} | "
                 + " | ".join(f(r[j]) if r else "—" for j in (1, 2, 3)) + " | "
                 + " | ".join(f(m[j]) if m else "—" for j in (1, 2, 3)) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default=os.environ.get("SCRATCH", ""))
    ap.add_argument("--cmech-dumps", default=None)
    ap.add_argument("--nodes", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50])
    ap.add_argument("--context", type=int, default=1000)
    ap.add_argument("--cs-d", nargs="+", default=CS_D_DEFAULT)
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--malc-tag", default="malc")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    SC = a.scratch
    keep_d = {str(x) for x in a.cs_d}

    L = ["# ATE-interval calibration — case studies and ComplexMech", "",
         "One interval for the AVERAGE effect per realization, not one per query —",
         "a different claim from the CATE coverage in the main tables. Cells are",
         "pooled weighted by the number of realizations each contributed.", ""]

    cs = case_study(SC, keep_d, a.cases, a.malc_tag)
    if cs:
        L += ["", "---", "", "# Case studies", ""]
        for case in a.cases:
            if case in cs:
                _emit(L, f"Case study — {case}   (pooled over shifts and d in "
                         f"{{{', '.join(sorted(keep_d, key=int))}}})", cs[case])
    else:
        print(f"note: no case-study calib_*_ate_*.md found under {SC}", file=sys.stderr)

    dumps = a.cmech_dumps or os.path.join(SC, "cmech_dumps")
    cm = cmech(dumps, a.nodes, a.context, a.malc_tag)
    if cm:
        L += ["", "---", "", f"# ComplexMech  ({os.path.basename(dumps)})", ""]
        for n in a.nodes:
            if n in cm:
                _emit(L, f"ComplexMech — n={n}", cm[n])
    else:
        msg = (f"ComplexMech ATE calibration is ABSENT under {dumps}. The scorer that "
               f"was run (submit_cmech_score_one.sbatch) has only point/raw/malc CATE "
               f"stages; run submit_ate_calib.sbatch to produce it.")
        print(f"note: {msg}", file=sys.stderr)
        L += ["", "---", "", "# ComplexMech", "", f"_{msg}_", ""]

    txt = "\n".join(L)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
