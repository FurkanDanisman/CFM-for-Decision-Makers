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
import json
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
    """-> {method: (n, pehe, ate, pehe_str, ate_str)} from a point_raw_em table.

    The STRINGS are kept because point_raw_em writes "0.1860 ± 0.0021" and this
    parser used to extract only the leading float, so every table reprinted the
    point estimate with its SEM silently discarded. Floats are still returned for
    sorting and for the RMS pooling of PEHE.
    """
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
        nums, strs = [], []
        for c in cells[:2]:                     # PEHE (raw) | eps_ATE (raw)
            f = _NUM.search(c)
            nums.append(float(f.group(1)) if f else float("nan"))
            strs.append(c if c else "—")
        if len(nums) == 2:
            out[method] = (n, nums[0], nums[1], strs[0], strs[1])
    return out


_CAL_ROW = re.compile(r"^\|\s*([A-Za-z0-9_\-]+)\s*\|(.*)\|\s*$")


def parse_calib(path):
    """-> {method: (n_files, coverage, length, is05)} from a cate_density_metrics table.

    This is the ATE-target calibration, which no table has ever shown: the scorer
    writes calib_*_ate.md files and final_table only ever read the CATE side, so the
    ATE interval's coverage was computed and then discarded. Columns are
    method | n_files | n_query | coverage95 | length | is05 | ...
    """
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        m = _CAL_ROW.match(line.rstrip("\n"))
        if not m:
            continue
        meth, rest = m.group(1), m.group(2)
        if meth.lower() == "method":
            continue
        cells = [c.strip() for c in rest.split("|")]
        if len(cells) < 5:
            continue
        def num(i):
            f = _NUM.search(cells[i]) if i < len(cells) else None
            return float(f.group(1)) if f else float("nan")
        n_files = num(0)
        if not np.isfinite(n_files):
            continue
        out[meth] = (int(n_files), num(2), num(3), num(4))
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


# sd(Y) per reporting group, from length_normalizers.py. Empty unless
# --normalizers is given, in which case every Len/sd(Y) cell prints as "—".
NORM: dict = {}


def nsd(group, key):
    """sd(Y) for one reporting group, or None when no normaliser is available."""
    try:
        v = float(NORM.get(group, {}).get(str(key)))
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 0 else None


def fmtn(s, sd, prec=3):
    """Length in units of outcome spread. Mean and sd both scale by the same
    constant, so the +- carries over unchanged."""
    if s is None or sd is None:
        return "—"
    return f"{s[0] / sd:.{prec}f} ± {s[1] / sd:.{prec}f}"


def display_name(label, method, single):
    """Row name. A single-model root normally supplies its own name, but a root can
    still carry SEVERAL method rows -- uwyk_bin has both a noanc and a v3a ancestry
    variant -- and collapsing those to the root label reported two different
    configurations under one name. Keep the distinguishing suffix when there is one.
    """
    if not single:
        return method
    for suffix in ("-noanc", "-v3a", "-v3ab"):
        if method.endswith(suffix):
            return single + suffix
    return single


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perreal", required=True)
    ap.add_argument("--scratch", default=os.environ.get("SCRATCH", ""))
    ap.add_argument("--out", default=None)
    ap.add_argument("--malc-tag", default="malc")
    ap.add_argument("--normalizers", default=None,
                    help="JSON from length_normalizers.py; adds Len/sd(Y) columns")
    ap.add_argument("--cs-d", nargs="+", default=CS_D_DEFAULT,
                    help="case-study d values to report (default drops d=2 and d=3)")
    a = ap.parse_args()
    SC = a.scratch or os.path.dirname(a.perreal.rstrip("/"))
    keep_d = [str(x) for x in a.cs_d]
    if a.normalizers:
        global NORM
        with open(a.normalizers) as fh:
            NORM = json.load(fh)
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
                n, pehe, ate, ps, as_ = pt.get(
                    meth, (None, float("nan"), float("nan"), "—", "—"))
                r = {k: stat(raw.get(meth, {}).get(k, [])) for k in _KEYS}
                m = {k: stat(mal.get(meth, {}).get(k, [])) for k in _KEYS}
                i = {k: stat(ind.get(meth, {}).get(k, [])) for k in _KEYS}
                im = {k: stat(indm.get(meth, {}).get(k, [])) for k in _KEYS}
                if all(v is None for v in r.values()) and n is None:
                    continue
                rows.append((display_name(label, meth, single), n, ps, as_,
                             r, m, i, im, pehe))
        if not rows:
            continue
        emit(f"\n## RealCause — {ds}   (eps_ATE is RELATIVE)\n")
        emit("| model | n | PEHE | eps_ATE "
             "| Cov (raw) | Len (raw) | Len/sd(Y) (raw) | IS (raw) "
             "| Cov (MALC) | Len (MALC) | Len/sd(Y) (MALC) | IS (MALC) "
             "| Cov (indep) | Len (indep) | Len/sd(Y) (indep) | IS (indep) "
             "| Cov (indep+MALC) | Len (indep+MALC) | Len/sd(Y) (indep+MALC) "
             "| IS (indep+MALC) |")
        emit("|" + "---|" * 20)
        _sd = nsd("realcause", ds)
        for nm, n, ps, as_, r, m, i, im, _srt in sorted(rows, key=lambda t: t[8]):
            tag = " *(released)*" if nm in RELEASED else ""
            emit(f"| {nm}{tag} | {n if n else '—'} | "
                 f"{ps} | {as_} | "
                 f"{fmt(r['cover'])} | {fmt(r['length'], 4)} | "
                 f"{fmtn(r['length'], _sd)} | {fmt(r['is05'], 4)} | "
                 f"{fmt(m['cover'])} | {fmt(m['length'], 4)} | "
                 f"{fmtn(m['length'], _sd)} | {fmt(m['is05'], 4)} | "
                 f"{fmt(i['cover'])} | {fmt(i['length'], 4)} | "
                 f"{fmtn(i['length'], _sd)} | {fmt(i['is05'], 4)} | "
                 f"{fmt(im['cover'])} | {fmt(im['length'], 4)} | "
                 f"{fmtn(im['length'], _sd)} | {fmt(im['is05'], 4)} |")

        # ── ATE-interval calibration, from the --target ate tables ──────────
        # A different claim from CATE coverage: one interval for the average effect
        # per realization, not one per query. Read from markdown rather than perreal
        # because the ATE stage was never routed through coverage_by_realization --
        # it needs no per-realization aggregation, since a realization yields exactly
        # one ATE interval and one truth.
        ate_rows = []
        for label, rc, cs, single in ROOTS:
            R = os.path.join(SC, rc)
            craw = parse_calib(os.path.join(R, f"calib_{ds}_raw_ate.md"))
            cmal = {}
            for cand in sorted(glob.glob(os.path.join(R, f"calib_{ds}_T_*_ate.md"))):
                cmal = parse_calib(cand)
            for meth in sorted(set(craw) | set(cmal)):
                nm = display_name(label, meth, single)
                a1 = craw.get(meth); a2 = cmal.get(meth)
                if a1 is None and a2 is None:
                    continue
                ate_rows.append((nm, a1, a2))
        if ate_rows:
            emit(f"\n### RealCause — {ds} — ATE interval (one per realization)\n")
            emit("| model | n | Cov (raw) | Len (raw) | Len/sd(Y) (raw) | IS (raw) "
                 "| Cov (MALC) | Len (MALC) | Len/sd(Y) (MALC) | IS (MALC) |")
            emit("|" + "---|" * 10)
            for nm, a1, a2 in sorted(ate_rows, key=lambda t: t[0]):
                tg = " *(released)*" if nm in RELEASED else ""
                n_ = (a1 or a2)[0]
                f3 = lambda a, j: ("—" if a is None or not np.isfinite(a[j])
                                   else f"{a[j]:.4f}")
                f3n = lambda a, j: ("—" if a is None or _sd is None
                                    or not np.isfinite(a[j])
                                    else f"{a[j] / _sd:.4f}")
                emit(f"| {nm}{tg} | {n_} | {f3(a1,1)} | {f3(a1,2)} | "
                     f"{f3n(a1,2)} | {f3(a1,3)} | "
                     f"{f3(a2,1)} | {f3(a2,2)} | {f3n(a2,2)} | {f3(a2,3)} |")

    # ── Case study: one table PER CASE, aggregated over d AND shifts ────────
    # The mechanism (confounder, mediator, frontdoor, ...) is the thing being
    # compared, so cases must stay separate; d is a nuisance axis and pools. Groups
    # are named d<D>_<Case> in the per-realization files, so the case is recovered
    # from the group name rather than from a directory walk.
    by_case = {}
    for label, rc, cs, single in ROOTS:
        raw = load_perreal(a.perreal, label, "raw", "*__shift*", keep_d)
        mal = load_perreal(a.perreal, label, a.malc_tag, "*__shift*", keep_d)
        # point tables, split by case
        pt_by_case = defaultdict(lambda: defaultdict(
            lambda: {"pehe2": [], "ate": [], "cells": 0}))
        for f in glob.glob(os.path.join(SC, cs, "shift*", "d*", "ctx*",
                                        "point_raw_em_*.md")):
            dm = re.search(r"/d(\d+)/", f)
            if dm and dm.group(1) not in keep_d:
                continue
            case = os.path.basename(f)[len("point_raw_em_"):-3]
            for meth, (n, pehe, ate, _ps, _as) in parse_point(f).items():
                d = pt_by_case[case][meth]
                if np.isfinite(pehe):
                    d["pehe2"].append(pehe ** 2)
                if np.isfinite(ate):
                    d["ate"].append(ate)
                d["cells"] += 1
        # per-realization arrays, split by case via the group name
        cov_by_case = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        mal_by_case = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for stage_acc, dest in ((raw, cov_by_case), (mal, mal_by_case)):
            for meth, keyed in stage_acc.items():
                for k, arrs in keyed.items():
                    dest["__ALL__"][meth][k].extend(arrs)
        # Re-read per file so the case can be recovered from each name.
        for stage, dest in ((("raw"), cov_by_case), ((a.malc_tag), mal_by_case)):
            for f in sorted(glob.glob(os.path.join(a.perreal, label,
                                                   f"{stage}__*__shift*.npz"))):
                gm = re.match(r"^[a-z_]+__d(\d+)_(.+)__shift", os.path.basename(f))
                if not gm or gm.group(1) not in keep_d:
                    continue
                case = gm.group(2)
                try:
                    z = np.load(f)
                except Exception:
                    continue
                for k in z.files:
                    if "__" not in k:
                        continue
                    meth, key = k.rsplit("__", 1)
                    if key not in _KEYS:
                        continue
                    arr = np.asarray(z[k], dtype=float).ravel()
                    if arr.size:
                        dest[case][meth][key].append(arr)
        cases = sorted(set(pt_by_case) | set(k for k in cov_by_case if k != "__ALL__"))
        for case in cases:
            for meth in sorted(set(pt_by_case.get(case, {}))
                               | set(cov_by_case.get(case, {}))):
                d = pt_by_case.get(case, {}).get(meth)
                pehe = (float(np.sqrt(np.mean(d["pehe2"])))
                        if d and d["pehe2"] else float("nan"))
                ate = float(np.mean(d["ate"])) if d and d["ate"] else float("nan")
                cells = d["cells"] if d else 0
                r = {k: stat(cov_by_case.get(case, {}).get(meth, {}).get(k, []))
                     for k in _KEYS}
                m = {k: stat(mal_by_case.get(case, {}).get(meth, {}).get(k, []))
                     for k in _KEYS}
                if all(v is None for v in r.values()) and not np.isfinite(pehe):
                    continue
                by_case.setdefault(case, []).append(
                    (display_name(label, meth, single), cells, pehe, ate, r, m))

    for case in sorted(by_case):
        emit(f"\n## Case study — {case}   "
             f"(pooled over shifts 0/+2/-2 and d in {{{', '.join(keep_d)}}}; "
             f"L1_ATE is ABSOLUTE)\n")
        emit("| model | cells | PEHE (rms) | L1_ATE "
             "| Cov (raw) | Len (raw) | Len/sd(Y) (raw) | IS (raw) "
             "| Cov (MALC) | Len (MALC) | Len/sd(Y) (MALC) | IS (MALC) |")
        emit("|" + "---|" * 12)
        _sd = nsd("case_study", case)
        for nm, cells, pehe, ate, r, m in sorted(by_case[case], key=lambda t: t[2]):
            tg = " *(released)*" if nm in RELEASED else ""
            emit(f"| {nm}{tg} | {cells} | {pehe:.4f} | {ate:.4f} | "
                 f"{fmt(r['cover'])} | {fmt(r['length'], 4)} | "
                 f"{fmtn(r['length'], _sd)} | {fmt(r['is05'], 4)} | "
                 f"{fmt(m['cover'])} | {fmt(m['length'], 4)} | "
                 f"{fmtn(m['length'], _sd)} | {fmt(m['is05'], 4)} |")

    # ── ComplexMech: one table per node count ───────────────────────────────
    # Per node count, not pooled: graph size is the axis this benchmark varies, so
    # collapsing it would hide the trend it exists to show. `total` = nonzero + zero,
    # and ATE error is L1 here rather than the relative form.
    CM = os.path.join(SC, "cmech_dumps")
    if os.path.isdir(CM):
        for n in (5, 10, 20, 30, 40, 50):
            rows = []
            for d in sorted(glob.glob(os.path.join(CM, "*"))):
                if not os.path.isdir(d):
                    continue
                label = os.path.basename(d)
                pt = parse_point(os.path.join(d, "N1000",
                                              f"point_raw_em_CMECH_n{n}.md"))
                raw = load_perreal(a.perreal, label, "raw", f"cmech_n{n}__-")
                mal = load_perreal(a.perreal, label, a.malc_tag, f"cmech_n{n}__-")
                for meth in sorted(set(pt) | set(raw) | set(mal)):
                    nn, pehe, ate, ps, as_ = pt.get(
                        meth, (None, float("nan"), float("nan"), "—", "—"))
                    r = {k: stat(raw.get(meth, {}).get(k, [])) for k in _KEYS}
                    m = {k: stat(mal.get(meth, {}).get(k, [])) for k in _KEYS}
                    if all(v is None for v in r.values()) and nn is None:
                        continue
                    rows.append((display_name(label, meth, label), nn, ps, as_,
                                 r, m, pehe))
            if not rows:
                continue
            emit(f"\n## ComplexMech — n={n} nodes, N=1000, subset=total   "
                 f"(L1_ATE is ABSOLUTE)\n")
            emit("| model | n | PEHE | L1_ATE "
                 "| Cov (raw) | Len (raw) | Len/sd(Y) (raw) | IS (raw) "
                 "| Cov (MALC) | Len (MALC) | Len/sd(Y) (MALC) | IS (MALC) |")
            emit("|" + "---|" * 12)
            _sd = nsd("cmech", n)
            for nm, nn, ps, as_, r, m, _srt in sorted(rows, key=lambda t: t[6]):
                tg = " *(released)*" if nm in RELEASED else ""
                emit(f"| {nm}{tg} | {nn if nn else '—'} | {ps} | {as_} | "
                     f"{fmt(r['cover'])} | {fmt(r['length'], 4)} | "
                     f"{fmtn(r['length'], _sd)} | {fmt(r['is05'], 4)} | "
                     f"{fmt(m['cover'])} | {fmt(m['length'], 4)} | "
                     f"{fmtn(m['length'], _sd)} | {fmt(m['is05'], 4)} |")

    # ── Do-PFN semi-real: one table per dataset ─────────────────────────────
    SR = os.path.join(SC, "semireal_dumps")
    if os.path.isdir(SR):
        for ds in ("sales", "law_race"):
            rows = []
            for d in sorted(glob.glob(os.path.join(SR, "*"))):
                if not os.path.isdir(d):
                    continue
                label = os.path.basename(d)
                pt = parse_point(os.path.join(d, f"point_raw_em_SEMIREAL_{ds}.md"))
                raw = load_perreal(a.perreal, label, "raw", f"{ds}__semireal")
                mal = load_perreal(a.perreal, label, a.malc_tag, f"{ds}__semireal")
                for meth in sorted(set(pt) | set(raw) | set(mal)):
                    nn, pehe, ate, ps, as_ = pt.get(
                        meth, (None, float("nan"), float("nan"), "—", "—"))
                    r = {k: stat(raw.get(meth, {}).get(k, [])) for k in _KEYS}
                    m = {k: stat(mal.get(meth, {}).get(k, [])) for k in _KEYS}
                    if all(v is None for v in r.values()) and nn is None:
                        continue
                    rows.append((display_name(label, meth, label), nn, ps, as_,
                                 r, m, pehe))
            if not rows:
                continue
            emit(f"\n## Do-PFN semi-real — {ds}   "
                 f"(5 SPLITS ONLY -- wide error bars; eps_ATE is RELATIVE)\n")
            emit("| model | n | PEHE | eps_ATE "
                 "| Cov (raw) | Len (raw) | Len/sd(Y) (raw) | IS (raw) "
                 "| Cov (MALC) | Len (MALC) | Len/sd(Y) (MALC) | IS (MALC) |")
            emit("|" + "---|" * 12)
            _sd = nsd("semireal", ds)
            for nm, nn, ps, as_, r, m, _srt in sorted(rows, key=lambda t: t[6]):
                tg = " *(released)*" if nm in RELEASED else ""
                emit(f"| {nm}{tg} | {nn if nn else '—'} | {ps} | {as_} | "
                     f"{fmt(r['cover'])} | {fmt(r['length'], 4)} | "
                     f"{fmtn(r['length'], _sd)} | {fmt(r['is05'], 4)} | "
                     f"{fmt(m['cover'])} | {fmt(m['length'], 4)} | "
                     f"{fmtn(m['length'], _sd)} | {fmt(m['is05'], 4)} |")

    emit("\n---\n")
    emit("Cov/Len/IS pool per-realization arrays by concatenation, so how the work")
    emit("was split across jobs does not change the number. PEHE pools as an RMS")
    emit("across cells because PEHE is itself an RMSE; ATE error pools as a mean.")
    emit("RealCause reports RELATIVE eps_ATE, the case studies absolute L1_ATE.")
    if NORM:
        emit("")
        emit("Len/sd(Y) divides each length by that dataset's outcome spread, so it")
        emit("reads in units of outcome SD and is comparable across models. For a")
        emit("calibrated 95% interval Len/sd(Y) = 3.92*sd(tau|X)/sd(Y), which is")
        emit("bounded by 5.54 (independent arms, sigma <= sd(Y)) -- a value above")
        emit("that is over-dispersed on any dataset. On the case studies the arms")
        emit("share one noise draw, so tau is deterministic given X and the oracle")
        emit("length is 0: there, lower is strictly better and the column measures")
        emit("excess width rather than calibration.")
    emit("Models marked *(released)* are upstream DoPFN weights, not this project's.")
    emit("Cov/Len/IS (indep) and (indep+MALC) are the forced-independent ablation:")
    emit("RealCause only, and meaningful only for 2D heads -- a 1D head has no joint")
    emit("to discard, so its indep columns restate the raw ones.")
    emit("The ATE-interval tables are a DIFFERENT claim from CATE coverage: one")
    emit("interval for the average effect per realization, not one per query.")
    emit("A dash means that stage has not been scored yet.")

    txt = "\n".join(L)
    print(txt)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(txt + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
