"""PEHE / ATE-error / CRPS table, from results already on disk. No compute.

CRPS is the one score that is defined on all three benchmarks without picking a
nominal level: it compares a predicted distribution against a single observed
tau, and for a point forecast it reduces exactly to |error|. That matters here
because on ComplexMech and the case studies tau is deterministic given x (the
DGP shares arm noise, rho = 1), so no 95% interval target exists and coverage
has nothing to converge to -- whereas CRPS is still minimised by the right
answer.

Columns, per method:
    PEHE, ATE error      from the point tables (no MALC)
    CRPS-CATE, CRPS-ATE  from the calibration json, raw and variant T

CRPS carries the units of tau, so values are NOT comparable across datasets
(CPS is dollars, IHDP is not). Compare within a column, not across.

Case-study rows pool the (shift x d) cells by n_query weighting, which is exact
for per-query means.

Usage:
    python benchmarks/crps_report.py                     # all three, defaults
    python benchmarks/crps_report.py --tag B1000_K1       # pick the T variant
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_RC = ["IHDP", "ACIC", "CPS", "PSID", "PSID_bal"]
_ND = [5, 10, 20, 30, 40, 50]
_CASES = ["Observed_Confounder", "Observed_Mediator",
          "Observed_Mediator_and_Confounder", "Unobserved_Confounder",
          "Frontdoor_Criterion", "Backdoor_Criterion"]


def jrows(path_md):
    """{method: row} from the .json beside a calib .md (it carries crps)."""
    j = path_md[:-3] + ".json" if path_md.endswith(".md") else path_md
    if not os.path.isfile(j):
        return {}
    try:
        rows = json.load(open(j))
    except Exception:
        return {}
    return {r["method"]: r for r in rows if isinstance(r, dict) and "method" in r}


def points(path_md):
    """{method: (pehe, ate_err)} from a point table, raw mode."""
    if not os.path.isfile(path_md):
        return {}
    out = {}
    g = lambda s: (float(re.match(rf"({_NUM})", s).group(1))
                   if re.match(rf"({_NUM})", s) else float("nan"))
    for ln in open(path_md):
        if not ln.startswith("| ") or "---" in ln:
            continue
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) < 4 or c[0] == "method":
            continue
        out[c[0]] = (g(c[2]), g(c[3]))
    return out


def pool(ds, key="crps"):
    """n_query-weighted mean of `key` per method across a list of {method: row}."""
    acc = {}
    for d in ds:
        for m, r in d.items():
            w, v = r.get("n_query") or 0, r.get(key)
            if not w or v is None:
                continue
            a = acc.setdefault(m, [0.0, 0])
            a[0] += w * float(v); a[1] += w
    return {m: a[0] / a[1] for m, a in acc.items() if a[1]}


def pool_pt(ds):
    acc = {}
    for d in ds:
        for m, t in d.items():
            a = acc.setdefault(m, [0.0, 0.0, 0])
            a[0] += t[0]; a[1] += t[1]; a[2] += 1
    return {m: (a[0] / a[2], a[1] / a[2]) for m, a in acc.items() if a[2]}


def emit(title, pt, cr, ca, tr, ta, note=None):
    print(f"\n### {title}" + (f"   ({note})" if note else ""))
    print(f"{'method':17s} {'PEHE':>11s} {'ATEerr':>10s} | "
          f"{'CRPS-CATE':>11s} {'CRPS-ATE':>11s} | "
          f"{'CRPS-CATE':>11s} {'CRPS-ATE':>11s}")
    print(f"{'':17s} {'':>11s} {'':>10s} | {'---- raw ----':^23s} | {'----- T -----':^23s}")
    print("-" * 100)
    f = lambda v: "     —     " if v is None else f"{v:11.4f}"
    order = sorted(set(cr) | set(pt), key=lambda m: cr.get(m, 9e18))
    for m in order:
        p = pt.get(m)
        print(f"{m:17s} "
              f"{(f'{p[0]:11.4f}' if p else '     —     ')} "
              f"{(f'{p[1]:10.4f}' if p else '    —     ')} | "
              f"{f(cr.get(m))} {f(ca.get(m))} | {f(tr.get(m))} {f(ta.get(m))}")


def main():
    S = os.environ.get("SCRATCH", "")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rc-root", default=f"{S}/rc_dens_uni")
    ap.add_argument("--cm-root", default=f"{S}/cmech_dens")
    ap.add_argument("--cs-root", default=f"{S}/cs_dvar_dens")
    ap.add_argument("--tag", default="B100_K1")
    ap.add_argument("--ctx", default="1000")
    a = ap.parse_args()

    print(f"CRPS is in units of tau -- compare within a column, not across "
          f"datasets.\nT variant = {a.tag}")

    print("\n" + "=" * 100 + "\n  REALCAUSE\n" + "=" * 100)
    for ds in _RC:
        emit(ds, points(f"{a.rc_root}/point_raw_em_{ds}.md"),
             pool([jrows(f"{a.rc_root}/calib_{ds}_raw_cate.md")]),
             pool([jrows(f"{a.rc_root}/calib_{ds}_raw_ate.md")]),
             pool([jrows(f"{a.rc_root}/calib_{ds}_T_{a.tag}_cate.md")]),
             pool([jrows(f"{a.rc_root}/calib_{ds}_T_{a.tag}_ate.md")]))

    print("\n" + "=" * 100 + "\n  COMPLEXMECH  (N=1000)\n" + "=" * 100)
    for d in _ND:
        emit(f"d = {d}", points(f"{a.cm_root}/point_raw_em_CMECH_d{d}.md"),
             pool([jrows(f"{a.cm_root}/calib_CMECH_d{d}_raw_cate.md")]),
             pool([jrows(f"{a.cm_root}/calib_CMECH_d{d}_raw_ate.md")]),
             pool([jrows(f"{a.cm_root}/calib_CMECH_d{d}_T_{a.tag}_cate.md")]),
             pool([jrows(f"{a.cm_root}/calib_CMECH_d{d}_T_{a.tag}_ate.md")]))

    cells = sorted(glob.glob(f"{a.cs_root}/shift*/d*/ctx{a.ctx}"))
    print("\n" + "=" * 100 + f"\n  CASE STUDIES  (pooled over {len(cells)} shift x d cells)\n"
          + "=" * 100)
    for c in _CASES:
        emit(c, pool_pt([points(f"{x}/point_raw_em_{c}.md") for x in cells]),
             pool([jrows(f"{x}/calib_raw_cate_{c}.md") for x in cells]),
             pool([jrows(f"{x}/calib_raw_ate_{c}.md") for x in cells]),
             pool([jrows(f"{x}/calib_T_{a.tag}_cate_{c}.md") for x in cells]),
             pool([jrows(f"{x}/calib_T_{a.tag}_ate_{c}.md") for x in cells]),
             note=f"{len(cells)} cells")


if __name__ == "__main__":
    main()
