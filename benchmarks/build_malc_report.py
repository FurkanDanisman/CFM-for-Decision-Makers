"""One report: 5 RealCause tables, 6 ComplexMech tables, 6 case-study tables.

Each table is one row per method with every variation side by side:

    PEHE raw | PEHE em | ATEerr raw | ATEerr em | cov/len/IS raw | cov/len/IS T

so the raw-vs-EM and raw-vs-MALC comparisons are read across a row rather than
by flipping between files.

  RealCause    one table per dataset (IHDP, ACIC, CPS, PSID, PSID_bal)
  ComplexMech  one table per node count d
  Case studies one table per CASE, pooled over all 3 shifts x 8 d values

POOLING (case studies only). Every metric in these tables is a per-query mean,
so pooling across cells is the n_query-weighted mean -- exact, identical to
scoring all cells in one pass. The pooled SEM combines as
sqrt(sum w_i^2 sem_i^2) / sum w_i.

Coverage is scale-free and pools cleanly. length and IS_0.05 carry the
outcome's units, which differ across d, so pooling them summarises
differently-scaled quantities; that is what pooling over d means, and the
caveat is printed above the tables rather than left implicit.

IS_0.05 is the only scoring rule reported. CRPS and WIS exist in the stored
json and are deliberately not carried through: IS scores exactly the 95%
interval being reported, whereas WIS averages 11 interval levels plus a median
term and is really a CRPS approximation, so it answers a different question.

Reads the .json beside each table when present (it carries the SEMs the
markdown drops) and falls back to parsing the markdown.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

_RC = ["IHDP", "ACIC", "CPS", "PSID", "PSID_bal"]
_CASES = ["Observed_Confounder", "Observed_Mediator",
          "Observed_Mediator_and_Confounder", "Unobserved_Confounder",
          "Frontdoor_Criterion", "Backdoor_Criterion"]
_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def load_calib(path_md):
    """{method: row-dict} from the .json beside path_md, else from the md."""
    j = path_md[:-3] + ".json"
    if os.path.isfile(j):
        try:
            rows = json.load(open(j))
            if isinstance(rows, list):
                return {r["method"]: r for r in rows if "method" in r}
        except Exception:
            pass
    if not os.path.isfile(path_md):
        return {}
    out = {}
    for ln in open(path_md):
        if not ln.startswith("| ") or "---" in ln:
            continue
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) < 10 or c[0] == "method":
            continue
        g = lambda s: float(re.match(rf"({_NUM})", s).group(1))
        try:
            out[c[0]] = dict(method=c[0], n_query=int(c[2]), coverage95=float(c[3]),
                             length=g(c[4]), is05=g(c[5]), pred_sd=float(c[6]),
                             true_sd=float(c[7]), sd_ratio=float(c[8]),
                             bias=float(c[9]))
        except Exception:
            continue
    return out


def load_point(path_md):
    """{method: (pehe_raw, ate_raw, pehe_em, ate_em)} from a point table.

    point_raw_em emits two columns per mode, so `--modes raw em` gives six
    columns and `--modes raw` gives four. Requiring six silently dropped every
    raw-only table and showed the point columns as blank -- handle both, with
    NaN for a mode that was not run.
    """
    if not os.path.isfile(path_md):
        return {}
    out = {}
    nan = float("nan")
    for ln in open(path_md):
        if not ln.startswith("| ") or "---" in ln:
            continue
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) < 4 or c[0] == "method":
            continue
        g = lambda s: (float(re.match(rf"({_NUM})", s).group(1))
                       if re.match(rf"({_NUM})", s) else nan)
        try:
            if len(c) >= 6:
                out[c[0]] = (g(c[2]), g(c[3]), g(c[4]), g(c[5]))
            else:
                out[c[0]] = (g(c[2]), g(c[3]), nan, nan)
        except Exception:
            continue
    return out


def pool(dicts):
    """n_query-weighted mean per method over a list of {method: row}."""
    acc = {}
    for d in dicts:
        for m, r in d.items():
            a = acc.setdefault(m, {"n": 0, "cells": 0})
            w = r.get("n_query", 0) or 0
            if not w:
                continue
            a["n"] += w; a["cells"] += 1
            for k in ("coverage95", "length", "is05", "pred_sd", "sd_ratio", "bias"):
                if k in r and r[k] is not None:
                    a[k] = a.get(k, 0.0) + w * float(r[k])
            for k in ("length_sem", "is05_sem"):
                if k in r and r[k] is not None:
                    a[k] = a.get(k, 0.0) + (w * float(r[k])) ** 2
    for m, a in acc.items():
        if not a["n"]:
            continue
        for k in ("coverage95", "length", "is05", "pred_sd", "sd_ratio", "bias"):
            if k in a:
                a[k] /= a["n"]
        for k in ("length_sem", "is05_sem"):
            if k in a:
                a[k] = a[k] ** 0.5 / a["n"]
    return acc


def pool_point(dicts):
    """Unweighted mean of point metrics over cells (each already a mean)."""
    acc = {}
    for d in dicts:
        for m, t in d.items():
            a = acc.setdefault(m, [0, 0.0, 0.0, 0.0, 0.0])
            a[0] += 1
            for i in range(4):
                a[i + 1] += t[i]
    return {m: tuple(v / a[0] for v in a[1:]) for m, a in acc.items() if a[0]}


def table(title, point, raw, sm, ate_label="eps_ATE", note=None,
          calib_only=False, rank_by="is", alpha=0.05):
    f = lambda v: "—" if v is None else f"{v:.4f}"
    lines = [f"### {title}"]
    if note:
        lines.append(f"*{note}*")
    if calib_only:
        # IS = length + 40 * E[miss] exactly (alpha = 0.05 -> 2/alpha = 40), so
        # the penalty and the typical miss magnitude are ALGEBRA on the three
        # reported columns -- no rescoring needed:
        #
        #   pen        = IS - len              = 40 * E[miss] over all queries
        #   miss|miss  = pen / (40 * (1-cov))  = mean miss distance among misses
        #
        # This is what makes "coverage up, length down, IS up" legible instead
        # of paradoxical: coverage counts HOW OFTEN you miss, IS prices HOW FAR.
        # A method can miss far less often, with tighter intervals, and still
        # lose on IS because the misses it does make are much further out.
        lines += ["",
                  "| method | cov raw | len raw | IS raw "
                  "| cov T | len T | IS T |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
    else:
        lines += ["",
                  f"| method | PEHE raw | PEHE em | {ate_label} raw | {ate_label} em "
                  f"| cov raw | len raw | IS raw | cov T | len T | IS T |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    # Ranking key. `coverage` implements the set-predictor (conformal)
    # convention: closest to nominal coverage first, shortest interval as the
    # tie-break, and no distance-to-truth term -- a miss is a miss. `is` uses
    # the quantile-forecasting convention, where a miss costs 40x its distance.
    # The two disagree exactly when a method trades many near-misses for few
    # far ones, which is what MALC smoothing does, so the choice is stated in
    # the header rather than left to whoever reads the table.
    def _key(m):
        d = sm.get(m) or raw.get(m) or {}
        if rank_by == "coverage":
            c, L = d.get("coverage95"), d.get("length")
            return (abs((c if c is not None else 0.0) - 0.95),
                    L if L is not None else 9e9)
        return d.get("is05", 9e9)

    methods = sorted(set(raw) | set(sm) | set(point), key=_key)
    for m in methods:
        p = point.get(m)
        r, s = raw.get(m, {}), sm.get(m, {})
        def mm(d):
            """Mean miss distance among the misses, from cov/len/IS."""
            c, L, I = d.get("coverage95"), d.get("length"), d.get("is05")
            if c is None or L is None or I is None or c >= 0.99995:
                return "—"
            return f"{max(I - L, 0.0) / (40.0 * (1.0 - c)):.4f}"

        if calib_only:
            lines.append(
                f"| {m} | "
                f"{f(r.get('coverage95'))} | {f(r.get('length'))} | "
                f"{f(r.get('is05'))} | "
                f"{f(s.get('coverage95'))} | {f(s.get('length'))} | "
                f"{f(s.get('is05'))} |")
            continue
        lines.append(
            f"| {m} | " +
            ((lambda q: " | ".join(q) + " | ")(
                [("—" if v != v else f"{v:.4f}") for v in (p[0], p[2], p[1], p[3])])
             if p else "— | — | — | — | ") +
            f"{f(r.get('coverage95'))} | {f(r.get('length'))} | {f(r.get('is05'))} | "
            f"{f(s.get('coverage95'))} | {f(s.get('length'))} | {f(s.get('is05'))} |")
    covs = [r.get("coverage95") for r in raw.values() if r.get("coverage95") is not None]
    if covs and all(c >= 0.9995 for c in covs):
        lines.append("")
        lines.append("> **coverage saturated in the raw column** — every method covered "
                     "every query, so IS equals length there and the ranking is by width.")
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rc-root"); ap.add_argument("--cmech-root"); ap.add_argument("--cs-root")
    ap.add_argument("--ctx", default="1000"); ap.add_argument("--tag", default="B100_K1")
    ap.add_argument("--target", default="cate", choices=["cate", "ate"])
    ap.add_argument("--nodes", nargs="+", default=["5", "10", "20", "30", "40", "50"])
    ap.add_argument("--calib-only", action="store_true",
                    help="drop the PEHE / ATE-error columns; coverage, length "
                         "and IS_0.05 only")
    ap.add_argument("--rank-by", choices=["is", "coverage"], default="is",
                    help="is: mean IS_0.05 (default). "
                         "coverage: |cov-0.95| then length.")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="interval level of the reported intervals; sets the "
                         "2/alpha coefficient (40 at the 95%% level)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = a.target

    out = [f"# MALC study — {T.upper()} ({a.tag})", "",
           "Each row carries every variation: point estimate under raw-mean vs",
           "EM-mean (no MALC), then calibration from the raw tau density and",
           "after MALC-1D smoothing of it (variant T).", "",
           "Rank on **IS_0.05**: it equals length whenever the truth is covered",
           "and exceeds it by 40x the miss distance when it is not.", ""]

    if a.rc_root:
        out += ["## RealCause", ""]
        for ds in _RC:
            out += table(ds,
                         load_point(f"{a.rc_root}/point_raw_em_{ds}.md"),
                         load_calib(f"{a.rc_root}/calib_{ds}_raw_{T}.md"),
                         load_calib(f"{a.rc_root}/calib_{ds}_T_{a.tag}_{T}.md"),
                         calib_only=a.calib_only, rank_by=a.rank_by, alpha=a.alpha)

    if a.cmech_root:
        out += ["## ComplexMech", "", "One table per node count d.", ""]
        for d in a.nodes:
            out += table(f"d = {d}",
                         load_point(f"{a.cmech_root}/point_raw_em_CMECH_d{d}.md"),
                         load_calib(f"{a.cmech_root}/calib_CMECH_d{d}_raw_{T}.md"),
                         load_calib(f"{a.cmech_root}/calib_CMECH_d{d}_T_{a.tag}_{T}.md"),
                         ate_label="L1_ATE", calib_only=a.calib_only, rank_by=a.rank_by, alpha=a.alpha)

    if a.cs_root:
        cells = sorted(glob.glob(f"{a.cs_root}/shift*/d*/ctx{a.ctx}"))
        out += ["## Case studies", "",
                f"One table per case, pooled over {len(cells)} (shift x d) cells by "
                "n_query weighting.", "",
                "> coverage and sd_ratio are scale-free and pool cleanly. length and "
                "IS carry the outcome's units, which differ across d — pooling those "
                "summarises differently-scaled quantities.", ""]
        for c in _CASES:
            out += table(c,
                         pool_point([load_point(f"{x}/point_raw_em_{c}.md") for x in cells]),
                         pool([load_calib(f"{x}/calib_raw_{T}_{c}.md") for x in cells]),
                         pool([load_calib(f"{x}/calib_T_{a.tag}_{T}_{c}.md") for x in cells]),
                         ate_label="L1_ATE", calib_only=a.calib_only, rank_by=a.rank_by, alpha=a.alpha,
                         note=f"pooled over {len(cells)} cells")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write("\n".join(out) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
