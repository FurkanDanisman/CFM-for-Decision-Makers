"""Pool the 24 case-study cells into ONE table per construction.

The case-study sweep writes a table per (shift, d) cell -- 3 shifts x 8 d
values. This pools them into a single row per method, so the case studies
report like the other two benchmarks instead of as 24 separate tables.

HOW THE POOLING WORKS. Every metric in those tables is a per-query mean, so
pooling across cells is the n_query-weighted mean -- exact, not an
approximation, and identical to what scoring all cells at once would give.
Coverage pools cleanly because it is scale-free: it is the fraction of queries
whose interval contained the truth, and that fraction is comparable whatever
the outcome scale.

LENGTH AND IS_0.05 DO NOT POOL CLEANLY, and the output says so. They carry the
units of the outcome, which differs across d (more covariates -> a different
tau scale). A weighted mean of lengths across d is therefore a summary of
differently-scaled quantities. It is reported because it is what "aggregate
the d variations" means, but the per-shift / per-d breakdowns are printed
alongside so a ranking that only holds at one d is visible rather than hidden.
Coverage and sd_ratio are the scale-free columns; rank on those first.

Usage:
    python benchmarks/aggregate_case_study_tables.py \\
        --cs-root $SCRATCH/cs_dvar_dens --tag B100_K1 --target cate
"""
from __future__ import annotations

import argparse
import glob
import os
import re

_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def parse_table(path):
    """{method: dict} from one calib markdown, or {} if unreadable."""
    if not os.path.isfile(path):
        return {}
    out = {}
    for ln in open(path):
        if not ln.startswith("| ") or "---" in ln:
            continue
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) < 10 or c[0] == "method":
            continue
        def val(s):                       # "0.5387 ± 0.0427" -> 0.5387
            m = re.match(rf"({_NUM})", s)
            return float(m.group(1)) if m else float("nan")
        try:
            out[c[0]] = dict(n_files=int(c[1]), n_query=int(c[2]),
                             cover=float(c[3]), length=val(c[4]), is05=val(c[5]),
                             pred_sd=float(c[6]), true_sd=float(c[7]),
                             sd_ratio=float(c[8]), bias=float(c[9]))
        except (ValueError, IndexError):
            continue
    return out


def pool(tables):
    """n_query-weighted mean per method across a list of per-cell dicts."""
    acc = {}
    for t in tables:
        for m, r in t.items():
            a = acc.setdefault(m, dict(n=0, cells=0, **{k: 0.0 for k in
                               ("cover", "length", "is05", "pred_sd", "sd_ratio", "bias")}))
            w = r["n_query"]
            a["n"] += w; a["cells"] += 1
            for k in ("cover", "length", "is05", "pred_sd", "sd_ratio", "bias"):
                a[k] += w * r[k]
    for m, a in acc.items():
        if a["n"]:
            for k in ("cover", "length", "is05", "pred_sd", "sd_ratio", "bias"):
                a[k] /= a["n"]
    return acc


def render(acc, title, note=None):
    lines = [f"#### {title}"]
    if note:
        lines.append(f"*{note}*")
    lines += ["", "| method | cells | n_query | coverage95 | length | is05 | pred_sd | sd_ratio | bias |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in sorted(acc, key=lambda k: acc[k]["is05"]):
        a = acc[m]
        lines.append(f"| {m} | {a['cells']} | {a['n']} | {a['cover']:.3f} | "
                     f"{a['length']:.4f} | {a['is05']:.4f} | {a['pred_sd']:.4f} | "
                     f"{a['sd_ratio']:.1f} | {a['bias']:+.4f} |")
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cs-root", required=True)
    ap.add_argument("--ctx", default="1000")
    ap.add_argument("--tag", default="B100_K1")
    ap.add_argument("--target", default="cate", choices=["cate", "ate"])
    ap.add_argument("--by-shift", action="store_true",
                    help="also break the pooled table out per shift")
    ap.add_argument("--by-d", action="store_true",
                    help="also break the pooled table out per d")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cells = sorted(glob.glob(f"{a.cs_root}/shift*/d*/ctx{a.ctx}"))
    if not cells:
        raise SystemExit(f"no cells under {a.cs_root}/shift*/d*/ctx{a.ctx}")

    raw, sm, meta = [], [], []
    for c in cells:
        sh = c.split("/shift")[1].split("/")[0]
        d = int(re.search(r"/d(\d+)/", c).group(1))
        tr = parse_table(f"{c}/calib_raw_{a.target}.md")
        tt = parse_table(f"{c}/calib_T_{a.tag}_{a.target}.md")
        if tr: raw.append(tr); meta.append((sh, d, "raw", tr))
        if tt: sm.append(tt);  meta.append((sh, d, "T", tt))

    out = [f"# Case studies — pooled over shifts and d  (target={a.target}, tag={a.tag})", "",
           f"{len(cells)} cells found; {len(raw)} raw and {len(sm)} T tables read.",
           "Pooling is the n_query-weighted mean, which is exact for per-query means.",
           "",
           "**coverage and sd_ratio are scale-free and pool cleanly. length and",
           "is05 carry the outcome's units, which differ across d — read those as",
           "a summary of differently-scaled quantities, not as one number.**", ""]
    out += render(pool(raw), f"RAW — pooled over all shifts and d")
    out += render(pool(sm),  f"T ({a.tag}) — pooled over all shifts and d")

    if a.by_shift:
        for sh in sorted({m[0] for m in meta}):
            out += render(pool([t for s, d, k, t in meta if s == sh and k == "raw"]),
                          f"RAW — shift {sh}, pooled over d")
            out += render(pool([t for s, d, k, t in meta if s == sh and k == "T"]),
                          f"T — shift {sh}, pooled over d")
    if a.by_d:
        for dv in sorted({m[1] for m in meta}):
            out += render(pool([t for s, d, k, t in meta if d == dv and k == "raw"]),
                          f"RAW — d = {dv}, pooled over shifts")
            out += render(pool([t for s, d, k, t in meta if d == dv and k == "T"]),
                          f"T — d = {dv}, pooled over shifts")

    txt = "\n".join(out)
    print(txt)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(txt + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
