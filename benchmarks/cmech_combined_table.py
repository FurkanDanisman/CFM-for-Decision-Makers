#!/usr/bin/env python
"""One table per node count with BOTH ComplexMech versions: the main benchmark and
the rho>0.99 cells, side by side.

Sources, deliberately split by which one is correct for what:

  PEHE / L1_ATE / eps_ATE   cmech_point_total_*.md   (100 realizations, queries
                            grouped by source realization)
  Cov / Len / Len-sd(Y)     final_table*.md          (per-realization arrays)

final_table's OWN PEHE and ATE columns are ignored: they come from
point_raw_em's positional nonzero/zero pairing, which truncates to min(len) and
joins unrelated realizations -- that is the n=9 artefact. Coverage was never
affected, so each number is taken from the file that computes it correctly.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

_HDR_PT = re.compile(r"^##\s*ComplexMech total\s*[-—]\s*n=(\d+)")
_HDR_FT = re.compile(r"^##\s*ComplexMech\s*[-—]\s*n=(\d+)\s*nodes")
_NUM = r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)"


def _cells(line):
    s = line.strip()
    if not s.startswith("|"):
        return None
    parts = [c.strip() for c in s.strip("|").split("|")]
    return parts if parts else None


def parse_point(path):
    """-> {node: {model: (reals, pehe, l1, rel)}} from cmech_point_total output."""
    out, n = {}, None
    if not path or not os.path.isfile(path):
        return out
    for line in open(path):
        m = _HDR_PT.match(line)
        if m:
            n = int(m.group(1)); out.setdefault(n, {}); continue
        c = _cells(line)
        if not c or n is None or len(c) < 5 or c[0] in ("model", "---"):
            continue
        if set(c[1]) <= set("-"):
            continue
        g = lambda s: (float(re.search(_NUM, s).group(1))
                       if re.search(_NUM, s) else float("nan"))
        try:
            # Keep the cells verbatim so the "+- SEM" survives; carry a parsed
            # float alongside purely for row ordering. Reformatting the number
            # dropped every error bar.
            out[n][c[0].replace("*(released)*", "").strip()] = dict(
                reals=c[1], cols=c[2:], sort=g(c[2]))
        except Exception:
            continue
    return out


def parse_cov(path):
    """-> {node: {model: (cov_raw, len_raw, lensd_raw, cov_malc, len_malc,
    lensd_malc)}} from a final_table ComplexMech section."""
    out, n = {}, None
    if not path or not os.path.isfile(path):
        return out
    for line in open(path):
        m = _HDR_FT.match(line)
        if m:
            n = int(m.group(1)); out.setdefault(n, {}); continue
        if line.startswith("## ") and not _HDR_FT.match(line):
            n = None
            continue
        c = _cells(line)
        if not c or n is None or len(c) < 12 or c[0] in ("model", "---"):
            continue
        if set(c[1]) <= set("-"):
            continue
        name = c[0].replace("*(released)*", "").strip()
        # 0 model | 1 n | 2 PEHE | 3 ATE | 4 Cov | 5 Len | 6 Len/sd | 7 IS
        #        | 8 Cov(M) | 9 Len(M) | 10 Len/sd(M) | 11 IS(M)
        out[n][name] = (c[4], c[5], c[6], c[8], c[9], c[10])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point-main", required=True)
    ap.add_argument("--point-rho99", required=True)
    ap.add_argument("--table-main", required=True)
    ap.add_argument("--table-rho99", required=True)
    ap.add_argument("--nodes", type=int, nargs="+",
                    default=[5, 10, 20, 30, 40, 50])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    P = {"main": parse_point(a.point_main), "rho99": parse_point(a.point_rho99)}
    C = {"main": parse_cov(a.table_main), "rho99": parse_cov(a.table_rho99)}
    for tag in ("main", "rho99"):
        print(f"[parsed] {tag}: point {sum(len(v) for v in P[tag].values())} rows, "
              f"cov {sum(len(v) for v in C[tag].values())} rows", file=sys.stderr)

    L = ["# ComplexMech — main benchmark vs rho>0.99 cells", "",
         "PEHE / L1_ATE / eps_ATE come from cmech_point_total (100 realizations,",
         "queries grouped by source realization). Cov / Len / Len/sd(Y) come from",
         "final_table. final_table's own PEHE/ATE columns are NOT used: they are",
         "produced by point_raw_em's positional nonzero/zero pairing, which",
         "truncates to min(len) and joins unrelated realizations (the n=9 artefact).",
         ""]
    for n in a.nodes:
        for tag, title in (("main", "main benchmark (cmech_data_v2)"),
                           ("rho99", "rho>0.99 cells (cmech_data_rho99)")):
            pt, cv = P[tag].get(n, {}), C[tag].get(n, {})
            models = sorted(set(pt) | set(cv),
                            key=lambda m: (pt.get(m, {}).get("sort", 9e9), m))
            if not models:
                continue
            ncol = max((len(v["cols"]) for v in pt.values()), default=4)
            pt_hdr = (["PEHE (mean)", "PEHE (rms)", "L1_ATE", "eps_ATE"]
                      if ncol >= 4 else ["PEHE", "L1_ATE", "eps_ATE"])
            pt_hdr = pt_hdr[:ncol]
            L += ["", f"## ComplexMech n={n} — {title}", "",
                  "| model | reals | " + " | ".join(pt_hdr)
                  + " | Cov (raw) | Len (raw) | Len/sd(Y) (raw) | Cov (MALC) "
                    "| Len (MALC) | Len/sd(Y) (MALC) |",
                  "|" + "---|" * (2 + len(pt_hdr) + 6)]
            for m in models:
                p = pt.get(m)
                c = cv.get(m, ("—",) * 6)
                pcols = (p["cols"][:len(pt_hdr)] if p
                         else ["—"] * len(pt_hdr))
                pcols += ["—"] * (len(pt_hdr) - len(pcols))
                L.append(f"| {m} | {p['reals'] if p else '—'} | "
                         + " | ".join(pcols) + " | " + " | ".join(c) + " |")
    txt = "\n".join(L)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        print(f"\nwrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
