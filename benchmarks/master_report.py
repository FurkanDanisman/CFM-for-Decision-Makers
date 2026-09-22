#!/usr/bin/env python
"""One report with every benchmark, each number taken from whichever file computes
it correctly.

  RealCause          verbatim from final_table (nothing wrong with it)
  Case study         final_table's table, PLUS an eps_ATE column joined from
                     cs_point_total. Case-study PEHE/L1 in final_table are fine --
                     point_raw_em scores ONE dataset per call there, so the
                     nonzero/zero pairing bug cannot arise.
  ComplexMech        BOTH versions (main + rho>0.99). PEHE/L1/eps_ATE come from
                     cmech_point_total; Cov/Len/Len-sd(Y) from final_table.
                     final_table's own ComplexMech PEHE/ATE are DROPPED -- they are
                     the n=9 positional-pairing artefact.
  Do-PFN semi-real   verbatim from final_table.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

_NUM = r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)"


def split_sections(path):
    """-> [(header_line, [body lines])]; text before the first '## ' is preamble."""
    if not path or not os.path.isfile(path):
        return []
    out, cur = [], ("", [])
    for line in open(path):
        if line.startswith("## "):
            out.append(cur); cur = (line.rstrip("\n"), [])
        else:
            cur[1].append(line.rstrip("\n"))
    out.append(cur)
    return out


def _cells(line):
    s = line.strip()
    if not s.startswith("|"):
        return None
    return [c.strip() for c in s.strip("|").split("|")]


def parse_cs_rel(path):
    """-> {case: {model: (PEHE, L1_ATE, eps_ATE)}} from cs_point_total output.

    All three are taken, not just eps_ATE: final_table prints PEHE/L1 without
    their SEM (parse_point discards it), so substituting the point file's cells
    restores the error bars AND keeps one pooling convention across the report.
    """
    out = {}
    for hdr, body in split_sections(path):
        m = re.match(r"^##\s*Case study\s*[-—]\s*(\S+)", hdr)
        if not m:
            continue
        case = m.group(1)
        out[case] = {}
        for line in body:
            c = _cells(line)
            if not c or len(c) < 6 or c[0] in ("model", "---") or set(c[1]) <= set("-"):
                continue
            # model | cells | realizations | PEHE | L1_ATE | eps_ATE
            out[case][c[0].replace("*(released)*", "").strip()] = (c[3], c[4], c[5])
    return out


def augment_case(hdr, body, rel_by_case):
    """Insert an eps_ATE column after L1_ATE, joined on (case, model)."""
    m = re.match(r"^##\s*Case study\s*[-—]\s*(\S+)", hdr)
    rel = rel_by_case.get(m.group(1), {}) if m else {}
    out = []
    for line in body:
        c = _cells(line)
        if not c:
            out.append(line); continue
        if c[0] == "model":
            # Relabel: the values are now mean +- SEM from cs_point_total, so
            # final_table's "PEHE (rms)" heading would misdescribe them.
            hdr_cols = list(c[:4])
            hdr_cols[2] = "PEHE"
            out.append("| " + " | ".join(hdr_cols + ["eps_ATE (rel)"] + c[4:]) + " |")
        elif set(c[0]) <= set("-"):
            out.append("|" + "---|" * (len(c) + 1))
        else:
            name = c[0].replace("*(released)*", "").strip()
            got = rel.get(name)
            if got:
                pehe, l1, eps = got          # with SEM, from cs_point_total
            else:
                pehe, l1, eps = c[2], c[3], "—"
            out.append("| " + " | ".join(
                c[:2] + [pehe, l1, eps] + c[4:]) + " |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-table", required=True)
    ap.add_argument("--cs-point", default=None)
    ap.add_argument("--cmech-both", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rel = parse_cs_rel(a.cs_point) if a.cs_point else {}
    if a.cs_point and not rel:
        print(f"note: no case-study eps_ATE parsed from {a.cs_point}; that column "
              f"will be em-dashes", file=sys.stderr)

    secs = split_sections(a.final_table)
    rc, cs, sr, foot = [], [], [], []
    n_cm = 0
    for hdr, body in secs:
        if not hdr:
            foot = body                       # preamble/footer of final_table
            continue
        if hdr.startswith("## RealCause") or hdr.startswith("### RealCause"):
            rc.append((hdr, body))
        elif hdr.startswith("## Case study"):
            cs.append((hdr, augment_case(hdr, body, rel)))
        elif hdr.startswith("## ComplexMech"):
            n_cm += 1                          # dropped; replaced by --cmech-both
        elif hdr.startswith("## Do-PFN semi-real"):
            sr.append((hdr, body))
        else:
            sr.append((hdr, body))
    print(f"[assembled] RealCause {len(rc)}, case study {len(cs)}, "
          f"semi-real/other {len(sr)}, ComplexMech dropped {n_cm}", file=sys.stderr)

    L = ["# R-PFN — full results", "",
         "Each number comes from whichever scorer computes it correctly:", "",
         "- **RealCause**, **semi-real**: as produced by `final_table`.",
         "- **Case study**: PEHE / L1_ATE / eps_ATE from `cs_point_total` (with SEM); "
         "Cov/Len from `final_table`. Its PEHE/L1 are unaffected by the ComplexMech pairing "
         "bug (one dataset per scoring call).",
         "- **ComplexMech**: both the main benchmark and the rho>0.99 cells. "
         "PEHE/L1/eps_ATE from `cmech_point_total` (100 realizations, queries "
         "grouped by source realization); Cov/Len/Len-sd(Y) from `final_table`. "
         "`final_table`'s own ComplexMech PEHE/ATE columns are dropped -- they are "
         "the n=9 positional-pairing artefact.", "",
         "`Len/sd(Y)` divides each length by that dataset's outcome spread, so it "
         "reads in units of outcome SD. It is comparable across models WITHIN a "
         "dataset, never across datasets, and the two ComplexMech versions use "
         "different constants.", ""]

    for title, group in (("RealCause", rc), ("Case studies", cs)):
        if group:
            L += ["", f"---", "", f"# {title}", ""]
            for hdr, body in group:
                L += [hdr] + body
    if a.cmech_both and os.path.isfile(a.cmech_both):
        L += ["", "---", "", "# ComplexMech (both versions)", ""]
        body = open(a.cmech_both).read().splitlines()
        # drop cmech_both's own H1 and its preamble, keep the tables
        start = next((i for i, l in enumerate(body) if l.startswith("## ")), 0)
        L += body[start:]
    elif n_cm:
        print("note: --cmech-both not given; ComplexMech sections were dropped and "
              "nothing replaced them", file=sys.stderr)
    if sr:
        L += ["", "---", "", "# Do-PFN semi-real / other", ""]
        for hdr, body in sr:
            L += [hdr] + body
    if foot:
        L += ["", "---", ""] + foot

    txt = "\n".join(L)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
