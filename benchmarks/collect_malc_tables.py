"""Collect the MALC study's tables from all three benchmarks into one document.

The runs write their tables next to the dumps, which means they are scattered
across three roots and, for the case studies, across 24 (shift x d) cells. This
walks all of them and emits a single markdown file.

What it gathers, per benchmark:
  point   PEHE + eps_ATE (RealCause) / L1_ATE (others), raw-mean vs EM-mean
  raw     coverage / length / IS_0.05 from the unsmoothed tau density
  T       the same after MALC-1D smoothing of the tau density

READ IS_0.05, NOT length. IS equals length whenever the truth is covered and
exceeds it by 40x the miss distance when it is not, so it is the only column
that prices coverage and width together. Where coverage saturates at 1.000 the
two columns are identical by construction and "best" degenerates to
"narrowest" -- the header of each section says whether that happened.

Usage:
    python benchmarks/collect_malc_tables.py \\
        --rc-root $SCRATCH/rc_dens_uni \\
        --cmech-root $SCRATCH/cmech_1d2d \\
        --cs-root $SCRATCH/cs_dvar_dens \\
        --tag B100_K1 --out $SCRATCH/MALC_RESULTS.md
"""
from __future__ import annotations

import argparse
import glob
import os
import re

_RC = ["IHDP", "ACIC", "CPS", "PSID", "PSID_bal"]


def _read_rows(path):
    """The markdown table rows of a file, or None if it has none."""
    if not os.path.isfile(path):
        return None
    rows = [ln.rstrip("\n") for ln in open(path) if ln.startswith("| ")]
    return rows or None


def _emit(out, title, rows, note=None):
    out.append(f"#### {title}")
    if note:
        out.append(f"*{note}*")
    if not rows:
        out.append("_(no table)_"); out.append(""); return
    out.extend(rows)
    out.append("")


def _saturated(rows):
    """True when every method's coverage is 1.000 -- IS then equals length."""
    covs = []
    for r in rows[2:] if len(rows) > 2 else []:
        cells = [c.strip() for c in r.strip("|").split("|")]
        for c in cells:
            if re.fullmatch(r"[01]\.\d{3}", c):
                covs.append(float(c)); break
    return bool(covs) and all(c >= 0.9995 for c in covs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rc-root");  ap.add_argument("--cmech-root")
    ap.add_argument("--cs-root");  ap.add_argument("--ctx", default="1000")
    ap.add_argument("--tag", default="B100_K1", help="MALC tag in the T filenames")
    ap.add_argument("--targets", nargs="+", default=["cate", "ate"])
    ap.add_argument("--cs-d", nargs="+", default=None,
                    help="restrict the case-study d values (default: all found)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = ["# MALC calibration study — all benchmarks", "",
           f"Variant T tag: `{a.tag}`. Point estimates carry no MALC.", "",
           "`IS_0.05` is the column to rank on. Where a section is flagged",
           "**coverage saturated**, every method covered every query, so IS is",
           "identical to length and the ranking is by width alone.", ""]

    if a.rc_root:
        out += ["## RealCause", ""]
        for ds in _RC:
            out.append(f"### {ds}")
            _emit(out, "Point (raw vs EM)", _read_rows(f"{a.rc_root}/point_raw_em_{ds}.md"))
            for t in a.targets:
                r = _read_rows(f"{a.rc_root}/calib_{ds}_raw_{t}.md")
                _emit(out, f"Calibration raw — {t}", r,
                      "coverage saturated" if r and _saturated(r) else None)
                r = _read_rows(f"{a.rc_root}/calib_{ds}_T_{a.tag}_{t}.md")
                _emit(out, f"Calibration T — {t}", r,
                      "coverage saturated" if r and _saturated(r) else None)

    if a.cmech_root:
        out += ["## ComplexMech", ""]
        for t in a.targets:
            r = _read_rows(f"{a.cmech_root}/calib_CMECH_raw_{t}.md")
            _emit(out, f"Calibration raw — {t}", r,
                  "coverage saturated" if r and _saturated(r) else None)
            r = _read_rows(f"{a.cmech_root}/calib_CMECH_T_{a.tag}_{t}.md")
            _emit(out, f"Calibration T — {t}", r,
                  "coverage saturated" if r and _saturated(r) else None)

    if a.cs_root:
        out += ["## Case studies", "",
                "One cell per (shift, d). The shift is a DGP-level constant",
                "treatment effect beta, present so the true ATE is never near",
                "zero and the relative error has a usable denominator.", ""]
        cells = sorted(glob.glob(f"{a.cs_root}/shift*/d*/ctx{a.ctx}"),
                       key=lambda p: (p.split("/shift")[1].split("/")[0],
                                      int(re.search(r"/d(\d+)/", p).group(1))))
        for cell in cells:
            sh = cell.split("/shift")[1].split("/")[0]
            d = re.search(r"/d(\d+)/", cell).group(1)
            if a.cs_d and d not in a.cs_d:
                continue
            out.append(f"### shift {sh}, d = {d}")
            pts = sorted(glob.glob(f"{cell}/point_raw_em_*.md"))
            for p in pts:
                case = os.path.basename(p)[len("point_raw_em_"):-3]
                _emit(out, f"Point — {case}", _read_rows(p))
            for t in a.targets:
                r = _read_rows(f"{cell}/calib_raw_{t}.md")
                _emit(out, f"Calibration raw — {t}", r,
                      "coverage saturated" if r and _saturated(r) else None)
                r = _read_rows(f"{cell}/calib_T_{a.tag}_{t}.md")
                _emit(out, f"Calibration T — {t}", r,
                      "coverage saturated" if r and _saturated(r) else None)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("\n".join(out) + "\n")
    n_tables = sum(1 for ln in out if ln.startswith("| ") and "---" not in ln)
    print(f"wrote {a.out}  ({n_tables} table rows)")


if __name__ == "__main__":
    main()
