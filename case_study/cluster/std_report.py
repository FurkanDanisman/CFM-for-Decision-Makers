"""Report PEHE per standardization scheme × case for cpfn2d and dopfn_bb.

Reads <root>/<model>/<scheme>/<case>/... from std_sweep.sh (dispatcher writes
OUT_ROOT/<case>), plus the 1D references under <root>/ref/<model>/<case>.
Per model, prints a scheme × case table (PEHE raw, and em where it differs),
with the matching 1D reference row as the target line. per_arm* rows are marked
INVALID (they reintroduce confounded arm-mean gaps — reference only).

Usage:
    python case_study/cluster/std_report.py --root $DEPLOY_ROOT/results_case_study/std_sweep
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

_KIND = {"dopfn_bb": "dopfn_bb", "cpfn2d": "uniform",
         "dopfn_native": "uniform", "cpfn1d": "uniform"}


def _read_cell(A, cell, kind, readout):
    """Mean PEHE for a <scheme>/<case> dir. readout ∈ {raw, em}."""
    if kind == "dopfn_bb":
        f = os.path.join(cell, "summary.npz")
        if not os.path.isfile(f):
            return None
        with np.load(f, allow_pickle=True) as z:
            v = A._first(z, ["pehe_em", "pehe"] if readout == "em" else ["pehe"])
        return float(np.mean(v)) if v is not None and len(v) else None
    keys = (["pehe_em", "pehe_full", "pehe_raw", "pehe"] if readout == "em"
            else ["pehe_raw", "pehe"])
    paths = (sorted(glob.glob(os.path.join(cell, "r*.npz")))
             or sorted(glob.glob(os.path.join(cell, "*_r*.npz"))))
    vals = []
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            pe = A._first(z, keys)
        if pe is not None:
            vals.append(float(pe))
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.repo, "realcause_eval"))
    import aggregate_scm_ctx_sweep as A
    cases = A.CASES
    short = {c: c.replace("Observed_", "Obs").replace("_Criterion", "")
              .replace("_and_Confounder", "+Cf").replace("_Confounder", "Cf")
              .replace("_Mediator", "Med") for c in cases}

    def table(model, ref_model):
        mdir = os.path.join(a.root, model)
        if not os.path.isdir(mdir):
            print(f"\n(no results for {model})"); return
        schemes = sorted(d for d in os.listdir(mdir)
                         if os.path.isdir(os.path.join(mdir, d)))
        print(f"\n══ {model}  PEHE (raw)  by standardization scheme ══")
        hdr = "scheme".ljust(14) + "".join(short[c][:8].rjust(9) for c in cases) + "   mean"
        print(hdr); print("-" * len(hdr))
        # reference 1D row first
        refdir = os.path.join(a.root, "ref", ref_model)
        if os.path.isdir(refdir):
            vals = [_read_cell(A, os.path.join(refdir, c), _KIND[ref_model], "raw") for c in cases]
            _row(f"[ref] {ref_model}"[:14], vals)
        for s in schemes:
            vals = [_read_cell(A, os.path.join(mdir, s, c), _KIND[model], "raw") for c in cases]
            tag = s + (" !" if "perarm" in s else "")
            _row(tag[:14], vals)

    def _row(label, vals):
        finite = [v for v in vals if v is not None]
        m = np.mean(finite) if finite else float("nan")
        line = label.ljust(14) + "".join("—".rjust(9) if v is None else f"{v:9.3f}" for v in vals)
        line += f"{m:8.3f}" if finite else "     —"
        print(line)

    table("dopfn_bb", "dopfn_native")
    table("cpfn2d", "cpfn1d")
    print("\n! per_arm rows are INVALID for confounded cases (reference upper-bound only).")


if __name__ == "__main__":
    main()
