"""Report PEHE (raw + em) on the ORIGINAL DoPFN data run (eval_original.sh).

Reads <root>/ctx<N>/<model>/<case>/... including the two cpfn1d variants
(cpfn1d_perarm, cpfn1d_pooled). Prints a model × case table for the raw and
em/full readouts, matching the layout of your Table 3.

Usage:
    python case_study/cluster/orig_report.py --root <RESULTS_ROOT> --n 200
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

DISPLAY = [
    ("dopfn_native", "Do-PFN"), ("dopfn_bb", "Do-PFN 2D"),
    ("graph2d_noanc", "Graph2d_noanc"), ("uwyk_noanc", "UWYK_noanc"),
    ("graph2d_v3a", "Graph2d_v3a"), ("uwyk_v3a", "UWYK_v3a"),
    ("graph2d_v3b", "Graph2d_v3b"), ("uwyk_v3b", "UWYK_v3b"),
    ("cpfn1d_perarm", "cpfn1d per-arm"), ("cpfn1d_pooled", "cpfn1d pooled"),
    ("cpfn2d_pooled", "cpfn2d pooled"), ("cpfn2d_log", "cpfn2d log"),
]
_ORDER = {n: i for i, (n, _) in enumerate(DISPLAY)}
_LABEL = dict(DISPLAY)


def _pehe(A, sweep, ctx, spec, case, readout):
    name, dirn, kind, tag = spec
    cell = os.path.join(sweep, f"ctx{ctx}", dirn, case)
    if not os.path.isdir(cell):
        return None
    if kind == "dopfn_bb":
        f = os.path.join(cell, "summary.npz")
        if not os.path.isfile(f):
            return None
        with np.load(f, allow_pickle=True) as z:
            v = A._first(z, ["pehe_em", "pehe"] if readout == "em" else ["pehe"])
        return float(np.mean(v)) if v is not None and len(v) else None
    if kind == "graph2d":
        keys = ([f"pehe_em_{tag}", f"pehe_full_{tag}", f"pehe_raw_{tag}"]
                if readout == "em" else [f"pehe_raw_{tag}"])
        paths = sorted(glob.glob(os.path.join(cell, f"{case}_r*.npz")))
    else:
        keys = (["pehe_em", "pehe_full", "pehe_raw", "pehe"] if readout == "em"
                else ["pehe_raw", "pehe"])
        paths = (sorted(glob.glob(os.path.join(cell, "r*.npz")))
                 or sorted(glob.glob(os.path.join(cell, f"{case}_r*.npz"))))
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
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.repo, "realcause_eval"))
    import aggregate_scm_ctx_sweep as A

    # Replace single cpfn1d/cpfn2d with their pooled + per_arm variants.
    specs = [s for s in A.MODEL_SPECS if s[0] not in ("cpfn1d", "cpfn2d")]
    specs += [("cpfn1d_perarm", "cpfn1d_perarm", "uniform", None),
              ("cpfn1d_pooled", "cpfn1d_pooled", "uniform", None),
              ("cpfn2d_pooled", "cpfn2d_pooled", "uniform", None),
              ("cpfn2d_log", "cpfn2d_log", "uniform", None)]
    specs.sort(key=lambda s: _ORDER.get(s[0], 99))

    cases = A.CASES
    short = {c: c.replace("Observed_", "Obs").replace("_Criterion", "")
              .replace("_and_Confounder", "+Cf").replace("_Confounder", "Cf")
              .replace("_Mediator", "Med") for c in cases}

    for readout in ("raw", "em"):
        rows = []
        for spec in specs:
            vals = [_pehe(A, a.root, a.n, spec, c, readout) for c in cases]
            if all(v is None for v in vals):
                continue
            rows.append((spec[0], vals))
        if not rows:
            continue
        print(f"\n══ ORIGINAL DoPFN data  PEHE ({readout})  N={a.n} ══")
        hdr = "model".ljust(16) + "".join(short[c][:9].rjust(10) for c in cases)
        print(hdr); print("-" * len(hdr))
        for name, vals in rows:
            print(_LABEL.get(name, name).ljust(16)
                  + "".join("—".rjust(10) if v is None else f"{v:10.3f}" for v in vals))


if __name__ == "__main__":
    main()
