"""Report PEHE (raw and em/full) vs σ_ε scale k, per case, for the noise sweep.

Reads <root>/k<K>/ctx500/<model>/<case>/... produced by noise_sweep.sh and, for
each case, prints a model × k table (rows = model, columns = noise scale) for
both the raw/inner and em/full readouts. Shows the coarse-bin 2D methods
converging to the 1D methods as σ_ε widens the CID past the J=10 bin width.

Usage:
    python case_study/cluster/noise_report.py \\
        --root $DEPLOY_ROOT/results_case_study/noise_sweep/results --ks 1 3 10 30 100
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

DISPLAY = [
    ("dopfn_native", "Do-PFN_native"), ("dopfn_bb", "Do-PFN_bb"),
    ("graph2d_noanc", "Graph2d_noanc"), ("uwyk_noanc", "UWYK_noanc"),
    ("graph2d_v3a", "Graph2d_v3a"), ("uwyk_v3a", "UWYK_v3a"),
    ("graph2d_v3b", "Graph2d_v3b"), ("uwyk_v3b", "UWYK_v3b"),
    ("cpfn1d", "cpfn1d"), ("cpfn2d", "cpfn2d"),
]
_ORDER = {n: i for i, (n, _) in enumerate(DISPLAY)}
_LABEL = dict(DISPLAY)


def _pehe(A, sweep, spec, case, readout, ctx=500):
    """Mean PEHE for one cell; readout ∈ {'raw','em'}."""
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
        keys = (["pehe_em", "pehe_full", "pehe_raw", "pehe"]
                if readout == "em" else ["pehe_raw", "pehe"])
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
    ap.add_argument("--root", required=True, help="Holds k<K>/ctx500/<model>/<case>/.")
    ap.add_argument("--ks", nargs="*", type=float, required=True)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--out", default=None, help="Optional CSV path.")
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(a.repo, "realcause_eval"))
    import aggregate_scm_ctx_sweep as A

    ks = [int(k) if float(k).is_integer() else k for k in a.ks]
    rows_csv = []
    for case in A.CASES:
        for readout in ("raw", "em"):
            printed = False
            lines = []
            for spec in sorted(A.MODEL_SPECS, key=lambda s: _ORDER.get(s[0], 99)):
                cells = []
                for k in ks:
                    v = _pehe(A, os.path.join(a.root, f"k{k}"), spec, case, readout)
                    cells.append(v)
                    if v is not None and readout == "raw":
                        rows_csv.append((case, spec[0], k, v))
                if all(c is None for c in cells):
                    continue
                printed = True
                line = _LABEL.get(spec[0], spec[0]).ljust(16)
                line += "".join("—".rjust(10) if c is None else f"{c:10.3f}" for c in cells)
                lines.append(line)
            if printed:
                print(f"\n══ {case}  PEHE ({readout})  vs σ_ε scale k ══")
                print("model".ljust(16) + "".join(f"k={k}".rjust(10) for k in ks))
                print("-" * (16 + 10 * len(ks)))
                print("\n".join(lines))

    if a.out and rows_csv:
        with open(a.out, "w") as f:
            f.write("case,model,k,pehe_raw\n")
            for r in rows_csv:
                f.write("%s,%s,%s,%.6f\n" % r)
        print(f"\n[noise_report] wrote {a.out}")


if __name__ == "__main__":
    main()
