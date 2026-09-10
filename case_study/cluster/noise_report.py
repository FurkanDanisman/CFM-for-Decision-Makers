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
    ap.add_argument("--scoreboard", action="store_true",
                    help="Also print per-model win counts (lowest PEHE per case) "
                         "for each k, plus the per-case winner list.")
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

    if a.scoreboard:
        _scoreboard(A, a.root, ks)

    if a.out and rows_csv:
        with open(a.out, "w") as f:
            f.write("case,model,k,pehe_raw\n")
            for r in rows_csv:
                f.write("%s,%s,%s,%.6f\n" % r)
        print(f"\n[noise_report] wrote {a.out}")


# Model groupings for the scoreboard. cpfn2d = CausalPFN's 2D head ("theirs").
_TWOD = ["dopfn_bb", "cpfn2d", "graph2d_noanc", "graph2d_v3a", "graph2d_v3b"]
_OURS_2D = ["dopfn_bb", "graph2d_noanc", "graph2d_v3a", "graph2d_v3b"]
_REFERENCE = "cpfn2d"


def _scoreboard(A, root, ks):
    spec_of = {s[0]: s for s in A.MODEL_SPECS}

    def pehe_at(k):  # {case: {model: pehe_raw}}
        out = {}
        for case in A.CASES:
            d = {}
            for name, spec in spec_of.items():
                v = _pehe(A, os.path.join(root, f"k{k}"), spec, case, "raw")
                if v is not None:
                    d[name] = v
            out[case] = d
        return out

    def wins(pe, models):
        w = {m: 0 for m in models}
        for case in A.CASES:
            cand = {m: pe[case][m] for m in models if m in pe[case]}
            if cand:
                w[min(cand, key=cand.get)] += 1
        return w

    print("\n" + "═" * 60)
    print("SCOREBOARD — wins /6 (lowest PEHE-raw per case), by σ_ε scale k")
    print("═" * 60)

    for scope, models in (("ALL models", [s[0] for s in A.MODEL_SPECS]),
                          ("2D heads only (cpfn2d = theirs)", _TWOD)):
        print(f"\n── {scope} ──")
        print("model".ljust(16) + "".join(f"k={k}".rjust(7) for k in ks))
        tallies = {k: wins(pehe_at(k), models) for k in ks}
        for m in sorted(models, key=lambda x: _ORDER.get(x, 99)):
            print(_LABEL.get(m, m).ljust(16)
                  + "".join(f"{tallies[k].get(m, 0):7d}" for k in ks))

    # Head-to-head: each of OURS vs their 2D head (cpfn2d), collapsing graph2d
    # to its best anc variant per case.
    print("\n── head-to-head vs cpfn2d (theirs), wins /6 ──")
    print("matchup".ljust(28) + "".join(f"k={k}".rjust(7) for k in ks))
    for label, ours in (("Do-PFN_bb  vs cpfn2d", ["dopfn_bb"]),
                        ("Graph2d(best) vs cpfn2d",
                         ["graph2d_noanc", "graph2d_v3a", "graph2d_v3b"])):
        row = label.ljust(28)
        for k in ks:
            pe = pehe_at(k); ow = 0
            for case in A.CASES:
                ours_v = [pe[case][m] for m in ours if m in pe[case]]
                if ours_v and _REFERENCE in pe[case] and min(ours_v) < pe[case][_REFERENCE]:
                    ow += 1
            row += f"{ow:7d}"
        print(row)
    print()


if __name__ == "__main__":
    main()
