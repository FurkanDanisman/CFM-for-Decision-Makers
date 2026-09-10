"""Aggregate the case-study eval sweep into PEHE + L1-ATE tables.

Reads $SWEEP/ctx<N>/<model>/<case>/... produced by 02_submit_eval.sh and, for
each (model, context, case), reports
    PEHE   = mean over realizations of sqrt-MSE(pred_cate, true_cate)
    L1_ATE = mean over realizations of |mean(pred_cate) - true_ate|

The per-model output-shape handling (uniform / graph2d suffixed keys /
dopfn_bb summary.npz) is reused verbatim from
realcause_eval/aggregate_scm_ctx_sweep.py — this script only parameterizes the
context list (that one hardcodes 50/100/250/500/1000) and prints/writes a
compact table for our contexts.

Usage:
    python case_study/cluster/03_aggregate.py \\
        --sweep $DEPLOY_ROOT/results_case_study/shift+2 \\
        --contexts 200 500 1000 \\
        --out $DEPLOY_ROOT/results_case_study/shift+2/summary.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


def _pehe_em(A, sweep, ctx, spec, case):
    """Per-realization EM/full PEHE array for one cell, mirroring
    aggregate_scm_ctx_sweep._cell_pehe_l1's file layout but reading the
    finer readout key. Models without a density readout (dopfn_native fine 1D
    head; uwyk point regressor) have no pehe_em → fall back to their raw PEHE,
    so em == raw for them (correct: their mean is already the full readout)."""
    name, dirn, kind, tag = spec
    cell = os.path.join(sweep, f"ctx{ctx}", dirn, case)
    if not os.path.isdir(cell):
        return None
    if kind == "dopfn_bb":
        f = os.path.join(cell, "summary.npz")
        if not os.path.isfile(f):
            return None
        with np.load(f, allow_pickle=True) as z:
            return A._first(z, ["pehe_em", "pehe"])
    if kind == "graph2d":
        keys = [f"pehe_em_{tag}", f"pehe_full_{tag}", f"pehe_raw_{tag}"]
        paths = sorted(glob.glob(os.path.join(cell, f"{case}_r*.npz")))
    else:  # uniform
        keys = ["pehe_em", "pehe_full", "pehe_raw", "pehe"]
        paths = (sorted(glob.glob(os.path.join(cell, "r*.npz")))
                 or sorted(glob.glob(os.path.join(cell, f"{case}_r*.npz"))))
    out = []
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            pe = A._first(z, keys)
        if pe is not None:
            out.append(float(pe))
    return np.array(out) if out else None

# Fixed display order + labels (internal aggregator name -> shown label).
DISPLAY = [
    ("dopfn_native",  "Do-PFN_native"),
    ("dopfn_bb",      "Do-PFN_bb"),
    ("graph2d_noanc", "Graph2d_noanc"),
    ("uwyk_noanc",    "UWYK_noanc"),
    ("graph2d_v3a",   "Graph2d_v3a"),
    ("uwyk_v3a",      "UWYK_v3a"),
    ("graph2d_v3b",   "Graph2d_v3b"),
    ("uwyk_v3b",      "UWYK_v3b"),
    ("cpfn1d",        "cpfn1d"),
    ("cpfn2d",        "cpfn2d"),
]
_ORDER = {name: i for i, (name, _) in enumerate(DISPLAY)}
_LABEL = dict(DISPLAY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="Sweep root (holds ctx<N>/<model>/<case>/).")
    ap.add_argument("--contexts", nargs="*", type=int, default=[200, 500, 1000])
    ap.add_argument("--out", default=None, help="Optional CSV output path.")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), help="R-PFN repo root.")
    a = ap.parse_args()

    # Reuse the tested per-model cell reader.
    sys.path.insert(0, os.path.join(a.repo, "realcause_eval"))
    import aggregate_scm_ctx_sweep as A

    records = []  # (model, ctx, case, pehe_raw, pehe_em, l1, n)
    for spec in A.MODEL_SPECS:
        model = spec[0]
        for ctx in a.contexts:
            for case in A.CASES:
                pehe, l1, _ = A._cell_pehe_l1(a.sweep, ctx, tuple(spec), case)
                if pehe is None or not len(pehe):
                    continue
                pe_em = _pehe_em(A, a.sweep, ctx, tuple(spec), case)
                pehe_em = float(np.mean(pe_em)) if pe_em is not None and len(pe_em) \
                    else float(np.mean(pehe))
                l1_mean = float(np.mean(l1)) if l1 is not None and len(l1) else float("nan")
                records.append((model, ctx, case, float(np.mean(pehe)),
                                pehe_em, l1_mean, int(len(pehe))))

    if not records:
        print(f"[aggregate] no results found under {a.sweep} for contexts {a.contexts}.\n"
              f"            (jobs still running, or wrong --sweep path?)")
        return

    cases = A.CASES
    _short = {c: c.replace("Observed_", "Obs").replace("_Criterion", "")
               .replace("_and_Confounder", "+Conf").replace("_Confounder", "Conf")
               .replace("_Mediator", "Med") for c in cases}

    for metric, idx in (("PEHE (raw/inner)", 3), ("PEHE (em/full)", 4),
                        ("L1-ATE", 5)):
        for ctx in a.contexts:
            rows = {m: {} for m in dict.fromkeys(r[0] for r in records)}
            for m, c, case, pehe, pehe_em, l1, n in records:
                if c == ctx:
                    rows[m][case] = {3: pehe, 4: pehe_em, 5: l1}[idx]
            present = [m for m in rows if rows[m]]
            present.sort(key=lambda m: _ORDER.get(m, len(_ORDER)))
            if not present:
                continue
            print(f"\n══ {metric}   ctx N={ctx} ══")
            hdr = "model".ljust(16) + "".join(_short[c][:10].rjust(11) for c in cases)
            print(hdr); print("-" * len(hdr))
            for m in present:
                line = _LABEL.get(m, m).ljust(16)
                for c in cases:
                    v = rows[m].get(c)
                    line += ("—".rjust(11) if v is None else f"{v:11.3f}")
                print(line)

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            f.write("model,context,case,pehe,pehe_em,l1_ate,n\n")
            for rec in sorted(records, key=lambda r: (_ORDER.get(r[0], 99), r[1],
                                                      cases.index(r[2]) if r[2] in cases else 99)):
                f.write("%s,%d,%s,%.6f,%.6f,%.6f,%d\n" % rec)
        print(f"\n[aggregate] wrote {len(records)} rows -> {a.out}")


if __name__ == "__main__":
    main()
