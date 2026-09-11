"""Aggregate the d_variation eval grid into one tidy long-form CSV.

Walks <root>/shift<S>/d<K>/ctx<N>/<model>/<case>/... and writes rows:
    shift, d, N, case, model, pehe_raw, pehe_em, l1_raw, l1_em, n
so you can pivot however you like (PEHE vs d, vs N, per shift, etc.).

Metric sources (per realization, then mean over realizations):
  uniform (cpfn*, native, uwyk*): pehe_raw / pehe_em ; err_raw / err_em (already
      L1 |ate_hat - true_ate| for the case studies)
  dopfn_bb (summary.npz arrays): pehe / pehe_em ; |ate_pred - true_ate| and
      |ate_pred_em - true_ate|
  graph2d (per-realization, per anc tag): pehe_raw_<tag> / pehe_em_<tag> ;
      |ate_raw_<tag> - true_ate| (em L1 = |ate_em_<tag> - true_ate| if present)

Usage:
    python case_study/cluster/dsweep_report.py --root <RESULTS_ROOT> --out grid.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

# (dir_name, kind, tag)
MODELS = [
    ("dopfn_native", "uniform", None), ("dopfn_bb", "dopfn_bb", None),
    ("cpfn2d_pooled", "uniform", None), ("cpfn2d_log", "uniform", None),
    ("cpfn1d_perarm", "uniform", None), ("cpfn1d_pooled", "uniform", None),
    ("graph2d", "graph2d", "noanc"), ("graph2d", "graph2d", "v3a"),
    ("graph2d", "graph2d", "v3b"),
    ("uwyk", "uniform", None), ("uwyk_v3a", "uniform", None),
    ("uwyk_noanc", "uniform", None),
]


def _ms(a):
    """(mean, SEM) over finite values."""
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    sem = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return float(a.mean()), sem


def read_cell(A, cell, kind, tag):
    """Return dict {pehe_raw:(m,sem), pehe_em, l1_raw, l1_em, n} for one cell."""
    if kind == "dopfn_bb":
        f = os.path.join(cell, "summary.npz")
        if not os.path.isfile(f):
            return None
        with np.load(f, allow_pickle=True) as z:
            pr = A._first(z, ["pehe"]); pe = A._first(z, ["pehe_em", "pehe"])
            ap = A._first(z, ["ate_pred"]); ae = A._first(z, ["ate_pred_em", "ate_pred"])
            ta = A._first(z, ["true_ate"])
        if pr is None:
            return None
        l1r = np.abs(ap - ta) if ap is not None and ta is not None else [None]
        l1e = np.abs(ae - ta) if ae is not None and ta is not None else [None]
        return {"pehe_raw": _ms(pr), "pehe_em": _ms(pe),
                "l1_raw": _ms(l1r), "l1_em": _ms(l1e), "n": len(pr)}

    if kind == "graph2d":
        pr_k = [f"pehe_raw_{tag}"]; pe_k = [f"pehe_em_{tag}", f"pehe_full_{tag}", f"pehe_raw_{tag}"]
        ar_k = [f"ate_raw_{tag}"]; ae_k = [f"ate_em_{tag}", f"ate_raw_{tag}"]
        paths = sorted(glob.glob(os.path.join(cell, "*_r*.npz")))
    else:  # uniform
        pr_k = ["pehe_raw", "pehe"]; pe_k = ["pehe_em", "pehe_full", "pehe_raw", "pehe"]
        paths = (sorted(glob.glob(os.path.join(cell, "r*.npz")))
                 or sorted(glob.glob(os.path.join(cell, "*_r*.npz"))))
        er_k = ["err_raw"]; ee_k = ["err_em"]  # already L1 for case studies
    if not paths:
        return None
    prs, pes, l1rs, l1es = [], [], [], []
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            prs.append(_scalar(A._first(z, pr_k)))
            pes.append(_scalar(A._first(z, pe_k)))
            if kind == "graph2d":
                ar = _scalar(A._first(z, ar_k)); ae = _scalar(A._first(z, ae_k))
                ta = _scalar(A._first(z, ["true_ate"]))
                l1rs.append(abs(ar - ta) if ar is not None and ta is not None else None)
                l1es.append(abs(ae - ta) if ae is not None and ta is not None else None)
            else:
                l1rs.append(_scalar(A._first(z, er_k)))
                l1es.append(_scalar(A._first(z, ee_k)))
    return {"pehe_raw": _ms(prs), "pehe_em": _ms(pes),
            "l1_raw": _ms(l1rs), "l1_em": _ms(l1es),
            "n": len([x for x in prs if x is not None])}


def _scalar(v):
    if v is None:
        return None
    v = np.asarray(v).reshape(-1)
    return float(v[0]) if v.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Grid root (holds shift<S>/d<K>/ctx<N>/...).")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--flat", action="store_true",
                    help="Root is a flat ctx<N>/<model>/<case> tree (original-data "
                         "/ table3 layout, no shift/d nesting). Emits shift=orig, d=0.")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.repo, "realcause_eval"))
    import aggregate_scm_ctx_sweep as A
    cases = A.CASES

    if a.flat:
        rx = re.compile(r".*/ctx(\d+)$")
        ctx_dirs = sorted(glob.glob(os.path.join(a.root, "ctx*")))
    else:
        rx = re.compile(r".*/shift([^/]+)/d(\d+)/ctx(\d+)$")
        ctx_dirs = sorted(glob.glob(os.path.join(a.root, "shift*", "d*", "ctx*")))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n_rows = 0
    with open(a.out, "w") as fh:
        fh.write("shift,d,N,case,model,"
                 "pehe_raw,pehe_raw_sem,pehe_em,pehe_em_sem,"
                 "l1_raw,l1_raw_sem,l1_em,l1_em_sem,n\n")
        for cd in ctx_dirs:
            m = rx.match(cd)
            if not m:
                continue
            if a.flat:
                shift, d, N = "orig", 0, int(m.group(1))
            else:
                shift, d, N = m.group(1), int(m.group(2)), int(m.group(3))
            for dirn, kind, tag in MODELS:
                label = dirn if tag is None else f"{dirn}_{tag}"
                for case in cases:
                    cell = os.path.join(cd, dirn, case)
                    if not os.path.isdir(cell):
                        continue
                    r = read_cell(A, cell, kind, tag)
                    if r is None:
                        continue
                    fh.write("shift%s,%d,%d,%s,%s,"
                             "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n" % (
                                 shift, d, N, case, label,
                                 r["pehe_raw"][0], r["pehe_raw"][1],
                                 r["pehe_em"][0], r["pehe_em"][1],
                                 r["l1_raw"][0], r["l1_raw"][1],
                                 r["l1_em"][0], r["l1_em"][1], r["n"]))
                    n_rows += 1
    print(f"[dsweep_report] wrote {n_rows} rows -> {a.out}")
    if n_rows == 0:
        print(f"[dsweep_report] (nothing found under {a.root} — jobs running or wrong path?)")


if __name__ == "__main__":
    main()
