"""Aggregate the d_variation eval grid into one tidy long-form CSV.

Walks <root>/shift<S>/d<K>/ctx<N>/<model>/<case>/... and writes, per
(shift, d, N, case, model), the mean / SEM / median of each metric over the
realizations:
    shift,d,N,case,model,
    pehe_raw,pehe_raw_sem,pehe_raw_med, pehe_em,pehe_em_sem,pehe_em_med,
    l1_raw,l1_raw_sem,l1_raw_med,      l1_em,l1_em_sem,l1_em_med, n

Modes:
  (default)         grid layout shift<S>/d<K>/ctx<N>/...
  --flat            flat layout ctx<N>/<model>/<case>/... (original/table3 run);
                    rows tagged shift=orig, d=0.
  --combine-shifts S1 S2 ...   pool realizations ACROSS those shift roots for
                    each (d,N,case,model) — one row per cell, tagged with
                    --combine-label. Only cells present in ALL listed shifts are
                    emitted (so mismatched contexts drop out automatically).
                    Pooling is done on the raw per-realization values, so the
                    median and SEM are correct for the combined set.

Usage:
    python dsweep_report.py --root <RESULTS_ROOT> --out grid.csv
    python dsweep_report.py --root <RESULTS_ROOT> --combine-shifts shift0 shift+2 \
        --combine-label shift0+2 --out combined.csv
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
    ("cpfn2d_pooled", "uniform", None),          # cpfn2d: pooled only
    ("cpfn1d_perarm", "uniform", None),          # cpfn1d: per-arm only
    ("graph2d", "graph2d", "noanc"), ("graph2d", "graph2d", "v3a"),
    ("graph2d", "graph2d", "v3b"),
    ("uwyk", "uniform", None), ("uwyk_v3a", "uniform", None),
    ("uwyk_noanc", "uniform", None),
]
METRICS = ["pehe_raw", "pehe_em", "l1_raw", "l1_em"]


def _scalar(v):
    if v is None:
        return None
    v = np.asarray(v).reshape(-1)
    return float(v[0]) if v.size else None


def _stats(vals):
    """(mean, sem, median, q1, q3) over finite values."""
    a = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return (float("nan"),) * 5
    sem = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    q1, q3 = (float(x) for x in np.percentile(a, [25, 75]))
    return float(a.mean()), sem, float(np.median(a)), q1, q3


def collect(A, cell, kind, tag):
    """Return {metric: [per-realization values]} for one model/case dir, or None."""
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
        l1r = list(np.abs(ap - ta)) if ap is not None and ta is not None else []
        l1e = list(np.abs(ae - ta)) if ae is not None and ta is not None else []
        return {"pehe_raw": list(pr), "pehe_em": list(pe), "l1_raw": l1r, "l1_em": l1e}

    if kind == "graph2d":
        pr_k = [f"pehe_raw_{tag}"]; pe_k = [f"pehe_em_{tag}", f"pehe_full_{tag}", f"pehe_raw_{tag}"]
        ar_k = [f"ate_raw_{tag}"]; ae_k = [f"ate_em_{tag}", f"ate_raw_{tag}"]
        paths = sorted(glob.glob(os.path.join(cell, "*_r*.npz")))
    else:  # uniform
        pr_k = ["pehe_raw", "pehe"]; pe_k = ["pehe_em", "pehe_full", "pehe_raw", "pehe"]
        er_k = ["err_raw"]; ee_k = ["err_em"]  # already L1 for the case studies
        paths = (sorted(glob.glob(os.path.join(cell, "r*.npz")))
                 or sorted(glob.glob(os.path.join(cell, "*_r*.npz"))))
    if not paths:
        return None
    out = {m: [] for m in METRICS}
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            out["pehe_raw"].append(_scalar(A._first(z, pr_k)))
            out["pehe_em"].append(_scalar(A._first(z, pe_k)))
            if kind == "graph2d":
                ar = _scalar(A._first(z, ar_k)); ae = _scalar(A._first(z, ae_k))
                ta = _scalar(A._first(z, ["true_ate"]))
                out["l1_raw"].append(abs(ar - ta) if ar is not None and ta is not None else None)
                out["l1_em"].append(abs(ae - ta) if ae is not None and ta is not None else None)
            else:
                out["l1_raw"].append(_scalar(A._first(z, er_k)))
                out["l1_em"].append(_scalar(A._first(z, ee_k)))
    return out


_SUFFIX = ["", "_sem", "_med", "_q1", "_q3"]        # 5 stats per metric
_HEADER = ",".join(["shift", "d", "N", "case", "model"]
                   + [m + s for m in METRICS for s in _SUFFIX] + ["n"]) + "\n"


def _row(shift, d, N, case, label, coll):
    n = max((len(coll[m]) for m in METRICS), default=0)
    vals = []
    for m in METRICS:
        vals.extend(_stats(coll[m]))
    return ("%s,%d,%d,%s,%s," + ",".join(["%.6f"] * (5 * len(METRICS))) + ",%d\n") % (
        (shift, d, N, case, label) + tuple(vals) + (n,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flat", action="store_true",
                    help="Flat ctx<N>/<model>/<case> root (original/table3); shift=orig,d=0.")
    ap.add_argument("--combine-shifts", nargs="*", default=None,
                    help="Pool realizations across these shift subdirs of --root.")
    ap.add_argument("--combine-label", default=None, help="shift label for the pooled rows.")
    ap.add_argument("--d-values", nargs="*", type=int, default=None,
                    help="Only read these d (skips npz for others — big I/O saving).")
    ap.add_argument("--pool-cases", action="store_true",
                    help="Combine mode: also pool realizations across the 6 cases -> "
                         "one row per (d,N,model) with case=ALL.")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.repo, "realcause_eval"))
    import aggregate_scm_ctx_sweep as A
    cases = A.CASES
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n_rows = 0

    with open(a.out, "w") as fh:
        fh.write(_HEADER)

        # ── Combine mode: pool across shift roots per (d,N,case,model) ──
        if a.combine_shifts:
            shifts = a.combine_shifts
            label = a.combine_label or "+".join(shifts)
            base = os.path.join(a.root, shifts[0])
            rx = re.compile(r".*/d(\d+)/ctx(\d+)$")
            for cd in sorted(glob.glob(os.path.join(base, "d*", "ctx*"))):
                m = rx.match(cd)
                if not m:
                    continue
                d, N = int(m.group(1)), int(m.group(2))
                if a.d_values and d not in a.d_values:
                    continue
                for dirn, kind, tag in MODELS:
                    lab = dirn if tag is None else f"{dirn}_{tag}"
                    if a.pool_cases:
                        # pool realizations across ALL shifts AND ALL cases
                        pooled = {mt: [] for mt in METRICS}
                        got = False
                        for sh in shifts:
                            for case in cases:
                                c = collect(A, os.path.join(a.root, sh, f"d{d}",
                                            f"ctx{N}", dirn, case), kind, tag)
                                if c is None:
                                    continue
                                got = True
                                for mt in METRICS:
                                    pooled[mt] += c[mt]
                        if got:
                            fh.write(_row(label, d, N, "ALL", lab, pooled)); n_rows += 1
                        continue
                    for case in cases:
                        cells = [os.path.join(a.root, sh, f"d{d}", f"ctx{N}", dirn, case)
                                 for sh in shifts]
                        colls = [collect(A, c, kind, tag) for c in cells]
                        if any(c is None for c in colls):   # need all shifts present
                            continue
                        pooled = {mt: sum((c[mt] for c in colls), []) for mt in METRICS}
                        fh.write(_row(label, d, N, case, lab, pooled)); n_rows += 1
            print(f"[dsweep_report] combined {shifts} -> '{label}': {n_rows} rows -> {a.out}")
            return

        # ── Normal (grid or flat) ──
        if a.flat:
            rx = re.compile(r".*/ctx(\d+)$")
            ctx_dirs = sorted(glob.glob(os.path.join(a.root, "ctx*")))
        else:
            rx = re.compile(r".*/shift([^/]+)/d(\d+)/ctx(\d+)$")
            ctx_dirs = sorted(glob.glob(os.path.join(a.root, "shift*", "d*", "ctx*")))
        for cd in ctx_dirs:
            m = rx.match(cd)
            if not m:
                continue
            if a.flat:
                shift, d, N = "orig", 0, int(m.group(1))
            else:
                shift, d, N = "shift" + m.group(1), int(m.group(2)), int(m.group(3))
            if a.d_values and d not in a.d_values:
                continue
            for dirn, kind, tag in MODELS:
                lab = dirn if tag is None else f"{dirn}_{tag}"
                for case in cases:
                    coll = collect(A, os.path.join(cd, dirn, case), kind, tag)
                    if coll is None:
                        continue
                    fh.write(_row(shift, d, N, case, lab, coll)); n_rows += 1

    print(f"[dsweep_report] wrote {n_rows} rows -> {a.out}")
    if n_rows == 0:
        print(f"[dsweep_report] (nothing under {a.root} — jobs running or wrong path?)")


if __name__ == "__main__":
    main()
