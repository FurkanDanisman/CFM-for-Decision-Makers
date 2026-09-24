#!/usr/bin/env python
"""Frequentist coverage of the two normal-approximation CIs, over resampled
observational datasets, for ONE fixed query.

For each model and each of the D replicate datasets: take the model's predictive
density for the query, form

    CATE = mean(Y1) - mean(Y0)
    v(x)      = Var(Y1 - Y0)   -- the head's own coupling, or rho = 0 for a 1D head
    v(x)|rho=1 = (s1 - s0)^2   -- same marginals, perfect positive dependence
    CI = CATE +- 1.96 * sqrt(v)

and report what fraction of the D intervals contain the true tau. Because the
query unit is frozen across replicates, that fraction is coverage for a single
fixed estimand -- the textbook frequentist quantity -- rather than an average over
queries and DGP draws, which is what the main tables report.

Also reports the sampling spread of the point estimate across datasets (sd of
CATE), which separates bias from variance: an interval can miss because it is
mis-centred or because it is too narrow, and coverage alone cannot tell you which.

    python benchmarks/fixedq_ci_coverage.py --root $SCRATCH/cs_fq_dumps/*/shift0/d0/ctx1000 \
        --dataset Observed_Confounder --true-tau -0.2445281297 --query 0
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "UWYK_Fig3_4"))

from cate_density_metrics import METHODS, _resolve_dir              # noqa: E402
from arm_sd_one_query import arms_for, _files_in                    # noqa: E402

Z = 1.959963984540054          # exact two-sided 95% normal quantile


def true_tau_from(data_cell, q):
    fs = sorted(glob.glob(os.path.join(data_cell, "*.npz")))
    if not fs:
        return float("nan")
    with np.load(fs[0], allow_pickle=True) as z:
        if "cate" not in z.files:
            return float("nan")
        c = np.asarray(z["cate"]).ravel()
        return float(c[q]) if q < c.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="+", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--query", type=int, default=0)
    ap.add_argument("--true-tau", type=float, default=None)
    ap.add_argument("--data-cell", default=None,
                    help="read the true tau from this cell instead of --true-tau")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tt = a.true_tau
    if tt is None and a.data_cell:
        tt = true_tau_from(a.data_cell, a.query)
    if tt is None or not np.isfinite(tt):
        sys.exit("need --true-tau or a --data-cell containing it")

    def _rootname(root):
        _LAYOUT = ("cs", "rc")
        for p in reversed([x for x in os.path.normpath(root).split(os.sep) if x]):
            if (p.startswith(("shift", "ctx")) or p in _LAYOUT
                    or re.fullmatch(r"d\d+", p)):
                continue
            return p
        return os.path.basename(root)

    rows = []
    for root in a.root:
        found = [(lab, sd, d) for lab, sd, _t in METHODS
                 for d in [_resolve_dir(root, sd, a.dataset)]
                 if d and os.path.isdir(d)]
        use_root = len({sd for _, sd, _ in found}) == 1
        rname = _rootname(root)
        for label, subdir, d in found:
            est, v_own, v_rho1 = [], [], []
            for f in _files_in(d):
                got = arms_for(f, a.query)
                if got is None:
                    continue
                m0, s0, m1, s1, st, _cp, _ck = got
                if not all(np.isfinite(x) for x in (m0, s0, m1, s1)):
                    continue
                est.append(m1 - m0)
                v_own.append(st ** 2 if np.isfinite(st) else s0 ** 2 + s1 ** 2)
                v_rho1.append((s1 - s0) ** 2)
            if not est:
                continue
            e = np.asarray(est); vo = np.asarray(v_own); v1 = np.asarray(v_rho1)
            suf = next((x for x in ("-noanc", "-v3ab", "-v3a", "-v3b")
                        if label.endswith(x)), "")
            name = (rname + suf) if use_root else label
            def cov(v):
                h = Z * np.sqrt(np.maximum(v, 0.0))
                return float(np.mean((e - h <= tt) & (tt <= e + h))), float(np.mean(2 * h))
            c_o, w_o = cov(vo)
            c_1, w_1 = cov(v1)
            rows.append((name, e.size, float(e.mean()), float(e.mean() - tt),
                         float(e.std(ddof=1)) if e.size > 1 else float("nan"),
                         c_o, w_o, c_1, w_1))

    ttl = f"  ({a.label})" if a.label else ""
    L = [f"## Fixed-query CI coverage — {a.dataset}, query {a.query}{ttl}", "",
         f"true tau = {tt:.10f}", "",
         "| model | datasets | mean est | bias | sd(est) | cover v(x) | mean width "
         "| cover rho=1 | mean width rho=1 |", "|" + "---|" * 9]
    for nm, n, m, b, sdv, co, wo, c1, w1 in sorted(rows, key=lambda r: abs(r[3])):
        L.append(f"| {nm} | {n} | {m:+.4f} | {b:+.4f} | "
                 + (f"{sdv:.4f}" if np.isfinite(sdv) else "—")
                 + f" | {co:.3f} | {wo:.4f} | {c1:.3f} | {w1:.4f} |")
    L += ["",
          "Coverage is over RESAMPLED OBSERVATIONAL DATASETS with the query unit",
          "held fixed, so it is the frequentist coverage of one estimand, not an",
          "average over queries and DGP draws like the main tables.",
          "",
          "bias = mean(est) - true tau;  sd(est) is the sampling spread of the",
          "point estimate across datasets. A model can miss by being mis-centred",
          "(large |bias|) or too narrow (width small relative to sd(est)), and the",
          "coverage column alone does not distinguish them.",
          "",
          "CI = est +- 1.96*sqrt(v). Note sd(est) is the ACTUAL sampling spread,",
          "while sqrt(v) is what the model claims; a calibrated head would have",
          "them comparable, and width << sd(est) means the head understates its",
          "own sampling variability."]
    txt = "\n".join(L)
    print(txt)
    if not rows:
        print(f"\nno dumps found under {a.root}", file=sys.stderr)
        return 1
    if a.out:
        open(a.out, "w").write(txt + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
