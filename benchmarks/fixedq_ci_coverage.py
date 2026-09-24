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
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "UWYK_Fig3_4"))

from cate_density_metrics import METHODS, _resolve_dir              # noqa: E402
from arm_sd_one_query import arms_for, _files_in                    # noqa: E402

Z = 1.959963984540054          # exact two-sided 95% normal quantile


def _dumped_point(path):
    """(the harness's own point estimate, its own true tau) for the fixed query.

    The case-study dumps store `ate_pred` / `true_ate` -- means over the queries --
    and NOT a per-query cate_pred vector. Every fixed-query dump is made with
    SCM_N_QUERY=1, so that mean is over exactly one query and `ate_pred` IS the
    point estimate for it. `true_ate` is returned so the caller can verify that:
    if it does not equal the estimand under test, the file is not a one-query dump
    and `ate_pred` is a mean over several queries, which would be meaningless here.
    """
    with np.load(path, allow_pickle=True) as z:
        k = z.files
        ap = (float(np.asarray(z["ate_pred"]).ravel()[0])
              if "ate_pred" in k else float("nan"))
        ta = (float(np.asarray(z["true_ate"]).ravel()[0])
              if "true_ate" in k else float("nan"))
    return ap, ta


def _one_file(args):
    """(est, v_own, v_rho1) for one replicate, or None.

    Module level and tuple-argument so ProcessPoolExecutor can pickle it. The work
    is one np.load per replicate and there are ~1000 per model x 13 models, so the
    cost is thousands of small reads off shared storage -- latency-bound, which is
    exactly what parallel workers fix.
    """
    path, q = args
    got = arms_for(path, q)
    if got is None:
        return None
    m0, s0, m1, s1, st, cp, _ck = got
    if not all(np.isfinite(x) for x in (m0, s0, m1, s1)):
        return None
    if not np.isfinite(cp):
        # No per-query cate vector in these dumps; ate_pred at SCM_N_QUERY=1 is it.
        cp, ta = _dumped_point(path)
    else:
        ta = float("nan")
    return (m1 - m0,
            st ** 2 if np.isfinite(st) else s0 ** 2 + s1 ** 2,
            (s1 - s0) ** 2,
            cp, ta)


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
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1") or 1),
                    help="parallel worker processes (default: $SLURM_CPUS_PER_TASK)")
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

    nw = max(1, int(a.workers))
    ex = ProcessPoolExecutor(max_workers=nw) if nw > 1 else None
    def read_all(files):
        it = (ex.map(_one_file, [(f, a.query) for f in files], chunksize=8) if ex
              else map(_one_file, [(f, a.query) for f in files]))
        return [r for r in it if r is not None]

    rows = []
    for root in a.root:
        found = [(lab, sd, d) for lab, sd, _t in METHODS
                 for d in [_resolve_dir(root, sd, a.dataset)]
                 if d and os.path.isdir(d)]
        use_root = len({sd for _, sd, _ in found}) == 1
        rname = _rootname(root)
        for label, subdir, d in found:
            fs = _files_in(d)
            print(f"[{_rootname(root)}/{label}] {len(fs)} replicate(s)",
                  file=sys.stderr, flush=True)
            got = read_all(fs)
            if not got:
                continue
            e = np.asarray([g[0] for g in got])
            vo = np.asarray([g[1] for g in got])
            v1 = np.asarray([g[2] for g in got])
            # The harness also dumps its OWN point estimate. mean(Y1)-mean(Y0)
            # under the dumped density should reproduce it; where it does not, the
            # density (or this reader's un-scaling of it) is wrong and every
            # interval built from it is meaningless. The main pipeline gates on
            # exactly this via density_scale_r2, but that gate is a regression
            # ACROSS queries and is NaN at SCM_N_QUERY=1 -- i.e. unavailable for
            # every fixed-query run -- so check it directly, per replicate.
            cp = np.asarray([g[3] for g in got])
            ta = np.asarray([g[4] for g in got])
            ok = np.isfinite(cp)
            dmax = float(np.abs(e[ok] - cp[ok]).max()) if ok.any() else float("nan")
            cpm = float(cp[ok].mean()) if ok.any() else float("nan")
            # The dump's OWN true effect. It should BE the estimand under test; if
            # it is not, the harness scored a different target (or averaged over
            # more than one query) and its point estimate cannot be compared here.
            # Reported rather than used to blank the row: a blank says nothing,
            # while a wrong number names the problem.
            okt = np.isfinite(ta)
            tam = float(ta[okt].mean()) if okt.any() else float("nan")
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
                         c_o, w_o, c_1, w_1, cpm, tam, dmax))

    ttl = f"  ({a.label})" if a.label else ""
    L = [f"## Fixed-query CI coverage — {a.dataset}, query {a.query}{ttl}", "",
         f"true tau = {tt:.10f}", "",
         "| model | datasets | mean est | bias | sd(est) | cover v(x) | mean width "
         "| cover rho=1 | mean width rho=1 | dumped est | dumped true "
         "| max|density-dumped| |", "|" + "---|" * 12]
    for nm, n, m, b, sdv, co, wo, c1, w1, cpm, tam, dmax in sorted(
            rows, key=lambda r: abs(r[3])):
        L.append(f"| {nm} | {n} | {m:+.4f} | {b:+.4f} | "
                 + (f"{sdv:.4f}" if np.isfinite(sdv) else "—")
                 + f" | {co:.3f} | {wo:.4f} | {c1:.3f} | {w1:.4f} | "
                 + (f"{cpm:+.4f}" if np.isfinite(cpm) else "—") + " | "
                 + (f"{tam:+.4f}" if np.isfinite(tam) else "—") + " | "
                 + (f"{dmax:.2e}" if np.isfinite(dmax) else "—") + " |")
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
          "dumped true is the harness's own true effect for what it scored. It",
          "should equal the true tau above; where it does not, that harness scored",
          "a different target (or averaged over more than one query) and its rows",
          "are not comparable. A dash means the dump carries neither a per-query",
          "cate vector nor ate_pred/true_ate at all.",
          "",
          "dumped est is the harness's OWN point estimate, and",
          "max|density-dumped| is the largest per-replicate disagreement with",
          "mean(Y1)-mean(Y0) read off the dumped density. Near zero means the",
          "density faithfully represents the model, so a large bias is the MODEL's;",
          "large means the density or its un-scaling is broken and every interval",
          "here is void. This is the density_scale_r2 check done per replicate,",
          "because that gate is a regression across queries and is NaN whenever",
          "SCM_N_QUERY=1 -- which is every fixed-query run.",
          "",
          "CI = est +- 1.96*sqrt(v). Note sd(est) is the ACTUAL sampling spread,",
          "while sqrt(v) is what the model claims; a calibrated head would have",
          "them comparable, and width << sd(est) means the head understates its",
          "own sampling variability."]
    if ex is not None:
        ex.shutdown()
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
