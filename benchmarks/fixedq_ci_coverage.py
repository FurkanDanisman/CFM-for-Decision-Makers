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


def _file_truth(path, q):
    """The dump's OWN true tau for query q, in that file's units.

    ComplexMech squashes the outcome by a scaling refit on each training sample, so
    its stored truth is in per-replicate units and no single number is the truth for
    every replicate. But the model's interval is in those same units, so comparing
    the two inside one file gives the right cover/miss -- coverage is invariant to
    the affine, even though the number is not. This reads that per-file truth
    instead of a shared constant.

    WIDTHS still are not comparable across replicates, which is why the caller
    reports them relative to sd(Y0) as well.
    """
    with np.load(path, allow_pickle=True) as z:
        for k in ("true_cate_per_query", "true_cate", "cate"):
            if k in z.files:
                v = np.asarray(z[k], dtype=np.float64).ravel()
                if q < v.size:
                    return float(v[q])
    return float("nan")


def _one_file(args):
    """(est, v_own, v_rho1) for one replicate, or None.

    Module level and tuple-argument so ProcessPoolExecutor can pickle it. The work
    is one np.load per replicate and there are ~1000 per model x 13 models, so the
    cost is thousands of small reads off shared storage -- latency-bound, which is
    exactly what parallel workers fix.
    """
    path, q, per_file_truth = args
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
            cp, ta, s0, s1,
            _file_truth(path, q) if per_file_truth else float("nan"))


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
    ap.add_argument("--query", type=int, nargs="+", default=[0],
                    help="one or more frozen rows. Each has its OWN true tau and "
                         "its own coverage; the reported figure pools them, which "
                         "is the mean of several single-estimand coverages and not "
                         "the same thing as coverage averaged over random queries")
    ap.add_argument("--true-tau", type=float, default=None,
                    help="only valid with a single --query; otherwise the truths "
                         "come from --data-cell")
    ap.add_argument("--data-cell", default=None,
                    help="read the true tau from this cell instead of --true-tau")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json-out", default=None,
                    help="write the per-model, per-query coverages here. This is "
                         "what an across-realization aggregator consumes; re-deriving "
                         "them from the markdown would lose the per-query detail the "
                         "average-of-average needs.")
    ap.add_argument("--per-file-truth", action="store_true",
                    help="take each replicate's true tau from the dump itself rather "
                         "than one shared value. Needed for ComplexMech, whose "
                         "stored truth is in per-replicate units; coverage is still "
                         "exact because the interval is in those same units, but "
                         "widths are then only comparable via sd(Y0).")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1") or 1),
                    help="parallel worker processes (default: $SLURM_CPUS_PER_TASK)")
    a = ap.parse_args()

    if a.per_file_truth:
        # A nominal value only, for the header: every indicator uses its own file's.
        truths = {q: 0.0 for q in a.query}
    elif a.true_tau is not None and len(a.query) > 1:
        sys.exit("--true-tau takes one value; use --data-cell for several queries")
    elif a.true_tau is not None:
        truths = {a.query[0]: float(a.true_tau)}
    elif a.data_cell:
        truths = {q: true_tau_from(a.data_cell, q) for q in a.query}
    else:
        sys.exit("need --true-tau or a --data-cell containing it")
    bad = ([] if a.per_file_truth
           else [q for q, v in truths.items() if v is None or not np.isfinite(v)])
    if bad:
        sys.exit(f"no true tau for quer{'y' if len(bad) == 1 else 'ies'} {bad}")
    tt = truths[a.query[0]]      # for the header when there is only one

    def _rootname(root):
        """The MODEL directory, skipping layout components.

        ComplexMech roots end .../<model>/N1000, so without N<int> here every row is
        named "N1000" and thirteen models become indistinguishable -- which is how the
        first ComplexMech table came out with sixteen identical labels.
        """
        _LAYOUT = ("cs", "rc")
        for p in reversed([x for x in os.path.normpath(root).split(os.sep) if x]):
            if (p.startswith(("shift", "ctx")) or p in _LAYOUT
                    or re.fullmatch(r"d\d+", p) or re.fullmatch(r"N\d+", p)):
                continue
            return p
        return os.path.basename(root)

    nw = max(1, int(a.workers))
    ex = ProcessPoolExecutor(max_workers=nw) if nw > 1 else None
    def read_all(files, q):
        arg = [(f, q, a.per_file_truth) for f in files]
        it = (ex.map(_one_file, arg, chunksize=8) if ex else map(_one_file, arg))
        return [r for r in it if r is not None]

    rows = []
    detail = {}
    for root in a.root:
        found = [(lab, sd, d) for lab, sd, _t in METHODS
                 for d in [_resolve_dir(root, sd, a.dataset)]
                 if d and os.path.isdir(d)]
        use_root = len({sd for _, sd, _ in found}) == 1
        rname = _rootname(root)
        for label, subdir, d in found:
            fs = _files_in(d)
            print(f"[{_rootname(root)}/{label}] {len(fs)} replicate(s) x "
                  f"{len(a.query)} quer{'y' if len(a.query) == 1 else 'ies'}",
                  file=sys.stderr, flush=True)
            # Every query is its own fixed estimand with its own truth, so each is
            # scored separately and the indicators are concatenated. Pooling the
            # ESTIMATES instead would average unrelated numbers.
            E, RAW, VO, V1, CP, TA, S0, S1 = ([] for _ in range(8))
            HIT_O, HIT_1, W_O, W_1 = ([] for _ in range(4))
            # AVERAGE OF AVERAGES: each query's own coverage over its replicates,
            # then the unweighted mean over queries. Concatenating the indicators
            # instead weights a query with more completed replicates more heavily,
            # which silently changes the estimand whenever a dump is short.
            QC_O, QC_1, QW_O, QW_1, QN = ([] for _ in range(5))
            for q in a.query:
                got = read_all(fs, q)
                if not got:
                    continue
                eq = np.asarray([g[0] for g in got])
                voq = np.asarray([g[1] for g in got])
                v1q = np.asarray([g[2] for g in got])
                # Per file when asked, else the one shared value. Either way tq
                # lines up elementwise with eq, so the indicator is computed in the
                # units each estimate actually lives in.
                tq = (np.asarray([g[7] for g in got]) if a.per_file_truth
                      else np.full(eq.shape, truths[q], dtype=np.float64))
                keep = np.isfinite(tq)
                if not keep.any():
                    continue
                eq, voq, v1q, tq = eq[keep], voq[keep], v1q[keep], tq[keep]
                got = [g for g, k in zip(got, keep) if k]
                for v, hit, wid, qc, qw in ((voq, HIT_O, W_O, QC_O, QW_O),
                                            (v1q, HIT_1, W_1, QC_1, QW_1)):
                    h = Z * np.sqrt(np.maximum(v, 0.0))
                    ind = (eq - h <= tq) & (tq <= eq + h)
                    hit.append(ind)
                    wid.append(2 * h)
                    qc.append(float(ind.mean()))        # this query's coverage
                    qw.append(float((2 * h).mean()))
                QN.append(int(eq.size))
                E.append(eq - tq)          # centred, so several queries pool
                RAW.append(eq)             # uncentred, for the density check
                VO.append(voq); V1.append(v1q)
                CP.append(np.asarray([g[3] for g in got]))
                TA.append(np.asarray([g[4] for g in got]))
                S0.append(np.asarray([g[5] for g in got]))
                S1.append(np.asarray([g[6] for g in got]))
            if not E:
                continue
            e = np.concatenate(E)          # now a BIAS series, not an estimate
            eraw = np.concatenate(RAW)     # the estimates themselves
            vo = np.concatenate(VO); v1 = np.concatenate(V1)
            cp = np.concatenate(CP); ta = np.concatenate(TA)
            sd0 = float(np.concatenate(S0).mean())
            sd1 = float(np.concatenate(S1).mean())
            # The harness also dumps its OWN point estimate. mean(Y1)-mean(Y0)
            # under the dumped density should reproduce it; where it does not, the
            # density (or this reader's un-scaling of it) is wrong and every
            # interval built from it is meaningless. The main pipeline gates on
            # exactly this via density_scale_r2, but that gate is a regression
            # ACROSS queries and is NaN at SCM_N_QUERY=1 -- i.e. unavailable for
            # every fixed-query run -- so check it directly, per replicate.
            # ate_pred / true_ate are means over EVERY query in the file. That is
            # the per-query value only when the dump has one query, which is true of
            # the case-study fixed-query runs and false here -- ComplexMech keeps all
            # n_test queries. Comparing a 100-query mean against a per-query estimate
            # produced differences like 6.7e-01 that mean nothing, so the columns are
            # withheld rather than shown as a density mismatch.
            multi_query = a.per_file_truth
            ok = np.isfinite(cp) & (not multi_query)
            dmax = (float(np.abs(eraw[ok] - cp[ok]).max()) if ok.any()
                    else float("nan"))
            cpm = float(cp[ok].mean()) if ok.any() else float("nan")
            # The dump's OWN true effect. It should BE the estimand under test; if
            # it is not, the harness scored a different target (or averaged over
            # more than one query) and its point estimate cannot be compared here.
            # Reported rather than used to blank the row: a blank says nothing,
            # while a wrong number names the problem.
            okt = np.isfinite(ta) & (not multi_query)
            tam = float(ta[okt].mean()) if okt.any() else float("nan")
            suf = next((x for x in ("-noanc", "-v3ab", "-v3a", "-v3b")
                        if label.endswith(x)), "")
            name = (rname + suf) if use_root else label
            c_o = float(np.mean(QC_O)); w_o = float(np.mean(QW_O))
            c_1 = float(np.mean(QC_1)); w_1 = float(np.mean(QW_1))
            per_query = {"n_queries": len(QC_O), "replicates_per_query": QN,
                         "cover_vx": QC_O, "width_vx": QW_O,
                         "cover_rho1": QC_1, "width_rho1": QW_1}
            rows.append((name, e.size, float(e.mean() + tt), float(e.mean()),
                         float(e.std(ddof=1)) if e.size > 1 else float("nan"),
                         c_o, w_o, c_1, w_1, cpm, tam, dmax, sd0, sd1))
            detail[name] = per_query

    ttl = f"  ({a.label})" if a.label else ""
    qs = ",".join(str(q) for q in a.query)
    if a.per_file_truth:
        tline = ("Each replicate scored against ITS OWN stored true tau (the "
                 "benchmark's truth is in per-replicate units). Coverage is exact; "
                 "widths are comparable only relative to sd(Y0).")
    elif len(a.query) == 1:
        tline = f"true tau = {tt:.10f}"
    else:
        vals = ", ".join(f"{q}:{truths[q]:+.4f}" for q in a.query)
        tline = (f"{len(a.query)} frozen queries, each its own estimand — {vals}."
                 f" 'mean est' and 'bias' are relative to each query's own truth,"
                 f" so only bias is meaningful across queries.")
    L = [f"## Fixed-query CI coverage — {a.dataset}, quer"
         f"{'y' if len(a.query) == 1 else 'ies'} {qs}{ttl}", "",
         tline, "",
         "| model | datasets | mean est | bias | sd(est) | cover v(x) | mean width "
         "| cover rho=1 | mean width rho=1 | sd(Y0) | sd(Y1) | dumped est "
         "| dumped true | max|density-dumped| |", "|" + "---|" * 14]
    for nm, n, m, b, sdv, co, wo, c1, w1, cpm, tam, dmax, sd0, sd1 in sorted(
            rows, key=lambda r: abs(r[3])):
        L.append(f"| {nm} | {n} | {m:+.4f} | {b:+.4f} | "
                 + (f"{sdv:.4f}" if np.isfinite(sdv) else "—")
                 + f" | {co:.3f} | {wo:.4f} | {c1:.3f} | {w1:.4f} "
                 + f"| {sd0:.4f} | {sd1:.4f} | "
                 + (f"{cpm:+.4f}" if np.isfinite(cpm) else "—") + " | "
                 + (f"{tam:+.4f}" if np.isfinite(tam) else "—") + " | "
                 + (f"{dmax:.2e}" if np.isfinite(dmax) else "—") + " |")
    L += ["",
          "Coverage is over RESAMPLED OBSERVATIONAL DATASETS with the query unit",
          "held fixed, so it is the frequentist coverage of one estimand, not an",
          "average over queries and DGP draws like the main tables.",
          "",
          "Each figure is an AVERAGE OF AVERAGES: every query's coverage over its own",
          "replicates, then the unweighted mean over queries. Pooling the indicators",
          "would weight a query with more completed replicates more heavily, which",
          "changes the quantity whenever a dump is short.",
          "",
          "bias = mean(est) - true tau;  sd(est) is the sampling spread of the",
          "point estimate across datasets. A model can miss by being mis-centred",
          "(large |bias|) or too narrow (width small relative to sd(est)), and the",
          "coverage column alone does not distinguish them.",
          "",
          "dumped est / dumped true / max|density-dumped| are blank under",
          "--per-file-truth: the dump stores them as means over every query in the",
          "file, which equals the per-query value only for a one-query dump.",
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
    if a.json_out:
        import json
        with open(a.json_out, "w") as fh:
            json.dump({"dataset": a.dataset, "query": list(a.query),
                       "label": a.label, "per_file_truth": bool(a.per_file_truth),
                       "models": detail}, fh, indent=2)
        print(f"wrote {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
