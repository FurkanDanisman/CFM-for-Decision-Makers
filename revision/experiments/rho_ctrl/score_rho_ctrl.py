"""Score the controlled-rho benchmarks: one row per (benchmark, bin, model).
python score_rho_ctrl.py DATA_ROOT OUT_PREFIX          DATA_ROOT = $SCRATCH/rho_ctrl  (cs/, rc/, cm/, dumps/)

Per dataset, exactly as the paper's scorers do it:
  interval  the hist 95% interval of each query's ITE density (_query_pmfs + interval_hist, interval_95 on a
            non-uniform atom grid), scored against the query's REALIZED ITE:
              coverage c = mean 1{l <= delta <= u};  length = mean (u - l);
              IS_0.05  = mean (u - l) + (2/a)(l - delta)+ + (2/a)(delta - u)+,  a = 0.05
  point     tau_hat = point_raw_em.cate_for_file(f, tag, "raw") (the paper's PEHE estimator):
              sqrt(PEHE) = sqrt(mean (tau_hat - tau)^2)
              eps_ATE    = |mean tau_hat - mean tau| / max(|mean tau|, 0.1)   (means over the queries)
Truth per benchmark (the paper's targets): realized ITE delta for the intervals; for the point metrics
tau = mu_1 - mu_0 on IHDP, ACIC and the case studies, and the stored true_cate on ComplexMech.
Reported: mean (SE over datasets) per metric, |c - 0.95| on the mean coverage, n = datasets scored. The pooled
row [-0.5, 1] is the three bins together.
"""
import csv
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
for p in (os.path.join(REPO, "UWYK_Fig3_4"), os.path.join(REPO, "realcause_eval"), os.path.join(REPO, "benchmarks"),
          os.path.join(REPO, "revision", "experiments", "E2_in_prior")):
    sys.path.insert(0, p)
from cate_density_metrics import _query_pmfs, interval_95     # noqa: E402
from hist_interval import interval_hist                       # noqa: E402
from point_raw_em import cate_for_file                        # noqa: E402

A = 0.05
BINS = ("[-0.5, 0)", "[0, 0.5)", "[0.5, 1]")
MODELS = [("Do-PFN", "dopfn1d", None), ("Do-PFN 2D", "dopfn2d", None),
          ("UWYK No-Anc", "uwyk1d", "noanc"), ("UWYK Anc", "uwyk1d", "v3a"),
          ("UWYK 2D No-Anc", "graph2d", "noanc"), ("UWYK 2D Anc", "graph2d", "v3a"),
          ("CausalPFN-C", "cpfn1d", None), ("CausalPFN-C 2D", "cpfn2d", None)]
CASES = ("Observed_Confounder", "Backdoor_Criterion", "Frontdoor_Criterion", "Unobserved_Confounder")


def rid(f):
    return int(re.search(r"r(\d+)\.npz$", f).group(1))


def truth_cs(D, b, case, k):
    z = np.load(os.path.join(D, "cs", f"bin{b}", case, "N1000", f"{case}_{k}.npz"), allow_pickle=True)
    return z["delta"][:100].astype(float), z["cate"][:100].astype(float), z["cate"][:100].astype(float)


def truth_rc(D, b, ds, k):
    z = np.load(os.path.join(D, "rc", f"bin{b}", ds, "truths", f"r{k:03d}.npz"))
    return z["delta_shared"].astype(float), z["cate"].astype(float), z["delta_shared"].astype(float)


def truth_cm(D, b, k):
    z = np.load(os.path.join(D, "cm", f"bin{b}", "complexmech", "5node", "path_TY", "hide_0.0", f"r{k}.npz"),
                allow_pickle=True)
    tc = z["true_cate"].astype(float)
    return tc, tc, tc


def score_file(f, tag, delta, tau, dump_truth):
    """(c, length, IS, sqrt_pehe, eps_ate) for one dump, or None if it has no usable density."""
    out = [np.nan] * 5
    got = cate_for_file(f, tag, "raw")
    if got is not None:
        th = np.asarray(got[0], float).ravel()
        assert th.size == tau.size, f"{f}: {th.size} estimates vs {tau.size} queries"
        assert np.allclose(np.asarray(got[1], float).ravel(), dump_truth, rtol=1e-3, atol=1e-4), f"{f}: truth mismatch"
        out[3] = float(np.sqrt(np.mean((th - tau) ** 2)))
        ate = tau.mean()
        out[4] = float(abs(th.mean() - ate) / max(abs(ate), 0.1))
    got = _query_pmfs(f, tag)
    if got is not None:
        atoms, pmfs, _ = got
        assert pmfs.shape[0] == delta.size, f"{f}: {pmfs.shape[0]} densities vs {delta.size} queries"
        uni = np.allclose(np.diff(atoms), atoms[1] - atoms[0], rtol=1e-3)
        iv = np.array([interval_hist(atoms, p) if uni else interval_95(atoms, p) for p in pmfs])
        lo, hi = iv[:, 0], iv[:, 1]
        out[0] = float(np.mean((lo <= delta) & (delta <= hi)))
        out[1] = float(np.mean(hi - lo))
        out[2] = float(np.mean((hi - lo) + (2 / A) * np.maximum(0, lo - delta) + (2 / A) * np.maximum(0, delta - hi)))
    return out


def cells(D):
    """(benchmark, bin, dump-dir-relative, truth function)"""
    for b in range(3):
        for case in CASES:
            yield (case.replace("_", " "), b, os.path.join("cs", f"bin{b}", case), lambda k, b=b, c=case: truth_cs(D, b, c, k))
        for ds in ("IHDP", "ACIC"):
            yield (ds, b, os.path.join("rc", f"bin{b}", ds), lambda k, b=b, d=ds: truth_rc(D, b, d, k))
        yield ("ComplexMech n=5", b, os.path.join("cm", f"bin{b}"), lambda k, b=b: truth_cm(D, b, k))


def ms(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    return (float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan"), int(v.size)) \
        if v.size else (float("nan"), float("nan"), 0)


def main(D, out):
    per = {}                                                    # (bench, model) -> {bin: [rows]}
    for bench, b, rel, truth in cells(D):
        for label, mdir, tag in MODELS:
            files = sorted(glob.glob(os.path.join(D, "dumps", mdir, rel, "*r[0-9]*.npz")), key=rid)
            rows = []
            for f in files:
                delta, tau, dump_truth = truth(rid(f))
                rows.append(score_file(f, tag, delta, tau, dump_truth))
            per.setdefault((bench, label), {})[b] = rows
            print(f"{bench:24s} bin{b} {label:16s} {len(rows)} dumps", flush=True)
    recs = []
    for (bench, label), byb in per.items():
        for key in (0, 1, 2, "all"):
            rows = sum(byb.values(), []) if key == "all" else byb.get(key, [])
            if not rows:
                continue
            R = np.array(rows, float)
            c, cse, n = ms(R[:, 0]); L, Lse, _ = ms(R[:, 1]); s, sse, _ = ms(R[:, 2])
            p, pse, npt = ms(R[:, 3]); e, ese, _ = ms(R[:, 4])
            recs.append(dict(benchmark=bench, bin="[-0.5, 1]" if key == "all" else BINS[key], model=label,
                             n=n, n_point=npt, coverage=c, coverage_se=cse, abs_cov_err=abs(c - 0.95) if n else np.nan,
                             length=L, length_se=Lse, IS=s, IS_se=sse, sqrt_pehe=p, sqrt_pehe_se=pse,
                             eps_ate=e, eps_ate_se=ese))
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    f3 = lambda m, se: "—" if not np.isfinite(m) else (f"{m:.3f} ({se:.3f})" if np.isfinite(se) else f"{m:.3f}")
    md = ["# Controlled-ρ benchmarks", "",
          "Intervals scored against the realized ITE; √PEHE and ε_ATE against each benchmark's point target "
          "(μ1 − μ0 for IHDP, ACIC, case studies; stored true_cate for ComplexMech). Mean (SE over datasets).", ""]
    for bench in dict.fromkeys(r["benchmark"] for r in recs):
        md += [f"## {bench}", "", "| ρ bin | model | n | |ĉ − 0.95| | ĉ | length | IS₀.₀₅ | √PEHE | ε_ATE |",
               "|---|---|---|---|---|---|---|---|---|"]
        for binl in BINS + ("[-0.5, 1]",):
            for r in recs:
                if r["benchmark"] == bench and r["bin"] == binl:
                    md.append(f"| {binl} | {r['model']} | {r['n']} | {r['abs_cov_err']:.3f} | {f3(r['coverage'], r['coverage_se'])} | "
                              f"{f3(r['length'], r['length_se'])} | {f3(r['IS'], r['IS_se'])} | "
                              f"{f3(r['sqrt_pehe'], r['sqrt_pehe_se'])} | {f3(r['eps_ate'], r['eps_ate_se'])} |")
        md.append("")
    open(out + ".md", "w").write("\n".join(md) + "\n")
    print(f"wrote {out}.csv / .md ({len(recs)} rows)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
