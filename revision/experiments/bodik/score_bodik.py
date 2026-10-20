"""Score Bodik's IHDP-rho and Twins datasets: one row per (benchmark, setting, model).
python score_bodik.py DATA_ROOT OUT_PREFIX          DATA_ROOT = $SCRATCH/bodik  (ihdp_a/, ihdp_b/, twins/, dumps/)

Per dataset, with the scorer of the controlled-rho benchmarks (score_rho_ctrl.score_file, i.e. the paper's):
  hist 95% interval of each query's ITE density against the REALIZED ITE (coverage, length, IS_0.05);
  sqrt(PEHE) and eps_ATE of the paper's point estimate against cate (mu1 - mu0 on IHDP; the realized ITE on
  Twins, where no other truth exists).
Twins only (binary outcome, ITE in {-1, 0, 1}): each query's ITE density is read as the three probabilities
  P(delta < -1/2), P(|delta| <= 1/2), P(delta > 1/2), and the realized ITE is scored with
    log score = -mean log P(realized class)   (probabilities floored at 1e-6)
    Brier     =  mean sum_k (P_k - 1{k = realized})^2
  For a single-outcome model the ITE density is its two marginals combined as independent, exactly as in
  every interval of the paper; for a 2D model it is the learned joint.
Reported: mean (SE over datasets). IHDP: 50 datasets per setting; Twins: 11 query blocks of one dataset.
"""
import csv
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rho_ctrl"))
import score_rho_ctrl as S                                     # noqa: E402  (sets sys.path for the scorer)
from cate_density_metrics import _query_pmfs                   # noqa: E402

MODELS = [m for m in S.MODELS if m[1] != "cpfn1d_j32"]          # the paper's main models
RHOS = ("1", "0.75", "0.5", "0.25", "0")


def rid(f):
    return int(re.search(r"r(\d+)\.npz$", f).group(1))


def truth(D, rel, k):
    z = np.load(os.path.join(D, rel, "truths", f"r{k:03d}.npz"))
    return z["delta_shared"].astype(float), z["cate"].astype(float), z["delta_shared"].astype(float)


def binary_scores(f, tag, delta):
    got = _query_pmfs(f, tag)
    if got is None:
        return np.nan, np.nan
    atoms, pmfs, _ = got
    assert pmfs.shape[0] == delta.size, f"{f}: {pmfs.shape[0]} densities vs {delta.size} queries"
    a = np.asarray(atoms, float)
    P = np.stack([pmfs[:, a < -0.5].sum(1), pmfs[:, np.abs(a) <= 0.5].sum(1), pmfs[:, a > 0.5].sum(1)], 1)
    P = P / P.sum(1, keepdims=True)
    k = np.rint(delta).astype(int) + 1
    onehot = np.eye(3)[k]
    return float(-np.mean(np.log(np.maximum(P[np.arange(k.size), k], 1e-6)))), float(np.mean(((P - onehot) ** 2).sum(1)))


def cells():
    for mode, label in (("ihdp_a", "IHDP-ρ (held-out queries)"), ("ihdp_b", "IHDP-ρ (in-sample)")):
        for d in (10, 1):
            for r in RHOS:
                yield label, f"d={d}, ρ={r}", f"{mode}/d{d}/rho{r}", f"{mode}/d{d}/rho{r}"
    for d in (71, 10, 1):
        yield "Twins", f"d={d}", f"twins/d{d}", f"twins/d{d}"


def main(D, out):
    recs = []
    for bench, setting, data_rel, dump_rel in cells():
        for label, mdir, tag in MODELS:
            files = sorted(glob.glob(os.path.join(D, "dumps", mdir, dump_rel, "*r[0-9]*.npz")), key=rid)
            if not files:
                continue
            truth_rel = data_rel if not (bench == "Twins" and setting == "d=71" and mdir in ("uwyk1d", "graph2d")) \
                else "twins/d71_u50"
            rows, bins = [], []
            for f in files:
                delta, tau, dump_truth = truth(D, truth_rel, rid(f))
                rows.append(S.score_file(f, tag, delta, tau, dump_truth, "2D" in label))
                if bench == "Twins":
                    bins.append(binary_scores(f, tag, delta))
            R = np.array(rows, float)
            rec = dict(benchmark=bench, setting=setting, model=label)
            for j, name in enumerate(("coverage", "length", "IS", "sqrt_pehe", "eps_ate")):
                m, se, n = S.ms(R[:, j]); rec[name] = m; rec[name + "_se"] = se
                if j == 0:
                    rec["n"] = n; rec["abs_cov_err"] = abs(m - 0.95) if n else np.nan
            if bins:
                B = np.array(bins, float)
                for j, name in enumerate(("log_score", "brier")):
                    m, se, _ = S.ms(B[:, j]); rec[name] = m; rec[name + "_se"] = se
            recs.append(rec)
            print(f"{bench:28s} {setting:14s} {label:16s} {len(files)} dumps", flush=True)
    keys = list(dict.fromkeys(k for r in recs for k in r))
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(recs)
    f3 = lambda r, k: "—" if not np.isfinite(r.get(k, np.nan)) else (
        f"{r[k]:.3f} ({r[k + '_se']:.3f})" if np.isfinite(r.get(k + "_se", np.nan)) else f"{r[k]:.3f}")
    md = ["# Bodik's IHDP-ρ and Twins (CLeaR 2026), the paper's models", "",
          "Intervals scored against the realized ITE; √PEHE and ε_ATE against μ1 − μ0 (IHDP) or the realized ITE "
          "(Twins). Mean (SE over datasets).", ""]
    for bench in dict.fromkeys(r["benchmark"] for r in recs):
        tw = bench == "Twins"
        md += [f"## {bench}", "",
               "| setting | model | n | |ĉ − 0.95| | ĉ | length | IS₀.₀₅ | √PEHE | ε_ATE |" + (" log score | Brier |" if tw else ""),
               "|---|---|---|---|---|---|---|---|---|" + ("---|---|" if tw else "")]
        for r in recs:
            if r["benchmark"] == bench:
                md.append(f"| {r['setting']} | {r['model']} | {r['n']} | {r['abs_cov_err']:.3f} | {f3(r, 'coverage')} | "
                          f"{f3(r, 'length')} | {f3(r, 'IS')} | {f3(r, 'sqrt_pehe')} | {f3(r, 'eps_ate')} |"
                          + (f" {f3(r, 'log_score')} | {f3(r, 'brier')} |" if tw else ""))
        md.append("")
    open(out + ".md", "w").write("\n".join(md) + "\n")
    print(f"wrote {out}.csv / .md ({len(recs)} rows)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
