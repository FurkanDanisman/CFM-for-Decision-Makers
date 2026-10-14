"""Score the rho sweep: one row per (model, rho).
python rho_score.py DATA_ROOT DUMP_ROOT OUT_PREFIX       -> OUT_PREFIX.csv, OUT_PREFIX.md

Reads the harness dumps DUMP_ROOT/<model dir>/rho<+r.rr>/*r###.npz (DENSITY_DUMP=1) and the truths
DATA_ROOT/rho<+r.rr>/truths/r###.npz written by gen_rho.py.

Interval: per query, exactly as E2_in_prior/score_shard.py -- the hist 95% interval of tau
(hist_interval.interval_hist) on the pmf UWYK_Fig3_4/cate_density_metrics._query_pmfs builds,
interval_95 when the atom grid is not uniform. Tag = the ancestry mode the paper's tables read
(METHODS in cate_density_metrics: noanc / v3a for the UWYK models, untagged for the rest).
Point estimate: realcause_eval/point_raw_em.cate_for_file(f, tag, "raw"), the paper's PEHE estimator.

Per dataset (Q = 100 queries):
  c       mean over queries of 1{lo <= delta <= hi}, delta = Y(1) - Y(0) realized
  length  mean of hi - lo (outcome units)
  IS      mean of (hi - lo) + (2/a)(lo - delta)+ + (2/a)(delta - hi)+, a = 0.05
          (cate_density_metrics.score_query)
  PEHE    sqrt(mean (tau_hat - (mu1 - mu0))^2)
Reported: mean over datasets, SE = sd / sqrt(n datasets); |c - 0.95| on the mean.
Coverage counts a non-finite bound as a miss (as e2_tables.py); length / IS then come out non-finite
and n_nonfinite says how many datasets carry such a bound. Datasets whose dump has no usable density
(the scorer's own gate, e.g. Do-PFN's density_scale_r2) are left out of the interval columns only
(n_density vs n_point).
"""
import csv
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
for p in (os.path.join(REPO, "UWYK_Fig3_4"), os.path.join(REPO, "realcause_eval"),
          os.path.join(REPO, "benchmarks"), os.path.join(HERE, "..", "E2_in_prior")):
    sys.path.insert(0, p)
from cate_density_metrics import _query_pmfs, interval_95     # noqa: E402
from hist_interval import interval_hist                       # noqa: E402
from point_raw_em import cate_for_file                        # noqa: E402

RHOS = (-0.5, 0.0, 0.5, 0.9, 1.0)
A = 0.05
#        row label            dump dir      tag
ROWS = [("Do-PFN 1D",          "dopfn1d",   None),
        ("Do-PFN 2D",          "dopfn2d",   None),
        ("UWYK 1D No-Anc",     "uwyk1d",    "noanc"),
        ("UWYK 1D Anc",        "uwyk1d",    "v3a"),
        ("UWYK 2D No-Anc",     "graph2d",   "noanc"),
        ("UWYK 2D Anc",        "graph2d",   "v3a"),
        ("CausalPFN-C 1D",     "cpfn1d",    None),
        ("CausalPFN-C 2D",     "cpfn2d",    None)]


def ms(v):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan"))


def score_cell(data_root, dump_dir, rho_tag, tag):
    files = sorted(glob.glob(os.path.join(dump_dir, "*r[0-9][0-9][0-9].npz")))
    cov, length, is05, pehe, nonfinite = [], [], [], [], 0
    for f in files:
        r = int(re.search(r"r(\d+)\.npz$", f).group(1))
        tr = np.load(os.path.join(data_root, rho_tag, "truths", f"r{r:03d}.npz"))
        delta, cate = tr["delta_shared"].astype(np.float64), tr["cate"].astype(np.float64)
        got = cate_for_file(f, tag, "raw")
        if got is not None:
            tau_hat = np.asarray(got[0], dtype=np.float64).ravel()
            assert tau_hat.size == cate.size, f"{f}: {tau_hat.size} point estimates, {cate.size} queries"
            assert np.allclose(got[1], delta, rtol=1e-4, atol=1e-5), f"{f}: dump/truth mismatch (point)"
            pehe.append(float(np.sqrt(np.mean((tau_hat - cate) ** 2))))
        got = _query_pmfs(f, tag)
        if got is None:                                            # no usable density (scorer's gate)
            continue
        atoms, pmfs, y_dump = got
        assert pmfs.shape[0] == delta.size, f"{f}: {pmfs.shape[0]} densities, {delta.size} queries"
        assert np.allclose(y_dump, delta, rtol=1e-4, atol=1e-5), f"{f}: dump/truth mismatch (density)"
        uniform = np.allclose(np.diff(atoms), atoms[1] - atoms[0], rtol=1e-3)
        iv = np.array([interval_hist(atoms, p) if uniform else interval_95(atoms, p) for p in pmfs])
        lo, hi = iv[:, 0], iv[:, 1]
        nonfinite += int(not (np.isfinite(lo).all() and np.isfinite(hi).all()))
        cov.append(np.mean((lo <= delta) & (delta <= hi)))
        length.append(np.mean(hi - lo))
        is05.append(np.mean((hi - lo) + (2 / A) * np.maximum(0.0, lo - delta)
                            + (2 / A) * np.maximum(0.0, delta - hi)))
    c, c_se = ms(cov)
    L, L_se = ms(length)
    s, s_se = ms(is05)
    p, p_se = ms(pehe)
    return dict(n_files=len(files), n_point=len(pehe), n_density=len(cov), n_nonfinite=nonfinite,
                coverage=c, coverage_se=c_se, abs_cov_err=abs(c - 0.95), length=L, length_se=L_se,
                is05=s, is05_se=s_se, sqrt_pehe=p, sqrt_pehe_se=p_se)


def main(data_root, dump_root, out_prefix):
    out = []
    for label, d, tag in ROWS:
        for rho in RHOS:
            rt = f"rho{rho:+.2f}"
            dd = os.path.join(dump_root, d, rt)
            if not os.path.isdir(dd):
                print(f"missing {dd}", flush=True)
                continue
            row = dict(model=label, dump=d, tag=tag or "", rho=rho, **score_cell(data_root, dd, rt, tag))
            out.append(row)
            print(f"{label:16s} rho={rho:+.2f}  n={row['n_density']:3d}/{row['n_files']:3d}  "
                  f"c={row['coverage']:.3f} ({row['coverage_se']:.3f})  len={row['length']:.3f}  "
                  f"IS={row['is05']:.3f}  sqrtPEHE={row['sqrt_pehe']:.3f}", flush=True)
    if not out:
        sys.exit("nothing scored")
    keys = list(out[0])
    with open(out_prefix + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    f3 = lambda m, s: f"{m:.3f} ({s:.3f})"
    md = ["# rho sweep: same checkpoints, controlled cross-world noise correlation", "",
          "Data (revision/experiments/rho_sweep/gen_rho.py): X ~ N(0, I_5), T|X ~ Bern(sigmoid(0.5 X1)), "
          "mu0 = x1 + 0.5 x2, mu1 = mu0 + 1 + 0.5 x3, (eps0, eps1) unit-variance normal with corr rho. "
          "Per rho 100 datasets, N = 1000 context units, Q = 100 query units.", "",
          "c-hat: coverage of the hist 95% interval of tau against the realized delta = Y(1) - Y(0) "
          "(mean within dataset; mean (SE) over datasets). Length and IS_0.05 in outcome units. "
          "sqrt(PEHE) of the raw point estimate against mu1 - mu0. n = datasets with a usable density / dumps.", "",
          "| model | rho | n | c-hat | abs(c-hat - 0.95) | length | IS_0.05 | sqrt(PEHE) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in out:
        nf = f" ({r['n_nonfinite']} non-finite)" if r["n_nonfinite"] else ""
        md.append(f"| {r['model']} | {r['rho']:+.1f} | {r['n_density']}/{r['n_files']}{nf} | "
                  f"{f3(r['coverage'], r['coverage_se'])} | {r['abs_cov_err']:.3f} | "
                  f"{f3(r['length'], r['length_se'])} | {f3(r['is05'], r['is05_se'])} | "
                  f"{f3(r['sqrt_pehe'], r['sqrt_pehe_se'])} |")
    open(out_prefix + ".md", "w").write("\n".join(md) + "\n")
    print(f"wrote {out_prefix}.csv / .md ({len(out)} rows)")


if __name__ == "__main__":
    main(*sys.argv[1:4])
