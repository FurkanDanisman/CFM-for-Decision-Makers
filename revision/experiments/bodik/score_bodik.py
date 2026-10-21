"""Score Bodik's IHDP-rho and Twins datasets: one row per (benchmark, setting, model).
python score_bodik.py DATA_ROOT OUT_PREFIX          DATA_ROOT = $SCRATCH/bodik  (ihdp_a/, ihdp_b/, twins/, dumps/)

Per dataset, with the scorer of the controlled-rho benchmarks (score_rho_ctrl.score_file, i.e. the paper's):
  hist 95% interval of each query's ITE density against the REALIZED ITE (coverage, length, IS_0.05);
  sqrt(PEHE) and eps_ATE of the paper's point estimate against cate (mu1 - mu0 on IHDP; the realized ITE on
  Twins, where no other truth exists).
Twins only (binary outcome, one-year mortality): each model's predicted 2 x 2 table of the PAIR,
  P(Y0 = a, Y1 = b | x), is scored against both twins' observed outcomes:
    log score = -mean log P(observed cell)   (floored at 1e-6)
    Brier     =  mean sum_cells (P_cell - 1{cell observed})^2
  Each arm's grid is put in outcome units (y_raw = y_scaled * y_scale + y_shift, per arm where the dump has
  per-arm maps) and P(Y >= 1/2) is its mass above 1/2 (the bin containing 1/2 split linearly). A 2D model's
  table is its learned joint; a 1D model's is the product of its two marginals (independence, as everywhere in
  the paper) or, in the coupling runs, its marginals under that coupling (comonotone: P(1,1) = min(q0, q1);
  Gaussian copula: P(1,1) = Phi_2(Phi^-1(q0), Phi^-1(q1); rho)). The ITE alone (-1, 0, +1) is not scored: its
  class 0 merges "both survive" with "both die", so it rewards a model that predicts no deaths.
Reported: mean (SE over datasets). IHDP: 50 datasets per setting; Twins: 11 query blocks of one dataset.

Coupling runs (as for the paper's coupling tables): with COUPLING = comonotonic | gauss:<rho> set, only the 1D
models are scored, their two marginals combined under that coupling instead of independently; the point metrics
are skipped (they do not depend on the coupling).
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
import cate_density_metrics as _cdm                            # noqa: E402

MODELS = [m for m in S.MODELS if m[1] != "cpfn1d_j32"] + [     # the paper's main models, then its appendix models
    ("Do-PFN (matched resolution)", "dopfn1d_mr", None), ("Do-PFN (matched label)", "dopfn1d_ml", None),
    ("UWYK No-Anc (matched resolution)", "uwyk1d_mr", "noanc"), ("UWYK Anc (matched resolution)", "uwyk1d_mr", "v3a"),
    ("UWYK No-Anc (matched label)", "uwyk1d_ml", "noanc"), ("UWYK Anc (matched label)", "uwyk1d_ml", "v3a"),
    ("CausalPFN-C (matched resolution)", "cpfn1d_mr", None), ("CausalPFN-C (matched label)", "cpfn1d_ml", None),
    ("CausalPFN (published two-stage)", "cpfn_v0", None),
    ("UWYK-B No-Anc", "uwyk_bin", "noanc"), ("UWYK-B Anc", "uwyk_bin", "v3a")]
ONLY_DIR = os.environ.get("ONLY_DIR", "")
TWINS_ONLY = os.environ.get("TWINS_ONLY", "") == "1"           # rescore Twins only (minutes, not hours)                     # score one model directory only (parallel parts)
CELLS4 = ("00", "01", "10", "11")                              # (Y0, Y1): both survive, heavier dies, lighter dies, both die
RHOS = ("1", "0.75", "0.5", "0.25", "0")


def rid(f):
    return int(re.search(r"r(\d+)\.npz$", f).group(1))


def truth(D, rel, k):
    z = np.load(os.path.join(D, rel, "truths", f"r{k:03d}.npz"))
    return z["delta_shared"].astype(float), z["cate"].astype(float), z["delta_shared"].astype(float)


def _edges(z, J, arm):
    e = (np.asarray(z["edges"], dtype=np.float64).reshape(-1) if "edges" in z.files
         else np.linspace(-1.0, 1.0, J + 1))
    if f"arm{arm}_shift" in z.files:
        return e * float(np.asarray(z[f"arm{arm}_scale"]).reshape(-1)[0]) + float(np.asarray(z[f"arm{arm}_shift"]).reshape(-1)[0])
    sc = float(z["y_scale"]) if "y_scale" in z.files else 1.0
    sh = float(z["y_shift"]) if "y_shift" in z.files else 0.0
    return e * sc + sh


def _above(e, thr=0.5):
    lo, hi = e[:-1], e[1:]
    return np.clip((hi - thr) / (hi - lo), 0.0, 1.0)              # share of each bin at or above thr


def _q_above(z, p, arm, tag):
    """P(Y_arm >= 1/2) per query from one arm's dumped pmf (n_q, J or K+2)."""
    e = np.asarray(z["edges"], dtype=np.float64).reshape(-1) if "edges" in z.files else None
    J = p.shape[-1]
    p = p / p.sum(1, keepdims=True)
    if e is None or e.size == J + 1:                          # one probability per bin
        return (p * _above(_edges(z, J, arm))).sum(1)
    if e.size == J - 1:                                       # UWYK bar distribution: [left tail, K bars, right tail]
        from scipy.special import expit
        from scipy.stats import norm
        sc = float(z["y_scale"]) if "y_scale" in z.files else 1.0
        sh = float(z["y_shift"]) if "y_shift" in z.files else 0.0
        thr = (0.5 - sh) / sc                                 # the threshold in the dump's scaled units
        sfx = f"_{tag}" if tag and f"sL_raw_{arm}_{tag}" in z.files else ""
        soft = lambda x: np.logaddexp(0.0, x)                 # softplus, as in density_common.UWYK1D
        fl = float(z["scale_floor"])
        sL = float(z["base_s_left"]) * (soft(np.asarray(z[f"sL_raw_{arm}{sfx}"], np.float64)) + fl)
        sR = float(z["base_s_right"]) * (soft(np.asarray(z[f"sR_raw_{arm}{sfx}"], np.float64)) + fl)
        lo, hi = e[:-1], e[1:]
        bars = (p[:, 1:-1] * np.clip((hi - thr) / (hi - lo), 0.0, 1.0)).sum(1)
        # half-normal tails: left mass lies below e[0], decaying with scale sL; right mass above e[-1] with sR
        left = np.where(e[0] > thr, 2 * norm.cdf((e[0] - thr) / sL) - 1, 0.0)
        right = np.where(e[-1] >= thr, 1.0, 2 * norm.sf((thr - e[-1]) / sR))
        return bars + p[:, 0] * left + p[:, -1] * right
    raise SystemExit(f"unrecognised density layout: {J} probabilities, {e.size} edges")


def table_2x2(f, tag):
    """(n_q, 4) predicted P(Y0, Y1) in the order 00, 01, 10, 11, or None."""
    with np.load(f, allow_pickle=True) as z:
        def g(base):
            k = f"{base}_{tag}" if tag else base
            return np.asarray(z[k], dtype=np.float64) if k in z.files else None
        joint, p0, p1 = g("p_joint_scaled"), g("p_y0_scaled"), g("p_y1_scaled")
        if joint is not None and joint.ndim == 3 and not S.COUPLING:
            J = joint.shape[-1]
            a0, a1 = _above(_edges(z, J, 0)), _above(_edges(z, J, 1))
            joint = joint / joint.sum((1, 2), keepdims=True)
            p11 = np.einsum("qij,i,j->q", joint, a0, a1)
            p10 = np.einsum("qij,i,j->q", joint, a0, 1 - a1)
            p01 = np.einsum("qij,i,j->q", joint, 1 - a0, a1)
            return np.stack([1 - p11 - p10 - p01, p01, p10, p11], 1)
        if p0 is None or p1 is None:
            return None
        q0, q1 = _q_above(z, p0, 0, tag), _q_above(z, p1, 1, tag)
    q0, q1 = np.clip(q0, 1e-9, 1 - 1e-9), np.clip(q1, 1e-9, 1 - 1e-9)
    if not S.COUPLING:
        p11 = q0 * q1
    elif S.COUPLING == "comonotonic":
        p11 = np.minimum(q0, q1)
    elif S.COUPLING.startswith("gauss:"):
        from scipy.stats import multivariate_normal, norm
        r = float(S.COUPLING.split(":", 1)[1])
        mvn = multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, r], [r, 1.0]])
        p11 = mvn.cdf(np.stack([norm.ppf(q0), norm.ppf(q1)], 1))
    else:
        raise SystemExit(f"coupling {S.COUPLING} not supported for the 2 x 2 table")
    p11 = np.clip(p11, np.maximum(0.0, q0 + q1 - 1), np.minimum(q0, q1))   # Frechet bounds
    return np.stack([1 - q0 - q1 + p11, q1 - p11, q0 - p11, p11], 1)


PAIR_KEYS = ("log_score", "brier", "log_factual", "log_cf", "brier_cf") + tuple(f"P{c}" for c in CELLS4)


def pair_scores(f, tag, y0, y1, t):
    """One dump's scores of the pair (PAIR_KEYS). The 2 x 2 log score splits exactly into the factual twin's part
    and the counterfactual twin's given the factual one:
        -log P(y0, y1 | x) = -log P(y_f | x) - log P(y_cf | y_f, x)
    The second is the retrospective counterfactual prediction of the observed pair (Bodik's task on Twins); under
    independence P(y_cf | y_f) = P(y_cf), so only a joint can use the factual outcome."""
    P = table_2x2(f, tag)
    if P is None:
        return (np.nan,) * len(PAIR_KEYS)
    assert P.shape[0] == y0.size, f"{f}: {P.shape[0]} tables vs {y0.size} queries"
    P = np.clip(P, 1e-12, 1.0); P = P / P.sum(1, keepdims=True)
    a, b, t = np.rint(y0).astype(int), np.rint(y1).astype(int), np.rint(t).astype(int)
    k, q = a * 2 + b, np.arange(a.size)
    M = P.reshape(-1, 2, 2)                                      # M[q, y0, y1]
    pc = M[q, a, b]
    pf = np.where(t == 1, M[q, :, b].sum(1), M[q, a, :].sum(1))   # P(factual twin's outcome)
    p_cf1 = np.where(t == 1, M[q, 1, b] / M[q, :, b].sum(1), M[q, a, 1] / M[q, a, :].sum(1))   # P(y_cf = 1 | y_f)
    ycf = np.where(t == 1, a, b)
    L = lambda p: -np.log(np.maximum(p, 1e-6))
    return (float(np.mean(L(pc))), float(np.mean(((P - np.eye(4)[k]) ** 2).sum(1))),
            float(np.mean(L(pf))), float(np.mean(L(pc / pf))), float(np.mean((p_cf1 - ycf) ** 2))) \
        + tuple(float(v) for v in P.mean(0))


def cells():
    for mode, label in (("ihdp_a", "IHDP-ρ (held-out queries)"), ("ihdp_b", "IHDP-ρ (in-sample)")):
        for d in (10, 1):
            for r in RHOS:
                yield label, f"d={d}, ρ={r}", f"{mode}/d{d}/rho{r}", f"{mode}/d{d}/rho{r}"
    for d in (10, 1):
        yield "Twins", f"d={d}", f"twins/d{d}", f"twins/d{d}"


def observed_2x2(D, rel):
    k = []
    for f in sorted(glob.glob(os.path.join(D, rel, "truths", "r*.npz"))):
        z = np.load(f); k.append(np.rint(z["y0"]).astype(int) * 2 + np.rint(z["y1"]).astype(int))
    k = np.concatenate(k)
    return np.bincount(k, minlength=4) / k.size


def main(D, out):
    recs = []
    for bench, setting, data_rel, dump_rel in cells():
        if TWINS_ONLY and bench != "Twins":
            continue
        if bench == "Twins" and ONLY_DIR in ("", "observed"):
            o = observed_2x2(D, data_rel)                       # the data's own 2 x 2 table, for the md
            recs.append(dict(benchmark=bench, setting=setting, model="observed",
                             **{f"P{c}": float(v) for c, v in zip(CELLS4, o)}))
        for label, mdir, tag in MODELS:
            if (S.COUPLING and "2D" in label) or (ONLY_DIR and mdir != ONLY_DIR):     # ONLY_DIR=observed: no model
                continue
            files = sorted(glob.glob(os.path.join(D, "dumps", mdir, dump_rel, "*r[0-9]*.npz")), key=rid)
            if not files:
                continue
            rows, bins = [], []
            for f in files:
                delta, tau, dump_truth = truth(D, data_rel, rid(f))
                rows.append(S.score_file(f, tag, delta, tau, dump_truth, "2D" in label))
                if bench == "Twins":
                    tz = np.load(os.path.join(D, data_rel, "truths", f"r{rid(f):03d}.npz"))
                    bins.append(pair_scores(f, tag, tz["y0"].astype(float), tz["y1"].astype(float), tz["t_test"].astype(float)))
            R = np.array(rows, float)
            rec = dict(benchmark=bench, setting=setting, model=label)
            for j, name in enumerate(("coverage", "length", "IS", "sqrt_pehe", "eps_ate")):
                m, se, n = S.ms(R[:, j]); rec[name] = m; rec[name + "_se"] = se
                if j == 0:
                    rec["n"] = n; rec["abs_cov_err"] = abs(m - 0.95) if n else np.nan
            if bins:
                B = np.array(bins, float)
                for j, name in enumerate(PAIR_KEYS):
                    m, se, _ = S.ms(B[:, j]); rec[name] = m; rec[name + "_se"] = se
            recs.append(rec)
            print(f"{bench:28s} {setting:14s} {label:34s} {len(files)} dumps", flush=True)
    write(out, recs)


def write(out, recs):
    keys = list(dict.fromkeys(k for r in recs for k in r))
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(recs)
    num = lambda r, k: float(r.get(k, np.nan)) if r.get(k, "") not in ("", None) else np.nan
    f3 = lambda r, k: "—" if not np.isfinite(num(r, k)) else (
        f"{num(r, k):.3f} ({num(r, k + '_se'):.3f})" if np.isfinite(num(r, k + "_se")) else f"{num(r, k):.3f}")
    order = {m[0]: i for i, m in enumerate(MODELS)}
    md = ["# Bodik's IHDP-ρ and Twins (CLeaR 2026), the paper's models"
          + (f" — 1D marginals combined under {S.COUPLING}" if S.COUPLING else ""), "",
          "IHDP-ρ: intervals scored against the realized ITE; √PEHE and ε_ATE against μ1 − μ0. Twins: the predicted "
          "2×2 table of the pair P(Y0, Y1 | x) against both twins' observed outcomes (log score, Brier) and its mean "
          "cells (00 both survive, 01 heavier twin dies, 10 lighter twin dies, 11 both die). Mean (SE over datasets).", ""]
    for bench in dict.fromkeys(r["benchmark"] for r in recs):
        rs = sorted([r for r in recs if r["benchmark"] == bench],
                    key=lambda r: (r["setting"], -1 if r["model"] == "observed" else order.get(r["model"], 99)))
        if bench == "Twins":
            md += [f"## {bench}", "",
                   "Log score of the pair = factual twin + counterfactual twin given the factual one (exact split).", "",
                   "| setting | model | 2×2 log score | 2×2 Brier | log, factual | log, cf given factual | Brier, cf given factual "
                   "| P(00) | P(01) | P(10) | P(11) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in rs:
                md.append(f"| {r['setting']} | {r['model']} | {f3(r, 'log_score')} | {f3(r, 'brier')} | {f3(r, 'log_factual')} | "
                          f"{f3(r, 'log_cf')} | {f3(r, 'brier_cf')} | " + " | ".join(f"{num(r, 'P' + c):.3f}" for c in CELLS4) + " |")
        else:
            md += [f"## {bench}", "", "| setting | model | n | |ĉ − 0.95| | ĉ | length | IS₀.₀₅ | √PEHE | ε_ATE |",
                   "|---|---|---|---|---|---|---|---|---|"]
            for r in rs:
                md.append(f"| {r['setting']} | {r['model']} | {r['n']} | {num(r, 'abs_cov_err'):.3f} | {f3(r, 'coverage')} | "
                          f"{f3(r, 'length')} | {f3(r, 'IS')} | {f3(r, 'sqrt_pehe')} | {f3(r, 'eps_ate')} |")
        md.append("")
    open(out + ".md", "w").write("\n".join(md) + "\n")
    print(f"wrote {out}.csv / .md ({len(recs)} rows)")


def merge(out, parts):
    recs = []
    for p in parts:
        with open(p) as fh:
            recs += list(csv.DictReader(fh))
    write(out, recs)


if __name__ == "__main__":
    if sys.argv[1] == "merge":                                 # python score_bodik.py merge OUT_PREFIX part.csv ...
        merge(sys.argv[2], sys.argv[3:])
    else:
        main(sys.argv[1], sys.argv[2])
