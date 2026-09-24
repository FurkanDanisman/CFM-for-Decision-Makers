#!/usr/bin/env python
"""Frequentist coverage for ONE case study, ONE query: resample the observational
dataset 100 times with the SCM and the query held fixed.

WHY THIS IS DIFFERENT from the main case-study tables. There, each realization is a
NEW SCM with a new query, so the reported coverage is averaged over DGPs and over
queries. That answers "across the prior, how often does the interval cover?" It
does NOT answer "for this one estimand, does the interval cover at 95%?" -- the
textbook frequentist question, where the truth is a fixed number and only the
sample varies.

CONSTRUCTION. Mechanisms (`_weights`, `_activation`) and the treatment threshold
are drawn once and FROZEN; only the exogenous draws (`_root`, `_noise`) are
resampled. Row 0's exogenous values are frozen too, so that unit's covariates,
treatment, outcome, mu_0, mu_1 and therefore tau are IDENTICAL in every replicate
-- it is the same estimand each time. Rows 1..N-1 are freshly drawn, so each
replicate is a different observational dataset from the same DGP.

The treatment threshold is frozen deliberately: it is calibrated from the sample
(median of the latent treatment score), so letting it float would make the DGP
itself differ between replicates and the coverage statement meaningless.

Output is an ordinary case-study cell, so the existing dump + score pipeline reads
it unchanged. Score with SCM_N_QUERY=1 so exactly the fixed query is evaluated:
coverage over the replicates is then the frequentist coverage for that estimand.

    python case_study/gen_fixed_query.py --case Observed_Confounder \
        --seed 0 --n-context 1000 --n-draws 100 --out $SCRATCH/cs_fixedq
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from generation import CASE_STUDIES, build_dag, _SampledSCM   # noqa: E402

sys.path.insert(0, os.path.join(_HERE, "d_variation"))
try:
    from generation_d import build_dag_d                        # noqa: E402
except Exception:                                               # pragma: no cover
    build_dag_d = None


def resample_exogenous(scm, nodes, rng, frozen_row0):
    """Fresh exogenous draws for every row, then restore row 0 from `frozen_row0`."""
    for n in nodes:
        if n.kind == "root_normal":
            scm._root[n.name] = rng.normal(0.0, scm.exo_std, size=scm.N)
        elif n.kind == "root_bernoulli":
            scm._root[n.name] = (rng.random(scm.N) < 0.5).astype(np.float64)
        elif n.kind == "structural":
            scm._noise[n.name] = rng.normal(0.0, scm.noise_std, size=scm.N)
    for name, v in frozen_row0["root"].items():
        scm._root[name][frozen_row0["i"]] = v
    for name, v in frozen_row0["noise"].items():
        scm._noise[name][frozen_row0["i"]] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="Observed_Confounder", choices=list(CASE_STUDIES))
    ap.add_argument("--seed", type=int, default=0, help="picks the SCM (frozen)")
    ap.add_argument("--from-npz", default=None,
                    help="reproduce an EXISTING realization's SCM: reads its "
                         "seed / n_context / cate_shift so the estimand matches "
                         "that file's query exactly")
    ap.add_argument("--d", type=int, default=None,
                    help="covariate count. Required with --from-npz; on its own "
                         "it selects the d_variation DAG instead of the plain one, "
                         "which is how a (d, seed) pair can be searched for")
    ap.add_argument("--query", type=int, default=0,
                    help="first row of the source realization to freeze")
    ap.add_argument("--n-queries", type=int, default=1,
                    help="freeze this many CONSECUTIVE rows from --query, all as "
                         "fixed estimands. K queries x D datasets then costs D "
                         "realizations, not K*D: the rows not frozen still "
                         "resample, so one replicate serves every query at once.")
    ap.add_argument("--query-list", default=None,
                    help="comma-separated row indices to freeze, e.g. 3,17,204. "
                         "Overrides --query/--n-queries")
    ap.add_argument("--random-queries", type=int, default=None,
                    help="freeze this many rows chosen uniformly at random "
                         "(see --query-seed). Overrides --query/--n-queries")
    ap.add_argument("--query-seed", type=int, default=12345,
                    help="seed for --random-queries; independent of the SCM seed "
                         "so the same SCM can be re-probed at different queries")
    ap.add_argument("--n-context", type=int, default=1000)
    ap.add_argument("--n-draws", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draw-seed", type=int, default=777000)
    a = ap.parse_args()

    shift, src_tau = 0.0, None
    if a.from_npz:
        # Rebuild the SOURCE realization's SCM exactly -- same DAG, seed,
        # n_context and cate_shift. Anything else and the estimand is a different
        # number, so the draws would answer a different question.
        with np.load(a.from_npz, allow_pickle=True) as z:
            a.seed = int(np.asarray(z["seed"]).reshape(-1)[0])
            a.n_context = int(np.asarray(z["n_context"]).reshape(-1)[0])
            if "cate_shift" in z.files:
                shift = float(np.asarray(z["cate_shift"]).reshape(-1)[0])
            if "case_study" in z.files:
                a.case = str(np.asarray(z["case_study"]).reshape(-1)[0])
            src_tau = float(np.asarray(z["cate"]).ravel()[a.query])
        if a.d is None:
            sys.exit("--d is required with --from-npz (the DAG depends on it)")
        if build_dag_d is None:
            sys.exit("generation_d unavailable; cannot rebuild a d_variation DAG")
        nodes = build_dag_d(a.case, a.d)
        print(f"[source] {a.from_npz}\n  case={a.case} d={a.d} seed={a.seed} "
              f"n_context={a.n_context} cate_shift={shift}\n"
              f"  true tau at query {a.query} = {src_tau:.10f}")
    elif a.d is not None:
        # --d alone: the d_variation DAG without a source npz. Needed to SEARCH for
        # the (d, seed) that produced an existing set of replicates, since those
        # files record neither. build_dag_d(case, d) is a different DAG from
        # build_dag(case), so the two cannot be substituted for one another.
        if build_dag_d is None:
            sys.exit("generation_d unavailable; cannot build a d_variation DAG")
        nodes = build_dag_d(a.case, a.d)
    else:
        nodes = build_dag(a.case)

    # Rebuild at the SOURCE size; the chosen query is frozen in place and the
    # other rows are resampled, so row `a.query` keeps its identity.
    N = a.n_context
    scm = _SampledSCM(nodes, N=N, rng=np.random.default_rng(a.seed),
                      cate_shift=shift)

    # One pass to calibrate the treatment threshold, then FREEZE it: it is part of
    # the DGP, and letting it be re-derived per replicate would change the DGP.
    scm.forward()
    thr = scm._t_threshold

    # Which rows keep their identity across every replicate. Three ways to say it:
    # an explicit list, a random sample, or (the original) a consecutive run from
    # --query. Random matters because consecutive rows of one realization are not a
    # sample of its queries -- they are whatever the generator happened to emit
    # first, and their taus can be atypically clustered.
    if a.query_list:
        qidx = np.unique(np.asarray([int(v) for v in a.query_list.split(",")],
                                    dtype=np.int64))
        if qidx.min() < 0 or qidx.max() >= N:
            sys.exit(f"--query-list has an index outside [0, {N})")
    elif a.random_queries:
        K = int(a.random_queries)
        if K >= N:
            sys.exit(f"--random-queries {K} leaves no rows to resample (N={N})")
        qidx = np.sort(np.random.default_rng(a.query_seed).choice(
            N, size=K, replace=False))
    else:
        K = max(1, int(a.n_queries))
        qidx = np.arange(a.query, min(a.query + K, N))
    K = qidx.size
    if K >= N:
        sys.exit(f"{K} frozen rows leaves none to resample (n_context={N})")
    frozen = {"i": qidx,
              "root": {k: np.asarray(v)[qidx].copy() for k, v in scm._root.items()},
              "noise": {k: np.asarray(v)[qidx].copy() for k, v in scm._noise.items()}}

    cell = os.path.join(a.out, a.case, f"N{a.n_context}")
    os.makedirs(cell, exist_ok=True)
    truths, qx, qt, qy = [], None, None, None
    for r in range(a.n_draws):
        rng = np.random.default_rng(a.draw_seed + r)
        resample_exogenous(scm, nodes, rng, frozen)
        scm._t_threshold = thr               # re-freeze (forward() would re-derive)
        obs = scm.forward()
        scm._t_threshold = thr
        mu_0 = scm.forward(do_T=0.0, y_noiseless=True)[scm.y_name]
        scm._t_threshold = thr
        mu_1 = scm.forward(do_T=1.0, y_noiseless=True)[scm.y_name]

        feats = [n.name for n in nodes if n.observed]
        X = (np.stack([obs[f] for f in feats], axis=-1) if feats
             else np.zeros((N, 0)))
        T, Y = obs[scm.t_name], obs[scm.y_name]
        cate = mu_1 - mu_0
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
            print(f"  draw {r}: non-finite, skipped", file=sys.stderr)
            continue

        # Move the frozen rows to positions 0..K-1. The loader takes
        # X_test = X[:n_q], so SCM_N_QUERY=K evaluates exactly those; leaving them
        # at their original indices would score units that are resampled every
        # replicate and therefore have no fixed estimand at all.
        order = np.r_[qidx, np.delete(np.arange(N), qidx)]
        X, T, Y = X[order], T[order], Y[order]
        cate, mu_0, mu_1 = cate[order], mu_0[order], mu_1[order]
        truths.append(np.asarray(cate[:K], dtype=np.float64).copy())
        if qx is None:
            qx, qt, qy = X[0].copy(), float(T[0]), float(Y[0])

        np.savez_compressed(
            os.path.join(cell, f"{a.case}_{r}.npz"),
            X=X.astype(np.float32), T=T.astype(np.float32), Y=Y.astype(np.float32),
            cate=cate.astype(np.float32),
            mu_0=mu_0.astype(np.float32), mu_1=mu_1.astype(np.float32),
            feature_names=np.array(feats), exo_std=np.float32(scm.exo_std),
            noise_std=np.float32(scm.noise_std), cate_shift=np.float32(0.0))

    t = np.asarray(truths)   # (draws, K)
    # The whole design rests on the estimand being identical across replicates;
    # assert it rather than trust it.
    # Per query, the spread ACROSS replicates must be zero; the spread across
    # queries is the thing we want to be large.
    spread = float(np.abs(t - t[0]).max()) if t.size else float("nan")
    print(f"\ncase={a.case} seed={a.seed} draws={t.shape[0]} "
          f"queries={t.shape[1] if t.ndim > 1 else 1} n_context={a.n_context}")
    print(f"  true tau, query 0:           {t[0][0]:.10f}")
    if t.ndim > 1 and t.shape[1] > 1:
        print(f"  true tau across queries:     min {t[0].min():+.4f}  "
              f"max {t[0].max():+.4f}  sd {t[0].std():.4f}")
    print(f"  spread across replicates:    {spread:.3e}   (must be ~0)")
    if src_tau is not None and t.size and int(a.query) in set(int(v) for v in qidx):
        # The whole point of --from-npz is that the estimand is the SOURCE
        # realization's. If it is not, the 1000 draws describe a different
        # number and the exercise is void, so fail loudly.
        # --random-queries / --query-list need not include --query, and the frozen
        # rows are reordered to 0..K-1, so the source tau must be compared against
        # the position --query LANDED at -- not blindly against t[0][0], which
        # reports MISMATCH on a perfectly good cell.
        pos = int(np.where(np.asarray(qidx) == int(a.query))[0][0])
        dev = abs(float(t[0][pos]) - src_tau)
        print(f"  source tau:                  {src_tau:.10f}")
        print(f"  |ours - source|:             {dev:.3e}   "
              f"{'OK' if dev < 1e-6 else 'MISMATCH -- not the same estimand'}")
    ok = t.size > 1 and spread < 1e-6
    print(f"  FIXED ESTIMAND: {'OK' if ok else 'NO -- design violated'}")
    man = {"case": a.case, "seed": a.seed, "n_context": a.n_context,
           "query_index": [int(v) for v in qidx],
           "n_draws": int(t.shape[0]), "n_queries": int(t.shape[1]),
           "true_tau": [float(v) for v in t[0]] if t.size else None,
           "tau_spread": spread, "t_threshold": thr,
           "query_T": qt, "query_Y": qy,
           "query_X": [float(v) for v in (qx if qx is not None else [])]}
    with open(os.path.join(a.out, f"manifest_{a.case}_seed{a.seed}.json"), "w") as fh:
        json.dump(man, fh, indent=2)
    print(f"\nwrote {t.size} replicate(s) to {cell}")
    print(f"score with:\n"
          f"  export CASE_STUDY_DATA_ROOT={a.out}\n"
          f"  export CASE_STUDY_N={a.n_context}\n"
          f"  export SCM_N_QUERY=1        # evaluate ONLY the fixed query\n"
          f"  # then the usual dump + coverage_by_realization for case {a.case}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
