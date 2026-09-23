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
        scm._root[name][0] = v
    for name, v in frozen_row0["noise"].items():
        scm._noise[name][0] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="Observed_Confounder", choices=list(CASE_STUDIES))
    ap.add_argument("--seed", type=int, default=0, help="picks the SCM (frozen)")
    ap.add_argument("--n-context", type=int, default=1000)
    ap.add_argument("--n-draws", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draw-seed", type=int, default=777000)
    a = ap.parse_args()

    nodes = build_dag(a.case)
    N = a.n_context + 1                      # row 0 is the fixed query
    scm = _SampledSCM(nodes, N=N, rng=np.random.default_rng(a.seed))

    # One pass to calibrate the treatment threshold, then FREEZE it: it is part of
    # the DGP, and letting it be re-derived per replicate would change the DGP.
    scm.forward()
    thr = scm._t_threshold

    frozen = {"root": {k: float(v[0]) for k, v in scm._root.items()},
              "noise": {k: float(v[0]) for k, v in scm._noise.items()}}

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

        truths.append(float(cate[0]))
        if qx is None:
            qx, qt, qy = X[0].copy(), float(T[0]), float(Y[0])

        np.savez_compressed(
            os.path.join(cell, f"{a.case}_{r}.npz"),
            X=X.astype(np.float32), T=T.astype(np.float32), Y=Y.astype(np.float32),
            cate=cate.astype(np.float32),
            mu_0=mu_0.astype(np.float32), mu_1=mu_1.astype(np.float32),
            feature_names=np.array(feats), exo_std=np.float32(scm.exo_std),
            noise_std=np.float32(scm.noise_std), cate_shift=np.float32(0.0))

    t = np.asarray(truths)
    # The whole design rests on the estimand being identical across replicates;
    # assert it rather than trust it.
    spread = float(t.max() - t.min()) if t.size else float("nan")
    print(f"\ncase={a.case} seed={a.seed} draws={t.size} n_context={a.n_context}")
    print(f"  true tau at the fixed query: {t[0]:.10f}")
    print(f"  spread across replicates:    {spread:.3e}   (must be ~0)")
    ok = t.size > 1 and spread < 1e-6
    print(f"  FIXED ESTIMAND: {'OK' if ok else 'NO -- design violated'}")
    man = {"case": a.case, "seed": a.seed, "n_context": a.n_context,
           "n_draws": int(t.size), "true_tau": float(t[0]) if t.size else None,
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
