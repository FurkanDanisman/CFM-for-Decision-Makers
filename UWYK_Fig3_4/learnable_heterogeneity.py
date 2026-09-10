"""How much of the true CATE is actually learnable from the covariates?

`cate_ate_distributions.py` reports the within-dataset sd of the true ITE, which
is the sqrt(PEHE) of a constant predictor. That is a *bar*, not a floor: a model
that uses X can go below it, down to sd(tau - E[tau|X]). The gap between the two
is the only heterogeneity a CATE method can be rewarded for finding.

This script estimates that gap directly. Within each dataset it splits the test
units in half, fits a random forest tau ~ X on one half, and scores it on the
other:

    pehe_constant  sqrt(PEHE) of predicting the train-half mean effect (uses no X)
    pehe_oracle    sqrt(PEHE) of the fitted forest        (uses X)
    r2_tau         out-of-sample R^2 of tau on X
    learnable      1 - (pehe_oracle / pehe_constant)^2, clipped at 0
                   = fraction of within-dataset CATE variance explained by X

The forest is a generous stand-in for "the best a CATE model could do given X",
not a competitor -- it sees the true tau as its target, which no real method does.
So `learnable` is an optimistic estimate of the ceiling. If it is ~0, no method
can beat a constant on that cell and PEHE there is an ATE benchmark in disguise.

Only `path_TY` is worth running: the other two regimes have tau identically 0.

Usage
-----
    export UWYK_SRC=/tmp/uwyk/src UWYK_ROOT=/tmp/uwyk
    python UWYK_Fig3_4/learnable_heterogeneity.py --nodes 2 5 20 35 50 \
        --n-realizations 30
"""
from __future__ import annotations

# See generate_pehe_benchmark for why OMP is pinned before torch is imported.
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _one(rec: dict, seed: int) -> dict | None:
    """Split-half R^2 of the true tau on X for a single dataset."""
    from sklearn.ensemble import RandomForestRegressor

    X = np.asarray(rec["X_test"], dtype=np.float64)
    tau = np.asarray(rec["true_cate"], dtype=np.float64).reshape(-1)
    n = tau.shape[0]
    if n < 20:
        return None
    # Drop all-zero padding columns; a 2-node SCM has no real covariates at all.
    keep = X.std(axis=0) > 0
    X = X[:, keep]
    if X.shape[1] == 0:
        return {"n_features": 0, "pehe_constant": float(tau.std()),
                "pehe_oracle": float(tau.std()), "r2_tau": 0.0, "learnable": 0.0}

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    a, b = idx[: n // 2], idx[n // 2:]
    if tau[a].std() == 0.0 and tau[b].std() == 0.0:
        # Constant effect: nothing to learn, and R^2 is undefined.
        return {"n_features": int(X.shape[1]), "pehe_constant": 0.0,
                "pehe_oracle": 0.0, "r2_tau": 0.0, "learnable": 0.0}

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                               random_state=seed, n_jobs=1)
    rf.fit(X[a], tau[a])
    pred = rf.predict(X[b])

    const = float(np.sqrt(np.mean((tau[a].mean() - tau[b]) ** 2)))
    oracle = float(np.sqrt(np.mean((pred - tau[b]) ** 2)))
    var_b = float(np.var(tau[b]))
    r2 = 1.0 - float(np.mean((pred - tau[b]) ** 2)) / var_b if var_b > 0 else 0.0
    learnable = max(0.0, 1.0 - (oracle / const) ** 2) if const > 0 else 0.0
    return {"n_features": int(X.shape[1]), "pehe_constant": const,
            "pehe_oracle": oracle, "r2_tau": r2, "learnable": learnable}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prior", nargs="+", default=["lingaus", "complexmech"],
                   choices=("lingaus", "complexmech"))
    p.add_argument("--nodes", type=int, nargs="+", default=[2, 5, 20, 35, 50])
    p.add_argument("--n-realizations", type=int, default=30)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--out-dir", default=os.path.join(_HERE, "distributions"))
    args = p.parse_args()

    from generate_pehe_benchmark import (
        _import_uwyk, config_path, generate_realization, load_config,
    )
    uwyk = _import_uwyk()

    rows = []
    for prior in args.prior:
        for n in args.nodes:
            cfg_file = config_path(prior, n, "path_TY", 0.0)
            if not os.path.exists(cfg_file):
                continue
            cfg = load_config(cfg_file)
            sampler = uwyk[0](cfg["scm_config"], seed=args.seed_base * 31 + 17)
            per = []
            for r in range(args.n_realizations):
                try:
                    rec = generate_realization(
                        cfg, sampler, uwyk, "path_TY",
                        seed=args.seed_base + 1_000_000 * r,
                        hide_fraction=0.0, test_feature_mask_fraction=0.0)
                except Exception as exc:  # noqa: BLE001
                    print(f"[fail] {prior} n={n} r={r}: {exc}", flush=True)
                    continue
                got = _one(rec, seed=r)
                if got:
                    per.append(got)
            if not per:
                continue
            agg = {"prior": prior, "n_nodes": n, "n_realizations": len(per)}
            for k in ("n_features", "pehe_constant", "pehe_oracle", "r2_tau",
                      "learnable"):
                agg[k] = float(np.mean([d[k] for d in per]))
            # Means over datasets are easily dragged by a few extreme
            # realizations (the effect distribution has a heavy tail plus a spike
            # at exactly zero), so report medians alongside and judge on those.
            for k in ("r2_tau", "learnable"):
                agg[k + "_median"] = float(np.median([d[k] for d in per]))
            agg["frac_learnable_gt_0.1"] = float(
                np.mean([d["learnable"] > 0.1 for d in per]))
            rows.append(agg)
            print(f"[cell] {prior:<12} n={n:<3} feats={agg['n_features']:.0f}  "
                  f"pehe_const={agg['pehe_constant']:.4g}  "
                  f"pehe_oracle={agg['pehe_oracle']:.4g}  "
                  f"R2 mean/med={agg['r2_tau']:.3f}/{agg['r2_tau_median']:.3f}  "
                  f"learnable mean/med={agg['learnable']:.3f}/"
                  f"{agg['learnable_median']:.3f}  "
                  f"frac>0.1={agg['frac_learnable_gt_0.1']:.2f}", flush=True)

    if not rows:
        raise SystemExit("no cells produced data")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "learnable_heterogeneity.json"), "w") as f:
        json.dump(rows, f, indent=2)

    cols = ["prior", "n_nodes", "n_features", "pehe_constant", "pehe_oracle",
            "r2_tau", "r2_tau_median", "learnable", "learnable_median",
            "frac_learnable_gt_0.1"]
    md = ["# Learnable CATE heterogeneity (regime path_TY)", "",
          "`pehe_constant` ignores X; `pehe_oracle` is a random forest fitted on the",
          "true tau, i.e. an optimistic ceiling for any CATE method. `learnable` is",
          "the fraction of within-dataset CATE variance X can explain. Near 0 means",
          "PEHE on that cell cannot separate a CATE method from an ATE estimator.", "",
          "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        md.append("| " + " | ".join(
            f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])
            for c in cols) + " |")
    path = os.path.join(args.out_dir, "learnable_heterogeneity.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[table] {path}", flush=True)


if __name__ == "__main__":
    main()
