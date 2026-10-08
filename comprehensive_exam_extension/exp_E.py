"""Experiment E: compared with methods without a prior (extension plan, Section 4).

    python exp_E.py --list-tasks
    python exp_E.py --task-id 0 --out results/exp_E

Regenerates the exact data sets of Experiments B and C (same seeds, same 50/50 split) and
scores two baselines with the same protocol and metrics (context = train half; CATE on the
test units; MSE_CATE and Rel. ATE error):
  S-learner (TabPFN)   one TabPFN v2 regressor on [T, X], tau = f(1, x) - f(0, x)   (paper D.3.2)
  Causal forest DML    econml CausalForestDML, discrete treatment, hyperparameters tuned
                       with its built-in search (.tune), default nuisance models  (paper D.3.2)
One array task = one (experiment, condition, case study); writes <out>/<exp>/<task>.csv.
Do-PFN's errors on the same data sets are in results/exp_B and results/exp_C (join on
case, dial, level, target, r); exp_E_analysis.py computes Delta and the breakdown level.

The condition lists below MUST match exp_B.py and exp_C.py (seeds and data depend on them).
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generation import _cell_seed, generate_realization  # noqa: E402

CASES = ["Observed_Confounder", "Unobserved_Confounder"]
B_DIALS = {
    "B1_tails": [("t", dict(noise_dist="t", noise_param=v)) for v in (30, 10, 5, 3)],
    "B2_skew": [("k", dict(noise_dist="gamma", noise_param=v)) for v in (16, 4, 1)],
    "B3_scale": [("c", dict(noise_scale=v)) for v in (0.5, 2, 4, 8)],
    "B4_hetero": [("gamma", dict(hetero_gamma=v)) for v in (0.5, 1, 2)],
}
C_G, C_LAMBDAS, C_TARGETS = ("sin", "sign", "cubic"), (0.1, 0.2, 0.3, 0.5, 0.75, 1.0), \
    ("outcome", "treatment", "both")


def b_conditions():
    yield "in_prior", "none", 0.0, "all", {}
    for dial, levels in B_DIALS.items():
        for name, kw in levels:
            level = kw.get("noise_param", kw.get("noise_scale", kw.get("hetero_gamma")))
            for target in ("all", "outcome"):
                yield dial, name, float(level), target, dict(kw, noise_target=target)


def c_conditions():
    yield "in_prior", "none", 0.0, "none", {}
    for g in C_G:
        for lam in C_LAMBDAS:
            for target in C_TARGETS:
                yield g, "lambda", lam, target, dict(mix_g=g, mix_lambda=lam, mix_target=target)


TASKS = [("B", c, ci) for c in b_conditions() for ci in range(len(CASES))] + \
        [("C", c, ci) for c in c_conditions() if c[0] != "in_prior" for ci in range(len(CASES))]

p = argparse.ArgumentParser()
p.add_argument("--task-id", type=int, default=None)
p.add_argument("--list-tasks", action="store_true")
p.add_argument("--n", type=int, default=200)
p.add_argument("--n-real", type=int, default=100)
p.add_argument("--seed-base", type=int, default=0)
p.add_argument("--out", default="results/exp_E")
a = p.parse_args()
if a.list_tasks:
    print(len(TASKS))
    sys.exit(0)
if a.task_id is None:
    a.task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

exp, (dial, name, level, target, kw), case_idx = TASKS[a.task_id]
case = CASES[case_idx]
path = os.path.join(os.path.abspath(a.out), exp, f"{case}__{dial}__{level:g}__{target}.csv")
if os.path.exists(path):
    print("exists:", path)
    sys.exit(0)

from econml.dml import CausalForestDML  # noqa: E402
from tabpfn import TabPFNRegressor  # noqa: E402
from tabpfn.constants import ModelVersion  # noqa: E402

warnings.filterwarnings("ignore")


def realization(r):
    for attempt in range(50):
        seed = _cell_seed(a.seed_base, case_idx, a.n, r, attempt)
        try:
            return seed, generate_realization(case, a.n, seed, **kw)
        except ValueError:
            continue
    raise RuntimeError(f"{case} r{r}: 50 resamples all non-finite")


def metrics(tau_hat, tau):
    ate = tau.mean()
    return dict(mse_cate=float(np.mean((tau_hat - tau) ** 2)),
                rel_ate=float(abs(tau_hat.mean() - ate) / abs(ate)) if ate != 0 else np.nan)


def s_learner(Xtr, Ttr, Ytr, Xte, seed):
    reg = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=seed % 2**31)
    reg.fit(np.c_[Ttr, Xtr], Ytr)
    return reg.predict(np.c_[np.ones(len(Xte)), Xte]) - reg.predict(np.c_[np.zeros(len(Xte)), Xte])


def causal_forest(Xtr, Ttr, Ytr, Xte, seed):
    cf = CausalForestDML(discrete_treatment=True, random_state=seed % 2**31)
    cf.tune(Ytr, Ttr, X=Xtr)
    cf.fit(Ytr, Ttr, X=Xtr)
    return cf.effect(Xte)


rows = []
for r in range(a.n_real):
    seed, real = realization(r)
    X, T, Y, tau = real.X.astype(np.float64), real.T.astype(int), real.Y.astype(np.float64), \
        real.cate.astype(np.float64)
    perm = np.random.default_rng(seed).permutation(len(Y))
    tr, te = perm[: len(Y) // 2], perm[len(Y) // 2:]
    for method, fn in (("S-learner (TabPFN)", s_learner), ("Causal forest DML", causal_forest)):
        try:
            m = metrics(np.asarray(fn(X[tr], T[tr], Y[tr], X[te], seed), dtype=np.float64).ravel(), tau[te])
        except Exception as e:   # e.g. a train half with one treatment arm only
            print(f"r{r} {method} failed: {e!r}", flush=True)
            m = dict(mse_cate=np.nan, rel_ate=np.nan)
        rows.append(dict(exp=exp, case=case, dial=dial, param=name, level=level, target=target, r=r,
                         seed=seed, method=method, **m))
    print(f"{exp} {case} {dial} {name}={level:g} {target} r{r} done", flush=True)

os.makedirs(os.path.dirname(path), exist_ok=True)
pd.DataFrame(rows).to_csv(path, index=False)
print("wrote", path)
