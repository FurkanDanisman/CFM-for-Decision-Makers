"""Experiment C: change only the functions (extension plan, Section 4).

    python exp_C.py --n 200 --n-real 100 --out results/exp_C

Noise stays Gaussian at the in-prior variance. The targeted mechanism moves along
  f_lambda = (1 - lambda) f_train + lambda g,  lambda in {0.1, 0.2, 0.3, 0.5, 0.75, 1},
with g in {sin, sign, cubic} rescaled to the variance of f_train on the same inputs
(generation.py, _SampledSCM._f). Targets: outcome equation only, treatment equation
only, or both. Every level reuses the same seeds per realization.

Protocol and metrics as in Experiments A and B: random 50/50 split; context = train half;
queries = test units at T=0 and T=1; MSE_CATE and Rel. ATE error on the test units.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generation import _cell_seed, generate_realization  # noqa: E402

CASES = ["Observed_Confounder", "Unobserved_Confounder"]   # order fixes the seeds (as in data/)
G_FUNCS = ("sin", "sign", "cubic", "expsq", "sin3", "exp")
LAMBDAS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
TARGETS = ("outcome", "treatment", "both")

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=200)
p.add_argument("--n-real", type=int, default=100, help="realizations r < n-real")
p.add_argument("--r-start", type=int, default=0, help="first realization (to extend an earlier run)")
p.add_argument("--g-funcs", nargs="*", default=None, help="subset of g (to split work)")
p.add_argument("--no-baseline", action="store_true", help="skip the in-prior condition")
p.add_argument("--condition-index", type=int, default=None,
               help="run only this condition (0 = in-prior, 1..54 = g x lambda x target); "
                    "default SLURM_ARRAY_TASK_ID if --list-conditions is not given and it is set")
p.add_argument("--list-conditions", action="store_true", help="print the number of conditions and exit")
p.add_argument("--seed-base", type=int, default=0)
p.add_argument("--out", default="results/exp_C")
a = p.parse_args()
out = os.path.abspath(a.out)
os.makedirs(out, exist_ok=True)
csv = os.path.join(out, "per_dataset.csv")

os.chdir(os.path.join(HERE, "dopfn"))
sys.path.insert(0, os.path.join(HERE, "dopfn"))
from scripts.transformer_prediction_interface.base import DoPFNRegressor  # noqa: E402

reg = DoPFNRegressor()
reg.device = "cuda" if torch.cuda.is_available() else "cpu"


def conditions():
    if not a.no_baseline:
        yield "in_prior", "none", 0.0, "none", {}
    for g in G_FUNCS:
        if a.g_funcs and g not in a.g_funcs:
            continue
        for lam in LAMBDAS:
            for target in TARGETS:
                yield g, "lambda", lam, target, dict(mix_g=g, mix_lambda=lam, mix_target=target)


def realization(case_idx, case, r, kw):
    for attempt in range(50):
        seed = _cell_seed(a.seed_base, case_idx, a.n, r, attempt)
        try:
            return seed, generate_realization(case, a.n, seed, **kw)
        except ValueError:
            continue
    raise RuntimeError(f"{case} r{r}: 50 resamples all non-finite")


def score(real, seed):
    X, T, Y, tau = real.X, real.T, real.Y, real.cate.astype(np.float64)
    perm = np.random.default_rng(seed).permutation(len(Y))
    tr, te = perm[: len(Y) // 2], perm[len(Y) // 2:]
    reg.fit(torch.tensor(np.c_[T[tr], X[tr]].astype(np.float32)), torch.tensor(Y[tr].astype(np.float32)))
    m = [np.asarray(reg.predict_full(torch.tensor(np.c_[np.full(len(te), t), X[te]].astype(np.float32)))["mean"],
                    dtype=np.float64) for t in (0.0, 1.0)]
    tau_hat, tau_te = m[1] - m[0], tau[te]
    ate = tau_te.mean()
    return dict(mse_cate=float(np.mean((tau_hat - tau_te) ** 2)),
                rel_ate=float(abs(tau_hat.mean() - ate) / abs(ate)) if ate != 0 else np.nan,
                ate=float(ate), ate_hat=float(tau_hat.mean()))


done = set()
rows = []
if os.path.exists(csv):
    prev = pd.read_csv(csv)
    rows = prev.to_dict("records")
    done = set(zip(prev.case, prev.dial, prev.level, prev.target))

CONDS = list(conditions())
if a.list_conditions:
    print(len(CONDS))
    sys.exit(0)
if a.condition_index is not None:
    CONDS = [CONDS[a.condition_index]]
for dial, name, level, target, kw in CONDS:
    for case_idx, case in enumerate(CASES):
        if (case, dial, level, target) in done:
            continue
        for r in range(a.r_start, a.n_real):
            seed, real = realization(case_idx, case, r, kw)
            rows.append(dict(case=case, dial=dial, param=name, level=level, target=target, r=r, seed=seed,
                             **score(real, seed)))
        pd.DataFrame(rows).to_csv(csv, index=False)
        d = pd.DataFrame(rows)
        d = d[(d.case == case) & (d.dial == dial) & (d.level == level) & (d.target == target)]
        print(f"{dial:8s} {name}={level:<5g} {target:9s} {case:22s} "
              f"median MSE_CATE={d.mse_cate.median():.4g}  median relATE={d.rel_ate.median():.4g}", flush=True)
