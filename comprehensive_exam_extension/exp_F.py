"""Experiment F: real-data properties (design figures: results/design/, plot_design_figures.py).

    python exp_F.py --dial treat --n 200 --n-real 1000 --out results/exp_F/treat

One dial per run, each level reusing the same seeds per realization (generation.py):
  treat  treated fraction 0.1, ..., 0.9 (in-prior 0.5)          -> treat_frac
  covk   confounder C recorded in k = 20, 10, 5, 3, 2 categories -> cov_k
  irrel  m = 1, 2, 5, 10, 20 irrelevant columns z               -> extra_cols
  outk   outcome Y recorded in k = 20, 10, 5, 3, 2 categories    -> out_k
The in-prior condition runs with the treat dial (skip with --no-baseline).

Protocol and metrics as in Experiments A-C: random 50/50 split; context = train half;
queries = test units at T=0 and T=1; MSE_CATE and Rel. ATE error on the test units, against
each level's own target (generation.py docstring).
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
DIALS = {"treat": ("treat_frac", (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9)),
         "covk": ("cov_k", (20, 10, 5, 3, 2)),
         "irrel": ("extra_cols", (1, 2, 5, 10, 20)),
         "outk": ("out_k", (20, 10, 5, 3, 2))}

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=200)
p.add_argument("--n-real", type=int, default=100, help="realizations r < n-real")
p.add_argument("--r-start", type=int, default=0, help="first realization (to extend an earlier run)")
p.add_argument("--dial", required=True, choices=list(DIALS))
p.add_argument("--no-baseline", action="store_true", help="skip the in-prior condition")
p.add_argument("--seed-base", type=int, default=0)
p.add_argument("--out", default="results/exp_F")
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
        yield "in_prior", "none", 0.0, {}
    name, levels = DIALS[a.dial]
    for level in levels:
        yield a.dial, name, level, {name: level}


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
    done = set(zip(prev.case, prev.dial, prev.level))

for dial, name, level, kw in conditions():
    for case_idx, case in enumerate(CASES):
        if (case, dial, level) in done:
            continue
        for r in range(a.r_start, a.n_real):
            seed, real = realization(case_idx, case, r, kw)
            rows.append(dict(case=case, dial=dial, param=name, level=level, r=r, seed=seed,
                             **score(real, seed)))
        pd.DataFrame(rows).to_csv(csv, index=False)
        d = pd.DataFrame(rows)
        d = d[(d.case == case) & (d.dial == dial) & (d.level == level)]
        print(f"{dial:8s} {name}={level:<5g} {case:22s} "
              f"median MSE_CATE={d.mse_cate.median():.4g}  median relATE={d.rel_ate.median():.4g}", flush=True)
