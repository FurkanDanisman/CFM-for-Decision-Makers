"""Experiment B: change only the noise (extension plan, Section 4).

    python exp_B.py --n 200 --n-real 100 --out results/exp_B

Dials (in-prior level = Gaussian, scale 1, homoscedastic), each run with the noise
changed on all structural nodes ("all") and on the outcome only ("outcome"):
  B1 tails  Student-t, nu in {30, 10, 5, 3}           (rescaled to the in-prior variance)
  B2 skew   centred Gamma, shape k in {16, 4, 1}      (rescaled to the in-prior variance)
  B3 scale  noise s.d. x c, c in {0.5, 2, 4, 8}
  B4 hetero sigma(pa) ∝ sqrt(1 + gamma s(pa)^2), gamma in {0.5, 1, 2} (mean variance fixed)
Every level reuses the same seeds per realization (same SCM, covariates and base noise).

Protocol (as in Experiment A): random 50/50 split; context = train half (T, X, Y);
queries = test units at T=0 and T=1. Per dataset, on the test units:
  MSE_CATE = mean (tau_hat - tau)^2,  Rel. ATE error = |mean tau_hat - mean tau| / |mean tau|.
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
DIALS = {
    "B1_tails": [("t", dict(noise_dist="t", noise_param=v)) for v in (30, 10, 5, 3)],
    "B2_skew": [("k", dict(noise_dist="gamma", noise_param=v)) for v in (16, 4, 1)],
    "B3_scale": [("c", dict(noise_scale=v)) for v in (0.5, 2, 4, 8)],
    "B4_hetero": [("gamma", dict(hetero_gamma=v)) for v in (0.5, 1, 2)],
}

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=200)
p.add_argument("--n-real", type=int, default=100)
p.add_argument("--seed-base", type=int, default=0)
p.add_argument("--out", default="results/exp_B")
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
    yield "in_prior", "none", 0.0, "all", {}
    for dial, levels in DIALS.items():
        for name, kw in levels:
            level = kw.get("noise_param", kw.get("noise_scale", kw.get("hetero_gamma")))
            for target in ("all", "outcome"):
                yield dial, name, float(level), target, dict(kw, noise_target=target)


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

for dial, name, level, target, kw in conditions():
    for case_idx, case in enumerate(CASES):
        if (case, dial, level, target) in done:
            continue
        for r in range(a.n_real):
            seed, real = realization(case_idx, case, r, kw)
            rows.append(dict(case=case, dial=dial, param=name, level=level, target=target, r=r, seed=seed,
                             noise_std=real.noise_std, **score(real, seed)))
        pd.DataFrame(rows).to_csv(csv, index=False)
        d = pd.DataFrame(rows)
        d = d[(d.case == case) & (d.dial == dial) & (d.level == level) & (d.target == target)]
        print(f"{dial:10s} {name}={level:<5g} {target:8s} {case:22s} "
              f"median MSE_CATE={d.mse_cate.median():.4g}  median relATE={d.rel_ate.median():.4g}", flush=True)
