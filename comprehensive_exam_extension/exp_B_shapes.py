"""Experiment B, shape paths: one noise family at a time, its shape parameter moving away
from the Gaussian over 5 levels (plot_noise_shape_paths.py shows the densities).

    python exp_B_shapes.py --scale scaled   --out results/exp_B_shapes/scaled
    python exp_B_shapes.py --scale unscaled --out results/exp_B_shapes/unscaled

  scaled   : each level standardized to variance 1 times the in-prior sigma_eps (shape only)
  unscaled : textbook form with scale sigma_eps, not standardized (shape and variance change)
Families and levels (closest to Gaussian first):
  gamma k 16, 8, 4, 2, 1 | bimodal a 1, 1.5, 2, 2.5, 3 | contam c 2, 3, 5, 7, 10 |
  lognorm s 0.25, 0.5, 0.75, 1, 1.25 | gennorm beta 1.5, 1.25, 1, 0.75, 0.6
Each run with the noise changed on all structural nodes ("all") and on the outcome only
("outcome"); every level reuses the same seeds per realization. Protocol and metrics as in
exp_B.py (50/50 split, CATE on the test units, MSE_CATE and Rel. ATE error).
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
FAMILIES = {
    "gamma": ("k", (16, 8, 4, 2, 1)),
    "bimodal": ("a", (1, 1.5, 2, 2.5, 3)),
    "contam": ("c", (2, 3, 5, 7, 10)),
    "lognorm": ("s", (0.25, 0.5, 0.75, 1.0, 1.25)),
    "gennorm": ("beta", (1.5, 1.25, 1.0, 0.75, 0.6)),
}

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=200)
p.add_argument("--n-real", type=int, default=100, help="realizations r < n-real")
p.add_argument("--r-start", type=int, default=0, help="first realization (to extend an earlier run)")
p.add_argument("--families", nargs="*", default=None, help="subset of families (to split work)")
p.add_argument("--no-baseline", action="store_true", help="skip the in-prior condition")
p.add_argument("--seed-base", type=int, default=0)
p.add_argument("--scale", choices=("scaled", "unscaled"), required=True)
p.add_argument("--out", required=True)
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
        yield "in_prior", "none", 0.0, "all", {}
    for fam, (name, levels) in FAMILIES.items():
        if a.families and fam not in a.families:
            continue
        for level in levels:
            for target in ("all", "outcome"):
                yield fam, name, float(level), target, dict(
                    noise_dist=fam, noise_param=level, noise_target=target,
                    noise_standardize=(a.scale == "scaled"))


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
        for r in range(a.r_start, a.n_real):
            seed, real = realization(case_idx, case, r, kw)
            rows.append(dict(case=case, dial=dial, param=name, level=level, target=target,
                             scale=a.scale, r=r, seed=seed,
                             noise_std=real.noise_std, **score(real, seed)))
        pd.DataFrame(rows).to_csv(csv, index=False)
        d = pd.DataFrame(rows)
        d = d[(d.case == case) & (d.dial == dial) & (d.level == level) & (d.target == target)]
        print(f"{dial:10s} {name}={level:<5g} {target:8s} {case:22s} "
              f"median MSE_CATE={d.mse_cate.median():.4g}  median relATE={d.rel_ate.median():.4g}", flush=True)
