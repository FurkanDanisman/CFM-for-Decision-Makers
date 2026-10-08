"""Experiment D: does more data fix it, and at what rate? (extension plan, Section 4).

    python exp_D.py --part conv --list-tasks            # number of array tasks
    python exp_D.py --part conv --task-id 0 --out results/exp_D
    python exp_D.py --part rate --task-id 0 --out results/exp_D

Fixed SCMs: per case study, K SCMs are drawn once (generation.scm_constants fixes the
sample-dependent constants on a 100k reference sample), so replicates differ only in data.
Queries: per SCM, one fixed held-out set of m units (both arms); every replicate fits a
fresh context of n observational rows from the same SCM and predicts tau at those m units.
Replicate r uses the same data seed in every setting / lambda (common random numbers).

  conv : for each setting, SCM k and n in N_GRID, R replicates   -> conv/<case>/<setting>/k<k>.npz
  rate : Observed Confounder, n = RATE_N, the full lambda path of each g, targets
         outcome and both, R replicates                          -> rate/<g>_<target>/k<k>.npz
Each npz holds tau_hat (..., R, m) and the truth tau (..., m); exp_D_aggregate.py computes
variance, squared bias and the excess bias E(lambda) from them.
"""
import argparse
import itertools
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generation import generate_realization, scm_constants  # noqa: E402

CASES = ["Observed_Confounder", "Unobserved_Confounder"]
# In-prior plus the worst level of each dial of B (noise on all nodes) and C (lambda = 1,
# outcome equation). Finalize from the B and C results before submitting.
SETTINGS = {
    "in_prior": {},
    "B1_t3": dict(noise_dist="t", noise_param=3, noise_target="all"),
    "B2_gamma1": dict(noise_dist="gamma", noise_param=1, noise_target="all"),
    "B3_c8": dict(noise_scale=8, noise_target="all"),
    "B4_gamma2": dict(hetero_gamma=2, noise_target="all"),
    "C_sin_1": dict(mix_g="sin", mix_lambda=1.0, mix_target="outcome"),
    "C_sign_1": dict(mix_g="sign", mix_lambda=1.0, mix_target="outcome"),
    "C_cubic_1": dict(mix_g="cubic", mix_lambda=1.0, mix_target="outcome"),
}
N_GRID = (100, 250, 500, 1000, 2000)
G_FUNCS = ("sin", "sign", "cubic")
RATE_TARGETS = ("outcome", "both")
LAMBDAS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
RATE_N = 2000

p = argparse.ArgumentParser()
p.add_argument("--part", choices=("conv", "rate"), required=True)
p.add_argument("--task-id", type=int, default=None)
p.add_argument("--list-tasks", action="store_true")
p.add_argument("--K", type=int, default=10, help="fixed SCMs per case study")
p.add_argument("--R", type=int, default=50, help="replicate data sets per SCM and n")
p.add_argument("--m", type=int, default=500, help="fixed held-out query units per SCM")
p.add_argument("--seed-base", type=int, default=0)
p.add_argument("--out", default="results/exp_D")
a = p.parse_args()

if a.part == "conv":
    TASKS = list(itertools.product(range(len(CASES)), SETTINGS, range(a.K)))
else:
    TASKS = list(itertools.product(G_FUNCS, RATE_TARGETS, range(a.K)))
if a.list_tasks:
    print(len(TASKS))
    sys.exit(0)
if a.task_id is None:
    a.task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
out = os.path.abspath(a.out)


def seed(*parts):
    return int(np.random.SeedSequence([a.seed_base, *parts]).generate_state(1)[0])


def scm_seed(case_idx, k):
    """First SCM seed whose reference sample is finite (same rule as generate_sweep)."""
    for attempt in range(50):
        s = seed(1000 + case_idx, k, attempt)
        try:
            generate_realization(CASES[case_idx], 2000, s, data_seed=seed(7, case_idx, k))
            return s
        except ValueError:
            continue
    raise RuntimeError(f"case {case_idx} k{k}: no finite SCM in 50 attempts")


def draw(case_idx, s, n, data_seed, consts, kw):
    for attempt in range(50):
        try:
            return generate_realization(CASES[case_idx], n, s, data_seed=data_seed + attempt,
                                        consts=consts, **kw)
        except ValueError:
            continue
    raise RuntimeError("50 non-finite data draws")


os.chdir(os.path.join(HERE, "dopfn"))
sys.path.insert(0, os.path.join(HERE, "dopfn"))
from scripts.transformer_prediction_interface.base import DoPFNRegressor  # noqa: E402

reg = DoPFNRegressor()
reg.device = "cuda" if torch.cuda.is_available() else "cpu"


def tau_hat(ctx, q):
    reg.fit(torch.tensor(np.c_[ctx.T, ctx.X].astype(np.float32)), torch.tensor(ctx.Y.astype(np.float32)))
    m = [np.asarray(reg.predict_full(torch.tensor(np.c_[np.full(len(q.T), t), q.X].astype(np.float32)))["mean"],
                    dtype=np.float64) for t in (0.0, 1.0)]
    return m[1] - m[0]


def run(case_idx, k, kw, ns):
    """tau_hat (len(ns), R, m) and the truth tau (m,) for one SCM and one setting."""
    s = scm_seed(case_idx, k)
    consts = scm_constants(CASES[case_idx], s, ref_seed=seed(8, case_idx, k), **kw)
    q = draw(case_idx, s, a.m, seed(9, case_idx, k), consts, kw)        # fixed held-out units
    est = np.empty((len(ns), a.R, a.m))
    for i, n in enumerate(ns):
        for r in range(a.R):
            ctx = draw(case_idx, s, n, seed(10, case_idx, k, n, r), consts, kw)
            est[i, r] = tau_hat(ctx, q)
    return s, est, q.cate.astype(np.float64)


if a.part == "conv":
    case_idx, setting, k = TASKS[a.task_id]
    path = os.path.join(out, "conv", CASES[case_idx], setting, f"k{k}.npz")
    if os.path.exists(path):
        print("exists:", path); sys.exit(0)
    s, est, tau = run(case_idx, k, SETTINGS[setting], N_GRID)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, tau_hat=est, tau=tau, n_grid=np.array(N_GRID), scm_seed=s,
                        case=CASES[case_idx], setting=setting, k=k)
else:
    g, target, k = TASKS[a.task_id]
    path = os.path.join(out, "rate", f"{g}_{target}", f"k{k}.npz")
    if os.path.exists(path):
        print("exists:", path); sys.exit(0)
    est, taus = [], []
    for lam in LAMBDAS:
        kw = dict(mix_g=g, mix_lambda=lam, mix_target=target) if lam > 0 else {}
        s, e, tau = run(0, k, kw, (RATE_N,))
        est.append(e[0]); taus.append(tau)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, tau_hat=np.stack(est), tau=np.stack(taus), lambdas=np.array(LAMBDAS),
                        n=RATE_N, scm_seed=s, g=g, target=target, k=k)
print("wrote", path)
