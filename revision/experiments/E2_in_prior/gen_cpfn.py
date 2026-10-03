"""CausalPFN prior worlds.   python gen_cpfn.py {1d,2d} R out_root      (env CAUSALPFN = repo root)

1d: the stock prior (eta0, eta1 independent), as cpfn1d_j1024_headrand was trained.
2d: the stock prior with eta1 := eta0 (CPFN_SHARED_ARM_NOISE=1), as cpfn2d_j32_eta0_y01 was trained.
"""
import os, random, sys
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from common import write_world, N_CTX, N_Q

model, R, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SEED = 20261003

# the training prior: train.py +experiment=simple_configuration -> train_meta_dataset (synthetic_backdoor)
with initialize_config_dir(config_dir=os.path.join(os.environ["CAUSALPFN"], "conf"), version_base=None):
    cfg = compose(config_name="train", overrides=["+experiment=simple_configuration"])
prior = instantiate(cfg.train_meta_dataset)

# backdoor_prior.get_sample draws eta_0 = a * g0, with a the per-unit size and g0 = randn_like(a)
# ("center the noises"); degree_heterogeneity then multiplies it by a per-unit factor in [0.7, 1].
# The 2D's independent truth needs a second copy of the same unit's eta_0 with a fresh g'.
# Record every randn_like call; afterwards find the one that made eta_0. The prior's own code
# and RNG stream are untouched.
calls = []
_randn_like = torch.randn_like
def _record(x, *a, **k):
    g = _randn_like(x, *a, **k)
    calls.append((x.detach().double().reshape(-1), g.detach().double().reshape(-1)))
    return g


def fresh_copy(eta0):
    """eta0 / g0 * g': the same unit's eta_0 with g0 replaced by a fresh N(0,1)."""
    for x, g in calls:
        if x.numel() == eta0.numel():
            factor = eta0 / (x * g)                            # per-unit deg-hetero factor
            ok = (x * g).abs() > 1e-6 * (x * g).abs().max()
            if ((factor[ok] > 0.69) & (factor[ok] < 1.01)).float().mean() > 0.99:
                factor = torch.where(ok, factor, factor[ok].median())
                return factor * x * torch.randn_like(eta0)
    raise RuntimeError("eta_0 draw not found")


for r in range(R):
    torch.manual_seed(SEED + r); random.seed(SEED + r); np.random.seed(SEED + r)
    calls.clear()
    torch.randn_like = _record
    s = prior.get_sample()
    torch.randn_like = _randn_like
    X, t, y0, y1, E0, E1 = (s[k].double() for k in ("X", "t", "y0", "y1", "E_y0", "E_y1"))
    tau = E1 - E0
    eta0 = y0 - E0

    if model == "1d":
        y = s["y"].double()                                    # factual outcome of the stock prior
        truths = {"delta_shared": tau,                         # eta1 := eta0 cancels: y1 - y0 = tau
                  "delta_indep": y1 - y0,
                  "tau": tau}
    else:
        y = torch.where(t > 0.5, E1 + eta0, y0)                # trainer.py: y1 = E_y1 + (y0 - E_y0)
        eta0_new = fresh_copy(eta0)                            # same unit, same scale, fresh draw
        truths = {"delta_shared": tau,                         # (E1 + eta0) - (E0 + eta0) = tau
                  "delta_indep": tau + eta0_new - eta0}

    q = slice(N_CTX, N_CTX + N_Q)
    write_world(out, r, X[:N_CTX], t[:N_CTX], y[:N_CTX], X[q], {k: v[q] for k, v in truths.items()})
    print(f"r={r} n_cov={X.shape[1]} treated={t[:N_CTX].mean():.2f}", flush=True)
