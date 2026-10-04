"""CausalPFN prior.   python gen_cpfn.py {1d,2d} {new,same} R0:R1 out_root      (env CAUSALPFN = repo root)

1d: the stock prior (eta0, eta1 independent), as cpfn1d_j1024_headrand was trained.
2d: the stock prior with eta1 := eta0 (CPFN_SHARED_ARM_NOISE=1), as cpfn2d_j32_eta0_y01 was trained.
new : dataset r = world r, a fresh draw (a 2048-row table, as in training; first 1500 rows used).
same: ONE world; one table of R1 * 1500 rows, dataset r = rows r*1500 .. r*1500+1499.
      Rows of one table are i.i.d. units of the same world.
"""
import os, random, sys
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from common import write_world, done, N_CTX, N_Q

model, mode, (R0, R1), out = sys.argv[1], sys.argv[2], map(int, sys.argv[3].split(":")), sys.argv[4]
ROWS = N_CTX + N_Q
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
    if x.numel() == prior.n_samples:                           # only unit-shaped draws (eta is one)
        calls.append((x.detach().double().reshape(-1), g.detach().double().reshape(-1)))
    return g


def fresh_copy(eta0):
    """eta0 / g0 * g': the same unit's eta_0 with g0 replaced by a fresh N(0,1).
    get_sample's first unit-shaped randn_like call is eta_0's ("center the noises"; the table
    generators only draw multi-column noise, and the treatment model runs after). The per-unit
    deg-hetero factor eta0 / (a * g0) lies in [0.7, 1]; y0 - E_y0 is float32, so when |E_y0| >> |eta_0|
    it is rounded (world 337: |E_y0| ~ 1e4; world 1391: ~ 2e11) and the factor is clipped to that range."""
    if len(calls) < 2:
        raise RuntimeError("eta_0 draw not found")
    x, g = calls[0]
    z = x * g
    ok = z.abs() > 1e-6 * z.abs().max()
    factor = torch.where(ok, eta0 / z.where(ok, torch.ones_like(z)), torch.full_like(z, 0.85)).clamp(0.7, 1.0)
    return factor * x * torch.randn_like(eta0)


def draw(r):
    """One table of prior.n_samples units: X, t, factual y, and the truths."""
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
    return X, t, y, truths


def write(r, X, t, y, truths, off=0):
    c, q = slice(off, off + N_CTX), slice(off + N_CTX, off + ROWS)
    write_world(out, r, X[c], t[c], y[c], X[q], {k: v[q] for k, v in truths.items()})
    print(f"r={r} n_cov={X.shape[1]} treated={t[c].mean():.2f}", flush=True)


if mode == "new":
    for r in range(R0, R1):
        if not done(out, r):
            write(r, *draw(r))
else:
    prior.n_samples = R1 * ROWS                                # one world, R1 * 1500 i.i.d. units
    X, t, y, truths = draw(0)
    for r in range(R0, R1):
        if not done(out, r):
            write(r, X, t, y, truths, off=r * ROWS)
