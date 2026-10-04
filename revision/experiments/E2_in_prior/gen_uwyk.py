"""UWYK prior.   python gen_uwyk.py {1d,2d} {new,same} R0:R1 out_root      (UWYK src on PYTHONPATH)

2d: the SCM prior graph2d_step_50000 was trained on (graph2d_scm_config.py), binary T.
1d: the released UWYK 1D's own SCM prior (its best_model_config.yaml scm_config), with T binarized
    by the same BinarizingMechanism (binarize_treatment_prob = 1; it was 0 in training).
Generation = benchmarks/context_sweep/scm_prior.py (graph2d's PairedInterventionalDataset._generate_one,
raw Y kept, no X clip). n_test = 500 query units, as in graph2d training (BatchNorm sees the
same batch size).
new : dataset r = world r, a fresh draw.
same: ONE world (the draw of world 0). Dataset r = a new observational set of 1000 units and
      500 new query units from it, propagated exactly as the prior propagates them (context and
      queries are separate batches, as in training).
"""
import os, random, sys
import numpy as np
import networkx as nx
import torch
import yaml
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "../../..")
sys.path.insert(0, os.path.join(REPO, "benchmarks", "context_sweep"))
import scm_prior as SP
from common import write_world, done, N_CTX, N_Q

model, mode, (R0, R1), out = sys.argv[1], sys.argv[2], map(int, sys.argv[3].split(":")), sys.argv[4]
SEED = 20261003
if model == "2d":
    from graph2d_scm_config import DEFAULT_SCM_CONFIG as scm_config
else:
    uwyk = os.environ.get("UWYK", os.path.join(REPO, "external/uwyk_reproduce"))
    yml = os.path.join(uwyk, "experiments/checkpoints/full_conditioned_model/"
                             "final_earlytest_full_conditioning_16773252.0/best_model_config.yaml")
    scm_config = yaml.safe_load(open(yml))["scm_config"]

# Shared truth: the prior's own paired propagation (every noise draw tiled across the two arms).
# Independent truth: propagate again with the arm-0 rows of every descendant of T given fresh
# endogenous noise. Roots (incl. hidden confounders) and non-descendants keep their draws.
state = {}
_paired = SP._propagate_paired
def paired_with_indep(obs_scm, intv_scm, T, n_test, t0, t1):
    state.update(scm=obs_scm, intv=intv_scm, T=T, t0=t0, t1=t1)  # kept for the same-world run
    res0, res1 = _paired(obs_scm, intv_scm, T, n_test, t0, t1)
    shared = intv_scm._fixed_endogenous_vec.clone()
    with torch.random.fork_rng():                              # leave the prior's RNG stream untouched
        torch.manual_seed(state["seed"])
        intv_scm.sample_endogenous(2 * n_test)                 # fresh noise, same distributions
    fresh, vec = intv_scm._fixed_endogenous_vec, shared.clone()
    for v in nx.descendants(intv_scm.dag.g, T):
        s, e = intv_scm._endo_slices[v]
        vec[:n_test, s:e] = fresh[:n_test, s:e]                # arm 0 only
    intv_scm._fixed_endogenous_vec = vec
    intv_scm._fixed_endogenous = {v: vec[:, slice(*intv_scm._endo_slices[v])].reshape(
        2 * n_test, *intv_scm._node_shape.get(v, ())) for v in intv_scm._endo_order}
    # BatchNorm mechanisms (train mode) normalise over the whole 2*n_test batch, so changing arm 0's
    # noise also moves arm 1. Take BOTH arms of the independent truth from this second pass.
    state["indep"] = intv_scm.propagate(2 * n_test)             # first half arm 0, second half arm 1
    return res0, res1
SP._propagate_paired = paired_with_indep

def write(r, X_ctx, T_ctx, Y_ctx, X_q, y0, y1, target):
    ind = state["indep"][target].reshape(-1).float()
    write_world(out, r, X_ctx, T_ctx, Y_ctx, X_q,
                {"delta_shared": y1 - y0, "delta_indep": ind[N_Q:] - ind[:N_Q]})
    print(f"r={r} F={X_ctx.shape[1]} treated={T_ctx.float().mean():.2f}", flush=True)


def world(r):
    """Draw world r from the prior (graph2d's generation, unchanged). Returns its sample dict."""
    state["seed"] = SEED + 7 * r + 1
    random.seed(SEED + r); np.random.seed(SEED + r)              # the sampler also reads these RNGs
    return SP.generate_paired_sample_with_raw(scm_seed=SEED + r * 1_000, n_train=N_CTX, n_test=N_Q,
                                              scm_config=scm_config)


def dataset(r, s):
    """A new observational set + new query units from the world of sample s (same steps as the prior)."""
    scm, intv, T, t0, t1 = (state[k] for k in ("scm", "intv", "T", "t0", "t1"))
    feats, target = s["feature_nodes"], s["target_node"]
    torch.manual_seed(SEED + 1_000_003 + r)
    state["seed"] = SEED + 7 * r + 1
    cols = lambda o, n: torch.cat([o[v].reshape(n, -1).float() for v in feats], dim=1)
    scm.sample_exogenous(N_CTX); scm._fixed_endogenous_vec = None; scm.sample_endogenous(N_CTX)
    obs = scm.propagate(N_CTX)
    scm.sample_exogenous(N_Q); scm._fixed_endogenous_vec = None; scm.sample_endogenous(N_Q)
    obs_test = scm.propagate(N_Q)
    res0, res1 = paired_with_indep(scm, intv, T, N_Q, t0, t1)
    X_ctx, X_q = SP._standardize(cols(obs, N_CTX), cols(obs_test, N_Q))
    T_ctx = (obs[T].reshape(-1) > (t0 + t1) / 2.0).float()
    write(r, X_ctx, T_ctx, obs[target].reshape(-1).float(), X_q,
          res0[target].reshape(-1).float(), res1[target].reshape(-1).float(), target)


if mode == "new":
    for r in range(R0, R1):
        if not done(out, r):
            s = world(r)
            F = len(s["feature_nodes"])
            write(r, s["X_obs"][:, :F], s["T_obs"].reshape(-1), s["Y_obs_raw"].reshape(-1), s["X_intv"][:, :F],
                  s["Y_do0_raw"].reshape(-1), s["Y_do1_raw"].reshape(-1), s["target_node"])
else:
    s = world(0)                                               # fixes the world; every shard redraws the same one
    for r in range(R0, R1):
        if not done(out, r):
            dataset(r, s)
