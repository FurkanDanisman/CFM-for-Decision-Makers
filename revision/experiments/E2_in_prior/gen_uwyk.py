"""UWYK prior worlds.   python gen_uwyk.py {1d,2d} R out_root      (UWYK src on PYTHONPATH)

2d: the SCM prior graph2d_step_50000 was trained on (graph2d_scm_config.py), binary T.
1d: the released UWYK 1D's own SCM prior (its best_model_config.yaml scm_config), with T binarized
    by the same BinarizingMechanism (binarize_treatment_prob = 1; it was 0 in training).
Generation = benchmarks/context_sweep/scm_prior.py (graph2d's PairedInterventionalDataset._generate_one,
raw Y kept, no X clip). n_test = 500 as in graph2d training (BatchNorm sees the same batch size);
the first 100 query units are kept.
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
from common import write_world, N_CTX, N_Q

model, R, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SEED = 20261003
if model == "2d":
    from graph2d_scm_config import DEFAULT_SCM_CONFIG as scm_config
else:
    yml = os.path.join(REPO, "external/uwyk_reproduce/experiments/checkpoints/full_conditioned_model/"
                             "final_earlytest_full_conditioning_16773252.0/best_model_config.yaml")
    scm_config = yaml.safe_load(open(yml))["scm_config"]

# Shared truth: the prior's own paired propagation (every noise draw tiled across the two arms).
# Independent truth: propagate again with the arm-0 rows of every descendant of T given fresh
# endogenous noise. Roots (incl. hidden confounders) and non-descendants keep their draws.
state = {}
_paired = SP._propagate_paired
def paired_with_indep(obs_scm, intv_scm, T, n_test, t0, t1):
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

for r in range(R):
    state["seed"] = SEED + 7 * r + 1
    random.seed(SEED + r); np.random.seed(SEED + r)              # the sampler also reads these RNGs
    s = SP.generate_paired_sample_with_raw(scm_seed=SEED + r * 1_000, n_train=N_CTX, n_test=500,
                                           scm_config=scm_config)
    F = len(s["feature_nodes"])
    y0 = s["Y_do0_raw"].reshape(-1)
    ind = state["indep"][s["target_node"]].reshape(-1).float()
    y0_ind, y1_ind = ind[:500], ind[500:]
    y1 = s["Y_do1_raw"].reshape(-1)
    q = slice(0, N_Q)
    write_world(out, r, s["X_obs"][:, :F], s["T_obs"], s["Y_obs_raw"], s["X_intv"][q, :F],
                {"delta_shared": (y1 - y0)[q], "delta_indep": (y1_ind - y0_ind)[q]})
    print(f"r={r} F={F} treated={s['T_obs'].mean():.2f}", flush=True)
