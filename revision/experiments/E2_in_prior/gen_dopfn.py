"""Do-PFN prior worlds (training_dopfn_repro/prior.py, the prior dopfn_repro_joint2d was trained on).
python gen_dopfn.py R0:R1 out_root        (env DOPFN_SRC = Do-PFN repo root)
"""
import os, sys
import networkx as nx
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
import training_dopfn_repro.prior as P
from common import write_world, N_CTX, N_Q

(R0, R1), out = map(int, sys.argv[1].split(":")), sys.argv[2]
SEED = 20261003
cfg = P.PriorConfig(batch_size=1, on_nonfinite="resample")   # one world per draw; skip non-finite worlds

# Shared truth: the prior's own y_do1 - y_do0 (both arms reuse every noise draw, _propagate_arm).
# Independent truth: arm 0 propagated again with fresh additive noise for every descendant of T.
# Pre-treatment variables (exogenous roots, hidden confounders, non-descendants) keep their draws.
state = {}
_hp, _arm = P.sample_hyperparameters, P._propagate_arm
def hp_recorder(c):
    state["hp"] = _hp(c)
    return state["hp"]
def arm_with_indep(scm, graph, t_key, value, shared_exogenous):
    res = _arm(scm, graph, t_key, value, shared_exogenous)
    if bool((value == scm.t2s[:, None]).all()) == P._ARM1_IS_LOWER_RAW_LEVEL:   # this is arm 0
        saved = {}
        for v in nx.descendants(graph, t_key):
            f = scm.functions[v][0]
            saved[v] = f.additive_noise
            f.additive_noise = torch.normal(0.0, state["hp"].noise_std, saved[v].shape, generator=state["gen"])
        try:
            state["arm0_indep"] = _arm(scm, graph, t_key, value, shared_exogenous)
        finally:
            for v, a in saved.items():
                scm.functions[v][0].additive_noise = a
    return res
P.sample_hyperparameters, P._propagate_arm = hp_recorder, arm_with_indep

for r in range(R0, R1):
    state["gen"] = torch.Generator().manual_seed(SEED + 7 * r + 1)   # separate stream: prior draws unchanged
    rec = P.sample_batch(SEED + r * 1_000, cfg)
    x, y = rec["x_obs"][:, 0], rec["y_obs"][:, 0]                    # (S, F+1), col 0 = treatment 0/1
    y_do0, y_do1 = rec["y_do0"][:, 0], rec["y_do1"][:, 0]
    y_do0_ind = state["arm0_indep"][rec["meta"]["y_key"]][0].float()
    q = slice(N_CTX, N_CTX + N_Q)
    write_world(out, r, x[:N_CTX, 1:], x[:N_CTX, 0], y[:N_CTX], x[q, 1:],
                {"delta_shared": (y_do1 - y_do0)[q], "delta_indep": (y_do1 - y_do0_ind)[q]})
    print(f"r={r} F={x.shape[1] - 1} treated={x[:N_CTX, 0].mean():.2f}", flush=True)
