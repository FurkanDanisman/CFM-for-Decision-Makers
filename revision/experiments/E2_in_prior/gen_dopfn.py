"""Do-PFN prior (training_dopfn_repro/prior.py, the prior dopfn_repro_joint2d was trained on).
python gen_dopfn.py {new,same} R0:R1 out_root        (env DOPFN_SRC = Do-PFN repo root)

new : dataset r = world r, a fresh draw from the prior.
same: ONE world (seed of world 0); dataset r = rows r*1500 .. r*1500+1499 of a single long draw
      from it. Rows of one draw are i.i.d. units of the same SCM, so each block is a new
      observational set (1000 units) plus 500 new query units of that world.
"""
import os, sys
import networkx as nx
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
import training_dopfn_repro.prior as P
from common import write_world, done, N_CTX, N_Q

mode, (R0, R1), out = sys.argv[1], map(int, sys.argv[2].split(":")), sys.argv[3]
SEED = 20261003
ROWS = N_CTX + N_Q

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


def draw(r, n_rows):
    """One draw of n_rows units: x (S, F+1) with col 0 = T, y, delta_shared, delta_indep."""
    state["gen"] = torch.Generator().manual_seed(SEED + 7 * r + 1)   # separate stream: prior draws unchanged
    rec = P.sample_batch(SEED + r * 1_000, P.PriorConfig(seq_len=n_rows, batch_size=1, on_nonfinite="resample"))
    x, y = rec["x_obs"][:, 0], rec["y_obs"][:, 0]
    y_do0, y_do1 = rec["y_do0"][:, 0], rec["y_do1"][:, 0]
    y_do0_ind = state["arm0_indep"][rec["meta"]["y_key"]][0].float()
    return x, y, y_do1 - y_do0, y_do1 - y_do0_ind


def write(r, x, y, ds, di, off=0):
    c, q = slice(off, off + N_CTX), slice(off + N_CTX, off + ROWS)
    write_world(out, r, x[c, 1:], x[c, 0], y[c], x[q, 1:], {"delta_shared": ds[q], "delta_indep": di[q]})
    print(f"r={r} F={x.shape[1] - 1} treated={x[c, 0].mean():.2f}", flush=True)


if mode == "new":
    for r in range(R0, R1):
        if not done(out, r):
            write(r, *draw(r, 2200))                    # 2200 = the prior's seq_len; first 1500 rows used
else:
    x, y, ds, di = draw(0, R1 * ROWS)                   # one world, R1 * 1500 i.i.d. units
    for r in range(R0, R1):
        if not done(out, r):
            write(r, x, y, ds, di, off=r * ROWS)
