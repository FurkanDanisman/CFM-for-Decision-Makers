"""Do-PFN-bb prior worlds (training/data/PairedDoPFNDataset.py, the prior dopfn_bb_j10_step_150000 was
trained on: training_dopfn_base/train.py with NUM_FEATURES=6, n_train=1000, n_test=500, default config).
python gen_dopfn_bb.py R0:R1 out_root        (env DOPFN_SRC = Do-PFN repo root)

World r = PairedDoPFNDataset._generate_one with __getitem__'s seeds and acceptance rule (the first
finite attempt that passes the variance / unique-value thresholds). Context = its 1000 training
rows, queries = its 500 query rows. Truths as for the other priors: shared noise (the prior's own
arms) and independent noise (arm 0 again, fresh additive noise for every descendant of T).
"""
import os, sys
import networkx as nx
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
import training.data.PairedDoPFNDataset as PD
from common import write_world, done, N_CTX, N_Q

(R0, R1), out = map(int, sys.argv[1].split(":")), sys.argv[2]
SEED = 20261003
ds = PD.PairedDoPFNDataset(num_features=6, n_train=N_CTX, n_test=N_Q, seed_base=SEED)

state = {}
_arm = PD._propagate_arm
def arm_with_indep(scm, graph, treatment, value, shared_exogenous):
    res = _arm(scm, graph, treatment, value, shared_exogenous)
    if bool((value == scm.t2s[:, None]).all()):                 # arm 0 (upper raw level), drawn first
        saved = {}
        for v in nx.descendants(graph, treatment):
            f = scm.functions[v][0]
            saved[v] = f.additive_noise
            f.additive_noise = torch.normal(0.0, float(ds.config["noise_std"]), saved[v].shape,
                                            generator=state["gen"])
        try:
            state["arm0_indep"] = _arm(scm, graph, treatment, value, shared_exogenous)
        finally:
            for v, a in saved.items():
                scm.functions[v][0].additive_noise = a
    return res
PD._propagate_arm = arm_with_indep

for r in range(R0, R1):
    if done(out, r):
        continue
    for attempt in range(ds.max_sampling_attempts):              # __getitem__'s loop
        state["gen"] = torch.Generator().manual_seed(SEED + 7 * r + 1 + attempt)
        try:
            s = ds._generate_one(ds.seed_base + r + attempt * 1_000_003)
        except (AssertionError, ValueError, RuntimeError, IndexError):
            continue
        if PD._finite(s) and PD._passes_thresholds(s, ds.min_target_variance, ds.min_unique_target_fraction):
            break
    else:
        raise RuntimeError(f"world {r}: no accepted draw")
    d, q, c = ds.last_debug, slice(N_CTX, N_CTX + N_Q), slice(0, N_CTX)
    Y = d["outcome"]
    y0, y1, y0i = d["do0"][Y][0, q], d["do1"][Y][0, q], state["arm0_indep"][Y][0, q]
    write_world(out, r, s["X_obs"], s["T_obs"], d["obs"][Y][0, c].float(), s["X_intv"],
                {"delta_shared": (y1 - y0).float(), "delta_indep": (y1 - y0i).float()})
    print(f"r={r} treated={s['T_obs'].mean():.2f}", flush=True)
