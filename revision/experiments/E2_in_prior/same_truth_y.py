"""Both true outcomes of every query unit of the same-SCM Do-PFN data (gen_dopfn.py same mode, one world), in both
noise versions, which gen_dopfn.py did not save.
python same_truth_y.py DOPFN_DATA_DIR [WORLD]        (PYTHONHASHSEED=0 and DOPFN_SRC as in gen_e2.sbatch)

The world is redrawn exactly as gen_dopfn.py draws it, including its independent-noise arm (arm 0 propagated again
with fresh additive noise for every descendant of T, from the same separate generator). Dataset r's query units are
rows r*1500+1000 .. r*1500+1499. The redraw must reproduce both saved truths, delta_shared and delta_indep, of every
dataset, else the script stops. Writes DOPFN_DATA_DIR/truths_y/r###.npz with
  y0, y1      outcomes under do(0), do(1), shared noise (the prior's own draw)
  y0_ind      the outcome under do(0) with independent noise (y1 is unchanged), so delta_indep = y1 - y0_ind
"""
import os, sys
import networkx as nx
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
import training_dopfn_repro.prior as P
from common import N_CTX, N_Q

assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0 (as gen_e2.sbatch)"
root = sys.argv[1]; W = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SEED, ROWS, R = 20261003, N_CTX + N_Q, 1000

# --- gen_dopfn.py's independent-noise arm, verbatim ---
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

state["gen"] = torch.Generator().manual_seed(SEED + 7 * W + 1)
rec = P.sample_batch(SEED + W * 1_000, P.PriorConfig(seq_len=R * ROWS, batch_size=1, on_nonfinite="resample"))
y0, y1 = rec["y_do0"][:, 0].numpy(), rec["y_do1"][:, 0].numpy()
y0i = state["arm0_indep"][rec["meta"]["y_key"]][0].float().numpy()
out = os.path.join(root, "truths_y"); os.makedirs(out, exist_ok=True)
for r in range(R):
    q = slice(r * ROWS + N_CTX, (r + 1) * ROWS)
    t = np.load(os.path.join(root, "truths", f"r{r:03d}.npz"))
    for name, mine in (("delta_shared", y1[q] - y0[q]), ("delta_indep", y1[q] - y0i[q])):
        if not np.allclose(mine.astype(np.float32), t[name], rtol=1e-5, atol=1e-6):
            raise SystemExit(f"dataset {r}: redraw does not reproduce the saved {name} -- stopping")
    np.savez(os.path.join(out, f"r{r:03d}.npz"), y0=y0[q].astype(np.float32), y1=y1[q].astype(np.float32),
             y0_ind=y0i[q].astype(np.float32))
print(f"{R} datasets: y0, y1, y0_ind written; both saved ITE truths reproduced", flush=True)
