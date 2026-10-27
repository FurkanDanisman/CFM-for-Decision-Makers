"""First Do-PFN same-world seed (> 20261003) whose SCM has NO hidden confounder: no node outside the model's
covariates (x_keys) and Y is a common ancestor of T and Y. Built exactly as gen_dopfn.py same mode builds it
(seq_len 1,500,000, batch 1, resample); run with PYTHONHASHSEED=0 as gen_e2.sbatch does (the roles depend on it).
python find_seed_nohc.py [start]      prints the seed on the last line
"""
import os, sys
import networkx as nx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
import training_dopfn_repro.prior as P

assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0 (as gen_e2.sbatch)"
cap = {}
_arm = P._propagate_arm
def hook(scm, graph, t_key, value, shared):
    cap["g"] = graph
    return _arm(scm, graph, t_key, value, shared)
P._propagate_arm = hook

s = int(sys.argv[1]) if len(sys.argv) > 1 else 20261004
while True:
    m = P.sample_batch(s, P.PriorConfig(seq_len=1_500_000, batch_size=1, on_nonfinite="resample"))["meta"]
    g, T, Y = cap["g"], m["t_key"], m["y_key"]
    hidden = set(g.nodes) - set(map(str, m["x_keys"])) - {Y}
    conf = sorted(h for h in hidden if h in nx.ancestors(g, T) and h in nx.ancestors(g, Y))
    print(f"seed {s}: T {T} Y {Y} x {list(map(str, m['x_keys']))} hidden {sorted(hidden)} "
          f"hidden confounders {conf}", flush=True)
    if not conf:
        print(s); break
    s += 1
