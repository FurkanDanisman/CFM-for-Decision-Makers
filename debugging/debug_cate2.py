"""
Debug CATE=0 issue: check what happens inside propagate() for treatment_node.
"""
import sys, os, torch
sys.path.insert(0, '/tmp/g4cfm/src')

from priors.causal_prior.scm.SCMSampler import SCMSampler
from priors.causal_prior.mechanisms.BinarizingMechanism import BinarizingMechanism
from copy import deepcopy
import networkx as nx

scm_config = {
    "num_nodes": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 2, "high": 51}},
    "graph_edge_prob": {"distribution": "beta", "distribution_parameters": {"alpha": 2.0, "beta": 3.0}},
    "graph_seed": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 0, "high": 100000}},
    "xgboost_prob": {"distribution": "categorical", "distribution_parameters": {"choices": [0.0], "probabilities": [1.0]}},
    "mechanism_seed": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 0, "high": 100000}},
    "mlp_nonlins": {"value": "tabicl"},
    "mlp_num_hidden_layers": {"distribution": "categorical", "distribution_parameters": {"choices": [0, 1, 2, 3], "probabilities": [0.875, 0.1, 0.025, 0.01]}},
    "mlp_hidden_dim": {"distribution": "categorical", "distribution_parameters": {"choices": [1, 2, 4, 6, 8, 10, 12, 14, 16, 32], "probabilities": [0.7, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01]}},
    "mlp_activation_mode": {"distribution": "categorical", "distribution_parameters": {"choices": ["pre", "post", "mixed_in"], "probabilities": [0.3, 0.3, 0.3]}},
    "mlp_use_batch_norm": {"distribution": "categorical", "distribution_parameters": {"choices": [True, False], "probabilities": [0.5, 0.5]}},
    "mlp_node_shape": {"value": (1,)},
    "xgb_node_shape": {"value": (1,)},
    "xgb_num_hidden_layers": {"value": 0},
    "xgb_hidden_dim": {"distribution": "categorical", "distribution_parameters": {"choices": [0, 16, 32, 64]}},
    "xgb_activation_mode": {"distribution": "categorical", "distribution_parameters": {"choices": ["pre", "post", "mixed_in"], "probabilities": [0.33, 0.33, 0.34]}},
    "xgb_use_batch_norm": {"distribution": "categorical", "distribution_parameters": {"choices": [True, False], "probabilities": [0.5, 0.5]}},
    "xgb_n_training_samples": {"distribution": "categorical", "distribution_parameters": {"choices": [10, 50, 100, 200, 500], "probabilities": [0.1, 0.1, 0.3, 0.4, 0.5]}},
    "xgb_add_noise": {"distribution": "categorical", "distribution_parameters": {"choices": [True, False], "probabilities": [0.5, 0.5]}},
    "random_additive_std": {"value": True},
    "exo_std_distribution": {"distribution": "categorical", "distribution_parameters": {"choices": ["gamma", "pareto"], "probabilities": [1.0, 0.0]}},
    "endo_std_distribution": {"distribution": "categorical", "distribution_parameters": {"choices": ["gamma", "pareto"], "probabilities": [1.0, 0.0]}},
    "exo_std_mean": {"distribution": "lognormal", "distribution_parameters": {"mean": 1.0, "std": 1.0}},
    "exo_std_std": {"distribution": "uniform", "distribution_parameters": {"low": 0.1, "high": 0.4}},
    "endo_std_mean": {"distribution": "lognormal", "distribution_parameters": {"mean": -3.0, "std": 0.6}},
    "endo_std_std": {"distribution": "uniform", "distribution_parameters": {"low": 0.0, "high": 0.5}},
    "endo_p_zero": {"value": 0.0},
    "noise_mixture_proportions": {"value": [0.33, 0.33, 0.34]},
    "use_exogenous_mechanisms": {"value": True},
    "mechanism_generator_seed": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 0, "high": 100000}},
}

N_TRAIN, N_TEST = 1000, 500
scm_sampler = SCMSampler(scm_config, seed=42 * 31 + 17)

# Reproduce generate_sample(0) seed
idx, seed_base = 0, 42
torch.manual_seed(seed_base + idx)
seed = seed_base + idx

for outer_attempt in range(50):
    attempt_seed = seed + outer_attempt * 997
    scm = scm_sampler.sample(seed=attempt_seed)
    all_nodes = sorted(scm.dag.nodes())
    n_nodes = len(all_nodes)
    if n_nodes < 3:
        continue
    rng = torch.Generator()
    rng.manual_seed(attempt_seed)
    found_pair = False
    for pair_try in range(30):
        t_idx = torch.randint(0, n_nodes, (1,), generator=rng).item()
        treatment_node = all_nodes[t_idx]
        available = [n for n in all_nodes if n != treatment_node]
        y_idx = torch.randint(0, len(available), (1,), generator=rng).item()
        target_node = available[y_idx]
        if scm.exists_treatment_outcome_path(treatment_node, target_node):
            found_pair = True
            break
    if not found_pair:
        continue

    feature_nodes = [n for n in all_nodes if n != treatment_node and n != target_node]
    original_mech = scm.mechanisms[treatment_node]

    binarized_ok = False
    for bin_try in range(10):
        scm.sample_exogenous(N_TRAIN)
        scm._fixed_endogenous_vec = None
        scm.sample_endogenous(N_TRAIN)
        obs = scm.propagate(N_TRAIN)
        t_cont = obs[treatment_node].reshape(-1).float()
        q = min(0.5 + 0.05 * bin_try, 0.95)
        threshold = torch.quantile(t_cont, q).item()
        bin_mech = BinarizingMechanism(wrapped_mechanism=original_mech, threshold=threshold, t0=0.0, t1=1.0)
        scm.mechanisms[treatment_node] = bin_mech
        scm.sample_exogenous(N_TRAIN)
        scm._fixed_endogenous_vec = None
        scm.sample_endogenous(N_TRAIN)
        obs = scm.propagate(N_TRAIN)
        T_obs_raw = obs[treatment_node].reshape(-1, 1).float()
        if T_obs_raw.unique().numel() >= 2:
            binarized_ok = True
            break
        scm.mechanisms[treatment_node] = original_mech
    if not binarized_ok:
        continue

    Y_obs_raw = obs[target_node].reshape(-1, 1).float()
    if Y_obs_raw.var() < 1e-3:
        continue
    if torch.unique(Y_obs_raw).numel() < max(5, int(0.1 * N_TRAIN)):
        continue
    break

print(f"outer_attempt={outer_attempt}, treatment_node={treatment_node}, target_node={target_node}")

# Inspect DAG structure
print(f"\nDAG edges (from g4cfm): {list(scm.dag.g.edges())[:20]}")
print(f"Parents of treatment_node {treatment_node}: {scm.dag.parents(treatment_node)}")
print(f"Parents of target_node {target_node}: {scm.dag.parents(target_node)}")
print(f"Is treatment_node a root in obs_scm? {scm._is_root[treatment_node]}")

# Check path exists
print(f"\nexists_treatment_outcome_path({treatment_node}, {target_node}): {scm.exists_treatment_outcome_path(treatment_node, target_node)}")

# Check using networkx path
try:
    path = nx.shortest_path(scm.dag.g, treatment_node, target_node)
    print(f"Shortest path {treatment_node} -> {target_node}: {path}")
except nx.NetworkXNoPath:
    print(f"No path {treatment_node} -> {target_node} in networkx!")
except Exception as e:
    print(f"Error checking path: {e}")

# Create intv_scm
intv_scm = deepcopy(scm)
intv_scm.intervene(treatment_node)

print(f"\nAfter intervene:")
print(f"  treatment_node in intv_scm._exo_order: {treatment_node in intv_scm._exo_order}")
print(f"  treatment_node in intv_scm._endo_order: {treatment_node in intv_scm._endo_order}")
print(f"  intv_scm._is_root[treatment_node]: {intv_scm._is_root[treatment_node]}")
print(f"  intv_scm._parent_slices[treatment_node]: {intv_scm._parent_slices[treatment_node]}")
print(f"  intv_scm._parent_slices[target_node]: {intv_scm._parent_slices[target_node]}")

# What are the parents of target_node in intv_scm?
print(f"  Parents of target_node {target_node} in intv_scm: {intv_scm.dag.parents(target_node)}")

# Sample noise for N_TEST query points
scm.sample_exogenous(N_TEST)
scm._fixed_endogenous_vec = None
scm.sample_endogenous(N_TEST)
obs_test = scm.propagate(N_TEST)

# Manual transfer_noise_and_propagate
total_exo = intv_scm._total_exo_dim
fixed_exo_vec = torch.zeros(N_TEST, total_exo, dtype=torch.float32)
fixed_exo = {}
for v in intv_scm._exo_order:
    s, e = intv_scm._exo_slices[v]
    d = e - s
    if v == treatment_node:
        val = torch.full((N_TEST,), 0.0)
        fixed_exo_vec[:, s:e] = 0.0
        fixed_exo[v] = val
    else:
        old = scm._fixed_exogenous[v]
        fixed_exo_vec[:, s:e] = old.reshape(N_TEST, d)
        fixed_exo[v] = old.clone()

intv_scm._fixed_exogenous_vec = fixed_exo_vec
intv_scm._fixed_exogenous = fixed_exo
intv_scm._fixed_batch = N_TEST

total_endo = intv_scm._total_endo_dim
if total_endo == 0:
    fixed_endo_vec = torch.empty(N_TEST, 0)
    fixed_endo = {}
else:
    fixed_endo_vec = torch.zeros(N_TEST, total_endo, dtype=torch.float32)
    fixed_endo = {}
    for v in intv_scm._endo_order:
        s, e = intv_scm._endo_slices[v]
        d = e - s
        old = scm._fixed_endogenous.get(v) if scm._fixed_endogenous else None
        if old is not None:
            fixed_endo_vec[:, s:e] = old.reshape(N_TEST, d)
            fixed_endo[v] = old.clone()
        else:
            shp = intv_scm._node_shape.get(v, ())
            z = torch.zeros(N_TEST, *shp) if shp else torch.zeros(N_TEST)
            fixed_endo_vec[:, s:e] = z.reshape(N_TEST, d)
            fixed_endo[v] = z

intv_scm._fixed_endogenous_vec = fixed_endo_vec
intv_scm._fixed_endogenous = fixed_endo

print(f"\nBefore do(T=0) propagation:")
print(f"  intv_scm._fixed_exogenous[treatment_node][:5]: {intv_scm._fixed_exogenous[treatment_node][:5]}")
res0 = intv_scm.propagate(N_TEST)
print(f"After do(T=0) propagation:")
print(f"  res0[treatment_node][:5]: {res0[treatment_node][:5].reshape(-1)}")
print(f"  res0[target_node][:5]: {res0[target_node][:5].reshape(-1)}")

# Update for T=1
s, e = intv_scm._exo_slices[treatment_node]
intv_scm._fixed_exogenous_vec[:, s:e] = 1.0
intv_scm._fixed_exogenous[treatment_node] = torch.ones(N_TEST)

print(f"\nBefore do(T=1) propagation:")
print(f"  intv_scm._fixed_exogenous[treatment_node][:5]: {intv_scm._fixed_exogenous[treatment_node][:5]}")
res1 = intv_scm.propagate(N_TEST)
print(f"After do(T=1) propagation:")
print(f"  res1[treatment_node][:5]: {res1[treatment_node][:5].reshape(-1)}")
print(f"  res1[target_node][:5]: {res1[target_node][:5].reshape(-1)}")

print(f"\nCATEcheck:")
do0 = res0[target_node].reshape(-1).float()
do1 = res1[target_node].reshape(-1).float()
print(f"  do0 == do1: {torch.allclose(do0, do1)}")
print(f"  CATE mean: {(do1-do0).mean():.4f}, std: {(do1-do0).std():.4f}")

# Also check intermediate nodes on path if any
print(f"\n--- Check intermediate nodes ---")
try:
    path = nx.shortest_path(scm.dag.g, treatment_node, target_node)
    for node in path[1:]:  # skip treatment_node itself
        v0 = res0[node].reshape(-1)
        v1 = res1[node].reshape(-1)
        print(f"  node {node}: do0 mean={v0.mean():.4f}, do1 mean={v1.mean():.4f}, diff={( v1-v0).abs().mean():.4f}")
except Exception as e:
    print(f"  Error: {e}")
