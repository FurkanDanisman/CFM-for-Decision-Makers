"""
Debug the specific SCM from debug_cate.py (treatment=3, target=0, CATE=0).
Check the actual path and what's happening.
"""
import sys, os, torch
sys.path.insert(0, '/tmp/g4cfm/src')

from priors.causal_prior.scm.SCMSampler import SCMSampler
from priors.causal_prior.mechanisms.BinarizingMechanism import BinarizingMechanism
from copy import deepcopy
import networkx as nx

# Exact config from generate_paired_samples.py / debug_cate.py
scm_config = {
    "num_nodes": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 2, "high": 51}},
    "graph_edge_prob": {"distribution": "beta", "distribution_parameters": {"alpha": 2.0, "beta": 3.0}},
    "graph_seed": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 0, "high": 100000}},
    "xgboost_prob": {"distribution": "categorical", "distribution_parameters": {"choices": [0.0, 0.1, 0.2, 0.3], "probabilities": [1.0, 0.0, 0.0, 0.0]}},
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

# Same seed as generate_sample(0, seed_base=42)
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
    rng_gen = torch.Generator()
    rng_gen.manual_seed(attempt_seed)
    found_pair = False
    for pair_try in range(30):
        t_idx = torch.randint(0, n_nodes, (1,), generator=rng_gen).item()
        treatment_node = all_nodes[t_idx]
        available = [n for n in all_nodes if n != treatment_node]
        y_idx = torch.randint(0, len(available), (1,), generator=rng_gen).item()
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
print(f"n_nodes={n_nodes}")

# Check DAG structure
print(f"\nDAG edges: {sorted(scm.dag.g.edges())[:30]}")
print(f"Parents of treatment_node {treatment_node}: {scm.dag.parents(treatment_node)}")
print(f"Parents of target_node {target_node}: {scm.dag.parents(target_node)}")
print(f"Is treatment_node a root in obs_scm? {scm._is_root[treatment_node]}")

# NetworkX path check
g = scm.dag.g
print(f"\nNetworkX check:")
print(f"  Path {treatment_node}→{target_node} exists: {nx.has_path(g, treatment_node, target_node)}")
try:
    path = nx.shortest_path(g, treatment_node, target_node)
    print(f"  Shortest path: {path}")
except:
    print(f"  No path in networkx!")

print(f"  Path {target_node}→{treatment_node} exists: {nx.has_path(g, target_node, treatment_node)}")

# Does exists_treatment_outcome_path agree?
print(f"  exists_treatment_outcome_path({treatment_node}, {target_node}): {scm.exists_treatment_outcome_path(treatment_node, target_node)}")

# Create intv_scm and test propagation directly
intv_scm = deepcopy(scm)
intv_scm.intervene(treatment_node)

print(f"\nIntv_scm structure:")
print(f"  treatment in exo_order: {treatment_node in intv_scm._exo_order}")
print(f"  _parent_slices[target_node]: {intv_scm._parent_slices[target_node]}")
print(f"  _parent_slices[treatment_node]: {intv_scm._parent_slices[treatment_node]}")

# Sample noise
scm.sample_exogenous(N_TEST)
scm._fixed_endogenous_vec = None
scm.sample_endogenous(N_TEST)
obs_test = scm.propagate(N_TEST)

# Set up noise for do(T=0)
total_exo = intv_scm._total_exo_dim
fixed_exo = {}
for v in intv_scm._exo_order:
    if v == treatment_node:
        fixed_exo[v] = torch.zeros(N_TEST)
    else:
        fixed_exo[v] = scm._fixed_exogenous[v].clone()

intv_scm._fixed_exogenous = fixed_exo
intv_scm._fixed_batch = N_TEST

total_endo = intv_scm._total_endo_dim
fixed_endo = {}
if scm._fixed_endogenous:
    for v in intv_scm._endo_order:
        old = scm._fixed_endogenous.get(v)
        if old is not None:
            fixed_endo[v] = old.clone()
        else:
            fixed_endo[v] = torch.zeros(N_TEST)
intv_scm._fixed_endogenous = fixed_endo

res0 = intv_scm.propagate(N_TEST)
Y_do0 = res0[target_node].reshape(-1).float()
T_do0 = res0[treatment_node].reshape(-1).float()
print(f"\ndo(T=0): T[:5]={T_do0[:5].tolist()}, Y mean={Y_do0.mean():.4f}, std={Y_do0.std():.4f}")

# Set treatment to 1
intv_scm._fixed_exogenous[treatment_node] = torch.ones(N_TEST)
res1 = intv_scm.propagate(N_TEST)
Y_do1 = res1[target_node].reshape(-1).float()
T_do1 = res1[treatment_node].reshape(-1).float()
print(f"do(T=1): T[:5]={T_do1[:5].tolist()}, Y mean={Y_do1.mean():.4f}, std={Y_do1.std():.4f}")
print(f"CATE: mean={(Y_do1-Y_do0).mean():.4f}, std={(Y_do1-Y_do0).std():.4f}")
print(f"Y_do0==Y_do1: {torch.allclose(Y_do0, Y_do1)}")

# Test with very different T values to see if there's ANY effect
intv_scm._fixed_exogenous[treatment_node] = torch.full((N_TEST,), 100.0)
res_large = intv_scm.propagate(N_TEST)
Y_large = res_large[target_node].reshape(-1).float()
print(f"\ndo(T=100): Y mean={Y_large.mean():.4f}, std={Y_large.std():.4f}")
print(f"Y_do0==Y_large: {torch.allclose(Y_do0, Y_large)}")

# Check the mechanism for target_node
print(f"\nMechanism for target_node: {type(intv_scm.mechanisms[target_node]).__name__}")
print(f"Mechanism for treatment_node: {type(intv_scm.mechanisms[treatment_node]).__name__}")

# Check if target's parents include treatment at any level
parents_of_target = intv_scm._parent_slices[target_node]
print(f"Direct parents of target in intv_scm: {[p for p,s,e in parents_of_target]}")

# Now manually run target mechanism with T=0 vs T=1
mech_target = intv_scm.mechanisms[target_node]
B = N_TEST
D = intv_scm._parent_total_dim[target_node]
parents_feat_0 = torch.empty(B, D)
parents_feat_1 = torch.empty(B, D)
for p, s, e in intv_scm._parent_slices[target_node]:
    parents_feat_0[:, s:e] = res0[p].reshape(B, -1)
    parents_feat_1[:, s:e] = res1[p].reshape(B, -1)

eps_target = intv_scm._fixed_endogenous.get(target_node)
out0 = mech_target._forward(parents_feat_0, eps=eps_target)
out1 = mech_target._forward(parents_feat_1, eps=eps_target)
print(f"\nManual mechanism output:")
print(f"  parents_feat_0[:5,0] = {parents_feat_0[:5,0].tolist()}")
print(f"  parents_feat_1[:5,0] = {parents_feat_1[:5,0].tolist()}")
print(f"  parents_feat_0 == parents_feat_1: {torch.allclose(parents_feat_0, parents_feat_1)}")
print(f"  out0[:5]: {out0[:5].reshape(-1).tolist()}")
print(f"  out1[:5]: {out1[:5].reshape(-1).tolist()}")
print(f"  out0 == out1: {torch.allclose(out0, out1)}")
