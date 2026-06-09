"""
Debug script: trace Y_do0_raw and Y_do1_raw through preprocessing for sample 0.
"""
import sys, os, torch
sys.path.insert(0, '/tmp/g4cfm/src')

from priors.causal_prior.scm.SCMSampler import SCMSampler
from priors.causal_prior.mechanisms.BinarizingMechanism import BinarizingMechanism
from copy import deepcopy

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

N_TRAIN, N_TEST, EPS, OUTLIER_Q = 1000, 500, 1e-8, 0.99
scm_sampler = SCMSampler(scm_config, seed=42 * 31 + 17)

def clip_outliers(t, q=OUTLIER_Q):
    lo = torch.quantile(t.reshape(-1).float(), 1.0 - q)
    hi = torch.quantile(t.reshape(-1).float(), q)
    return t.clamp(lo, hi)

def scale_to_neg1_pos1(Y_train, Y_test0, Y_test1, eps=EPS):
    lo = Y_train.min()
    hi = Y_train.max()
    rng = (hi - lo).clamp(min=eps)
    def _scale(y):
        return 2.0 * (y - lo) / rng - 1.0
    return _scale(Y_train), _scale(Y_test0), _scale(Y_test1)

def transfer_noise_and_propagate(obs_scm, intv_scm, treatment_node, target_node, n_test, t_value):
    total_exo = intv_scm._total_exo_dim
    fixed_exo_vec = torch.zeros(n_test, total_exo, dtype=torch.float32)
    fixed_exo = {}
    for v in intv_scm._exo_order:
        s, e = intv_scm._exo_slices[v]
        d = e - s
        if v == treatment_node:
            val = torch.full((n_test,), float(t_value))
            fixed_exo_vec[:, s:e] = t_value
            fixed_exo[v] = val
        else:
            old = obs_scm._fixed_exogenous[v]
            fixed_exo_vec[:, s:e] = old.reshape(n_test, d)
            fixed_exo[v] = old.clone()
    intv_scm._fixed_exogenous_vec = fixed_exo_vec
    intv_scm._fixed_exogenous = fixed_exo
    intv_scm._fixed_batch = n_test

    total_endo = intv_scm._total_endo_dim
    if total_endo == 0:
        fixed_endo_vec = torch.empty(n_test, 0)
        fixed_endo = {}
    else:
        fixed_endo_vec = torch.zeros(n_test, total_endo, dtype=torch.float32)
        fixed_endo = {}
        for v in intv_scm._endo_order:
            s, e = intv_scm._endo_slices[v]
            d = e - s
            old = obs_scm._fixed_endogenous.get(v) if obs_scm._fixed_endogenous else None
            if old is not None:
                fixed_endo_vec[:, s:e] = old.reshape(n_test, d)
                fixed_endo[v] = old.clone()
            else:
                shp = intv_scm._node_shape.get(v, ())
                z = torch.zeros(n_test, *shp) if shp else torch.zeros(n_test)
                fixed_endo_vec[:, s:e] = z.reshape(n_test, d)
                fixed_endo[v] = z
    intv_scm._fixed_endogenous_vec = fixed_endo_vec
    intv_scm._fixed_endogenous = fixed_endo
    return intv_scm.propagate(n_test)


# Reproduce generate_sample(0, seed_base=42) — same seed logic
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
print(f"Y_obs_raw: mean={Y_obs_raw.mean():.4f}, std={Y_obs_raw.std():.4f}, min={Y_obs_raw.min():.4f}, max={Y_obs_raw.max():.4f}")

intv_scm = deepcopy(scm)
intv_scm.intervene(treatment_node)

scm.sample_exogenous(N_TEST)
scm._fixed_endogenous_vec = None
scm.sample_endogenous(N_TEST)
obs_test = scm.propagate(N_TEST)

res0 = transfer_noise_and_propagate(scm, intv_scm, treatment_node, target_node, N_TEST, 0.0)
Y_do0_raw = res0[target_node].reshape(-1, 1).float()

print(f"\nAfter do(T=0) propagation:")
print(f"  res0[target_node] id={id(res0[target_node])}, data_ptr={res0[target_node].data_ptr()}")
print(f"  Y_do0_raw id={id(Y_do0_raw)}, data_ptr={Y_do0_raw.data_ptr()}")
print(f"  Y_do0_raw: mean={Y_do0_raw.mean():.4f}, std={Y_do0_raw.std():.4f}, first5={Y_do0_raw[:5, 0].tolist()}")

# Clone to prevent aliasing
Y_do0_raw_cloned = Y_do0_raw.clone()
print(f"  Y_do0_raw_cloned id={id(Y_do0_raw_cloned)}, data_ptr={Y_do0_raw_cloned.data_ptr()}")

s, e = intv_scm._exo_slices[treatment_node]
intv_scm._fixed_exogenous_vec[:, s:e] = 1.0
intv_scm._fixed_exogenous[treatment_node] = torch.ones(N_TEST)
res1 = intv_scm.propagate(N_TEST)
Y_do1_raw = res1[target_node].reshape(-1, 1).float()

print(f"\nAfter do(T=1) propagation:")
print(f"  res1[target_node] id={id(res1[target_node])}, data_ptr={res1[target_node].data_ptr()}")
print(f"  Y_do1_raw id={id(Y_do1_raw)}, data_ptr={Y_do1_raw.data_ptr()}")
print(f"  Y_do1_raw: mean={Y_do1_raw.mean():.4f}, std={Y_do1_raw.std():.4f}, first5={Y_do1_raw[:5, 0].tolist()}")

print(f"\nCheck if res0 is res1: {res0 is res1}")
print(f"Check if res0[target_node] is res1[target_node]: {res0[target_node] is res1[target_node]}")
print(f"Check Y_do0_raw after second propagation: mean={Y_do0_raw.mean():.4f} (should still be do(T=0))")
print(f"Check Y_do0_raw_cloned: mean={Y_do0_raw_cloned.mean():.4f} (cloned, definitely safe)")

print(f"\nBEFORE preprocessing:")
print(f"  Y_do0_raw: mean={Y_do0_raw.mean():.4f}, std={Y_do0_raw.std():.4f}")
print(f"  Y_do1_raw: mean={Y_do1_raw.mean():.4f}, std={Y_do1_raw.std():.4f}")
print(f"  CATE (raw): mean={(Y_do1_raw - Y_do0_raw).mean():.4f}")

# Preprocessing
Y_obs_clipped = clip_outliers(Y_obs_raw.reshape(-1)).reshape(-1, 1)
print(f"\nY_obs_raw clipped: lo={Y_obs_clipped.min():.4f}, hi={Y_obs_clipped.max():.4f}")
lo = Y_obs_clipped.min()
hi = Y_obs_clipped.max()
rng_val = (hi - lo).clamp(min=EPS)
print(f"Scale params: lo={lo:.4f}, hi={hi:.4f}, rng={rng_val:.4f}")

Y_obs_s, Y_do0_s, Y_do1_s = scale_to_neg1_pos1(Y_obs_clipped, Y_do0_raw, Y_do1_raw)
print(f"\nAFTER scale_to_neg1_pos1:")
print(f"  Y_do0_s: mean={Y_do0_s.mean():.4f}, first5={Y_do0_s[:5, 0].tolist()}")
print(f"  Y_do1_s: mean={Y_do1_s.mean():.4f}, first5={Y_do1_s[:5, 0].tolist()}")
print(f"  CATE (scaled): mean={(Y_do1_s - Y_do0_s).mean():.4f}")

Y_do0_c = Y_do0_s.clamp(-2.0, 2.0)
Y_do1_c = Y_do1_s.clamp(-2.0, 2.0)
print(f"\nAFTER clamp:")
print(f"  Y_do0_c: min={Y_do0_c.min():.4f}, max={Y_do0_c.max():.4f}, mean={Y_do0_c.mean():.4f}")
print(f"  Y_do1_c: min={Y_do1_c.min():.4f}, max={Y_do1_c.max():.4f}, mean={Y_do1_c.mean():.4f}")
print(f"  CATE (final): mean={(Y_do1_c - Y_do0_c).mean():.4f}, std={(Y_do1_c - Y_do0_c).std():.4f}")
