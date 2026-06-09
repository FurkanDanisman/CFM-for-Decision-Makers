"""
Generate 5 sample outputs from the InterventionalDataset using the exact prior
from the complexmech_gcn_softatt.yaml config (the paper's main training config).
Prints a detailed breakdown of every output tensor.
Saves results to /Users/furkandanisman/R-PFN/outputs/
"""
import sys
import os

REPO_SRC = '/tmp/g4cfm/src'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, REPO_SRC)

import torch
from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset

# ── SCM config: exact copy of complexmech_gcn_softatt.yaml ──────────────────
scm_config = {
    "num_nodes": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 2, "high": 51}},
    "graph_edge_prob": {"distribution": "beta", "distribution_parameters": {"alpha": 2.0, "beta": 3.0}},
    "graph_seed": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 0, "high": 100000}},
    "xgboost_prob": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [0.0, 0.1, 0.2, 0.3], "probabilities": [0.9, 0.1, 0.01, 0.001]}
    },
    "mechanism_seed": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 0, "high": 100000}},
    "mlp_nonlins": {"value": "tabicl"},
    "mlp_num_hidden_layers": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [0, 1, 2, 3], "probabilities": [0.875, 0.1, 0.025, 0.01]}
    },
    "mlp_hidden_dim": {
        "distribution": "categorical",
        "distribution_parameters": {
            "choices": [1, 2, 4, 6, 8, 10, 12, 14, 16, 32],
            "probabilities": [0.7, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01]
        }
    },
    "mlp_activation_mode": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": ["pre", "post", "mixed_in"], "probabilities": [0.3, 0.3, 0.3]}
    },
    "mlp_use_batch_norm": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [True, False], "probabilities": [0.5, 0.5]}
    },
    "mlp_node_shape": {"value": (1,)},
    "xgb_node_shape": {"value": (1,)},
    "xgb_num_hidden_layers": {"value": 0},
    "xgb_hidden_dim": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [0, 16, 32, 64]}
    },
    "xgb_activation_mode": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": ["pre", "post", "mixed_in"], "probabilities": [0.33, 0.33, 0.34]}
    },
    "xgb_use_batch_norm": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [True, False], "probabilities": [0.5, 0.5]}
    },
    "xgb_n_training_samples": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [10, 50, 100, 200, 500], "probabilities": [0.1, 0.1, 0.3, 0.4, 0.5]}
    },
    "xgb_add_noise": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [True, False], "probabilities": [0.5, 0.5]}
    },
    "random_additive_std": {"value": True},
    "exo_std_distribution": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": ["gamma", "pareto"], "probabilities": [1.0, 0.0]}
    },
    "endo_std_distribution": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": ["gamma", "pareto"], "probabilities": [1.0, 0.0]}
    },
    "exo_std_mean": {"distribution": "lognormal", "distribution_parameters": {"mean": 1.0, "std": 1.0}},
    "exo_std_std": {"distribution": "uniform", "distribution_parameters": {"low": 0.1, "high": 0.4}},
    "endo_std_mean": {"distribution": "lognormal", "distribution_parameters": {"mean": -3.0, "std": 0.6}},
    "endo_std_std": {"distribution": "uniform", "distribution_parameters": {"low": 0.0, "high": 0.5}},
    "endo_p_zero": {"value": 0.0},
    "noise_mixture_proportions": {"value": [0.33, 0.33, 0.34]},
    "use_exogenous_mechanisms": {"value": True},
    "mechanism_generator_seed": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 0, "high": 100000}},
}

# ── Dataset config ───────────────────────────────────────────────────────────
dataset_config = {
    "dataset_size": {"value": 100_000_000},
    "max_number_samples_per_dataset": {"value": 500},
    "max_number_train_samples_per_dataset": {"value": 1000},
    "max_number_test_samples_per_dataset": {"value": 1000},
    "n_test_samples_per_dataset": {
        "distribution": "discrete_uniform",
        "distribution_parameters": {"low": 1, "high": 500}
    },
    "return_adjacency_matrix": {"value": False},
    "return_ancestor_matrix": {"value": True},
    "use_partial_graph_format": {"value": True},
    "hide_fraction_matrix": {
        "distribution": "uniform",
        "distribution_parameters": {"low": 0.0, "high": 1.0}
    },
    "min_target_variance": {"value": 1e-2},
    "min_unique_target_fraction": {"value": 0.2},
    "max_resample_attempts": {"value": 100},
    "max_number_features": {"value": 50},
    "seed": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 0, "high": 100000}},
    "binarize_treatment_prob": {"value": 0.0},
}

# ── Preprocessing config ─────────────────────────────────────────────────────
preprocessing_config = {
    "dropout_prob": {"value": 0.0},
    "shuffle_data": {"value": True},
    "target_feature": {"value": None},
    "negative_one_one_scaling": {"value": False},
    "remove_outliers": {"value": True},
    "outlier_quantile": {"value": 0.99},
    "yeo_johnson": {"value": False},
    "standardize": {"value": True},
    "y_clip_quantile": {"value": None},
    "eps": {"value": 1e-8},
    "increase_treatment_scale": {"value": False},
    "distribution_rescale_factor": {"value": 0.0},
    "interventional_distribution_type": {"value": "resample"},
    "test_feature_mask_fraction": {"value": 0.0},
    "random_seed": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 0, "high": 100000}},
}

print("Building InterventionalDataset...")
dataset = InterventionalDataset(
    scm_config=scm_config,
    preprocessing_config=preprocessing_config,
    dataset_config=dataset_config,
    seed=42,
)
print("Dataset built. Sampling 5 items...\n")

for i in range(5):
    print("=" * 70)
    print(f"SAMPLE {i}")
    print("=" * 70)
    try:
        out = dataset[i]
        names = ["X_obs", "T_obs", "Y_obs", "X_intv", "T_intv", "Y_intv", "anc_matrix"]
        sample_dict = {}
        for j, t in enumerate(out):
            name = names[j] if j < len(names) else f"extra_{j}"
            sample_dict[name] = t
            if hasattr(t, 'shape'):
                print(f"  {name}: shape={tuple(t.shape)}, dtype={t.dtype}, "
                      f"min={t.min().item():.4f}, max={t.max().item():.4f}, "
                      f"mean={t.float().mean().item():.4f}")
                if name == "anc_matrix":
                    vals, counts = t.unique(return_counts=True)
                    total = t.numel()
                    print(f"           unique values: " +
                          ", ".join(f"{v.item():.0f}({c.item()/total*100:.1f}%)"
                                    for v, c in zip(vals, counts)))
            else:
                print(f"  {name}: {t}")
        torch.save(sample_dict, os.path.join(OUT_DIR, f"sample_{i}.pt"))
        print(f"  -> saved to outputs/sample_{i}.pt")
        print()
    except Exception as e:
        import traceback
        print(f"  ERROR on sample {i}: {e}")
        traceback.print_exc()
        print()

print("Done. All outputs saved to R-PFN/outputs/")
