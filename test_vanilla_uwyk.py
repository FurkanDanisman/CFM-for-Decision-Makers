"""
Run UWYK's vanilla InterventionalDataset for N_ITERATIONS iterations.

This is a control experiment. NO modifications to UWYK's pipeline,
NO private-state mutation, NO deepcopy, NO `_propagate_paired`, NO
paired potential outcomes — just UWYK's published, supported,
public-API-only `InterventionalDataset.__getitem__(i)` in a tight loop.

faulthandler is enabled so any C-level crash gives us a real stack
trace including library / function names.

Decision matrix after running this:

    Cleanly completes 25K iterations
        -> The bug is in OUR code (paired-outcome mutation in
           `_propagate_paired`). UWYK's library + the cluster
           environment are fine.

    Crashes with `Fatal Python error: none_dealloc`
        -> The bug is in UWYK / its dependencies / the cluster env,
           NOT in our paired-propagation modifications. Different fix
           path entirely.

For context: with our code (which mutates UWYK's private state),
crashes on the cluster happen between ~3,000 and ~20,000 samples.
25K is comfortably past that range.
"""
import faulthandler
faulthandler.enable()

import os
import sys
import time

UWYK_SRC = os.environ.get("UWYK_SRC", "/scratch/furkanbd/g4cfm/src")
sys.path.insert(0, UWYK_SRC)

from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset


# === SCM config — verbatim from generate_paired_samples.py (the SCM prior we train on) ===
SCM_CONFIG = {
    "num_nodes": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 2, "high": 51}},
    "graph_edge_prob": {"distribution": "beta", "distribution_parameters": {"alpha": 2.0, "beta": 3.0}},
    "graph_seed": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 0, "high": 100000}},
    "xgboost_prob": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [0.0, 0.1, 0.2, 0.3], "probabilities": [1.0, 0.0, 0.0, 0.0]},
    },
    "mechanism_seed": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 0, "high": 100000}},
    "mlp_nonlins": {"value": "tabicl"},
    "mlp_num_hidden_layers": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [0, 1, 2, 3], "probabilities": [0.875, 0.1, 0.025, 0.01]},
    },
    "mlp_hidden_dim": {
        "distribution": "categorical",
        "distribution_parameters": {
            "choices": [1, 2, 4, 6, 8, 10, 12, 14, 16, 32],
            "probabilities": [0.7, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01],
        },
    },
    "mlp_activation_mode": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": ["pre", "post", "mixed_in"], "probabilities": [0.3, 0.3, 0.3]},
    },
    "mlp_use_batch_norm": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [True, False], "probabilities": [0.5, 0.5]},
    },
    "mlp_node_shape": {"value": (1,)},
    "xgb_node_shape": {"value": (1,)},
    "xgb_num_hidden_layers": {"value": 0},
    "xgb_hidden_dim": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [0, 16, 32, 64]},
    },
    "xgb_activation_mode": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": ["pre", "post", "mixed_in"], "probabilities": [0.33, 0.33, 0.34]},
    },
    "xgb_use_batch_norm": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [True, False], "probabilities": [0.5, 0.5]},
    },
    "xgb_n_training_samples": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [10, 50, 100, 200, 500], "probabilities": [0.1, 0.1, 0.3, 0.4, 0.5]},
    },
    "xgb_add_noise": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [True, False], "probabilities": [0.5, 0.5]},
    },
    "random_additive_std": {"value": True},
    "exo_std_distribution": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": ["gamma", "pareto"], "probabilities": [1.0, 0.0]},
    },
    "endo_std_distribution": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": ["gamma", "pareto"], "probabilities": [1.0, 0.0]},
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

# === preprocessing_config — verbatim from UWYK complexmech_gcn_softatt.yaml ===
PREPROCESSING_CONFIG = {
    "dropout_prob": {"value": 0.0},
    "shuffle_data": {"value": True},
    "target_feature": {"value": None},
    "feature_standardize": {"value": True},
    "feature_negative_one_one_scaling": {"value": False},
    "target_negative_one_one_scaling": {"value": True},
    "remove_outliers": {"value": True},
    "outlier_quantile": {"value": 0.99},
    "yeo_johnson": {"value": False},
    "increase_treatment_scale": {"value": False},
    "distribution_rescale_factor": {"value": 0.0},
    "interventional_distribution_type": {"value": "resample"},
    "test_feature_mask_fraction": {"value": 0.0},
    "random_seed": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 0, "high": 100000}},
}

# === dataset_config — verbatim from UWYK complexmech_gcn_softatt.yaml ===
DATASET_CONFIG = {
    "dataset_size": {"value": 100000000},
    "max_number_features": {"value": 50},
    "max_number_samples_per_dataset": {"value": 500},
    "max_number_train_samples_per_dataset": {"value": 1000},
    "max_number_test_samples_per_dataset": {"value": 1000},
    "n_test_samples_per_dataset": {
        "distribution": "discrete_uniform",
        "distribution_parameters": {"low": 1, "high": 500},
    },
    "return_adjacency_matrix": {"value": False},
    "return_ancestor_matrix": {"value": True},
    "use_partial_graph_format": {"value": True},
    "hide_fraction_matrix": {
        "distribution": "uniform",
        "distribution_parameters": {"low": 0.0, "high": 1.0},
    },
    "min_target_variance": {"value": 1e-2},
    "min_unique_target_fraction": {"value": 0.2},
    "max_resample_attempts": {"value": 100},
}


N_ITERATIONS = int(os.environ.get("N_ITERATIONS", 25000))


def main():
    print(f"PID         : {os.getpid()}", flush=True)
    print(f"UWYK_SRC    : {UWYK_SRC}", flush=True)
    print(f"N_ITERATIONS: {N_ITERATIONS}", flush=True)
    print(f"faulthandler enabled: {faulthandler.is_enabled()}", flush=True)

    ds = InterventionalDataset(
        scm_config=SCM_CONFIG,
        preprocessing_config=PREPROCESSING_CONFIG,
        dataset_config=DATASET_CONFIG,
        seed=42,
    )
    print("InterventionalDataset constructed.\n", flush=True)

    t0 = time.time()
    last_print = t0
    ok = 0
    py_err = 0

    # Y statistics — to see whether UWYK's data prior occasionally emits
    # extreme |Y_intv| values that would explain the loss spikes
    # (e.g., target at |Y|=100 + tail sigma=0.02 -> density ~ exp(-1e7) -> loss ~ 1e7).
    y_intv_abs_max = 0.0
    y_intv_max_seen_idx = -1
    n_yi_over_5    = 0
    n_yi_over_50   = 0
    n_yi_over_500  = 0
    n_yi_over_5000 = 0

    for i in range(N_ITERATIONS):
        try:
            sample = ds[i]
            ok += 1

            # Sample is a tuple — extract Y_intv (the interventional outcome).
            # Default ordering for InterventionalDataset is:
            #   (X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, [anc])
            # The 6th element (index 5) is Y_intv.
            y_intv = sample[5]
            y_abs = float(y_intv.abs().max().item())

            if y_abs > y_intv_abs_max:
                y_intv_abs_max = y_abs
                y_intv_max_seen_idx = i
            if y_abs > 5:    n_yi_over_5    += 1
            if y_abs > 50:   n_yi_over_50   += 1
            if y_abs > 500:  n_yi_over_500  += 1
            if y_abs > 5000: n_yi_over_5000 += 1
        except Exception as e:
            # Non-fatal Python exception (e.g., a rejected SCM). Log and continue.
            py_err += 1
            if py_err <= 10:
                print(f"  py exc idx={i}: {type(e).__name__}: {str(e)[:200]}", flush=True)

        now = time.time()
        if now - last_print > 30 or i == N_ITERATIONS - 1:
            rate = (i + 1) / (now - t0)
            eta = (N_ITERATIONS - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{int(now - t0):>5}s] {i + 1:>6}/{N_ITERATIONS}  "
                f"ok={ok}  py_err={py_err}  rate={rate:.1f}/s  ETA={eta / 60:.1f}min  "
                f"|Y_intv|max={y_intv_abs_max:.2f}  "
                f">5:{n_yi_over_5} >50:{n_yi_over_50} >500:{n_yi_over_500} >5000:{n_yi_over_5000}",
                flush=True,
            )
            last_print = now

    print(f"\nDONE: {ok}/{N_ITERATIONS} ok, {py_err} python errors, "
          f"total {time.time() - t0:.1f}s — NO fatal crash", flush=True)
    print(f"\nY_intv extremity summary:", flush=True)
    print(f"  global max |Y_intv| = {y_intv_abs_max:.4f}  (first seen at idx {y_intv_max_seen_idx})", flush=True)
    print(f"  samples with |Y_intv| > 5    : {n_yi_over_5}  ({100*n_yi_over_5/max(1,ok):.2f}%)", flush=True)
    print(f"  samples with |Y_intv| > 50   : {n_yi_over_50}  ({100*n_yi_over_50/max(1,ok):.2f}%)", flush=True)
    print(f"  samples with |Y_intv| > 500  : {n_yi_over_500}  ({100*n_yi_over_500/max(1,ok):.2f}%)", flush=True)
    print(f"  samples with |Y_intv| > 5000 : {n_yi_over_5000}  ({100*n_yi_over_5000/max(1,ok):.2f}%)", flush=True)


if __name__ == "__main__":
    main()
