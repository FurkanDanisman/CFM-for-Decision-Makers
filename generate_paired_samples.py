"""
Generate 5 paired potential outcome samples from the SCM prior.

For each sample:
  - Observational context: (X_obs, T_obs ∈ {0,1}, Y_obs) — n_train rows
  - Query points with paired outcomes: (X_intv, Y_do0, Y_do1) — n_test rows
    where Y_do0 = Y|do(T=0) and Y_do1 = Y|do(T=1), generated from the SAME
    exogenous noise draw ε (i.e., coupled potential outcomes, not independent)

Key differences from generate_5_samples.py:
  1. binarize_treatment_prob = 1.0 → T is always binary
  2. t0=0.0, t1=1.0 exactly (not sampled from quantiles) → T ∈ {0,1} literally
  3. T_obs is NOT standardized → stays as {0, 1}
  4. Two outcome columns: Y_do0 and Y_do1 (no T_intv column)
  5. Y_do0 and Y_do1 share the same ε draw (proper coupling)

Run from repo root:
  cd /tmp/g4cfm/src && python3 /Users/furkandanisman/R-PFN/generate_paired_samples.py
"""
import sys
import os
import torch
import numpy as np
from copy import deepcopy

REPO_SRC = '/tmp/g4cfm/src'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs_paired')
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, REPO_SRC)

from priors.causal_prior.scm.SCMSampler import SCMSampler
from priors.causal_prior.mechanisms.BinarizingMechanism import BinarizingMechanism

# ── SCM config: same as complexmech_gcn_softatt.yaml ─────────────────────────
scm_config = {
    "num_nodes": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 2, "high": 51}},
    "graph_edge_prob": {"distribution": "beta", "distribution_parameters": {"alpha": 2.0, "beta": 3.0}},
    "graph_seed": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 0, "high": 100000}},
    "xgboost_prob": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [0.0, 0.1, 0.2, 0.3], "probabilities": [1.0, 0.0, 0.0, 0.0]}
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

MAX_FEATURES = 50
N_TRAIN = 1000   # observational context size
N_TEST  = 500    # number of query points with paired outcomes
EPS = 1e-8
OUTLIER_Q = 0.99  # clip at 99th percentile

scm_sampler = SCMSampler(scm_config, seed=42 * 31 + 17)


def standardize(X_train, X_test=None, eps=EPS):
    """Zero-mean unit-variance using training statistics."""
    mu  = X_train.mean(0, keepdim=True)
    std = X_train.std(0, keepdim=True).clamp(min=eps)
    X_train_s = (X_train - mu) / std
    if X_test is None:
        return X_train_s
    return X_train_s, (X_test - mu) / std


def scale_to_neg1_pos1(Y_train, Y_test0, Y_test1, eps=EPS):
    """Scale all three Y arrays to [-1, 1] using training min/max."""
    lo = Y_train.min()
    hi = Y_train.max()
    rng = (hi - lo).clamp(min=eps)
    def _scale(y):
        return 2.0 * (y - lo) / rng - 1.0
    return _scale(Y_train), _scale(Y_test0), _scale(Y_test1)


def clip_outliers(t, q=OUTLIER_Q):
    """Clip tensor values to the q-th quantile range."""
    lo = torch.quantile(t.reshape(-1).float(), 1.0 - q)
    hi = torch.quantile(t.reshape(-1).float(), q)
    return t.clamp(lo, hi)


def propagate_paired(obs_scm, intv_scm, treatment_node, n_test, t0_value, t1_value):
    """
    Propagate intv_scm for BOTH do(T=t0_value) and do(T=t1_value) in a single
    doubled batch.

    t0_value / t1_value are the SCM's two binary treatment levels chosen by
    BinarizingMechanism.from_observational_data — sampled from observed T
    quantiles so downstream MLPs see in-distribution inputs. They are remapped
    to model-facing {0,1} by the caller, after propagation.

    Uses a 2*n_test batch: first n_test rows have T=t0_value, last n_test rows
    have T=t1_value. The same ε_k is shared between row k and row k+n_test,
    giving proper coupled potential outcomes.

    Using a mixed batch avoids the batch-norm cancellation bug that occurs when
    all samples have the same treatment value (BN would normalize it away).

    Returns:
        res0: dict of node values for the first n_test rows  (do(T=t0_value))
        res1: dict of node values for the last  n_test rows  (do(T=t1_value))
    """
    B2 = 2 * n_test  # doubled batch

    # --- Exogenous noise: tile n_test → 2*n_test, then override treatment ---
    total_exo = intv_scm._total_exo_dim
    fixed_exo_vec = torch.zeros(B2, total_exo, dtype=torch.float32)
    fixed_exo = {}

    for v in intv_scm._exo_order:
        s, e = intv_scm._exo_slices[v]
        d = e - s
        if v == treatment_node:
            # First half do(T=t0_value), second half do(T=t1_value).
            t_vals = torch.cat([
                torch.full((n_test,), t0_value),
                torch.full((n_test,), t1_value),
            ])
            fixed_exo_vec[:, s:e] = t_vals.reshape(B2, d)
            fixed_exo[v] = t_vals  # shape (B2,) with use_exo_mech=True
        else:
            old = obs_scm._fixed_exogenous[v]      # shape (n_test,) 1-D noise
            tiled = old.repeat(2)                  # shape (2*n_test,)
            fixed_exo_vec[:, s:e] = tiled.reshape(B2, d)
            fixed_exo[v] = tiled

    intv_scm._fixed_exogenous_vec = fixed_exo_vec
    intv_scm._fixed_exogenous = fixed_exo
    intv_scm._fixed_batch = B2

    # --- Endogenous noise: tile n_test → 2*n_test, using same view pattern
    #     as sample_endogenous() so shapes match what _sample_fast expects ---
    total_endo = intv_scm._total_endo_dim
    if total_endo == 0:
        fixed_endo_vec = torch.empty(B2, 0)
    else:
        fixed_endo_vec = torch.zeros(B2, total_endo, dtype=torch.float32)
        for v in intv_scm._endo_order:
            s, e = intv_scm._endo_slices[v]
            d = e - s
            old = obs_scm._fixed_endogenous.get(v) if obs_scm._fixed_endogenous else None
            if old is not None:
                old_flat = old.reshape(n_test, d)      # normalize to (n_test, d)
                fixed_endo_vec[:, s:e] = old_flat.repeat(2, 1)  # tile → (B2, d)
            # else leave zeros

    # Build per-node views exactly like sample_endogenous() does
    fixed_endo = {}
    for v in intv_scm._endo_order:
        s, e = intv_scm._endo_slices[v]
        flat = fixed_endo_vec[:, s:e]          # shape (B2, d)
        shp = intv_scm._node_shape.get(v, ())
        fixed_endo[v] = flat.reshape(B2, *shp) if shp else flat.reshape(B2)

    intv_scm._fixed_endogenous_vec = fixed_endo_vec
    intv_scm._fixed_endogenous = fixed_endo

    # Single forward pass with the doubled batch
    res_full = intv_scm.propagate(B2)

    # Split results back into per-row dicts
    res0 = {v: t[:n_test] for v, t in res_full.items()}
    res1 = {v: t[n_test:] for v, t in res_full.items()}
    return res0, res1


def generate_sample(idx, seed_base=42):
    """Generate one paired-outcome sample."""
    torch.manual_seed(seed_base + idx)
    seed = seed_base + idx

    # Outer loop: try different SCMs + T/Y node pairs until we get:
    #   (a) a causal path from T to Y, and
    #   (b) binary T with both 0s and 1s observed
    for outer_attempt in range(50):
        attempt_seed = seed + outer_attempt * 997

        # ── 1. Sample SCM ─────────────────────────────────────────────────────
        scm = scm_sampler.sample(seed=attempt_seed)

        all_nodes = sorted(scm.dag.nodes())
        n_nodes = len(all_nodes)
        if n_nodes < 3:
            continue  # need at least T, Y, one feature

        # ── 2. Find T and Y with a guaranteed causal path T → Y ──────────────
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
            continue  # try a new SCM

        feature_nodes = [n for n in all_nodes if n != treatment_node and n != target_node]
        original_mech = scm.mechanisms[treatment_node]

        # ── 3. Binarize treatment via UWYK's quantile-based factory ──────────
        # BinarizingMechanism.from_observational_data samples threshold and
        # t0/t1 from observed T quantiles, so downstream MLPs see T values
        # inside the SCM's natural range. T_obs gets remapped to {0,1} for the
        # model AFTER propagation (see step 8).
        # Retry if binarization collapses all T to one value (degenerate SCM).
        binarized_ok = False
        for bin_try in range(10):
            scm.sample_exogenous(N_TRAIN)
            scm._fixed_endogenous_vec = None
            scm.sample_endogenous(N_TRAIN)
            obs_cont = scm.propagate(N_TRAIN)
            t_cont = obs_cont[treatment_node].reshape(-1).float()

            try:
                bin_mech = BinarizingMechanism.from_observational_data(
                    wrapped_mechanism=original_mech,
                    obs_values=t_cont,
                )
            except ValueError:
                # Factory raises when sampled t0 == t1 (SCM emitted constant
                # T despite fresh noise). Retry with new noise; if all 10
                # attempts fail, the outer loop rejects the SCM.
                continue
            scm.mechanisms[treatment_node] = bin_mech
            t0_value = bin_mech.t0
            t1_value = bin_mech.t1

            # ── 4. Re-sample observational data with binary treatment ──────────
            scm.sample_exogenous(N_TRAIN)
            scm._fixed_endogenous_vec = None
            scm.sample_endogenous(N_TRAIN)
            obs = scm.propagate(N_TRAIN)

            T_obs_raw = obs[treatment_node].reshape(-1, 1).float()
            if T_obs_raw.unique().numel() >= 2:
                binarized_ok = True
                break
            # Reset mechanism and try next threshold
            scm.mechanisms[treatment_node] = original_mech

        if not binarized_ok:
            continue  # try a new SCM

        Y_obs_raw = obs[target_node].reshape(-1, 1).float()
        X_obs_raw = torch.cat(
            [obs[n].reshape(N_TRAIN, -1).float() for n in feature_nodes], dim=1
        ) if feature_nodes else torch.zeros(N_TRAIN, 0)

        # Reject if Y has near-zero variance or too few unique values
        if Y_obs_raw.var() < 1e-3:
            continue
        if torch.unique(Y_obs_raw).numel() < max(5, int(0.1 * N_TRAIN)):
            continue

        break  # all checks passed — proceed with this SCM + T/Y pair

    # ── 6. Create intervened SCM + sample query noise ─────────────────────────
    intv_scm = deepcopy(scm)
    intv_scm.intervene(treatment_node)

    # Sample noise for ALL non-intervention nodes via the observational SCM.
    # We sample from obs_scm directly for n_test points; this gives us ε_k
    # for k=1..N_TEST. Then we propagate the intervened SCM with T=0 and T=1
    # using the same ε_k.
    scm.sample_exogenous(N_TEST)
    scm._fixed_endogenous_vec = None   # force re-allocation for new batch size
    scm.sample_endogenous(N_TEST)
    obs_test = scm.propagate(N_TEST)   # pre-intervention covariates x^in_k

    # ── 7. Generate paired Y^do(0) and Y^do(1) ───────────────────────────────
    # Propagate a doubled batch (N_TEST T=0 rows + N_TEST T=1 rows) so that
    # batch norm sees mixed treatment values (avoids BN constant-cancellation).
    # Same ε_k for row k and row k+N_TEST → proper coupled potential outcomes.
    res0, res1 = propagate_paired(scm, intv_scm, treatment_node, N_TEST, t0_value, t1_value)
    Y_do0_raw = res0[target_node].reshape(-1, 1).float()
    Y_do1_raw = res1[target_node].reshape(-1, 1).float()

    # Pre-intervention covariate values for query points
    X_intv_raw = torch.cat(
        [obs_test[n].reshape(N_TEST, -1).float() for n in feature_nodes], dim=1
    ) if feature_nodes else torch.zeros(N_TEST, 0)

    # ── 8. Preprocess ─────────────────────────────────────────────────────────
    # Remove outliers from Y (clip at 99th percentile computed on training Y)
    Y_obs_raw = clip_outliers(Y_obs_raw.reshape(-1)).reshape(-1, 1)

    # Scale Y to [-1, 1] using training min/max; apply same scale to Y_do0, Y_do1
    Y_obs, Y_do0, Y_do1 = scale_to_neg1_pos1(Y_obs_raw, Y_do0_raw, Y_do1_raw)

    # Y_do0 and Y_do1 are NOT clamped — they can go outside [-1,1] because the
    # interventional distribution differs from the observational one. The
    # BarDistribution2D tail regions handle out-of-range values.

    # Standardize X features using training statistics (zero-mean, unit-var)
    n_feat = X_obs_raw.shape[1]
    if n_feat > 0:
        X_obs_s, X_intv_s = standardize(X_obs_raw, X_intv_raw)
    else:
        X_obs_s = X_obs_raw
        X_intv_s = X_intv_raw

    # Pad X to MAX_FEATURES columns with zeros
    def pad_x(x, max_f):
        if x.shape[1] >= max_f:
            return x[:, :max_f]
        pad = torch.zeros(x.shape[0], max_f - x.shape[1])
        return torch.cat([x, pad], dim=1)

    X_obs  = pad_x(X_obs_s,  MAX_FEATURES)
    X_intv = pad_x(X_intv_s, MAX_FEATURES)

    # T_obs_raw holds the SCM's two binary values (t0_value, t1_value sampled
    # from observed T quantiles, NOT necessarily {0,1}). Remap to model-facing
    # {0,1} using the midpoint for float-safe comparison.
    T_obs = (T_obs_raw > (t0_value + t1_value) / 2.0).float()

    # ── 9. Ancestor matrix ────────────────────────────────────────────────────
    try:
        from utils.graph_utils import adjacency_to_ancestor_matrix, propagate_ancestor_knowledge
        # Build node ordering: [T, Y, X_0, ..., X_{L-1}]
        ordered_nodes = [treatment_node, target_node] + feature_nodes
        adj_raw = scm.get_adjacency_matrix(node_order=ordered_nodes)
        anc_raw = adjacency_to_ancestor_matrix(adj_raw)
        # Convert {0,1} → {-1, 1}
        anc = 2.0 * anc_raw.float() - 1.0

        # Randomly hide a fraction of entries (U[0,1])
        hide_frac = torch.rand(1).item()
        L = len(feature_nodes)
        real_n = 2 + L
        rand_mat = torch.rand(real_n, real_n)
        hide_mask = rand_mat < hide_frac
        anc[:real_n, :real_n][hide_mask] = 0.0
        anc = propagate_ancestor_knowledge(anc)

        # Pad to (MAX_FEATURES+2) × (MAX_FEATURES+2)
        target_size = MAX_FEATURES + 2
        if anc.shape[0] < target_size:
            padded = torch.full((target_size, target_size), -1.0)
            padded[:anc.shape[0], :anc.shape[1]] = anc
            anc = padded
    except Exception:
        anc = torch.full((MAX_FEATURES + 2, MAX_FEATURES + 2), 0.0)

    return {
        "X_obs":    X_obs,     # (N_TRAIN, 50)
        "T_obs":    T_obs,     # (N_TRAIN, 1) — binary {0,1}, not standardized
        "Y_obs":    Y_obs,     # (N_TRAIN, 1) — scaled to [-1,1]
        "X_intv":   X_intv,   # (N_TEST, 50)
        "Y_do0":    Y_do0,     # (N_TEST, 1) — Y|do(T=0), same ε as Y_do1
        "Y_do1":    Y_do1,     # (N_TEST, 1) — Y|do(T=1), same ε as Y_do0
        "anc_matrix": anc,     # (52, 52) or (MAX+2, MAX+2)
        "_meta": {
            "treatment_node": treatment_node,
            "target_node": target_node,
            "n_feature_nodes": len(feature_nodes),
            "T_unique": T_obs.unique().tolist(),
        }
    }


print("Generating 5 paired-outcome samples...\n")

for i in range(5):
    print("=" * 70)
    print(f"SAMPLE {i}")
    print("=" * 70)
    try:
        sample = generate_sample(i, seed_base=42)
        meta = sample.pop("_meta")

        # Print tensor shapes and stats
        for name, t in sample.items():
            if torch.is_tensor(t) and t.numel() > 0:
                print(f"  {name}: shape={tuple(t.shape)}, dtype={t.dtype}, "
                      f"min={t.min().item():.4f}, max={t.max().item():.4f}, "
                      f"mean={t.float().mean().item():.4f}")
                if name == "anc_matrix":
                    vals, cnts = t.unique(return_counts=True)
                    total = t.numel()
                    print(f"    unique: " +
                          ", ".join(f"{v.item():.0f}({c.item()/total*100:.1f}%)"
                                    for v, c in zip(vals, cnts)))
                if name == "T_obs":
                    print(f"    unique T values: {t.unique().tolist()}")
                if name in ("Y_do0", "Y_do1"):
                    tau = (sample.get("Y_do1", t) - sample.get("Y_do0", t)).squeeze()
                    if name == "Y_do1":
                        print(f"    CATE τ=Y_do1-Y_do0: mean={tau.mean():.4f}, std={tau.std():.4f}")

        print(f"  meta: treatment={meta['treatment_node']}, target={meta['target_node']}, "
              f"n_features={meta['n_feature_nodes']}")
        print(f"  T unique values: {meta['T_unique']}")

        # Save
        sample["_meta"] = meta
        sample_dir = os.path.join(OUT_DIR, f"sample_{i}")
        os.makedirs(sample_dir, exist_ok=True)

        # Save tensors as CSVs
        tensor_names = ["X_obs", "T_obs", "Y_obs", "X_intv", "Y_do0", "Y_do1", "anc_matrix"]
        for name in tensor_names:
            t = sample.get(name)
            if t is not None and torch.is_tensor(t):
                np.savetxt(
                    os.path.join(sample_dir, f"{name}.csv"),
                    t.float().numpy(),
                    delimiter=",",
                    fmt="%.6f",
                )

        torch.save({k: v for k, v in sample.items() if k != "_meta"},
                   os.path.join(OUT_DIR, f"sample_{i}.pt"))
        print(f"  -> saved to outputs_paired/sample_{i}/")
        print()

    except Exception as e:
        import traceback
        print(f"  ERROR on sample {i}: {e}")
        traceback.print_exc()
        print()

print(f"Done. Outputs in {OUT_DIR}")
