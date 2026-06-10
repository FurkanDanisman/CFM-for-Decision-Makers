"""
Streaming Dataset for paired potential outcomes (Y_do0, Y_do1) from
fresh SCMs at every step.

Refactor of ``generate_paired_samples.py``:
  - Same SCM config, same binarisation, same paired-noise propagation, same
    preprocessing (X standardised, Y scaled to [-1,1], NO clamp).
  - Each ``__getitem__(idx)`` produces a fresh task with `idx` as the
    deterministic seed offset. Designed for ``DataLoader(num_workers=K)`` —
    each worker maintains its own ``SCMSampler``.

Required external code:
  Requires UWYK's ``src/`` on the Python path so we can import:
    - ``priors.causal_prior.scm.SCMSampler``
    - ``priors.causal_prior.mechanisms.BinarizingMechanism``
  Set ``UWYK_SRC`` env var to the path (default: ``/tmp/g4cfm/src``).

Returned tensors per item (no batch dim — DataLoader stacks):
    X_obs   : (n_train, max_features) float
    T_obs   : (n_train, 1)            float in {0,1}
    Y_obs   : (n_train, 1)            float in [-1,1]
    X_intv  : (n_test,  max_features) float
    Y_do0   : (n_test,  1)            float
    Y_do1   : (n_test,  1)            float
    anc_matrix : (max_features+2, max_features+2) float in {-1,0,1}
"""
from __future__ import annotations

import os
import sys
from copy import deepcopy
from typing import Any

import torch
from torch.utils.data import Dataset


# --- UWYK source path (set UWYK_SRC env var to override) ----------------------
_UWYK_SRC = os.environ.get("UWYK_SRC", "/tmp/g4cfm/src")


# --- Default SCM config (verbatim from generate_paired_samples.py) ------------

DEFAULT_SCM_CONFIG = {
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


# --- Helpers (verbatim from generate_paired_samples.py) ------------------------

def _standardize(X_train, X_test=None, eps=1e-8):
    mu = X_train.mean(0, keepdim=True)
    std = X_train.std(0, keepdim=True).clamp(min=eps)
    X_train_s = (X_train - mu) / std
    if X_test is None:
        return X_train_s
    return X_train_s, (X_test - mu) / std


def _scale_to_neg1_pos1(Y_train, Y_test0, Y_test1, eps=1e-8):
    lo = Y_train.min()
    hi = Y_train.max()
    rng = (hi - lo).clamp(min=eps)
    def _scale(y):
        return 2.0 * (y - lo) / rng - 1.0
    return _scale(Y_train), _scale(Y_test0), _scale(Y_test1)


def _clip_outliers(t, q=0.99):
    lo = torch.quantile(t.reshape(-1).float(), 1.0 - q)
    hi = torch.quantile(t.reshape(-1).float(), q)
    return t.clamp(lo, hi)


def _propagate_paired(obs_scm, intv_scm, treatment_node, n_test):
    """Propagate intv_scm for both do(T=0) and do(T=1) in one doubled batch."""
    B2 = 2 * n_test
    total_exo = intv_scm._total_exo_dim
    fixed_exo_vec = torch.zeros(B2, total_exo, dtype=torch.float32)
    fixed_exo: dict[str, torch.Tensor] = {}

    for v in intv_scm._exo_order:
        s, e = intv_scm._exo_slices[v]
        d = e - s
        if v == treatment_node:
            t_vals = torch.cat([torch.zeros(n_test), torch.ones(n_test)])
            fixed_exo_vec[:, s:e] = t_vals.reshape(B2, d)
            fixed_exo[v] = t_vals
        else:
            old = obs_scm._fixed_exogenous[v]
            tiled = old.repeat(2)
            fixed_exo_vec[:, s:e] = tiled.reshape(B2, d)
            fixed_exo[v] = tiled

    intv_scm._fixed_exogenous_vec = fixed_exo_vec
    intv_scm._fixed_exogenous = fixed_exo
    intv_scm._fixed_batch = B2

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
                old_flat = old.reshape(n_test, d)
                fixed_endo_vec[:, s:e] = old_flat.repeat(2, 1)

    fixed_endo: dict[str, torch.Tensor] = {}
    for v in intv_scm._endo_order:
        s, e = intv_scm._endo_slices[v]
        flat = fixed_endo_vec[:, s:e]
        shp = intv_scm._node_shape.get(v, ())
        fixed_endo[v] = flat.reshape(B2, *shp) if shp else flat.reshape(B2)

    intv_scm._fixed_endogenous_vec = fixed_endo_vec
    intv_scm._fixed_endogenous = fixed_endo

    res_full = intv_scm.propagate(B2)
    res0 = {v: t[:n_test] for v, t in res_full.items()}
    res1 = {v: t[n_test:] for v, t in res_full.items()}
    return res0, res1


def _pad_x(x: torch.Tensor, max_f: int) -> torch.Tensor:
    if x.shape[1] >= max_f:
        return x[:, :max_f]
    pad = torch.zeros(x.shape[0], max_f - x.shape[1])
    return torch.cat([x, pad], dim=1)


# --- Dataset class ------------------------------------------------------------

class PairedInterventionalDataset(Dataset):
    """
    Streaming dataset; each __getitem__(idx) samples a fresh SCM task.

    Args:
        scm_config: SCM prior config (defaults to the same one used in
            ``generate_paired_samples.py``).
        n_train: observational context size per task.
        n_test:  query (paired-outcome) count per task.
        max_features: pad/truncate X to this width.
        seed_base: per-process seed offset (the actual per-sample seed is
            ``seed_base + idx + 997·worker_id``).
        max_outer_attempts: retry count when SCM sampling produces a
            degenerate task (no T→Y path, binarisation failure, near-constant
            Y, etc.). Generally 50 is enough.
        outlier_q: Y clipping quantile (0.99).
        epsilon: numerical floor for normalisation.
    """

    def __init__(
        self,
        scm_config: dict | None = None,
        n_train: int = 1000,
        n_test: int = 500,
        max_features: int = 50,
        seed_base: int = 42,
        max_outer_attempts: int = 50,
        outlier_q: float = 0.99,
        epsilon: float = 1e-8,
        infinite_len: int = 10**9,
    ):
        super().__init__()
        if _UWYK_SRC not in sys.path:
            sys.path.insert(0, _UWYK_SRC)
        # Lazy imports — only happen when UWYK source is on the path
        from priors.causal_prior.scm.SCMSampler import SCMSampler          # noqa: F401
        from priors.causal_prior.mechanisms.BinarizingMechanism import BinarizingMechanism  # noqa: F401
        self._SCMSampler = SCMSampler
        self._BinarizingMechanism = BinarizingMechanism

        self.scm_config = scm_config if scm_config is not None else DEFAULT_SCM_CONFIG
        self.n_train = n_train
        self.n_test = n_test
        self.max_features = max_features
        self.seed_base = seed_base
        self.max_outer_attempts = max_outer_attempts
        self.outlier_q = outlier_q
        self.epsilon = epsilon
        self.infinite_len = infinite_len

        # One sampler per process. DataLoader workers each get their own via
        # ``worker_init_fn`` (set in :func:`make_streaming_loader`).
        self._sampler = self._SCMSampler(self.scm_config, seed=seed_base * 31 + 17)

    def __len__(self) -> int:
        return self.infinite_len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Robustness: if a specific SCM in the prior triggers a C-extension
        # bug (numpy/torch refcount error, etc.), retry up to 5 times with
        # different seed offsets so one bad task doesn't crash the worker.
        import gc
        last_err = None
        for retry in range(5):
            try:
                offset = retry * 1_000_003   # large prime, low collision
                out = self._generate_one(idx + offset)
                if retry > 0:
                    # one bad task burned — force GC to clear any partial state
                    gc.collect()
                return out
            except (RuntimeError, ValueError, ArithmeticError, FloatingPointError) as e:
                last_err = e
                gc.collect()
                continue
        raise RuntimeError(
            f"PairedInterventionalDataset.__getitem__({idx}): all 5 retries failed; "
            f"last error: {last_err}"
        )

    # ----- internal -----------------------------------------------------------

    def _generate_one(self, idx: int) -> dict[str, Any]:
        torch.manual_seed(self.seed_base + idx)
        seed = self.seed_base + idx

        scm = None
        treatment_node = None
        target_node = None
        feature_nodes: list = []
        obs = None
        T_obs_raw: torch.Tensor | None = None
        Y_obs_raw: torch.Tensor | None = None
        X_obs_raw: torch.Tensor | None = None

        for outer_attempt in range(self.max_outer_attempts):
            attempt_seed = seed + outer_attempt * 997
            scm = self._sampler.sample(seed=attempt_seed)

            all_nodes = sorted(scm.dag.nodes())
            n_nodes = len(all_nodes)
            if n_nodes < 3:
                continue

            rng = torch.Generator()
            rng.manual_seed(attempt_seed)
            found_pair = False
            for _ in range(30):
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

            # Binarise treatment until we get both classes
            binarised_ok = False
            for bin_try in range(10):
                scm.sample_exogenous(self.n_train)
                scm._fixed_endogenous_vec = None
                scm.sample_endogenous(self.n_train)
                obs_cont = scm.propagate(self.n_train)
                t_cont = obs_cont[treatment_node].reshape(-1).float()
                q = min(0.5 + 0.05 * bin_try, 0.95)
                threshold = torch.quantile(t_cont, q).item()
                bin_mech = self._BinarizingMechanism(
                    wrapped_mechanism=original_mech, threshold=threshold, t0=0.0, t1=1.0,
                )
                scm.mechanisms[treatment_node] = bin_mech

                scm.sample_exogenous(self.n_train)
                scm._fixed_endogenous_vec = None
                scm.sample_endogenous(self.n_train)
                obs = scm.propagate(self.n_train)

                T_obs_raw = obs[treatment_node].reshape(-1, 1).float()
                if T_obs_raw.unique().numel() >= 2:
                    binarised_ok = True
                    break
                scm.mechanisms[treatment_node] = original_mech

            if not binarised_ok:
                continue

            Y_obs_raw = obs[target_node].reshape(-1, 1).float()
            X_obs_raw = (
                torch.cat([obs[n].reshape(self.n_train, -1).float() for n in feature_nodes], dim=1)
                if feature_nodes
                else torch.zeros(self.n_train, 0)
            )

            if Y_obs_raw.var() < 1e-3:
                continue
            if torch.unique(Y_obs_raw).numel() < max(5, int(0.1 * self.n_train)):
                continue
            break
        else:
            # Couldn't find a usable SCM in max_outer_attempts tries
            raise RuntimeError(f"PairedInterventionalDataset: gave up after {self.max_outer_attempts} attempts at idx={idx}")

        # Interventional propagation with the paired-noise trick
        intv_scm = deepcopy(scm)
        intv_scm.intervene(treatment_node)

        scm.sample_exogenous(self.n_test)
        scm._fixed_endogenous_vec = None
        scm.sample_endogenous(self.n_test)
        obs_test = scm.propagate(self.n_test)

        res0, res1 = _propagate_paired(scm, intv_scm, treatment_node, self.n_test)
        Y_do0_raw = res0[target_node].reshape(-1, 1).float()
        Y_do1_raw = res1[target_node].reshape(-1, 1).float()

        X_intv_raw = (
            torch.cat([obs_test[n].reshape(self.n_test, -1).float() for n in feature_nodes], dim=1)
            if feature_nodes
            else torch.zeros(self.n_test, 0)
        )

        # --- preprocess (same as generate_paired_samples.py) ----------
        Y_obs_clipped = _clip_outliers(Y_obs_raw.reshape(-1), q=self.outlier_q).reshape(-1, 1)
        Y_obs, Y_do0, Y_do1 = _scale_to_neg1_pos1(Y_obs_clipped, Y_do0_raw, Y_do1_raw, eps=self.epsilon)

        if X_obs_raw.shape[1] > 0:
            X_obs_s, X_intv_s = _standardize(X_obs_raw, X_intv_raw, eps=self.epsilon)
        else:
            X_obs_s = X_obs_raw
            X_intv_s = X_intv_raw

        X_obs = _pad_x(X_obs_s, self.max_features)
        X_intv = _pad_x(X_intv_s, self.max_features)

        T_obs = T_obs_raw.clamp(0.0, 1.0)

        # --- ancestor matrix -------------------------------------------------
        try:
            from utils.graph_utils import (  # type: ignore
                adjacency_to_ancestor_matrix,
                propagate_ancestor_knowledge,
            )
            ordered_nodes = [treatment_node, target_node] + feature_nodes
            adj_raw = scm.get_adjacency_matrix(node_order=ordered_nodes)
            anc_raw = adjacency_to_ancestor_matrix(adj_raw)
            anc = 2.0 * anc_raw.float() - 1.0

            hide_frac = torch.rand(1).item()
            L = len(feature_nodes)
            real_n = 2 + L
            rand_mat = torch.rand(real_n, real_n)
            hide_mask = rand_mat < hide_frac
            anc[:real_n, :real_n][hide_mask] = 0.0
            anc = propagate_ancestor_knowledge(anc)

            target_size = self.max_features + 2
            if anc.shape[0] < target_size:
                padded = torch.full((target_size, target_size), -1.0)
                padded[:anc.shape[0], :anc.shape[1]] = anc
                anc = padded
        except Exception:
            anc = torch.full((self.max_features + 2, self.max_features + 2), 0.0)

        return {
            "X_obs":   X_obs,
            "T_obs":   T_obs,
            "Y_obs":   Y_obs,
            "X_intv":  X_intv,
            "Y_do0":   Y_do0,
            "Y_do1":   Y_do1,
            "anc_matrix": anc,
        }


# --- DataLoader factory with worker-aware seeding ------------------------------

def _worker_init_fn(worker_id: int):
    """Reseed each worker's sampler so we don't repeat the same SCMs."""
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    if isinstance(ds, PairedInterventionalDataset):
        # New sampler with a worker-distinct seed
        ds._sampler = ds._SCMSampler(ds.scm_config, seed=(ds.seed_base + worker_id) * 31 + 17)


def make_streaming_loader(
    batch_size: int = 1,
    num_workers: int = 4,
    scm_config: dict | None = None,
    **dataset_kwargs,
) -> torch.utils.data.DataLoader:
    """
    Convenience constructor. Returns a DataLoader yielding dicts of stacked
    tensors with leading batch dim ``batch_size``.
    """
    ds = PairedInterventionalDataset(scm_config=scm_config, **dataset_kwargs)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        # persistent_workers=False so workers can restart if one dies (e.g.,
        # from a transient C-extension crash) instead of taking the whole
        # training run down.
        persistent_workers=False,
        pin_memory=True,
    )
