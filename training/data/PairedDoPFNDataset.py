"""Paired potential outcomes generated from Do-PFN's original SCM prior.

CFM owns the marginal-to-joint conversion while reusing Do-PFN's graph,
mechanism, and noise APIs unchanged. Set ``DOPFN_SRC`` to the Do-PFN repository
root (default: ``/tmp/dopfn``).

Each item follows CFM's joint-training contract (no batch dimension):

    X_obs  : (n_train, num_features)
    T_obs  : (n_train, 1), Do-PFN model-facing values in {0, 1}
    Y_obs  : (n_train, 1), context-scaled to [-1, 1]
    X_intv : (n_test, num_features), factual pre-intervention covariates
    Y_do0  : (n_test, 1), same context-derived Y transform
    Y_do1  : (n_test, 1), same context-derived Y transform
"""

from __future__ import annotations

from copy import deepcopy
import os
import random
import sys
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset


_DOPFN_SRC = os.environ.get("DOPFN_SRC", "/tmp/dopfn")

DEFAULT_DOPFN_CONFIG: dict[str, Any] = {
    "num_unobserved": 1,
    "noise_std": 0.01,
    "noise_dist": "gaussian",
    "exo_std": 0.1,
    "exo_dist": "gaussian",
    "nonlins": "mixed",
    "max_hidden_layers": 0,
    "graph": None,
    "t_idx": None,
    "y_idx": None,
    "x_idcs": None,
    "binary_strategy": "extreme",
    "ensure_treatment_outcome_path": True,
}


def _load_dopfn_components():
    """Import Do-PFN lazily and reject ambiguous top-level package collisions."""
    loaded_priors = sys.modules.get("priors")
    loaded_path = getattr(loaded_priors, "__file__", "") if loaded_priors else ""
    if loaded_path and not os.path.abspath(loaded_path).startswith(
        os.path.abspath(_DOPFN_SRC)
    ):
        raise RuntimeError(
            "A different top-level 'priors' package is already loaded. Run the "
            "Do-PFN CFM data pipeline in a fresh process to avoid a collision "
            "with Graphs4CausalFoundationModels."
        )
    if _DOPFN_SRC not in sys.path:
        sys.path.insert(0, _DOPFN_SRC)

    from priors.playground_scm.MakeStructuralEquations import (
        MakeStructuralEquations,
        make_additive_noise_gaussian,
        make_additive_noise_mixed,
    )
    from priors.playground_scm.generators import SCMGenerator

    return (
        SCMGenerator,
        MakeStructuralEquations,
        make_additive_noise_gaussian,
        make_additive_noise_mixed,
    )


def _make_exogenous_sampler(name, shape, std, gaussian_factory, mixed_factory):
    if name == "gaussian":
        return gaussian_factory(shape, std)
    proportions = {
        "mixed": [0.25, 0.25, 0.25, 0.25],
        "laplace": [0.0, 1.0, 0.0, 0.0],
        "student": [0.0, 0.0, 1.0, 0.0],
        "gumbel": [0.0, 0.0, 0.0, 1.0],
    }
    if name not in proportions:
        raise ValueError(f"Unsupported Do-PFN exo_dist {name!r}")
    return mixed_factory(
        shape=shape,
        std=std,
        mixture_proportions=proportions[name],
    )


def _node_at(nodes: list[str], index: int, role: str) -> str:
    matches = [node for node in nodes if node[1:] == str(index)]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely resolve {role} index {index}: {matches}")
    return matches[0]


def _select_treatment_outcome(graph, config):
    nodes = list(graph.nodes)
    ensure_path = bool(config["ensure_treatment_outcome_path"])
    if config["t_idx"] is None:
        candidates = [n for n in nodes if not ensure_path or graph.out_degree(n) > 0]
        if not candidates:
            raise ValueError("Sampled graph has no treatment with descendants")
        treatment = random.choice(candidates)
    else:
        treatment = _node_at(nodes, int(config["t_idx"]), "treatment")

    descendants = list(nx.descendants(graph, treatment))
    if config["y_idx"] is None:
        candidates = descendants if ensure_path else [n for n in nodes if n != treatment]
        if not candidates:
            raise ValueError("Selected treatment has no outcome candidate")
        outcome = random.choice(candidates)
    else:
        outcome = _node_at(nodes, int(config["y_idx"]), "outcome")
    if outcome == treatment or (ensure_path and outcome not in descendants):
        raise ValueError("Treatment and outcome do not satisfy the requested path")
    return treatment, outcome


def _select_features(graph, treatment, outcome, num_features, x_idcs):
    candidates = [n for n in graph.nodes if n not in {treatment, outcome}]
    if x_idcs is None:
        if len(candidates) < num_features:
            raise ValueError("Sampled SCM does not contain enough covariate nodes")
        return list(np.random.choice(candidates, num_features, replace=False))
    selected = [_node_at(candidates, int(index), "covariate") for index in x_idcs]
    if len(selected) != num_features:
        raise ValueError("x_idcs must select exactly num_features covariates")
    return selected


def _clone_tensors(values):
    return {key: value.clone() for key, value in values.items()}


def _propagate_arm(scm, graph, treatment, value, shared_exogenous):
    """Propagate one arm while preserving all non-treatment SCM noise."""
    exogenous = _clone_tensors(shared_exogenous)
    is_endogenous = treatment in scm.endogenous_vars
    if is_endogenous:
        scm.do_interventions([(treatment, (lambda value=value: value, {}))])
    else:
        exogenous[treatment] = value.clone()
    try:
        endogenous, exogenous_out = scm.get_next_sample(
            exogenous_vars=exogenous,
            graph=graph,
        )
        return endogenous | exogenous_out
    finally:
        if is_endogenous:
            scm.undo_interventions()


def _matrix(sample, keys, row_slice):
    if not keys:
        return torch.zeros(row_slice.stop - row_slice.start, 0)
    # Do-PFN SCM values have shape (batch=1, samples).
    return torch.stack([sample[key][0, row_slice] for key in keys], dim=-1).float()


def _context_standardize(X_obs, X_intv, eps):
    if X_obs.shape[1] == 0:
        return X_obs, X_intv
    mean = X_obs.mean(0, keepdim=True)
    std = X_obs.std(0, keepdim=True, unbiased=False).clamp_min(eps)
    return (X_obs - mean) / std, (X_intv - mean) / std


def _context_scale_y(Y_obs, Y_do0, Y_do1, eps):
    """Fit the deployable outcome transform on observational context only."""
    lower = Y_obs.min()
    span = (Y_obs.max() - lower).clamp_min(eps)

    def transform(value):
        return 2.0 * (value - lower) / span - 1.0

    return transform(Y_obs), transform(Y_do0), transform(Y_do1)


def _finite(sample):
    return all(torch.isfinite(value).all() for value in sample.values())


def _passes_thresholds(sample, min_variance, min_unique_fraction):
    for key in ("Y_obs", "Y_do0", "Y_do1"):
        value = sample[key].reshape(-1)
        if (
            min_variance is not None
            and float(value.var(unbiased=False)) < min_variance
        ):
            return False
        if (
            min_unique_fraction is not None
            and torch.unique(value).numel() / value.numel() < min_unique_fraction
        ):
            return False
    return True


class PairedDoPFNDataset(Dataset):
    """Streaming CFM dataset backed by Do-PFN's original SCM prior."""

    def __init__(
        self,
        dopfn_config: dict[str, Any] | None = None,
        n_train: int = 1000,
        n_test: int = 500,
        num_features: int = 10,
        seed_base: int = 42,
        max_sampling_attempts: int = 50,
        epsilon: float = 1e-8,
        min_target_variance: float | None = 1e-2,
        min_unique_target_fraction: float | None = 0.2,
        infinite_len: int = 10**9,
    ):
        super().__init__()
        if n_train < 2 or n_test < 1 or num_features < 1:
            raise ValueError("Expected n_train>=2, n_test>=1, num_features>=1")
        self.config = dict(DEFAULT_DOPFN_CONFIG)
        if dopfn_config:
            self.config.update(dopfn_config)
        self.n_train = n_train
        self.n_test = n_test
        self.num_features = num_features
        self.seed_base = seed_base
        self.max_sampling_attempts = max_sampling_attempts
        self.epsilon = epsilon
        self.min_target_variance = min_target_variance
        self.min_unique_target_fraction = min_unique_target_fraction
        self.infinite_len = infinite_len
        self._components = _load_dopfn_components()
        self.last_debug: dict[str, Any] | None = None

    def __len__(self):
        return self.infinite_len

    def _build_scm(self, seed):
        (
            SCMGenerator,
            MakeStructuralEquations,
            make_additive_noise_gaussian,
            make_additive_noise_mixed,
        ) = self._components
        config = self.config
        total_samples = self.n_train + self.n_test
        shape = (1, total_samples)
        generator = SCMGenerator(
            all_functions={"nonlinear": MakeStructuralEquations},
            seed=seed,
            samples_shape=shape,
            noise_std=float(config["noise_std"]),
            noise_dist=str(config["noise_dist"]),
            nonlins=str(config["nonlins"]),
            max_hidden_layers=int(config["max_hidden_layers"]),
        )
        if config["graph"] is None:
            n_nodes = self.num_features + int(round(config["num_unobserved"])) + 2
            edge_min = 1.0 / (self.num_features + 1)
            graph = generator.create_graph_from_nodes(
                n_nodes,
                float(np.random.uniform(edge_min, 1.0)),
            )
        else:
            graph = deepcopy(config["graph"])  # Do-PFN relabels it in place.
        exogenous_sampler = _make_exogenous_sampler(
            str(config["exo_dist"]),
            shape,
            float(config["exo_std"]),
            make_additive_noise_gaussian,
            make_additive_noise_mixed,
        )
        scm = generator.create_scm_from_graph(
            graph,
            possible_functions=["nonlinear"],
            exo_distribution=exogenous_sampler,
            exo_distribution_kwargs={},
        )
        # Do-PFN's SCM accesses this dynamically during treatment binarization.
        scm.binary_strategy = str(config["binary_strategy"])
        return scm, scm.create_graph()

    def _generate_one(self, seed):
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        scm, graph = self._build_scm(seed)
        treatment, outcome = _select_treatment_outcome(graph, self.config)
        features = _select_features(
            graph,
            treatment,
            outcome,
            self.num_features,
            self.config["x_idcs"],
        )
        scm.t_key, scm.y_key = treatment, outcome
        endo_obs, exo_obs = scm.get_next_sample(binarize=True, graph=graph)
        obs = endo_obs | exo_obs

        total_samples = self.n_train + self.n_test
        # Do-PFN encodes its lower raw treatment level as model-facing 1 and
        # its upper level as 0 (StructuralCausalModel.get_zero_one_treatment).
        raw_do0 = scm.t2s[:, None].expand(-1, total_samples)
        raw_do1 = scm.t1s[:, None].expand(-1, total_samples)
        if torch.any(raw_do0 == raw_do1):
            raise ValueError("Do-PFN binarization produced identical treatment levels")
        do0 = _propagate_arm(scm, graph, treatment, raw_do0, exo_obs)
        do1 = _propagate_arm(scm, graph, treatment, raw_do1, exo_obs)

        context = slice(0, self.n_train)
        query = slice(self.n_train, total_samples)
        X_obs = _matrix(obs, features, context)
        X_intv = _matrix(obs, features, query)
        T_obs_raw = obs[treatment][0, context].reshape(-1, 1).float()
        midpoint = ((scm.t1s + scm.t2s) / 2).reshape(1, 1)
        T_obs = (T_obs_raw < midpoint).float()
        if T_obs.unique().numel() != 2:
            raise ValueError("Observational context does not contain both treatments")

        Y_obs = obs[outcome][0, context].reshape(-1, 1).float()
        Y_do0 = do0[outcome][0, query].reshape(-1, 1).float()
        Y_do1 = do1[outcome][0, query].reshape(-1, 1).float()
        X_obs, X_intv = _context_standardize(X_obs, X_intv, self.epsilon)
        Y_obs, Y_do0, Y_do1 = _context_scale_y(
            Y_obs,
            Y_do0,
            Y_do1,
            self.epsilon,
        )
        sample = { 
            "X_obs": X_obs,
            "T_obs": T_obs,
            "Y_obs": Y_obs,
            "X_intv": X_intv,
            "Y_do0": Y_do0,
            "Y_do1": Y_do1,
        }
        self.last_debug = {
            "scm": scm,
            "graph": graph,
            "treatment": treatment,
            "outcome": outcome,
            "features": features,
            "obs": obs,
            "do0": do0,
            "do1": do1,
            "raw_do0": raw_do0,
            "raw_do1": raw_do1,
        }
        return sample # we know this is correct from subsample task

    def __getitem__(self, index):
        last_error = None
        # Most recent NaN/Inf-free sample, even if it failed the variance /
        # unique-fraction thresholds. Mirrors PairedInterventionalDataset's
        # "use last finite sample" fallback (UWYK InterventionalDataset lines
        # 1124-1136): we would rather ship an imperfect-but-finite task than
        # crash a streaming DataLoader worker when thresholds never pass.
        last_finite = None
        for attempt in range(self.max_sampling_attempts):
            seed = self.seed_base + int(index) + attempt * 1_000_003
            try:
                sample = self._generate_one(seed)
            except (AssertionError, ValueError, RuntimeError, IndexError) as error:
                # Do-PFN's binarization can raise AssertionError (`assert t1 != t2`
                # in set_binarization_params) as well as ValueError; treat every
                # such degenerate draw as a retry rather than letting it escape.
                last_error = error
                continue
            if not _finite(sample):
                continue
            last_finite = sample
            if _passes_thresholds(
                sample,
                self.min_target_variance,
                self.min_unique_target_fraction,
            ):
                return sample
        if last_finite is not None:
            return last_finite
        raise RuntimeError(
            f"Unable to generate paired Do-PFN task for index {index} after "
            f"{self.max_sampling_attempts} attempts"
        ) from last_error


def make_dopfn_streaming_loader(
    batch_size: int = 1,
    num_workers: int = 4,
    **dataset_kwargs,
):
    """Create a DataLoader with the same batched dict contract as CFM training."""
    dataset = PairedDoPFNDataset(**dataset_kwargs)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
