"""SCM sampler from our training prior (Option A).

Delegates to UWYK's SCMSampler + BinarizingMechanism (the prior our model was
trained on) plus the paired-interventional propagation helpers in
`data.PairedInterventionalDataset`. Returns both scaled (model-space [-1,1])
and raw (SCM-native) arrays for a single SCM instance.
"""
from __future__ import annotations

import os
import sys
from copy import deepcopy
from typing import Any

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from training.data.PairedInterventionalDataset import (
    _propagate_paired,
    _standardize,
    _pad_x,
    _contains_nan_or_inf,
    _sample_passes_thresholds,
    DEFAULT_SCM_CONFIG,
)


def generate_paired_sample_with_raw(
    scm_seed: int = 2,
    idx: int = 0,
    n_train: int = 1000,
    n_test: int = 500,
    max_features: int = 50,
    max_outer_attempts: int = 50,
    max_nan_retries: int = 10,
    min_target_variance: float | None = 1e-2,
    min_unique_target_fraction: float | None = 0.2,
    epsilon: float = 1e-8,
    scm_config: dict | None = None,
) -> dict[str, Any]:
    """Draw one paired sample and return scaled + raw Y arrays + affine.

    Returns a dict with:
        # scaled ([-1,1] model-space)
        X_obs, T_obs, Y_obs, X_intv, Y_do0, Y_do1
        # raw (SCM-native units)
        Y_obs_raw, Y_do0_raw, Y_do1_raw
        # affine used to go raw -> scaled
        ymin, ymax
        # bookkeeping
        treatment_node, target_node, feature_nodes, t0_value, t1_value
    """
    from priors.causal_prior.scm.SCMSampler import SCMSampler
    from priors.causal_prior.mechanisms.BinarizingMechanism import BinarizingMechanism

    cfg = scm_config if scm_config is not None else DEFAULT_SCM_CONFIG
    sampler = SCMSampler(cfg, seed=scm_seed * 31 + 17)

    for attempt in range(max_nan_retries):
        seed = scm_seed + idx + attempt * 1_000_000
        torch.manual_seed(seed)

        scm = treatment_node = target_node = None
        feature_nodes: list = []
        obs = T_obs_raw = Y_obs_raw = X_obs_raw = None
        t0_value = t1_value = None

        for outer_attempt in range(max_outer_attempts):
            attempt_seed = seed + outer_attempt * 997
            scm = sampler.sample(seed=attempt_seed)

            all_nodes = sorted(scm.dag.nodes())
            n_nodes = len(all_nodes)
            if n_nodes < 3:
                continue

            rng = torch.Generator(); rng.manual_seed(attempt_seed)
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

            binarised_ok = False
            for _bin_try in range(10):
                scm.sample_exogenous(n_train)
                scm._fixed_endogenous_vec = None
                scm.sample_endogenous(n_train)
                obs_cont = scm.propagate(n_train)
                t_cont = obs_cont[treatment_node].reshape(-1).float()
                try:
                    bin_mech = BinarizingMechanism.from_observational_data(
                        wrapped_mechanism=original_mech, obs_values=t_cont,
                    )
                except ValueError:
                    continue
                scm.mechanisms[treatment_node] = bin_mech
                t0_value = bin_mech.t0
                t1_value = bin_mech.t1

                scm.sample_exogenous(n_train)
                scm._fixed_endogenous_vec = None
                scm.sample_endogenous(n_train)
                obs = scm.propagate(n_train)

                T_obs_raw = obs[treatment_node].reshape(-1, 1).float()
                if T_obs_raw.unique().numel() >= 2:
                    binarised_ok = True
                    break
                scm.mechanisms[treatment_node] = original_mech

            if not binarised_ok:
                continue

            Y_obs_raw = obs[target_node].reshape(-1, 1).float()
            X_obs_raw = (
                torch.cat([obs[n].reshape(n_train, -1).float() for n in feature_nodes], dim=1)
                if feature_nodes else torch.zeros(n_train, 0)
            )
            if Y_obs_raw.var() < 1e-3:
                continue
            if torch.unique(Y_obs_raw).numel() < max(5, int(0.1 * n_train)):
                continue
            break
        else:
            raise RuntimeError(f"scm_prior: gave up after {max_outer_attempts} attempts")

        intv_scm = deepcopy(scm)
        intv_scm.intervene(treatment_node)

        scm.sample_exogenous(n_test)
        scm._fixed_endogenous_vec = None
        scm.sample_endogenous(n_test)
        obs_test = scm.propagate(n_test)

        res0, res1 = _propagate_paired(scm, intv_scm, treatment_node, n_test, t0_value, t1_value)
        Y_do0_raw = res0[target_node].reshape(-1, 1).float()
        Y_do1_raw = res1[target_node].reshape(-1, 1).float()

        X_intv_raw = (
            torch.cat([obs_test[n].reshape(n_test, -1).float() for n in feature_nodes], dim=1)
            if feature_nodes else torch.zeros(n_test, 0)
        )

        y_all = torch.cat([Y_obs_raw.reshape(-1), Y_do0_raw.reshape(-1), Y_do1_raw.reshape(-1)])
        ymin = float(y_all.min())
        ymax = float(y_all.max())
        rng_y = max(ymax - ymin, epsilon)
        Y_obs = 2.0 * (Y_obs_raw - ymin) / rng_y - 1.0
        Y_do0 = 2.0 * (Y_do0_raw - ymin) / rng_y - 1.0
        Y_do1 = 2.0 * (Y_do1_raw - ymin) / rng_y - 1.0

        if X_obs_raw.shape[1] > 0:
            X_obs_s, X_intv_s = _standardize(X_obs_raw, X_intv_raw, eps=epsilon)
        else:
            X_obs_s, X_intv_s = X_obs_raw, X_intv_raw
        X_obs = _pad_x(X_obs_s, max_features)
        X_intv = _pad_x(X_intv_s, max_features)

        T_obs = (T_obs_raw > (t0_value + t1_value) / 2.0).float()

        out = {
            'X_obs': X_obs, 'T_obs': T_obs, 'Y_obs': Y_obs,
            'X_intv': X_intv, 'Y_do0': Y_do0, 'Y_do1': Y_do1,
            'Y_obs_raw': Y_obs_raw, 'Y_do0_raw': Y_do0_raw, 'Y_do1_raw': Y_do1_raw,
            'ymin': ymin, 'ymax': ymax,
            'treatment_node': treatment_node, 'target_node': target_node,
            'feature_nodes': feature_nodes, 't0_value': t0_value, 't1_value': t1_value,
        }

        if _contains_nan_or_inf(out):
            continue
        ok, _reason = _sample_passes_thresholds(
            out, min_target_variance, min_unique_target_fraction,
        )
        if not ok:
            continue

        return out

    raise RuntimeError(f"scm_prior: failed clean sample after {max_nan_retries} attempts")


def sample_as_cate_dataset(scm_seed: int, n_context: int, n_test: int = 50):
    """Wrap generate_paired_sample_with_raw as a CATE_Dataset-like namespace."""
    s = generate_paired_sample_with_raw(
        scm_seed=scm_seed, idx=0, n_train=n_context, n_test=n_test,
    )
    class _CD: pass
    cd = _CD()
    cd.X_train  = s['X_obs']
    cd.t_train  = s['T_obs'].reshape(-1)
    cd.y_train  = s['Y_obs_raw'].reshape(-1)
    cd.X_test   = s['X_intv']
    # per-query CATE = Y_do1_raw - Y_do0_raw (in original y units)
    cd.true_cate = (s['Y_do1_raw'] - s['Y_do0_raw']).reshape(-1)
    return cd, s
