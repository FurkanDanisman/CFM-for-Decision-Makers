"""R-PFN paired-outcome sampler with a controllable ρ knob.

This mirrors `context_sweep/scm_prior.py`'s `sample_as_cate_dataset` but
replaces the internal `_propagate_paired` call — which tiles the exact
same exogenous noise into both halves of the interventional batch
(shared ε, high natural ρ) — with a mixed-noise variant that draws
correlated exogenous noise per (target_ρ, seed).

The mix formula, per exogenous slot:
    ε_top = obs_scm._fixed_exogenous[v]                (length n_test)
    ε_fresh ~ N(0, std(ε_top))                          (length n_test)
    ε_bot = ρ * ε_top + sqrt(1 - ρ²) * ε_fresh         (length n_test)

This preserves ε_top's marginal (up to first two moments) and gives ε_bot
a matched marginal with correlation ρ to ε_top. For Gaussian exogenous
noise the mix is exact; for other distributions the marginal is preserved
in first two moments only. The resulting output-level
ρ(Y_do0, Y_do1 | X) will not equal `target_rho` exactly for nonlinear
SCMs but tracks it monotonically — we measure the empirical ρ per SCM
and use *that* as the x-axis in the scaling test.
"""
from __future__ import annotations
import os, sys
from copy import deepcopy

import numpy as np
import torch


# ── Path sanity — R-PFN first, then UWYK src ─────────────────────────────
def _bootstrap_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    uwyk_src = os.environ.get(
        'UWYK_SRC',
        '/Users/furkandanisman/.claude/jobs/7758df90/tmp/uwyk_upstream/src',
    )
    for p in (uwyk_src, repo_root):
        if os.path.isdir(p) and p in sys.path: sys.path.remove(p)
    for p in (uwyk_src, repo_root):
        if os.path.isdir(p): sys.path.insert(0, p)
_bootstrap_paths()

from training.data.PairedInterventionalDataset import (
    _pad_x, _standardize, _contains_nan_or_inf, _sample_passes_thresholds,
    DEFAULT_SCM_CONFIG,
)


def _propagate_paired_with_rho(obs_scm, intv_scm, treatment_node,
                                 n_test, t0_value, t1_value,
                                 target_rho: float, seed: int):
    """Drop-in replacement for `_propagate_paired` with a ρ knob.

    ρ = 1.0 recovers the original shared-noise behaviour (default in
    R-PFN training). Smaller ρ mixes in fresh noise per exogenous slot.
    """
    B2 = 2 * n_test
    total_exo = intv_scm._total_exo_dim
    fixed_exo_vec = torch.zeros(B2, total_exo, dtype=torch.float32)

    gen = torch.Generator(); gen.manual_seed(seed * 7919 + 31)
    rho = float(np.clip(target_rho, -0.999, 0.999))
    mix_a = rho
    mix_b = float(np.sqrt(max(0.0, 1.0 - rho * rho)))

    for v in intv_scm._exo_order:
        s, e = intv_scm._exo_slices[v]
        d = e - s
        if v == treatment_node:
            t_vals = torch.cat([
                torch.full((n_test,), t0_value),
                torch.full((n_test,), t1_value),
            ])
            fixed_exo_vec[:, s:e] = t_vals.reshape(B2, d)
        else:
            old = obs_scm._fixed_exogenous[v]
            eps_top = old.reshape(n_test, d) if old.dim() > 1 else old.reshape(n_test, 1)
            # ε_fresh sampled with the same marginal std as ε_top so
            # ε_bot preserves the empirical variance up to O(1/n_test).
            fresh = torch.randn((n_test, d), generator=gen) * eps_top.std().clamp(min=1e-6)
            eps_bot = mix_a * eps_top + mix_b * fresh
            top = eps_top.reshape(n_test, d)
            bot = eps_bot.reshape(n_test, d)
            stacked = torch.cat([top, bot], dim=0)
            fixed_exo_vec[:, s:e] = stacked.reshape(B2, d)

    fixed_exo: dict[str, torch.Tensor] = {}
    for v in intv_scm._exo_order:
        s, e = intv_scm._exo_slices[v]
        flat = fixed_exo_vec[:, s:e]
        if intv_scm.use_exogenous_mechanisms:
            fixed_exo[v] = flat.reshape(B2)
        else:
            shp = intv_scm._node_shape.get(v, ())
            fixed_exo[v] = flat.reshape(B2, *shp) if shp else flat.reshape(B2)

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


def sample_as_cate_dataset_with_rho(
    scm_seed: int,
    n_context: int,
    n_test: int = 50,
    target_rho: float = 1.0,
    max_features: int = 50,
    max_outer_attempts: int = 50,
    max_nan_retries: int = 10,
    min_target_variance: float | None = 1e-2,
    min_unique_target_fraction: float | None = 0.2,
    epsilon: float = 1e-8,
    scm_config: dict | None = None,
):
    """Sample one paired SCM from R-PFN's prior with imposed ρ knob.

    Returns (cd, s) matching the sibling `scm_prior.sample_as_cate_dataset`,
    plus:
        s['target_rho']  = requested ρ
        s['empirical_rho'] = measured Corr(Y_do0_raw, Y_do1_raw)
    """
    from priors.causal_prior.scm.SCMSampler import SCMSampler
    from priors.causal_prior.mechanisms.BinarizingMechanism import BinarizingMechanism

    cfg = scm_config if scm_config is not None else DEFAULT_SCM_CONFIG
    sampler = SCMSampler(cfg, seed=scm_seed * 31 + 17)

    for attempt in range(max_nan_retries):
        seed = scm_seed + attempt * 1_000_000
        torch.manual_seed(seed)

        scm = treatment_node = target_node = None
        feature_nodes: list = []
        obs = T_obs_raw = Y_obs_raw = X_obs_raw = None
        t0_value = t1_value = None

        for outer_attempt in range(max_outer_attempts):
            attempt_seed = seed + outer_attempt * 997
            scm = sampler.sample(seed=attempt_seed)
            all_nodes = sorted(scm.dag.nodes())
            if len(all_nodes) < 3: continue

            rng = torch.Generator(); rng.manual_seed(attempt_seed)
            found_pair = False
            for _ in range(30):
                t_idx = torch.randint(0, len(all_nodes), (1,), generator=rng).item()
                treatment_node = all_nodes[t_idx]
                avail = [n for n in all_nodes if n != treatment_node]
                y_idx = torch.randint(0, len(avail), (1,), generator=rng).item()
                target_node = avail[y_idx]
                if scm.exists_treatment_outcome_path(treatment_node, target_node):
                    found_pair = True; break
            if not found_pair: continue

            feature_nodes = [n for n in all_nodes
                              if n != treatment_node and n != target_node]
            original_mech = scm.mechanisms[treatment_node]

            binarised_ok = False
            for _bin in range(10):
                scm.sample_exogenous(n_context)
                scm._fixed_endogenous_vec = None
                scm.sample_endogenous(n_context)
                obs_cont = scm.propagate(n_context)
                t_cont = obs_cont[treatment_node].reshape(-1).float()
                try:
                    bin_mech = BinarizingMechanism.from_observational_data(
                        wrapped_mechanism=original_mech, obs_values=t_cont)
                except ValueError:
                    continue
                scm.mechanisms[treatment_node] = bin_mech
                t0_value = bin_mech.t0; t1_value = bin_mech.t1
                scm.sample_exogenous(n_context)
                scm._fixed_endogenous_vec = None
                scm.sample_endogenous(n_context)
                obs = scm.propagate(n_context)
                T_obs_raw = obs[treatment_node].reshape(-1, 1).float()
                if T_obs_raw.unique().numel() >= 2:
                    binarised_ok = True; break
                scm.mechanisms[treatment_node] = original_mech
            if not binarised_ok: continue

            Y_obs_raw = obs[target_node].reshape(-1, 1).float()
            X_obs_raw = (
                torch.cat([obs[n].reshape(n_context, -1).float()
                            for n in feature_nodes], dim=1)
                if feature_nodes else torch.zeros(n_context, 0))
            if Y_obs_raw.var() < 1e-3: continue
            if torch.unique(Y_obs_raw).numel() < max(5, int(0.1 * n_context)):
                continue
            break
        else:
            raise RuntimeError(f'sample_as_cate_dataset_with_rho: exhausted after {max_outer_attempts}')

        intv_scm = deepcopy(scm)
        intv_scm.intervene(treatment_node)

        scm.sample_exogenous(n_test)
        scm._fixed_endogenous_vec = None
        scm.sample_endogenous(n_test)
        obs_test = scm.propagate(n_test)

        res0, res1 = _propagate_paired_with_rho(
            scm, intv_scm, treatment_node, n_test, t0_value, t1_value,
            target_rho=target_rho, seed=scm_seed,
        )
        Y_do0_raw = res0[target_node].reshape(-1, 1).float()
        Y_do1_raw = res1[target_node].reshape(-1, 1).float()

        X_intv_raw = (
            torch.cat([obs_test[n].reshape(n_test, -1).float()
                        for n in feature_nodes], dim=1)
            if feature_nodes else torch.zeros(n_test, 0))

        y_all = torch.cat([Y_obs_raw.reshape(-1), Y_do0_raw.reshape(-1), Y_do1_raw.reshape(-1)])
        ymin = float(y_all.min()); ymax = float(y_all.max())
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

        y0 = Y_do0_raw.reshape(-1).cpu().numpy()
        y1 = Y_do1_raw.reshape(-1).cpu().numpy()
        emp_rho = float(np.corrcoef(y0, y1)[0, 1]) if y0.std() > 1e-6 and y1.std() > 1e-6 else float('nan')

        out = {
            'X_obs': X_obs, 'T_obs': T_obs, 'Y_obs': Y_obs,
            'X_intv': X_intv, 'Y_do0': Y_do0, 'Y_do1': Y_do1,
            'Y_obs_raw': Y_obs_raw, 'Y_do0_raw': Y_do0_raw, 'Y_do1_raw': Y_do1_raw,
            'ymin': ymin, 'ymax': ymax,
            'treatment_node': treatment_node, 'target_node': target_node,
            'feature_nodes': feature_nodes, 't0_value': t0_value, 't1_value': t1_value,
            'target_rho': target_rho, 'empirical_rho': emp_rho,
        }
        if _contains_nan_or_inf(out): continue
        ok, _ = _sample_passes_thresholds(
            out, min_target_variance, min_unique_target_fraction)
        if not ok: continue

        # Package as CATE_Dataset view
        class _CD: pass
        cd = _CD()
        cd.X_train  = X_obs
        cd.t_train  = T_obs.reshape(-1)
        cd.y_train  = Y_obs_raw.reshape(-1)
        cd.X_test   = X_intv
        cd.true_cate = (Y_do1_raw - Y_do0_raw).reshape(-1)
        return cd, out

    raise RuntimeError(f'sample_as_cate_dataset_with_rho: no clean sample after {max_nan_retries}')
