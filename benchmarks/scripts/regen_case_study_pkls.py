"""Regenerate DoPFN's 6 synthetic case-study pkls with bivariate Y-noise.

Each regenerated pkl matches DoPFN's `InterventionalDataset` schema but
adds three extra attributes:

  ds.mu_0_per_query : np.ndarray  shape (N_int_query,)
      Structural mean E[Y | do(T=0), X = x_i] with noise held at zero
      (evaluated per interventional query row, using the SCM's cached
      exogenous state).
  ds.mu_1_per_query : np.ndarray  shape (N_int_query,)
      Structural mean E[Y | do(T=1), X = x_i].
  ds.sigma_eps      : float
      Empirical std of the Y-node's noise tensor across the realization.
  ds.rho_y_noise    : float
      The correlation used between eps_obs and eps_int for the Y-node.

Given (eps_obs, eps_int) ~ N(0, sigma^2 * [[1, rho], [rho, 1]]):
  p(Y | do(t), x)   = N(mu_t(x), sigma^2)
  p(tau | x)        = N(mu_1(x) - mu_0(x), 2 * sigma^2 * (1 - rho))

The graph structure for each case study is fixed (see
`dopfn_patches/case_study_graphs.py`). Only the random draws (linear
weights, activation choices, additive noise, exo noise) vary
realization-to-realization, seeded by `seed_base + i`.

CLI
---
python regen_case_study_pkls.py \
    --dopfn-root /path/to/dopfn_upstream \
    --out-dir /path/to/output/root \
    --y-noise-corr 0.2 \
    --n-per-case 100 \
    --seed-base 42 \
    --seq-len 500 \
    --num-features 3 \
    --noise-std 0.01 \
    --exo-std 0.1

Pkls land at `<out-dir>/prior_sampling_rho02/<Case>/<Case>_{i}.pkl`.
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import time
from typing import Any, Dict, List

import numpy as np


# ── CLI ──────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dopfn-root", type=str, required=True,
                   help="Path to the dopfn_upstream repo root (has priors/, datasets/).")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Output root; pkls land at <out-dir>/prior_sampling_rho02/<Case>/...")
    p.add_argument("--y-noise-corr", type=float, default=0.2,
                   help="Correlation rho between eps_obs and eps_int on the Y-node.")
    p.add_argument("--n-per-case", type=int, default=100,
                   help="Number of realization pkls to generate per case study.")
    p.add_argument("--seed-base", type=int, default=42,
                   help="Base seed; realization i uses seed_base + i.")
    p.add_argument("--seq-len", type=int, default=500,
                   help="Rows per realization (must be > 200 to leave test queries).")
    p.add_argument("--num-features", type=int, default=3,
                   help="Case-study width parameter (# confounders / mediators / ...).")
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--exo-std", type=float, default=0.1)
    p.add_argument("--nonlins", type=str, default="sophisticated_sampling_1_rescaling_normalization",
                   help="DoPFN nonlinearity setting; matches the paper prior default.")
    p.add_argument("--noise-dist", type=str, default="gaussian",
                   help="Only 'gaussian' currently supported by the bivariate patch.")
    p.add_argument("--binary-strategy", type=str, default="extreme",
                   choices=("extreme", "mean"))
    p.add_argument("--out-subdir", type=str, default="prior_sampling_rho02",
                   help="Name of the subdirectory under out-dir.")
    p.add_argument("--only-cases", type=str, nargs="*", default=None,
                   help="Optional list of case studies to regenerate; default: all 6.")
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--n-mc-samples", type=int, default=0,
                   help="Number of Monte-Carlo Y-noise draws for empirical truth "
                        "densities per query. Each draw is a paired (eps_obs, eps_int) "
                        "with rho correlation; produces one sample of Y|do(0) and one "
                        "of Y|do(1) per query row. Writes sidecar "
                        "<case>_<i>_truth_samples.npz next to each pkl. Default 0 = off.")
    return p.parse_args()


# ── Bootstrap paths ──────────────────────────────────────────────────────────
def _install_paths(dopfn_root: str) -> None:
    """Put DoPFN + our patches on sys.path so imports resolve."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if dopfn_root not in sys.path:
        sys.path.insert(0, dopfn_root)


# ── Per-realization SCM construction + sampling ──────────────────────────────
def _build_scm(hyperparameters: Dict[str, Any], graph, samples_shape):
    """Instantiate DoPFN's SCMGenerator + build SCM from the given graph."""
    from priors.playground_scm.generators import SCMGenerator
    from priors.playground_scm.MakeStructuralEquations import (
        MakeStructuralEquations,
        make_additive_noise_gaussian,
    )

    gen = SCMGenerator(
        all_functions={"nonlinear": MakeStructuralEquations},
        seed=hyperparameters["seed"],
        samples_shape=samples_shape,
        noise_std=hyperparameters["noise_std"],
        noise_dist=hyperparameters["noise_dist"],
        nonlins=hyperparameters["nonlins"],
        max_hidden_layers=hyperparameters.get("max_hidden_layers", 3),
    )
    exo_dist = make_additive_noise_gaussian(samples_shape, hyperparameters["exo_std"])
    scm = gen.create_scm_from_graph(
        graph,
        possible_functions=["nonlinear"],
        exo_distribution=exo_dist,
        exo_distribution_kwargs={},
    )
    return scm


def _binarize_treatment_row(vec, strategy: str = "extreme"):
    """Return (thresh, t_low, t_high, binarized_vec_0_1). Matches
    StructuralCausalModel.set_binarization_params semantics for a single
    batch element (our batch_size = 1)."""
    import torch
    from priors.playground_scm.utils_playground_scm import torch_random_choice

    vec = torch.nan_to_num(vec)
    not_min_max = (vec > vec.min()) & (vec < vec.max())
    if not not_min_max.any():
        t_thresh = vec[0]
        t_low = vec[0]
        t_high = vec[0]
    elif strategy == "extreme":
        t_thresh = np.random.choice(vec[not_min_max].cpu().numpy())
        t_thresh = torch.tensor(float(t_thresh), dtype=vec.dtype)
        t_low = torch_random_choice(vec[vec < t_thresh])
        t_high = torch_random_choice(vec[vec > t_thresh])
    elif strategy == "mean":
        t_thresh = vec[not_min_max].mean()
        t_low = vec[vec < t_thresh].mean()
        t_high = vec[vec > t_thresh].mean()
    else:
        raise ValueError(strategy)

    lt_map = vec < t_thresh
    out = vec.clone()
    out[lt_map] = t_low
    out[~lt_map] = t_high
    return t_thresh, t_low, t_high, out


def _sample_case_study_realization(args, case_study: str, seed: int):
    """One realization for one case study. Returns a dict of arrays ready
    to be assembled into an InterventionalDataset."""
    import torch

    from dopfn_patches.case_study_graphs import build_case_study_graph
    from dopfn_patches.make_structural_equations_bivariate import (
        install_bivariate_y_noise, set_y_call_mode, get_y_noise_std_empirical,
    )

    # Seeding (mirrors DoSCM.forward's implicit reliance on np/torch/random)
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

    batch_size = 1
    samples_shape = (batch_size, args.seq_len)
    hyperparameters: Dict[str, Any] = {
        "seed": seed,
        "noise_std": float(args.noise_std),
        "exo_std": float(args.exo_std),
        "nonlins": args.nonlins,
        "noise_dist": args.noise_dist,
        "max_hidden_layers": 3,
    }

    graph, t_key, y_key, x_idcs_int = build_case_study_graph(
        case_study, num_features=int(args.num_features))
    scm = _build_scm(hyperparameters, graph=graph, samples_shape=samples_shape)
    scm.t_key = t_key
    scm.y_key = y_key
    scm.zero_one_treatment = True
    scm.binary_strategy = args.binary_strategy

    # Install bivariate Y-noise BEFORE the first (observational) call.
    install_bivariate_y_noise(
        scm, y_noise_correlation=float(args.y_noise_corr),
        noise_std=float(args.noise_std),
    )
    set_y_call_mode(scm, "obs")

    scm_graph = scm.create_graph()

    # ── Observational pass ──────────────────────────────────────────────────
    endo_obs, exo_obs = scm.get_next_sample(binarize=True, graph=scm_graph)
    sample_obs = {**endo_obs, **exo_obs}

    # T for the interventional call: coin-flip over the two binarised levels.
    coin_flips = torch.randint(0, 2, (batch_size, args.seq_len))
    t1s_exp = scm.t1s.unsqueeze(1).expand(-1, args.seq_len)
    t2s_exp = scm.t2s.unsqueeze(1).expand(-1, args.seq_len)
    t_int_vec = torch.where(coin_flips == 0, t1s_exp, t2s_exp)

    if t_key.startswith("X"):
        # Endogenous T: intercept with do_intervention.
        scm.do_interventions([(t_key, (lambda: t_int_vec, {}))])
    else:
        # Exogenous T: overwrite the exogenous draw directly.
        exo_obs[t_key] = t_int_vec

    # ── Interventional pass (Y-noise = eps_int) ─────────────────────────────
    set_y_call_mode(scm, "int")
    endo_int, exo_int = scm.get_next_sample(exogenous_vars=exo_obs, graph=scm_graph)
    sample_int = {**endo_int, **exo_int}
    scm.undo_interventions()

    # ── mu_0 / mu_1 per interventional query (noise held at 0) ──────────────
    # NOTE: mu is analytical but of limited use under DoPFN's LayerNorm-coupled
    # DGP (see docstring of _compute_mu_0_mu_1). Kept for backwards compat.
    mu_0, mu_1 = _compute_mu_0_mu_1(
        scm=scm, scm_graph=scm_graph, exo_obs=exo_obs,
        t_key=t_key, y_key=y_key,
        samples_shape=samples_shape,
        t_baseline_vec=t_int_vec,
    )

    # ── Empirical truth density samples (K MC noise draws per query) ────────
    y_do0_samples, y_do1_samples = None, None
    if int(args.n_mc_samples) > 0:
        y_do0_samples, y_do1_samples = _sample_empirical_truth(
            scm=scm, scm_graph=scm_graph, exo_obs=exo_obs,
            t_key=t_key, y_key=y_key,
            samples_shape=samples_shape,
            t_baseline_vec=t_int_vec,
            K=int(args.n_mc_samples),
            rho=float(args.y_noise_corr),
            noise_std=float(args.noise_std),
        )

    # ── Assemble x_obs, x_int, y_obs, y_int (T first column) ────────────────
    X_keys = [t_key]
    for var in list(scm_graph.nodes):
        if var == t_key or var == y_key:
            continue
        # int(var[-1]) matches DoSCM's x_idcs lookup semantics (single-digit)
        try:
            var_id = int(var[-1])
        except ValueError:
            continue
        if var_id in x_idcs_int:
            X_keys.append(var)

    x_obs = torch.stack([sample_obs[k] for k in X_keys]).permute(-1, 1, 0)
    x_int = torch.stack([sample_int[k] for k in X_keys]).permute(-1, 1, 0)

    if scm.zero_one_treatment:
        x_obs[:, :, 0] = scm.get_zero_one_treatment(x_obs[:, :, 0])
        x_int[:, :, 0] = scm.get_zero_one_treatment(x_int[:, :, 0])

    y_obs = sample_obs[y_key].T
    y_int = sample_int[y_key].T
    y_obs = y_obs.detach().unsqueeze(-1)
    y_int = y_int.detach().unsqueeze(-1)
    x_obs = x_obs.detach()
    x_int = x_int.detach()

    # NaN/Inf sentinel: mirror DoSCM's -100 sentinel behaviour.
    if (torch.any(torch.isnan(x_int)) or torch.any(torch.isnan(x_obs))
            or torch.any(torch.isnan(y_obs)) or torch.any(torch.isnan(y_int))
            or torch.any(torch.isinf(x_int)) or torch.any(torch.isinf(x_obs))
            or torch.any(torch.isinf(y_obs)) or torch.any(torch.isinf(y_int))):
        for t in (x_obs, x_int, y_obs, y_int):
            t[:] = -100

    sigma_eps = get_y_noise_std_empirical(scm)

    # Squeeze the batch dim: our pkls are single-batch by convention.
    x_obs_np = x_obs.squeeze(1).cpu().numpy().astype(np.float32)
    x_int_np = x_int.squeeze(1).cpu().numpy().astype(np.float32)
    y_obs_np = y_obs.squeeze(1).cpu().numpy().astype(np.float32).reshape(-1)
    y_int_np = y_int.squeeze(1).cpu().numpy().astype(np.float32).reshape(-1)

    attribute_names = list(X_keys)

    # Per-row true CATE = mu_1 - mu_0 (independent of the T actually drawn).
    cate_per_row = (mu_1 - mu_0).astype(np.float32)

    return {
        "x_obs": x_obs_np,
        "x_int": x_int_np,
        "y_obs": y_obs_np,
        "y_int": y_int_np,
        "cate": cate_per_row,
        "mu_0_per_query": mu_0.astype(np.float32),
        "mu_1_per_query": mu_1.astype(np.float32),
        "y_do0_samples": (y_do0_samples if y_do0_samples is not None
                          else np.zeros((0,), dtype=np.float32)),
        "y_do1_samples": (y_do1_samples if y_do1_samples is not None
                          else np.zeros((0,), dtype=np.float32)),
        "sigma_eps": float(sigma_eps),
        "rho_y_noise": float(args.y_noise_corr),
        "attribute_names": attribute_names,
        "case_study": case_study,
        "seed": int(seed),
        "graph_edges": [(str(u), str(v)) for u, v in scm_graph.edges],
        "t_key": t_key, "y_key": y_key, "x_keys": X_keys,
        "hyperparameters": hyperparameters,
    }


def _compute_mu_0_mu_1(scm, scm_graph, exo_obs, t_key, y_key, samples_shape,
                       t_baseline_vec=None):
    """Per-query mu_t(x_i) = E[Y | do(T=t), X=x_i, batch context = fixed]  (noise = 0).

    DoPFN's Y activation includes a batch-wide LayerNorm, so Y[i] is a function
    of the entire (batch × seq_len) tensor, not of x_i alone. There is no
    single "true mu(x_i)" — it depends on what the OTHER 499 rows' T values are.

    We resolve this with a **half-and-half T split** that matches DoPFN's natural
    sampling: use the observed `t_baseline_vec` (~50/50 coin-flipped t_low/t_high)
    for pass 1, and the bit-flipped version (t_low ↔ t_high) for pass 2. Then:

      - If t_baseline_vec[i] = t_low  : pass 1 gives mu_0[i], pass 2 gives mu_1[i].
      - If t_baseline_vec[i] = t_high : pass 1 gives mu_1[i], pass 2 gives mu_0[i].

    Both passes have LayerNorm see ~50/50 T variation across the batch (matching
    DoPFN's DGP), so mu magnitudes stay in the range DoPFN naturally produces.
    Only 2 forward passes per realization — fast.

    Trade-off vs single-row counterfactual: mu[i] here technically depends on
    which of the other 499 rows are at each T, but that dependence is
    intrinsic to DoPFN's LayerNorm-coupled DGP and stays in a natural range.

    Parameters
    ----------
    t_baseline_vec : torch.Tensor, shape samples_shape
        The mixed T vector used for the interventional sample. Must have both
        t_low and t_high values (coin-flipped). If None, falls back to a
        uniform-T intervention (kept for backwards compat only — will
        collapse under LayerNorm activations).

    Returns
    -------
    mu_0, mu_1 : np.ndarray, shape (seq_len,)
        Per-query analytical structural means (noise = 0), computed under
        DoPFN's natural half-and-half T distribution.
    """
    import numpy as np
    import torch

    y_fn, _ = scm.functions[y_key]
    saved_noise = y_fn.additive_noise
    zeros = torch.zeros_like(saved_noise)
    y_fn.additive_noise = zeros

    t_low = float(scm.t1s[0].item()) if hasattr(scm, "t1s") else 0.0
    t_high = float(scm.t2s[0].item()) if hasattr(scm, "t2s") else 1.0

    batch_size, seq_len = int(samples_shape[0]), int(samples_shape[1])
    assert batch_size == 1, (
        "Half-and-half mu assumes batch_size=1; got batch_size={batch_size}"
    )

    if t_baseline_vec is None:
        # Backwards-compat uniform-T path (WILL collapse under LayerNorm).
        t_baseline_vec = torch.full(samples_shape, t_low, dtype=torch.float32)

    def _do_with_vec(T_vec: torch.Tensor) -> torch.Tensor:
        if t_key.startswith("X"):
            scm.do_interventions([(t_key, (lambda: T_vec, {}))])
        else:
            _saved = exo_obs.get(t_key)
            exo_obs[t_key] = T_vec
        try:
            endo, _ = scm.get_next_sample(exogenous_vars=exo_obs, graph=scm_graph)
        finally:
            if t_key.startswith("X"):
                scm.undo_interventions()
            else:
                if _saved is not None:
                    exo_obs[t_key] = _saved
        return endo[y_key].detach()  # shape (batch, seq_len)

    # Bit-flipped baseline: swap t_low ↔ t_high per row.
    t_flipped_vec = torch.where(t_baseline_vec == t_low,
                                 torch.tensor(t_high, dtype=t_baseline_vec.dtype),
                                 torch.tensor(t_low,  dtype=t_baseline_vec.dtype))

    y_pass_baseline = _do_with_vec(t_baseline_vec)[0].cpu().numpy().astype(np.float32)
    y_pass_flipped  = _do_with_vec(t_flipped_vec )[0].cpu().numpy().astype(np.float32)

    # Row-level assignment: whichever pass had T[i]=t_low gives mu_0[i], other gives mu_1[i].
    mask_baseline_is_low = (t_baseline_vec[0].cpu().numpy() == t_low)
    mu_0 = np.where(mask_baseline_is_low, y_pass_baseline, y_pass_flipped).astype(np.float32)
    mu_1 = np.where(mask_baseline_is_low, y_pass_flipped, y_pass_baseline).astype(np.float32)

    y_fn.additive_noise = saved_noise
    return mu_0, mu_1


def _sample_empirical_truth(scm, scm_graph, exo_obs, t_key, y_key,
                             samples_shape, t_baseline_vec,
                             K: int, rho: float, noise_std: float):
    """Draw K bivariate Y-noise pairs, run half-and-half passes per k,
    collect empirical Y samples per query per arm.

    Under DoPFN's LayerNorm-coupled DGP, p(Y|do(t), x_i) is not Gaussian and
    has no closed form. Instead we sample the SCM K times per realization
    with fresh Y-noise, keeping the batch context (other rows' T mix and X)
    identical to the observed interventional pass. Each MC draw gives ONE
    paired (Y_do0, Y_do1) sample per query row, with the DGP's own rho=0.2
    correlation between them (via bivariate eps_obs, eps_int draws).

    Parameters
    ----------
    K : int   number of MC draws.
    rho : float  eps_obs / eps_int correlation (matches the pkl's rho_y_noise).
    noise_std : float  Y-node noise std (matches the SCM's noise_std).

    Returns
    -------
    y_do0_samples, y_do1_samples : np.ndarray of shape (seq_len, K)
        Per-query empirical Y samples under do(T=0) and do(T=1).
    """
    import math
    import numpy as np
    import torch

    y_fn, _ = scm.functions[y_key]
    saved_noise = y_fn.additive_noise

    t_low  = float(scm.t1s[0].item()) if hasattr(scm, "t1s") else 0.0
    t_high = float(scm.t2s[0].item()) if hasattr(scm, "t2s") else 1.0

    batch_size, seq_len = int(samples_shape[0]), int(samples_shape[1])
    assert batch_size == 1

    t_flipped_vec = torch.where(t_baseline_vec == t_low,
                                 torch.tensor(t_high, dtype=t_baseline_vec.dtype),
                                 torch.tensor(t_low,  dtype=t_baseline_vec.dtype))
    mask_baseline_is_low = (t_baseline_vec[0].cpu().numpy() == t_low)

    def _run(T_vec):
        if t_key.startswith("X"):
            scm.do_interventions([(t_key, (lambda: T_vec, {}))])
        else:
            _saved = exo_obs.get(t_key)
            exo_obs[t_key] = T_vec
        try:
            endo, _ = scm.get_next_sample(exogenous_vars=exo_obs, graph=scm_graph)
        finally:
            if t_key.startswith("X"):
                scm.undo_interventions()
            else:
                if _saved is not None:
                    exo_obs[t_key] = _saved
        return endo[y_key].detach()  # (batch, seq_len)

    y_do0 = np.empty((seq_len, K), dtype=np.float32)
    y_do1 = np.empty((seq_len, K), dtype=np.float32)

    coeff_partner = math.sqrt(max(1.0 - rho * rho, 0.0))
    for k in range(K):
        # Draw fresh bivariate Y-noise per DGP: (eps_obs, eps_int) ~ N(0, sigma^2*[[1,rho],[rho,1]])
        eps_obs   = torch.normal(0.0, noise_std, samples_shape)
        eps_indep = torch.normal(0.0, noise_std, samples_shape)
        eps_int   = rho * eps_obs + coeff_partner * eps_indep

        # Pass A: baseline T, eps_obs noise
        y_fn.additive_noise = eps_obs
        y_A = _run(t_baseline_vec)[0].cpu().numpy().astype(np.float32)

        # Pass B: flipped T, eps_int noise
        y_fn.additive_noise = eps_int
        y_B = _run(t_flipped_vec)[0].cpu().numpy().astype(np.float32)

        # Row-level assignment: pass with T[i]=t_low → y_do0[i,k]; the other → y_do1[i,k].
        y_do0[:, k] = np.where(mask_baseline_is_low, y_A, y_B)
        y_do1[:, k] = np.where(mask_baseline_is_low, y_B, y_A)

    y_fn.additive_noise = saved_noise
    return y_do0, y_do1


# ── Assemble the InterventionalDataset pkl ───────────────────────────────────
def _assemble_and_save(row: dict, out_path: str) -> None:
    """Build an InterventionalDataset with the extra fields attached, then
    pickle it. We rely on DoPFN's `datasets` package being importable so
    the pkl can be unpickled by SCMCaseStudyDataset later.
    """
    import torch
    from datasets import InterventionalDataset

    x_obs_t = torch.from_numpy(row["x_obs"])
    x_int_t = torch.from_numpy(row["x_int"])
    y_obs_t = torch.from_numpy(row["y_obs"])
    y_int_t = torch.from_numpy(row["y_int"])

    ds = InterventionalDataset(
        x_obs=x_obs_t,
        y_obs=y_obs_t,
        x_int=x_int_t,
        y_int=y_int_t,
        do_scm=None,   # not saved — pickling live SCMs is fragile across envs
        attribute_names=row["attribute_names"],
        name=f"{row['case_study']}_seed{row['seed']}",
        function_args={"case_study": row["case_study"], "seed": row["seed"],
                       "hyperparameters": row["hyperparameters"],
                       "graph_edges": row["graph_edges"],
                       "t_key": row["t_key"], "y_key": row["y_key"],
                       "x_keys": row["x_keys"]},
    )
    # Match the paired-row structure expected by the downstream fast-path in
    # SCMCaseStudyDataset._cate_from_paired_rows: split .x into (x_obs || x_int)
    # so that generate_valid_split can pull both halves as needed.
    #
    # We ALSO populate the extra fields the density-eval pipeline needs.
    ds.cate = torch.from_numpy(row["cate"])
    ds.mu_0_per_query = row["mu_0_per_query"]
    ds.mu_1_per_query = row["mu_1_per_query"]
    ds.sigma_eps = row["sigma_eps"]
    ds.rho_y_noise = row["rho_y_noise"]
    # Provide `.x` as (x_obs concat x_int) so downstream splits can index
    # both halves; density-eval will use `.mu_0_per_query` / `.mu_1_per_query`
    # aligned to the SECOND half (the query / x_int rows). See File 4 patch.
    ds.x = torch.cat([x_obs_t, x_int_t], dim=0)
    # Extend y correspondingly.
    ds.y = torch.cat([y_obs_t, y_int_t], dim=0)
    # Mark which rows correspond to the interventional queries.
    n_obs = x_obs_t.shape[0]
    n_int = x_int_t.shape[0]
    ds.n_obs_rows = int(n_obs)
    ds.n_int_rows = int(n_int)
    ds.int_row_indices = np.arange(n_obs, n_obs + n_int, dtype=np.int64)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(ds, f)

    # Sidecar .npz for empirical MC truth samples (if any were drawn).
    y_do0_s = row.get("y_do0_samples")
    y_do1_s = row.get("y_do1_samples")
    if (isinstance(y_do0_s, np.ndarray) and y_do0_s.ndim == 2
            and y_do0_s.size > 0):
        base = out_path[:-4] if out_path.endswith(".pkl") else out_path
        npz_path = f"{base}_truth_samples.npz"
        np.savez_compressed(
            npz_path,
            y_do0_samples=y_do0_s.astype(np.float32),
            y_do1_samples=y_do1_s.astype(np.float32),
            sigma_eps=np.float32(row["sigma_eps"]),
            rho_y_noise=np.float32(row["rho_y_noise"]),
        )


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = _parse_args()
    _install_paths(args.dopfn_root)

    from dopfn_patches.case_study_graphs import CASE_STUDIES

    only = set(args.only_cases) if args.only_cases else set(CASE_STUDIES)
    for c in only:
        if c not in CASE_STUDIES:
            raise SystemExit(f"unknown case study: {c!r}; pick from {CASE_STUDIES}")

    out_root = os.path.join(args.out_dir, args.out_subdir)
    print(f"[regen] out={out_root}  rho={args.y_noise_corr}  "
          f"n_per_case={args.n_per_case}  seq_len={args.seq_len}  "
          f"num_features={args.num_features}", flush=True)

    t0 = time.time()
    for case in CASE_STUDIES:
        if case not in only:
            continue
        case_dir = os.path.join(out_root, case)
        os.makedirs(case_dir, exist_ok=True)
        for i in range(1, args.n_per_case + 1):
            out_path = os.path.join(case_dir, f"{case}_{i}.pkl")
            if os.path.exists(out_path) and not args.overwrite:
                continue
            seed = int(args.seed_base) + i
            try:
                row = _sample_case_study_realization(args, case, seed=seed)
            except Exception as e:
                print(f"[regen] {case}_{i}  SEED={seed}  ERROR: "
                      f"{type(e).__name__}: {e}", flush=True)
                continue
            _assemble_and_save(row, out_path)
            if i == 1 or i % 25 == 0 or i == args.n_per_case:
                dt = time.time() - t0
                print(f"[regen] {case:32s}  i={i:3d}/{args.n_per_case}  "
                      f"seed={seed}  sigma_eps={row['sigma_eps']:.4f}  "
                      f"cate_mean={row['cate'].mean():+.3f}  "
                      f"({dt:.0f}s)", flush=True)
    print(f"[regen] done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
