"""Do-PFN's SCM prior, repaired and made reproducible.

This is a faithful re-implementation of ``Do-PFN/priors/doscm.py::get_batch``.
It is a re-implementation rather than a wrapper for three reasons:

  1. The shipped ``get_batch`` does not run (see REPAIR A/B/C below).
  2. It emits only one interventional arm per row; the joint head needs both.
  3. It reads RNG state from wherever the process happens to be, so a batch is
     not a pure function of a seed.

Everything else mirrors ``doscm.py`` statement for statement, including the
quirks: one graph and one set of structural-equation weights shared across the
whole batch, additive noise drawn once at SCM construction, and the inverted
treatment polarity.

Hyperparameters are read off the released checkpoint's training config
(``model_submitit_0ccc_id_171b69db_epoch_-1.cpkt``), not the repo's demo script:

    exo_std         ~ Uniform(1, 3)
    noise_std       ~ 0.3 * Beta(1, 5)
    num_unobserved  ~ UniformInt(0, 4)
    num_features    ~ UniformInt(1, 6)
    seq_len         = 2200

REPAIRS applied to the shipped prior
------------------------------------
A. ``exo_distribution`` is only assigned inside an ``elif`` that is unreachable
   when ``graph is None`` -- i.e. exactly the training path -- and is then
   passed to ``create_scm_from_graph``. Hoisted unconditionally.
B. ``StructuralCausalModel.binary_strategy`` is read by
   ``set_binarization_params`` but never assigned by ``__init__`` or by
   ``doscm.py``. Set explicitly; the checkpoint records no value, so this is a
   genuine free choice -- see ``PriorConfig.binary_strategy``.
C. ``doscm.py`` requires ``noise_dist`` / ``nonlins`` / ``exo_dist`` /
   ``max_hidden_layers``, none of which exist in the checkpoint's config. The
   shipped prior is newer than the released weights (those keys were added for
   the Do-PFN-Mixed ablation). Defaults here pin the v1 behaviour.
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOPFN_SRC = os.environ.get("DOPFN_SRC", os.path.join(_REPO_ROOT, "Do-PFN"))

# Model-facing treatment labels. Do-PFN's get_zero_one_treatment maps values
# BELOW the mean to 1.0, and binarization yields levels t1 < thresh < t2, so the
# LOWER raw level is the "treated" arm. Arbitrary, but load-bearing: flip it and
# every CATE changes sign.
_ARM1_IS_LOWER_RAW_LEVEL = True

SENTINEL = -100.0


def _load_dopfn_components():
    """Import Do-PFN's SCM machinery, guarding the top-level name collision.

    Do-PFN ships a top-level ``priors`` package and so does g4cfm; whichever
    imports first owns ``sys.modules['priors']``. Fail loudly rather than
    silently sampling from the wrong prior.
    """
    loaded = sys.modules.get("priors")
    loaded_path = getattr(loaded, "__file__", "") or ""
    if loaded_path and not os.path.abspath(loaded_path).startswith(
        os.path.abspath(DOPFN_SRC)
    ):
        raise RuntimeError(
            f"A different top-level 'priors' package is already imported from "
            f"{loaded_path!r}. Run this in a fresh process, or set DOPFN_SRC."
        )
    if DOPFN_SRC not in sys.path:
        sys.path.insert(0, DOPFN_SRC)

    from priors.playground_scm.generators import SCMGenerator
    from priors.playground_scm.MakeStructuralEquations import (
        MakeStructuralEquations,
        make_additive_noise_gaussian,
    )
    from priors.playground_scm.utils_playground_scm import torch_random_choice

    return (
        SCMGenerator,
        MakeStructuralEquations,
        make_additive_noise_gaussian,
        torch_random_choice,
    )


@dataclass
class PriorConfig:
    """Prior hyperparameters. Defaults reproduce the released checkpoint."""

    seq_len: int = 2200           # config: seq_len
    batch_size: int = 4           # config: batch_size

    min_num_features: int = 1     # config: num_features_sampler_config
    max_num_features: int = 6
    num_unobserved_min: int = 0   # differentiable_hyperparameters.num_unobserved
    num_unobserved_max: int = 4

    exo_std_min: float = 1.0      # differentiable_hyperparameters.exo_std
    exo_std_max: float = 3.0
    noise_std_k: float = 1.0      # differentiable_hyperparameters.noise_std
    noise_std_b: float = 5.0
    noise_std_scale: float = 0.3

    # REPAIR B / DECISION. Unrecorded anywhere in the released checkpoint, so
    # this is a free choice rather than a recovered value. Fixed to 'extreme'
    # (draws the two arm levels from either side of a sampled threshold; 'mean'
    # would use conditional means instead).
    #
    # Rationale: the parity harness favoured 'extreme' in both runs, judged on
    # excess NLL over a context-fitted Gaussian -- 0.182 vs 0.240, and -2.053 vs
    # -1.750. The margin is weak-to-moderate rather than decisive, so this is
    # recorded as a deviation, not a finding. It also matches what
    # PairedDoPFNDataset independently chose.
    #
    # It must stay identical across the 1-D and joint runs; that is what the
    # comparison depends on, more than which value is historically correct.
    binary_strategy: str = "extreme"

    # REPAIR C: keys the shipped prior needs but the checkpoint never set.
    nonlins: str = "mixed"        # pool: {x^2, ReLU, tanh, identity}
    noise_dist: str = "gaussian"
    exo_dist: str = "gaussian"
    max_hidden_layers: int = 0    # accepted then ignored by MakeStructuralEquations

    zero_one_treatment: bool = True
    num_outputs: int = 1
    num_treatments: int = 1

    # doscm.py overwrites the whole batch with -100 on any non-finite value.
    # 'sentinel' reproduces that; 'resample' redraws the SCM instead.
    on_nonfinite: str = "sentinel"
    max_attempts: int = 50


@dataclass
class Hyperparameters:
    num_features: int
    num_unobserved: int
    exo_std: float
    noise_std: float
    edge_prob: float
    num_nodes: int

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def sample_hyperparameters(cfg: PriorConfig) -> Hyperparameters:
    """Draw the per-step hyperparameters. Consumes the ambient numpy RNG."""
    num_features = int(
        np.random.randint(cfg.min_num_features, cfg.max_num_features + 1)
    )
    num_unobserved = int(
        np.random.randint(cfg.num_unobserved_min, cfg.num_unobserved_max + 1)
    )
    exo_std = float(np.random.uniform(cfg.exo_std_min, cfg.exo_std_max))
    noise_std = float(
        np.random.beta(cfg.noise_std_k, cfg.noise_std_b) * cfg.noise_std_scale
    )
    num_nodes = num_features + num_unobserved + cfg.num_outputs + cfg.num_treatments
    # doscm.DoSCM.__init__
    edge_prob = float(np.random.uniform(1.0 / (num_features + 1), 1.0))
    return Hyperparameters(
        num_features=num_features,
        num_unobserved=num_unobserved,
        exo_std=exo_std,
        noise_std=noise_std,
        edge_prob=edge_prob,
        num_nodes=num_nodes,
    )


def _seed_everything(seed: int) -> None:
    """The SCM machinery reads `random`, `np.random` and torch's global RNG."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def _select_treatment_and_outcome(scm_graph, torch_random_choice):
    """doscm.DoSCM.forward: treatment has descendants, outcome is one of them."""
    nodes = list(scm_graph.nodes)
    if len(list(scm_graph.edges)) == 0:
        t_key = torch_random_choice(nodes)
        y_key = torch_random_choice(list(set(nodes) - {t_key}))
        return t_key, y_key

    import networkx as nx

    t_choice = [v for v in nodes if scm_graph.out_degree(v) > 0]
    t_key = torch_random_choice(t_choice)
    y_key = torch_random_choice(list(nx.descendants(scm_graph, t_key)))
    return t_key, y_key


def _propagate_arm(scm, graph, t_key, value, shared_exogenous):
    """Propagate one pure arm, reusing every non-treatment exogenous draw.

    Holding the noise fixed across arms is what makes the two outcomes genuine
    counterfactuals for the same unit rather than independent draws (paper
    footnote 5). Mirrors doscm's endogenous/exogenous branch.
    """
    exogenous = {k: v.clone() for k, v in shared_exogenous.items()}
    is_endogenous = t_key in scm.endogenous_vars
    if is_endogenous:
        scm.do_interventions([(t_key, (lambda value=value: value, {}))])
    else:
        exogenous[t_key] = value.clone()
    try:
        endo, exo = scm.get_next_sample(exogenous_vars=exogenous, graph=graph)
        return endo | exo
    finally:
        if is_endogenous:
            scm.undo_interventions()


def _stack_features(sample, keys, ) -> torch.Tensor:
    """(n_keys, B, S) -> (S, B, n_keys), matching doscm's permute(-1, 1, 0)."""
    return torch.stack([sample[k] for k in keys]).permute(-1, 1, 0).float()


def _sample_once(seed: int, cfg: PriorConfig) -> dict[str, Any]:
    (
        SCMGenerator,
        MakeStructuralEquations,
        make_additive_noise_gaussian,
        torch_random_choice,
    ) = _load_dopfn_components()

    _seed_everything(seed)
    hp = sample_hyperparameters(cfg)
    samples_shape = (cfg.batch_size, cfg.seq_len)

    gen = SCMGenerator(
        all_functions={"nonlinear": MakeStructuralEquations},
        seed=seed,
        samples_shape=samples_shape,
        noise_std=hp.noise_std,
        noise_dist=cfg.noise_dist,
        nonlins=cfg.nonlins,
        max_hidden_layers=cfg.max_hidden_layers,
    )

    graph = gen.create_graph_from_nodes(num_nodes=hp.num_nodes, p=hp.edge_prob)

    # REPAIR A: hoisted out of the unreachable `elif` in doscm.py.
    if cfg.exo_dist != "gaussian":
        raise ValueError(
            f"exo_dist={cfg.exo_dist!r} is a Do-PFN-Mixed ablation setting; the "
            f"released v1 checkpoint records no such key."
        )
    exo_distribution = make_additive_noise_gaussian(samples_shape, hp.exo_std)

    scm = gen.create_scm_from_graph(
        graph,
        possible_functions=["nonlinear"],
        exo_distribution=exo_distribution,
        exo_distribution_kwargs={},
    )
    scm.zero_one_treatment = cfg.zero_one_treatment
    scm.binary_strategy = cfg.binary_strategy  # REPAIR B

    scm_graph = scm.create_graph()
    t_key, y_key = _select_treatment_and_outcome(scm_graph, torch_random_choice)
    scm.t_key, scm.y_key = t_key, y_key

    # Observational pass. binarize=True also populates scm.t1s / t2s / t_threshs.
    endo_obs, exo_obs = scm.get_next_sample(binarize=True, graph=scm_graph)
    sample_obs = endo_obs | exo_obs

    # Both pure arms, sharing every exogenous draw with the observational pass.
    # t1s is the LOWER raw level -> model-facing t=1.
    t1s = scm.t1s[:, None].expand(-1, cfg.seq_len)   # (B, S)
    t2s = scm.t2s[:, None].expand(-1, cfg.seq_len)
    if torch.any(t1s == t2s):
        raise ValueError("binarization produced identical treatment levels")

    raw_arm1, raw_arm0 = (t1s, t2s) if _ARM1_IS_LOWER_RAW_LEVEL else (t2s, t1s)
    sample_arm1 = _propagate_arm(scm, scm_graph, t_key, raw_arm1, exo_obs)
    sample_arm0 = _propagate_arm(scm, scm_graph, t_key, raw_arm0, exo_obs)

    # Covariate selection (doscm.DoSCM.forward). Column 0 is always the treatment.
    x_cand = list(set(scm_graph.nodes) - {y_key, t_key})
    x_keys = [t_key] + list(
        np.random.choice(x_cand, size=hp.num_features, replace=False)
    )

    x_obs = _stack_features(sample_obs, x_keys)          # (S, B, F+1)
    x_arm1 = _stack_features(sample_arm1, x_keys)
    x_arm0 = _stack_features(sample_arm0, x_keys)

    # Do-PFN's coin flip over the intervention value: coin==0 -> t1s -> model 1.
    coin = torch.randint(0, 2, (cfg.batch_size, cfg.seq_len))
    t_int01 = (coin == 0).float().T                      # (S, B)

    if cfg.zero_one_treatment:
        # get_zero_one_treatment normalises by the column mean, which degenerates
        # on a pure arm (every value equal -> all zeros). Use the midpoint of the
        # two raw levels instead: identical whenever both levels are present,
        # and correct when only one is.
        midpoint = ((scm.t1s + scm.t2s) / 2.0).view(1, -1)   # (1, B)
        x_obs[:, :, 0] = (x_obs[:, :, 0] < midpoint).float()
        x_arm1[:, :, 0] = 1.0
        x_arm0[:, :, 0] = 0.0

    y_obs = sample_obs[y_key].T.float()                  # (S, B)
    y_do1 = sample_arm1[y_key].T.float()
    y_do0 = sample_arm0[y_key].T.float()

    # Exactly doscm's single interventional draw: propagation is row-wise, so
    # selecting per row from the two pure arms is identical to propagating the
    # mixed treatment vector in one pass.
    sel = t_int01.bool()
    y_int = torch.where(sel, y_do1, y_do0)
    x_int = torch.where(sel.unsqueeze(-1), x_arm1, x_arm0)

    return {
        "x_obs": x_obs,        # (S, B, F+1), col 0 = observational treatment
        "y_obs": y_obs,        # (S, B)
        "t_int01": t_int01,    # (S, B) model-facing intervention label
        "y_int": y_int,        # (S, B) outcome under do(t_int01)
        "y_do0": y_do0,        # (S, B) outcome under do(t=0)
        "y_do1": y_do1,        # (S, B) outcome under do(t=1)
        "x_int": x_int,        # (S, B, F+1) post-intervention covariates
        "meta": {
            "seed": seed,
            "t_key": t_key,
            "y_key": y_key,
            "x_keys": x_keys,
            **hp.as_dict(),
        },
    }


def _is_finite(rec: dict[str, Any]) -> bool:
    return all(
        torch.isfinite(rec[k]).all()
        for k in ("x_obs", "y_obs", "y_int", "y_do0", "y_do1", "x_int")
    )


def sample_batch(seed: int, cfg: PriorConfig | None = None) -> dict[str, Any]:
    """Draw one training batch. A pure function of ``seed`` and ``cfg``.

    One graph and one set of structural-equation weights are shared across the
    batch dimension -- this is Do-PFN's own behaviour, not a simplification. The
    ``batch_size`` datasets differ only in their noise realisations, so a run of
    N optimizer steps sees N distinct SCMs, not N * batch_size.
    """
    cfg = cfg or PriorConfig()
    last_error: Exception | None = None

    for attempt in range(cfg.max_attempts):
        try:
            rec = _sample_once(seed + attempt * 1_000_003, cfg)
        except (AssertionError, ValueError, RuntimeError, IndexError) as err:
            # Degenerate draws are expected: `assert t1 != t2` in
            # set_binarization_params fires whenever a treatment is constant.
            last_error = err
            continue

        if _is_finite(rec):
            rec["meta"]["attempts"] = attempt + 1
            rec["meta"]["nonfinite"] = False
            return rec

        if cfg.on_nonfinite == "sentinel":
            # doscm.get_batch: overwrite the whole batch with -100.
            for key in ("x_obs", "y_obs", "y_int", "y_do0", "y_do1", "x_int"):
                rec[key] = torch.full_like(rec[key], SENTINEL)
            rec["meta"]["attempts"] = attempt + 1
            rec["meta"]["nonfinite"] = True
            return rec

    raise RuntimeError(
        f"could not draw a valid SCM for seed {seed} in {cfg.max_attempts} "
        f"attempts"
    ) from last_error
