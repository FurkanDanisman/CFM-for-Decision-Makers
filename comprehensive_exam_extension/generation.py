"""Regenerate Do-PFN's synthetic case studies, sweeping the context size.

CLI
---
    python case_study/generation.py --out-dir case_study/data
    python case_study/generation.py --out-dir case_study/data \
        --context-sizes 50 100 200 500 1000 --n-realizations 100 --seed-base 0

Output layout:  <out-dir>/<Case>/N{N}/{Case}_{r}.npz   (+ manifest.json)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np


# ── Activation pool (repo: nonlins='mixed' -> one of these per node) ──────────
ACTIVATIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "square":   lambda x: x ** 2,
    "relu":     lambda x: np.maximum(0.0, x),
    "tanh":     np.tanh,
    "identity": lambda x: x,
}
_ACTIVATION_NAMES = tuple(ACTIVATIONS)

# Target functions g for the function path of Experiment C (not in the training prior).
MIX_FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sin": np.sin,       # non-monotone, periodic
    "sign": np.sign,     # non-smooth (a step)
    "cubic": lambda x: x ** 3,   # grows faster than any training function
    "expsq": lambda x: np.exp(-x ** 2 / 2),   # exp(-x^2/2): localized, returns to 0 on both sides
    "sin3": lambda x: np.sin(3 * x),           # sin(3x): high-frequency oscillation
    "saw": lambda x: x - 2 * np.floor((x + 1) / 2),   # sawtooth x - 2 floor((x+1)/2): periodic with jumps
}


def _mixture_ppf(u: np.ndarray, cdf: Callable[[np.ndarray], np.ndarray], half_width: float) -> np.ndarray:
    """Numerical inverse of a continuous CDF on [-half_width, half_width]."""
    grid = np.linspace(-half_width, half_width, 400_001)
    return np.interp(u, cdf(grid), grid)


def _standardized_quantile(z: np.ndarray, dist: str, param: float,
                           standardize: bool = True) -> np.ndarray:
    """Map standard-normal draws z to a mean-0 draw of `dist` through the quantile
    function (z -> Φ(z) -> F⁻¹), so every level reuses the same z. With standardize=True
    the draw is divided by the distribution's s.d. (variance 1, only the shape differs);
    with False it keeps the textbook form with scale 1 (its own variance).
      t        Student-t_ν, ν = param                         variance ν/(ν-2)
      gamma    G - k, G ~ Gamma(k, 1), k = param                variance k
      lognorm  exp(s z) - exp(s²/2), s = param                  variance (e^{s²}-1) e^{s²}
      gennorm  generalized normal ∝ exp(-|x|^β), β = param     variance Γ(3/β)/Γ(1/β)
      bimodal  0.5 N(-a, 1) + 0.5 N(a, 1), a = param            variance 1 + a²
      contam   0.9 N(0, 1) + 0.1 N(0, c²), c = param            variance 0.9 + 0.1 c²"""
    if dist == "gaussian":
        return z
    from scipy import stats
    from scipy.special import gamma as G
    lo = z <= 0                       # use the lower/upper tail for precision
    p_lo, p_hi = stats.norm.cdf(z), stats.norm.sf(z)
    p = float(param)
    if dist == "t":
        if p <= 2:
            raise ValueError("Student-t needs nu > 2 for finite variance")
        q, var = np.where(lo, stats.t.ppf(p_lo, p), stats.t.isf(p_hi, p)), p / (p - 2.0)
    elif dist == "gamma":
        q, var = np.where(lo, stats.gamma.ppf(p_lo, p), stats.gamma.isf(p_hi, p)) - p, p
    elif dist == "lognorm":           # monotone in z: no quantile inversion needed
        w = np.exp(p ** 2)
        q, var = np.exp(p * z) - np.sqrt(w), (w - 1.0) * w
    elif dist == "gennorm":
        q = np.where(lo, stats.gennorm.ppf(p_lo, p), stats.gennorm.isf(p_hi, p))
        var = G(3.0 / p) / G(1.0 / p)
    elif dist == "bimodal":
        cdf = lambda x: 0.5 * (stats.norm.cdf(x + p) + stats.norm.cdf(x - p))
        q, var = _mixture_ppf(p_lo, cdf, p + 12.0), 1.0 + p ** 2
    elif dist == "contam":
        cdf = lambda x: 0.9 * stats.norm.cdf(x) + 0.1 * stats.norm.cdf(x / p)
        q, var = _mixture_ppf(p_lo, cdf, 12.0 * max(p, 1.0)), 0.9 + 0.1 * p ** 2
    else:
        raise ValueError(f"unknown noise_dist {dist!r}")
    return q / np.sqrt(var) if standardize else q


# ── DAG description ───────────────────────────────────────────────────────────
# kind ∈ {"root_normal", "root_bernoulli", "structural"}:
#   root_normal    exogenous root z ~ N(0, exo_std)      (confounders / hidden U)
#   root_bernoulli exogenous treatment T ~ Bernoulli(1/2)  (Observed_Mediator)
#   structural     gamma(W @ parents) + eps               (T, mediators, Y)
# `observed` nodes are the ones exposed in X (never T, never hidden nodes).
@dataclass(frozen=True)
class Node:
    name: str
    kind: str
    parents: Tuple[str, ...] = ()
    observed: bool = False
    binarize: bool = False        # median-split this structural node -> {0,1}
    is_treatment: bool = False
    is_outcome: bool = False


CASE_STUDIES: Tuple[str, ...] = (
    "Observed_Confounder",
    "Backdoor_Criterion",
    "Observed_Mediator",
    "Observed_Mediator_and_Confounder",
    "Unobserved_Confounder",
    "Frontdoor_Criterion",
)


def _observed_confounder() -> List[Node]:
    # C -> T, C -> Y, T -> Y   (confounder observed)
    return [
        Node("C", "root_normal", observed=True),
        Node("T", "structural", ("C",), binarize=True, is_treatment=True),
        Node("Y", "structural", ("C", "T"), is_outcome=True),
    ]


def _backdoor_criterion() -> List[Node]:
    # C -> T, C -> Y, T -> M -> Y   (confounder observed, mediator hidden,
    # no direct T -> Y). Structurally identical to Frontdoor in the release.
    return [
        Node("C", "root_normal", observed=True),
        Node("T", "structural", ("C",), binarize=True, is_treatment=True),
        Node("M", "structural", ("T",), observed=False),
        Node("Y", "structural", ("C", "M"), is_outcome=True),
    ]


def _observed_mediator() -> List[Node]:
    # T -> M -> Y, T -> Y   (treatment exogenous; mediator observed)
    return [
        Node("T", "root_bernoulli", is_treatment=True),
        Node("M", "structural", ("T",), observed=True),
        Node("Y", "structural", ("T", "M"), is_outcome=True),
    ]


def _observed_mediator_and_confounder() -> List[Node]:
    # C -> T, C -> Y, T -> Y, T -> M -> Y   (confounder + mediator both observed)
    return [
        Node("C", "root_normal", observed=True),
        Node("T", "structural", ("C",), binarize=True, is_treatment=True),
        Node("M", "structural", ("T",), observed=True),
        Node("Y", "structural", ("C", "T", "M"), is_outcome=True),
    ]


def _unobserved_confounder() -> List[Node]:
    # U -> T, U -> Y (hidden) ; C -> T, C -> Y (observed) ; T -> Y
    return [
        Node("U", "root_normal", observed=False),        # hidden confounder
        Node("C", "root_normal", observed=True),          # observed confounder
        Node("T", "structural", ("U", "C"), binarize=True, is_treatment=True),
        Node("Y", "structural", ("U", "C", "T"), is_outcome=True),
    ]


def _frontdoor_criterion() -> List[Node]:
    # C -> T, C -> Y, T -> M -> Y   (confounder observed, mediator hidden).
    # Same structure as Backdoor in the released data (see module docstring).
    return [
        Node("C", "root_normal", observed=True),
        Node("T", "structural", ("C",), binarize=True, is_treatment=True),
        Node("M", "structural", ("T",), observed=False),
        Node("Y", "structural", ("C", "M"), is_outcome=True),
    ]


_BUILDERS: Dict[str, Callable[[], List[Node]]] = {
    "Observed_Confounder":              _observed_confounder,
    "Backdoor_Criterion":               _backdoor_criterion,
    "Observed_Mediator":                _observed_mediator,
    "Observed_Mediator_and_Confounder": _observed_mediator_and_confounder,
    "Unobserved_Confounder":            _unobserved_confounder,
    "Frontdoor_Criterion":              _frontdoor_criterion,
}


def build_dag(case_study: str) -> List[Node]:
    """Return the fixed case-study template as a topologically-ordered list."""
    if case_study not in _BUILDERS:
        raise ValueError(f"unknown case study {case_study!r}; pick from {CASE_STUDIES}")
    nodes = _BUILDERS[case_study]()
    seen: set[str] = set()
    for n in nodes:
        for p in n.parents:
            if p not in seen:
                raise AssertionError(f"{case_study}: node {n.name!r} parent {p!r} "
                                     "not defined earlier (must be topological)")
        seen.add(n.name)
    return nodes


# ── One sampled SCM / realization ─────────────────────────────────────────────
@dataclass
class Realization:
    case_study: str
    n_context: int
    seed: int
    exo_std: float
    noise_std: float
    feature_names: List[str]
    X: np.ndarray       # (N, d)   observed covariates
    T: np.ndarray       # (N,)     binary treatment
    Y: np.ndarray       # (N,)     factual outcome
    cate: np.ndarray    # (N,)     mu_1 - mu_0
    mu_0: np.ndarray    # (N,)     E[Y | do(T=0), x]
    mu_1: np.ndarray    # (N,)     E[Y | do(T=1), x]
    graph_edges: List[Tuple[str, str]] = field(default_factory=list)
    cate_shift: float = 0.0   # applied beta (0 for mediated cases)


class _SampledSCM:
    """A single sampled SCM: fixed graph, weights, activations, and exogenous
    noise. Supports the observational treatment mechanism and interventions
    do(T=t); the same exogenous noise is reused across all passes (so the
    additive Y-noise cancels in mu_1 - mu_0)."""

    def __init__(self, nodes: List[Node], N: int, rng: np.random.Generator,
                 cate_shift: float = 0.0, noise_scale: float = 1.0,
                 noise_dist: str = "gaussian", noise_param: float = 0.0,
                 noise_standardize: bool = True,
                 hetero_gamma: float = 0.0, noise_target: str = "all",
                 mix_g: str = "", mix_lambda: float = 0.0, mix_target: str = "outcome",
                 rng_data: np.random.Generator | None = None, consts: dict | None = None):
        self.nodes = nodes
        self.t_name = next(n.name for n in nodes if n.is_treatment)
        self.y_name = next(n.name for n in nodes if n.is_outcome)
        self.N = int(N)
        # Constant additive treatment effect beta*T, injected into the outcome
        # ONLY when T is a direct parent of Y (a "meaningful" direct effect).
        # This shifts cate / mu_1 / treated-Y by beta and leaves mediated cases
        # (Backdoor, Frontdoor — no direct T->Y edge) untouched, so they stay
        # centered at 0 while the four direct-effect cases re-center on beta.
        self._shift = float(cate_shift)
        self._shift_outcome = (self._shift != 0.0
                               and self.t_name in next(n for n in nodes
                                                       if n.is_outcome).parents)

        # Per-realization noise scales (verified against the shipped pkls).
        # `noise_scale` multiplies σ_ε only — it does NOT change the true CATE
        # (μ are noiseless), just the width of the conditional Y distribution.
        # Used for the σ_ε resolution sweep (scale=1 == the original data).
        self.exo_std = float(rng.uniform(1.0, 3.0))
        self.noise_scale = float(noise_scale)
        self.noise_std = float(0.3 * rng.beta(1.0, 5.0))   # in-prior σ_ε (before noise_scale)

        # Noise perturbations (Experiment B) on the targeted structural nodes:
        # noise_target "all" = every structural node, "outcome" = Y only.
        # ε = σ_node · q(z), z ~ N(0,1) drawn once per node (common random numbers),
        # q = standardized quantile map of `noise_dist` (unit variance, mean 0):
        #   gaussian: q(z) = z;  t: Student-t_ν (ν = noise_param);
        #   gamma: centred Gamma with shape k = noise_param.
        # σ_node = σ_ε · noise_scale; heteroscedastic (hetero_gamma = γ > 0):
        # σ_i = σ_node · sqrt((1 + γ s_i²) / (1 + γ)), s_i = the node's standardized
        # linear predictor w·pa on the observational pass (keeps E[σ_i²] = σ_node²).
        if noise_target not in ("all", "outcome"):
            raise ValueError(f"noise_target must be 'all' or 'outcome', got {noise_target!r}")
        self.noise_dist, self.noise_param = noise_dist, float(noise_param)
        self.noise_standardize = bool(noise_standardize)   # False: textbook form, own variance
        self.hetero_gamma, self.noise_target = float(hetero_gamma), noise_target
        self._hetero_ref: Dict[str, Tuple[float, float]] = {}   # (mean, sd) of lin per node

        # Function perturbation (Experiment C) on the targeted node(s):
        # f_lambda(lin) = (1 - lambda) f_train(lin) + lambda * a * g(lin), with g in
        # MIX_FUNCTIONS and a = sd(f_train(lin)) / sd(g(lin)) on the observational pass
        # (same scale, new shape); a is reused for the interventional passes.
        # mix_target: "outcome" (Y), "treatment" (T) or "both".
        if mix_g and mix_g not in MIX_FUNCTIONS:
            raise ValueError(f"unknown mix_g {mix_g!r}; pick from {tuple(MIX_FUNCTIONS)}")
        if mix_target not in ("outcome", "treatment", "both"):
            raise ValueError(f"mix_target must be outcome/treatment/both, got {mix_target!r}")
        self.mix_g, self.mix_lambda, self.mix_target = mix_g, float(mix_lambda), mix_target
        self._mix_scale: Dict[str, float] = {}

        self._weights: Dict[str, np.ndarray] = {}
        self._activation: Dict[str, str] = {}
        self._noise: Dict[str, np.ndarray] = {}      # sampled ONCE, reused
        self._root: Dict[str, np.ndarray] = {}
        self._t_threshold: float | None = None

        # SCM parameters come from `rng`; the data (roots, base noise z) from `rng_data`.
        # With rng_data=None both come from `rng` in the original draw order, so the
        # default data are unchanged. A separate rng_data redraws data from the SAME SCM
        # (Experiment D). `consts` (from .constants() on a large reference sample) fixes
        # the sample-dependent SCM constants: the treatment threshold, the heteroscedastic
        # standardization and the function-path scale.
        rd = rng if rng_data is None else rng_data
        if consts:
            self._t_threshold = consts.get("t_threshold")
            self._hetero_ref = dict(consts.get("hetero_ref", {}))
            self._mix_scale = dict(consts.get("mix_scale", {}))

        for n in nodes:
            if n.kind == "root_normal":
                self._root[n.name] = rd.normal(0.0, self.exo_std, size=N)
            elif n.kind == "root_bernoulli":
                self._root[n.name] = (rd.random(N) < 0.5).astype(np.float64)
            elif n.kind == "structural":
                p = len(n.parents)
                bound = 1.0 / np.sqrt(p) if p > 0 else 1.0   # Kaiming (fan_in)
                self._weights[n.name] = rng.uniform(-bound, bound, size=p)
                self._activation[n.name] = _ACTIVATION_NAMES[
                    rng.integers(len(_ACTIVATION_NAMES))]
                z = rd.standard_normal(size=N)
                if self._targeted(n):
                    self._noise[n.name] = self.noise_std * self.noise_scale * _standardized_quantile(
                        z, self.noise_dist, self.noise_param, self.noise_standardize)
                else:
                    self._noise[n.name] = self.noise_std * z
            else:
                raise ValueError(f"unknown node kind {n.kind!r}")

    def _f(self, n: Node, lin: np.ndarray) -> np.ndarray:
        """The node's mechanism f(lin): the sampled activation, mixed with g if targeted."""
        f = ACTIVATIONS[self._activation[n.name]](lin)
        hit = {"outcome": n.is_outcome, "treatment": n.is_treatment,
               "both": n.is_outcome or n.is_treatment}[self.mix_target]
        if not self.mix_g or self.mix_lambda == 0.0 or not hit:
            return f
        g = MIX_FUNCTIONS[self.mix_g](lin)
        if n.name not in self._mix_scale:
            sg = g.std()
            self._mix_scale[n.name] = float(f.std() / sg) if sg > 0 else 0.0
        return (1.0 - self.mix_lambda) * f + self.mix_lambda * self._mix_scale[n.name] * g

    def _targeted(self, n: Node) -> bool:
        return self.noise_target == "all" or n.is_outcome

    def _eps(self, n: Node, lin: np.ndarray) -> np.ndarray:
        """The node's additive noise; the heteroscedastic standardization (mean, sd of
        lin) is fixed on the first (observational) pass unless given in `consts`."""
        if self.hetero_gamma == 0.0 or not self._targeted(n):
            return self._noise[n.name]
        if n.name not in self._hetero_ref:
            self._hetero_ref[n.name] = (float(lin.mean()), float(lin.std()))
        mu, sd = self._hetero_ref[n.name]
        s = (lin - mu) / sd if sd > 0 else np.zeros_like(lin)
        g = self.hetero_gamma
        return np.sqrt((1.0 + g * s ** 2) / (1.0 + g)) * self._noise[n.name]

    def constants(self) -> dict:
        """Sample-dependent SCM constants, to pass as `consts` (call after forward())."""
        return dict(t_threshold=self._t_threshold, hetero_ref=dict(self._hetero_ref),
                    mix_scale=dict(self._mix_scale))

    def graph_edges(self) -> List[Tuple[str, str]]:
        return [(p, n.name) for n in self.nodes for p in n.parents]

    def _linear(self, node: Node, values: Dict[str, np.ndarray]) -> np.ndarray:
        if len(node.parents) == 0:
            return np.zeros(self.N)
        return np.stack([values[p] for p in node.parents], axis=-1) @ self._weights[node.name]

    def _linear_activation(self, node: Node, values: Dict[str, np.ndarray]) -> np.ndarray:
        return ACTIVATIONS[self._activation[node.name]](self._linear(node, values))

    def forward(self, do_T: float | None = None,
                y_noiseless: bool = False) -> Dict[str, np.ndarray]:
        """Evaluate all nodes in topological order. `do_T` overrides the
        treatment; `y_noiseless` drops the outcome's additive noise (for mu_t)."""
        values: Dict[str, np.ndarray] = {}
        for n in self.nodes:
            if n.is_treatment:
                if do_T is not None:
                    values[n.name] = np.full(self.N, float(do_T))
                elif n.kind == "root_bernoulli":
                    values[n.name] = self._root[n.name]
                else:  # structural treatment -> median-split binarization
                    lin = self._linear(n, values)
                    cont = self._f(n, lin) + self._eps(n, lin)
                    if self._t_threshold is None:
                        self._t_threshold = float(np.median(cont))
                    values[n.name] = (cont > self._t_threshold).astype(np.float64)
            elif n.kind in ("root_normal", "root_bernoulli"):
                values[n.name] = self._root[n.name]
            else:  # structural mediator / outcome
                lin = self._linear(n, values)
                out = self._f(n, lin)
                if not (n.is_outcome and y_noiseless):
                    out = out + self._eps(n, lin)
                if n.is_outcome and self._shift_outcome:
                    out = out + self._shift * values[self.t_name]
                values[n.name] = out
        return values


def scm_constants(case_study: str, seed: int, n_ref: int = 100_000, ref_seed: int = 0,
                  **noise_kw) -> dict:
    """SCM constants of the SCM drawn with `seed`, from one large reference sample
    (data stream `ref_seed`); pass to generate_realization(consts=...) to keep the SCM
    fixed across data sets and sample sizes."""
    scm = _SampledSCM(build_dag(case_study), N=n_ref, rng=np.random.default_rng(seed),
                      rng_data=np.random.default_rng(ref_seed), **noise_kw)
    scm.forward()
    return scm.constants()


def generate_realization(case_study: str, n_context: int, seed: int,
                         cate_shift: float = 0.0,
                         noise_scale: float = 1.0, data_seed: int | None = None,
                         consts: dict | None = None, **noise_kw) -> Realization:
    """Sample one SCM realization. Raises ValueError on non-finite draws
    (nonlinearity blow-up) so the caller can resample with another seed.
    `noise_kw`: noise_dist, noise_param, hetero_gamma, noise_target, and
    mix_g, mix_lambda, mix_target (see _SampledSCM). `data_seed` draws the data from a
    separate stream (same SCM, new data); `consts` fixes the SCM constants."""
    nodes = build_dag(case_study)
    rng = np.random.default_rng(seed)
    rng_data = None if data_seed is None else np.random.default_rng(data_seed)
    scm = _SampledSCM(nodes, N=n_context, rng=rng, cate_shift=cate_shift,
                      noise_scale=noise_scale, rng_data=rng_data, consts=consts, **noise_kw)

    obs = scm.forward()                                    # observational
    mu_0 = scm.forward(do_T=0.0, y_noiseless=True)[scm.y_name]
    mu_1 = scm.forward(do_T=1.0, y_noiseless=True)[scm.y_name]

    feature_names = [n.name for n in nodes if n.observed]
    X = (np.stack([obs[name] for name in feature_names], axis=-1)
         if feature_names else np.zeros((n_context, 0)))
    T, Y, cate = obs[scm.t_name], obs[scm.y_name], mu_1 - mu_0

    for arr in (X, T, Y, mu_0, mu_1):
        if not np.all(np.isfinite(arr)):
            raise ValueError("non-finite values (nonlinearity blow-up); resample")

    return Realization(
        case_study=case_study, n_context=int(n_context), seed=int(seed),
        exo_std=scm.exo_std, noise_std=scm.noise_std, feature_names=feature_names,
        X=X.astype(np.float32), T=T.astype(np.float32), Y=Y.astype(np.float32),
        cate=cate.astype(np.float32),
        mu_0=mu_0.astype(np.float32), mu_1=mu_1.astype(np.float32),
        graph_edges=scm.graph_edges(),
        cate_shift=scm._shift if scm._shift_outcome else 0.0,
    )


# ── Saving + sweep driver ─────────────────────────────────────────────────────
def save_realization(r: Realization, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path, X=r.X, T=r.T, Y=r.Y, cate=r.cate, mu_0=r.mu_0, mu_1=r.mu_1,
        feature_names=np.asarray(r.feature_names, dtype=object),
        graph_edges=np.asarray(r.graph_edges, dtype=object),
        case_study=r.case_study, n_context=r.n_context, seed=r.seed,
        exo_std=r.exo_std, noise_std=r.noise_std, cate_shift=r.cate_shift,
    )


def _cell_seed(seed_base: int, case_idx: int, N: int, r: int, attempt: int) -> int:
    """Independent, reproducible seed per (case, N, realization, attempt)."""
    return int(np.random.SeedSequence([seed_base, case_idx, N, r, attempt])
               .generate_state(1)[0])


def generate_sweep(out_dir: str, cases: List[str], context_sizes: List[int],
                   n_realizations: int, seed_base: int, overwrite: bool,
                   cate_shift: float = 0.0, noise_scale: float = 1.0,
                   max_resamples: int = 50) -> dict:
    """Generate the full (case x N x realization) grid of .npz files and a
    manifest. Returns the manifest dict (also written to manifest.json)."""
    t0 = time.time()
    manifest = {
        "cases": cases, "context_sizes": context_sizes,
        "n_realizations": n_realizations, "seed_base": seed_base,
        "cate_shift": cate_shift, "noise_scale": noise_scale,
        "source": "reimplementation of github.com/jr2021/Do-PFN case studies",
        "cells": [],
    }
    for case_idx, case in enumerate(cases):
        for N in context_sizes:
            cell_dir = os.path.join(out_dir, case, f"N{N}")
            written = resampled = 0
            for r in range(n_realizations):
                path = os.path.join(cell_dir, f"{case}_{r}.npz")
                if os.path.exists(path) and not overwrite:
                    written += 1
                    continue
                for attempt in range(max_resamples):
                    seed = _cell_seed(seed_base, case_idx, N, r, attempt)
                    try:
                        real = generate_realization(case, N, seed, cate_shift,
                                                    noise_scale)
                        break
                    except ValueError:
                        resampled += 1
                else:
                    raise RuntimeError(f"{case} N{N} r{r}: {max_resamples} "
                                       "resamples all non-finite")
                save_realization(real, path)
                written += 1
            manifest["cells"].append({"case": case, "N": N, "n_written": written,
                                      "n_resampled": resampled})
            print(f"[gen] {case:34s} N={N:<5d} written={written:3d} "
                  f"resampled={resampled:3d} ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[gen] done: {len(manifest['cells'])} cells in "
          f"{time.time() - t0:.0f}s -> {out_dir}", flush=True)
    return manifest


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="case_study/data",
                   help="Output root for the .npz grid + manifest.json.")
    p.add_argument("--cases", nargs="*", default=list(CASE_STUDIES),
                   choices=list(CASE_STUDIES),
                   help="Which case studies to generate (default: all six).")
    p.add_argument("--context-sizes", nargs="*", type=int,
                   default=[50, 100, 200, 500, 1000],
                   help="Context-size sweep N (observational samples per dataset).")
    p.add_argument("--n-realizations", type=int, default=100,
                   help="Independent SCM realizations per (case, N) cell.")
    p.add_argument("--seed-base", type=int, default=0,
                   help="Base seed; each grid cell derives an independent stream.")
    p.add_argument("--cate-shift", type=float, default=0.0,
                   help="Constant additive treatment effect beta added as beta*T "
                        "to the outcome, but ONLY for cases with a direct T->Y edge "
                        "(Observed_Confounder, Observed_Mediator, "
                        "Observed_Mediator_and_Confounder, Unobserved_Confounder). "
                        "Re-centers their CATE/ATE on beta; Backdoor/Frontdoor "
                        "(no direct T->Y) stay centered at 0. Default 0.")
    p.add_argument("--noise-scale", type=float, default=1.0,
                   help="Multiplier on the endogenous additive-noise std σ_ε "
                        "(σ_ε = noise_scale · 0.3 · Beta(1,5)). Does NOT change the "
                        "true CATE, only the conditional-Y width. scale=1 == the "
                        "original data; large values widen the CID so coarse bins "
                        "resolve it (σ_ε resolution sweep). Default 1.0.")
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate cells even if the .npz already exists.")
    return p.parse_args()


def main():
    a = _parse_args()
    generate_sweep(out_dir=a.out_dir, cases=a.cases, context_sizes=a.context_sizes,
                   n_realizations=a.n_realizations, seed_base=a.seed_base,
                   cate_shift=a.cate_shift, noise_scale=a.noise_scale,
                   overwrite=a.overwrite)


if __name__ == "__main__":
    main()
