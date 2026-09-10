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
                 cate_shift: float = 0.0, noise_scale: float = 1.0):
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
        self.noise_std = float(noise_scale * 0.3 * rng.beta(1.0, 5.0))

        self._weights: Dict[str, np.ndarray] = {}
        self._activation: Dict[str, str] = {}
        self._noise: Dict[str, np.ndarray] = {}      # sampled ONCE, reused
        self._root: Dict[str, np.ndarray] = {}
        self._t_threshold: float | None = None

        for n in nodes:
            if n.kind == "root_normal":
                self._root[n.name] = rng.normal(0.0, self.exo_std, size=N)
            elif n.kind == "root_bernoulli":
                self._root[n.name] = (rng.random(N) < 0.5).astype(np.float64)
            elif n.kind == "structural":
                p = len(n.parents)
                bound = 1.0 / np.sqrt(p) if p > 0 else 1.0   # Kaiming (fan_in)
                self._weights[n.name] = rng.uniform(-bound, bound, size=p)
                self._activation[n.name] = _ACTIVATION_NAMES[
                    rng.integers(len(_ACTIVATION_NAMES))]
                self._noise[n.name] = rng.normal(0.0, self.noise_std, size=N)
            else:
                raise ValueError(f"unknown node kind {n.kind!r}")

    def graph_edges(self) -> List[Tuple[str, str]]:
        return [(p, n.name) for n in self.nodes for p in n.parents]

    def _linear_activation(self, node: Node, values: Dict[str, np.ndarray]) -> np.ndarray:
        w = self._weights[node.name]
        if len(node.parents) == 0:
            lin = np.zeros(self.N)
        else:
            lin = np.stack([values[p] for p in node.parents], axis=-1) @ w
        return ACTIVATIONS[self._activation[node.name]](lin)

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
                    cont = self._linear_activation(n, values) + self._noise[n.name]
                    if self._t_threshold is None:
                        self._t_threshold = float(np.median(cont))
                    values[n.name] = (cont > self._t_threshold).astype(np.float64)
            elif n.kind in ("root_normal", "root_bernoulli"):
                values[n.name] = self._root[n.name]
            else:  # structural mediator / outcome
                out = self._linear_activation(n, values)
                if not (n.is_outcome and y_noiseless):
                    out = out + self._noise[n.name]
                if n.is_outcome and self._shift_outcome:
                    out = out + self._shift * values[self.t_name]
                values[n.name] = out
        return values


def generate_realization(case_study: str, n_context: int, seed: int,
                         cate_shift: float = 0.0,
                         noise_scale: float = 1.0) -> Realization:
    """Sample one SCM realization. Raises ValueError on non-finite draws
    (nonlinearity blow-up) so the caller can resample with another seed."""
    nodes = build_dag(case_study)
    rng = np.random.default_rng(seed)
    scm = _SampledSCM(nodes, N=n_context, rng=rng, cate_shift=cate_shift,
                      noise_scale=noise_scale)

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
