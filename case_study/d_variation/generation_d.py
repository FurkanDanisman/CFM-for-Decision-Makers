"""Covariate-count sweep for the Do-PFN synthetic case studies.

Generalizes the six fixed case-study DAGs in `case_study/generation.py` so the
number of OBSERVED covariates `d` becomes a free parameter, by *scaling the
causal roles*: the single confounder C becomes C1..Ck (every one a genuine
confounder into T and Y) and the single observed mediator M becomes M1..Mj
(every one on a T -> M -> Y path).

HIDDEN nodes (U in Unobserved_Confounder, the mediator in Backdoor/Frontdoor)
scale at one per 10 covariates, floored at 1:
    d = 2, 3, 5, 10 -> 1 hidden;  20 -> 2;  30 -> 3;  40 -> 4;  50 -> 5.
They are not covariates, so X stays (N, d) regardless.

Every case therefore keeps its identification structure at every d:

    Observed_Confounder               C1..Cd -> T, C1..Cd -> Y, T -> Y
    Backdoor_Criterion                C1..Cd -> T/Y, T -> M1..Mh(hidden) -> Y
    Observed_Mediator                 T -> M1..Md -> Y, T -> Y
    Observed_Mediator_and_Confounder  C1..Ck -> T/Y, T -> M1..Mj -> Y, T -> Y
                                      with k = ceil(d/2), j = floor(d/2)
    Unobserved_Confounder             U1..Uh(hidden) + C1..Cd -> T/Y, T -> Y
    Frontdoor_Criterion               C1..Cd -> T/Y, T -> M1..Mh(hidden) -> Y

    with h = max(1, d // 10) hidden nodes.

At the baseline d of each case (d=1, or d=2 for Observed_Mediator_and_Confounder)
these builders reduce to exactly the templates in generation.py.

Weight init is unchanged (Kaiming fan-in, bound = 1/sqrt(#parents)), so the
pre-activation scale of each node stays O(1) as d grows; what grows with d is
the amount of confounding / mediation the estimator must handle.

CLI
---
    python case_study/d_variation/generation_d.py \
        --out-dir case_study/d_variation \
        --n-covariates 2 3 5 10 20 30 40 50

Output layout:  <out-dir>/d{K}/<Case>/N{N}/{Case}_{r}.npz  (+ per-d manifest.json,
                and a top-level manifest.json describing the sweep)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Tuple

import numpy as np

# Reuse the sampler, the Realization container and the writer verbatim, so the
# SCM semantics here are identical to the baseline generator.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generation import (  # noqa: E402
    CASE_STUDIES,
    Node,
    Realization,
    _SampledSCM,
    save_realization,
)


# ── d-parameterized DAG builders ─────────────────────────────────────────────
def _n_hidden(d: int) -> int:
    """Number of HIDDEN nodes as a function of the observed covariate count:
    one per 10 covariates, floored at 1 (d=2/3/5/10 -> 1, then 20 -> 2,
    30 -> 3, 40 -> 4, 50 -> 5). The floor is required: Backdoor/Frontdoor need
    at least one mediator and Unobserved_Confounder at least one U for the case
    to be well-defined. Hidden nodes are NOT covariates, so X stays (N, d)."""
    return max(1, d // 10)


def _confounders(k: int) -> List[Node]:
    return [Node(f"C{i + 1}", "root_normal", observed=True) for i in range(k)]


def _observed_confounder(d: int) -> List[Node]:
    Cs = _confounders(d)
    cn = tuple(n.name for n in Cs)
    return Cs + [
        Node("T", "structural", cn, binarize=True, is_treatment=True),
        Node("Y", "structural", cn + ("T",), is_outcome=True),
    ]


def _mediated_hidden(d: int) -> List[Node]:
    """Backdoor / Frontdoor: d observed confounders, one HIDDEN mediator,
    no direct T -> Y edge."""
    Cs = _confounders(d)
    cn = tuple(n.name for n in Cs)
    Ms = [Node(f"M{i + 1}", "structural", ("T",), observed=False)
          for i in range(_n_hidden(d))]
    mn = tuple(n.name for n in Ms)
    return (Cs
            + [Node("T", "structural", cn, binarize=True, is_treatment=True)]
            + Ms
            + [Node("Y", "structural", cn + mn, is_outcome=True)])


def _observed_mediator(d: int) -> List[Node]:
    Ms = [Node(f"M{i + 1}", "structural", ("T",), observed=True) for i in range(d)]
    mn = tuple(n.name for n in Ms)
    return ([Node("T", "root_bernoulli", is_treatment=True)] + Ms
            + [Node("Y", "structural", ("T",) + mn, is_outcome=True)])


def _observed_mediator_and_confounder(d: int) -> List[Node]:
    if d < 2:
        raise ValueError("Observed_Mediator_and_Confounder needs d >= 2 "
                         "(at least one confounder and one mediator)")
    k = (d + 1) // 2          # confounders  (ceil)
    j = d - k                 # mediators    (floor)
    Cs = _confounders(k)
    cn = tuple(n.name for n in Cs)
    Ms = [Node(f"M{i + 1}", "structural", ("T",), observed=True) for i in range(j)]
    mn = tuple(n.name for n in Ms)
    return (Cs
            + [Node("T", "structural", cn, binarize=True, is_treatment=True)]
            + Ms
            + [Node("Y", "structural", cn + ("T",) + mn, is_outcome=True)])


def _unobserved_confounder(d: int) -> List[Node]:
    Us = [Node(f"U{i + 1}", "root_normal", observed=False)
          for i in range(_n_hidden(d))]
    un = tuple(n.name for n in Us)
    Cs = _confounders(d)
    cn = tuple(n.name for n in Cs)
    return Us + Cs + [
        Node("T", "structural", un + cn, binarize=True, is_treatment=True),
        Node("Y", "structural", un + cn + ("T",), is_outcome=True),
    ]


_BUILDERS_D: Dict[str, Callable[[int], List[Node]]] = {
    "Observed_Confounder":              _observed_confounder,
    "Backdoor_Criterion":               _mediated_hidden,
    "Observed_Mediator":                _observed_mediator,
    "Observed_Mediator_and_Confounder": _observed_mediator_and_confounder,
    "Unobserved_Confounder":            _unobserved_confounder,
    "Frontdoor_Criterion":              _mediated_hidden,
}


def build_dag_d(case_study: str, d: int) -> List[Node]:
    """Return the d-covariate case-study DAG, topologically ordered."""
    if case_study not in _BUILDERS_D:
        raise ValueError(f"unknown case study {case_study!r}; pick from {CASE_STUDIES}")
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    nodes = _BUILDERS_D[case_study](d)

    seen: set[str] = set()
    for n in nodes:
        for p in n.parents:
            if p not in seen:
                raise AssertionError(f"{case_study} d={d}: node {n.name!r} parent "
                                     f"{p!r} not defined earlier (must be topological)")
        seen.add(n.name)
    n_obs = sum(1 for n in nodes if n.observed)
    if n_obs != d:
        raise AssertionError(f"{case_study} d={d}: built {n_obs} observed covariates")
    return nodes


# ── realization / sweep ──────────────────────────────────────────────────────
def generate_realization_d(case_study: str, d: int, n_context: int, seed: int,
                           cate_shift: float = 0.0) -> Realization:
    """Sample one d-covariate SCM realization. Raises ValueError on non-finite
    draws so the caller can resample with another seed."""
    nodes = build_dag_d(case_study, d)
    rng = np.random.default_rng(seed)
    scm = _SampledSCM(nodes, N=n_context, rng=rng, cate_shift=cate_shift)

    obs = scm.forward()
    mu_0 = scm.forward(do_T=0.0, y_noiseless=True)[scm.y_name]
    mu_1 = scm.forward(do_T=1.0, y_noiseless=True)[scm.y_name]

    feature_names = [n.name for n in nodes if n.observed]
    X = np.stack([obs[name] for name in feature_names], axis=-1)
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


def _cell_seed_d(seed_base: int, d: int, case_idx: int, N: int, r: int,
                 attempt: int) -> int:
    """Independent, reproducible stream per (d, case, N, realization, attempt)."""
    return int(np.random.SeedSequence([seed_base, d, case_idx, N, r, attempt])
               .generate_state(1)[0])


def generate_for_d(out_root: str, d: int, cases: List[str], context_sizes: List[int],
                   n_realizations: int, seed_base: int, overwrite: bool,
                   cate_shift: float, max_resamples: int = 50) -> dict:
    """Generate the (case x N x realization) grid for one covariate count."""
    t0 = time.time()
    d_dir = os.path.join(out_root, f"d{d}")
    manifest = {
        "n_covariates": d, "cases": cases, "context_sizes": context_sizes,
        "n_realizations": n_realizations, "seed_base": seed_base,
        "cate_shift": cate_shift,
        "covariate_scaling": "causal-roles (C -> C1..Ck confounders, "
                             "M -> M1..Mj mediators)",
        "n_hidden": _n_hidden(d),
        "hidden_scaling": "max(1, d // 10) hidden nodes "
                          "(U in Unobserved_Confounder, mediators in "
                          "Backdoor/Frontdoor); not counted as covariates",
        "source": "reimplementation of github.com/jr2021/Do-PFN case studies",
        "cells": [],
    }
    for case_idx, case in enumerate(cases):
        nodes = build_dag_d(case, d)
        feat = [n.name for n in nodes if n.observed]
        for N in context_sizes:
            cell_dir = os.path.join(d_dir, case, f"N{N}")
            written = resampled = 0
            for r in range(n_realizations):
                path = os.path.join(cell_dir, f"{case}_{r}.npz")
                if os.path.exists(path) and not overwrite:
                    written += 1
                    continue
                for attempt in range(max_resamples):
                    seed = _cell_seed_d(seed_base, d, case_idx, N, r, attempt)
                    try:
                        real = generate_realization_d(case, d, N, seed, cate_shift)
                        break
                    except ValueError:
                        resampled += 1
                else:
                    raise RuntimeError(f"d{d} {case} N{N} r{r}: {max_resamples} "
                                       "resamples all non-finite")
                save_realization(real, path)
                written += 1
            manifest["cells"].append({"case": case, "N": N, "n_written": written,
                                      "n_resampled": resampled,
                                      "feature_names": feat})
            print(f"[d{d:<3d}] {case:34s} N={N:<5d} written={written:3d} "
                  f"resampled={resampled:3d} d_obs={len(feat):2d} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(d_dir, exist_ok=True)
    with open(os.path.join(d_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def generate_sweep_d(out_dir: str, d_values: List[int], cases: List[str],
                     context_sizes: List[int], n_realizations: int,
                     seed_base: int, overwrite: bool, cate_shift: float) -> dict:
    t0 = time.time()
    top = {
        "sweep": "n_covariates", "d_values": d_values, "cases": cases,
        "context_sizes": context_sizes, "n_realizations": n_realizations,
        "seed_base": seed_base, "cate_shift": cate_shift,
        "covariate_scaling": "causal-roles",
        "hidden_scaling": "max(1, d // 10)",
        "n_hidden_by_d": {f"d{d}": _n_hidden(d) for d in d_values},
        "roots": {},
    }
    for d in d_values:
        m = generate_for_d(out_dir, d, cases, context_sizes, n_realizations,
                           seed_base, overwrite, cate_shift)
        top["roots"][f"d{d}"] = {
            "path": os.path.join(out_dir, f"d{d}"),
            "n_files": sum(c["n_written"] for c in m["cells"]),
            "feature_names": {c["case"]: c["feature_names"] for c in m["cells"]},
        }
        print(f"[sweep] d={d} done ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(top, f, indent=2)
    print(f"[sweep] all done: {len(d_values)} covariate counts in "
          f"{time.time() - t0:.0f}s -> {out_dir}", flush=True)
    return top


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="case_study/d_variation")
    p.add_argument("--n-covariates", nargs="*", type=int,
                   default=[2, 3, 5, 10, 20, 30, 40, 50],
                   help="Observed-covariate counts to sweep.")
    p.add_argument("--cases", nargs="*", default=list(CASE_STUDIES),
                   choices=list(CASE_STUDIES))
    p.add_argument("--context-sizes", nargs="*", type=int, default=[200, 500, 1000])
    p.add_argument("--n-realizations", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--cate-shift", type=float, default=0.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    a = _parse_args()
    generate_sweep_d(out_dir=a.out_dir, d_values=a.n_covariates, cases=a.cases,
                     context_sizes=a.context_sizes,
                     n_realizations=a.n_realizations, seed_base=a.seed_base,
                     overwrite=a.overwrite, cate_shift=a.cate_shift)


if __name__ == "__main__":
    main()
