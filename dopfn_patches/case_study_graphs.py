"""DAG builders for DoPFN's 6 synthetic case studies (App. D.1 Table 1).

Each case-study graph is an integer-labelled `networkx.DiGraph`; when it
is fed to `SCMGenerator.create_scm_from_graph`, node <n> becomes
    - "U<n>" if it has no parents (exogenous), or
    - "X<n>" if it has parents (endogenous).
So we deliberately choose integer node IDs to control which nodes wind
up as exogenous / endogenous under DoPFN's relabelling convention.

Return tuple: (graph, t_key, y_key, x_idcs)
  graph     : nx.DiGraph over integer nodes
  t_key     : final string node name of the treatment node
              (e.g. "U0" or "X3") — used to set scm.t_key downstream
  y_key     : final string node name of the outcome node
  x_idcs    : list of integer node IDs that are the OBSERVED features to
              include in x_obs / x_int (never includes T or Y)
"""
from __future__ import annotations

from typing import Tuple, List

import networkx as nx


def _relabel_node(n: int, parents_of: dict) -> str:
    return f"X{n}" if len(parents_of[n]) > 0 else f"U{n}"


def _finalize(
    edges: List[Tuple[int, int]],
    n_nodes: int,
    t_node: int,
    y_node: int,
    x_nodes: List[int],
):
    G = nx.DiGraph()
    for n in range(n_nodes):
        G.add_node(n)
    G.add_edges_from(edges)
    parents_of = {n: list(G.predecessors(n)) for n in G.nodes}
    t_key = _relabel_node(t_node, parents_of)
    y_key = _relabel_node(y_node, parents_of)
    return G, t_key, y_key, list(x_nodes)


# ── Case-study specs (paper App. D.1 Table 1) ────────────────────────────────
# Node IDs are chosen so that DoPFN's relabel produces intuitive names.
# Confounders / roots get low IDs; T and Y are placed with parents when
# they are endogenous (so they become "X<id>" and can be intervened on
# via `do_interventions`).

def build_observed_confounder(num_confounders: int = 3):
    """X_i -> T, X_i -> Y, T -> Y (X_i are all observed confounders)."""
    # nodes 0..K-1 are confounders (U), node K is T (X), node K+1 is Y (X)
    K = int(num_confounders)
    t_node = K
    y_node = K + 1
    edges = []
    for i in range(K):
        edges.append((i, t_node))  # X_i -> T
        edges.append((i, y_node))  # X_i -> Y
    edges.append((t_node, y_node))  # T -> Y
    x_nodes = list(range(K))
    return _finalize(edges, n_nodes=K + 2, t_node=t_node, y_node=y_node, x_nodes=x_nodes)


def build_backdoor_criterion(num_confounders: int = 3):
    """Same DAG as Observed_Confounder — the causal-inference distinction
    is that the observed X_i satisfy the back-door criterion for T -> Y."""
    return build_observed_confounder(num_confounders=num_confounders)


def build_observed_mediator(num_mediators: int = 3):
    """T -> X_i -> Y, T -> Y  (X_i are observed mediators)."""
    K = int(num_mediators)
    t_node = 0                       # root -> U0
    y_node = 1 + K                   # last node, has all parents -> X<K+1>
    med_nodes = list(range(1, 1 + K))
    edges = []
    for m in med_nodes:
        edges.append((t_node, m))    # T -> X_m
        edges.append((m, y_node))    # X_m -> Y
    edges.append((t_node, y_node))   # T -> Y (direct)
    return _finalize(edges, n_nodes=1 + K + 1, t_node=t_node, y_node=y_node, x_nodes=med_nodes)


def build_observed_mediator_and_confounder(num_mediators: int = 3):
    """X_0 confounder (X_0 -> T, X_0 -> Y); T -> X_i -> Y for i >= 1;
    T -> Y direct.
    """
    K = int(num_mediators)
    conf_node = 0            # -> U0
    t_node = 1               # -> X1 (has parent U0)
    med_nodes = list(range(2, 2 + K))     # -> X2..X{K+1}
    y_node = 2 + K           # -> X{K+2}
    edges = [(conf_node, t_node), (conf_node, y_node), (t_node, y_node)]
    for m in med_nodes:
        edges.append((t_node, m))
        edges.append((m, y_node))
    x_nodes = [conf_node] + med_nodes
    return _finalize(edges, n_nodes=2 + K + 1, t_node=t_node, y_node=y_node, x_nodes=x_nodes)


def build_unobserved_confounder(num_spurious: int = 3):
    """U (unobserved) -> T, U -> Y; T -> Y. Plus `num_spurious` isolated
    exogenous variables (observed but not on the causal path)."""
    u_node = 0               # unobserved -> U0
    t_node = 1               # -> X1
    y_node = 2               # -> X2
    spur_nodes = list(range(3, 3 + int(num_spurious)))    # -> U3..
    edges = [(u_node, t_node), (u_node, y_node), (t_node, y_node)]
    # spurious vars are isolated (no edges); they are still exogenous roots.
    return _finalize(edges, n_nodes=3 + int(num_spurious),
                     t_node=t_node, y_node=y_node, x_nodes=spur_nodes)


def build_frontdoor_criterion(num_mediators: int = 3):
    """U -> T, U -> Y (confound); T -> X_M -> Y (front-door)."""
    u_node = 0
    t_node = 1
    med_nodes = list(range(2, 2 + int(num_mediators)))
    y_node = 2 + int(num_mediators)
    edges = [(u_node, t_node), (u_node, y_node)]  # confounding
    for m in med_nodes:
        edges.append((t_node, m))   # T -> X_m
        edges.append((m, y_node))   # X_m -> Y
    # NOTE: no direct T->Y edge in the front-door structure.
    return _finalize(edges, n_nodes=2 + int(num_mediators) + 1,
                     t_node=t_node, y_node=y_node, x_nodes=med_nodes)


CASE_STUDIES = (
    "Observed_Confounder",
    "Observed_Mediator",
    "Observed_Mediator_and_Confounder",
    "Unobserved_Confounder",
    "Frontdoor_Criterion",
    "Backdoor_Criterion",
)


_BUILDERS = {
    "Observed_Confounder":               build_observed_confounder,
    "Observed_Mediator":                 build_observed_mediator,
    "Observed_Mediator_and_Confounder":  build_observed_mediator_and_confounder,
    "Unobserved_Confounder":             build_unobserved_confounder,
    "Frontdoor_Criterion":               build_frontdoor_criterion,
    "Backdoor_Criterion":                build_backdoor_criterion,
}


def build_case_study_graph(case_study: str, num_features: int = 3):
    """Return (graph, t_key, y_key, x_idcs) for the requested case study.

    `num_features` parameterises the width of the case (# of
    confounders / mediators / spurious vars). Default 3 matches
    DoPFN's prior_data_example.py.
    """
    if case_study not in _BUILDERS:
        raise ValueError(f"unknown case study {case_study!r}; pick from {CASE_STUDIES}")
    return _BUILDERS[case_study](num_features)
