"""Graph-conditioned model with our 2D BarDistribution joint head.

Wraps UWYK's `GraphConditionedInterventionalPFN` architecture (from
`external/uwyk/src/models/`) and:

  1. Overrides the final projection so the output has `total_params(J)`
     dimensions (= J**2 + 9 + 4) instead of `num_bars + 4`.
  2. Adds a learned `null_t_intv` parameter so the query no longer needs
     a hard-coded arm — the joint head predicts BOTH arms at once.

Everything else (feature/label embeddings, two-way attention blocks,
graph_encoder GCN + soft-attention on the adjacency matrix, masking) is
inherited from UWYK unchanged.

Forward signature:
    logits = model(X_obs, T_obs, Y_obs, X_intv, adjacency_matrix,
                   T_intv=None)   # T_intv defaults to the null token
    # logits: (B, M, J**2 + 9 + 4)
"""
from __future__ import annotations
import os
import sys
from typing import Optional

import torch
import torch.nn as nn


def _wire_uwyk_paths(uwyk_root: Optional[str] = None):
    """Insert UWYK's src/ AND its parent onto sys.path so both
    'models.*' and 'src.models.*' imports resolve."""
    uwyk_root = uwyk_root or os.environ.get(
        'UWYK_ROOT', '/scratch/furkanbd/rpfn_bench_kit/external/uwyk')
    uwyk_src = os.path.join(uwyk_root, 'src')
    for p in (uwyk_src, uwyk_root):
        if p not in sys.path:
            sys.path.insert(0, p)


_wire_uwyk_paths()

from models.GraphConditionedInterventionalPFN import (  # noqa: E402
    GraphConditionedInterventionalPFN,
)

_REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)
from losses.BarDistribution2D import total_params  # noqa: E402


class GraphConditioned2DHead(GraphConditionedInterventionalPFN):
    """Same architecture as UWYK's graph-conditioned model, but with our 2D
    BarDistribution joint head at the output.

    We do NOT modify the graph_encoder or any block-level attention path.
    Two changes from the parent:

    1. Final projection has `total_params(J)` outputs, not `num_bars + 4`.
    2. A learned `null_t_intv` parameter fills the query T_intv slot when
       T_intv is not supplied (the joint-head training regime).

    Parameters
    ----------
    J : int
        Grid side length for the 2D bar distribution.
        `total_params(J) = J**2 + 9 + 4`.
    **kwargs
        All other kwargs pass through to GraphConditionedInterventionalPFN's
        __init__ (num_features, d_model, depth, heads, hidden_mult, dropout,
        etc.). We inject `num_bars=J` so the parent's projection has a size
        we can swap out cleanly.
    """

    def __init__(self, *, J: int = 100, **kwargs):
        kwargs['num_bars'] = kwargs.get('num_bars', 0) or J
        super().__init__(**kwargs)
        self.J = J
        self.output_dim_2d = total_params(J)

        proj_attr = None
        for cand in ('output_proj', 'head', 'y_head', 'proj_out', 'output_head',
                     'output', 'final_proj', 'to_out'):
            if hasattr(self, cand) and isinstance(getattr(self, cand), nn.Linear):
                proj_attr = cand
                break
        if proj_attr is None:
            lin_names = [n for n, m in self.named_modules()
                         if isinstance(m, nn.Linear)]
            raise RuntimeError(
                'GraphConditioned2DHead could not locate the final Linear '
                'projection on the parent module. Linear submodules found: '
                f'{lin_names}. Add the correct name to the candidates list '
                'in model_graph_2d.py.'
            )
        old_proj: nn.Linear = getattr(self, proj_attr)
        new_proj = nn.Linear(old_proj.in_features, self.output_dim_2d,
                              bias=(old_proj.bias is not None))
        nn.init.xavier_uniform_(new_proj.weight)
        if new_proj.bias is not None:
            nn.init.zeros_(new_proj.bias)
        setattr(self, proj_attr, new_proj)
        self._final_proj_name = proj_attr

        self.null_t_intv = nn.Parameter(torch.zeros(1, 1, 1))
        nn.init.normal_(self.null_t_intv, std=0.02)

        print(f'[GraphConditioned2DHead] swapped {proj_attr}: '
              f'{old_proj.in_features} → {self.output_dim_2d} '
              f'(J={J}, total_params(J) = {self.output_dim_2d})')

    def forward(
        self,
        X_obs: torch.Tensor,
        T_obs: torch.Tensor,
        Y_obs: torch.Tensor,
        X_intv: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        T_intv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass. T_intv defaults to the learned null token; supply it
        only if you want single-arm predictions (rare — the joint head is
        trained to marginalise T_intv into its output distribution).

        Returns logits of shape (B, M, J**2 + 9 + 4).
        """
        B, M = X_intv.shape[0], X_intv.shape[1]
        if T_intv is None:
            T_intv = self.null_t_intv.expand(B, M, 1).to(X_intv.dtype)
        elif T_intv.dim() == 2:
            T_intv = T_intv.unsqueeze(-1)
        return super().forward(
            X_obs=X_obs, T_obs=T_obs, Y_obs=Y_obs,
            X_intv=X_intv, T_intv=T_intv,
            adjacency_matrix=adjacency_matrix,
        )
