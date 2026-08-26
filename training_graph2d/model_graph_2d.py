"""Graph-conditioned model with our 2D BarDistribution joint head.

Wraps UWYK's `GraphConditionedInterventionalPFN` and:
  1. Sets output_dim = total_params(J) so the parent's regression_head is
     built at the right size — no post-hoc swap needed.
  2. Adds a learned `null_t_intv` parameter so the query no longer needs
     a hard-coded arm — the joint head predicts BOTH arms at once.

Forward returns the parent's dict {"predictions": (B, M, J**2 + 9 + 4)}.
"""
from __future__ import annotations
import os
import sys
from typing import Optional

import torch
import torch.nn as nn


def _wire_uwyk_paths(uwyk_root: Optional[str] = None):
    uwyk_root = uwyk_root or os.environ.get(
        'UWYK_ROOT',
        os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'external', 'uwyk')))
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
    """UWYK's graph-conditioned model with a 2D BarDistribution head.

    Only two departures from the parent:
    1. output_dim = total_params(J)  (K^2 + 9 + 4)
    2. Learned null_t_intv fills the query T_intv slot at forward time
       so we can train a joint (Y_do0, Y_do1) predictor.

    Accepted kwargs mirror the parent __init__ exactly:
        num_features, d_model, depth, heads_feat, heads_samp,
        dropout, hidden_mult, normalize_features, use_same_row_mlp,
        n_sample_attention_sink_rows, n_feature_attention_sink_cols
    """

    def __init__(self, *, J: int = 100, **kwargs):
        kwargs.setdefault('output_dim', total_params(J))
        super().__init__(**kwargs)
        self.J = J
        self.output_dim_2d = total_params(J)

        self.null_t_intv = nn.Parameter(torch.zeros(1, 1, 1))
        nn.init.normal_(self.null_t_intv, std=0.02)

        print(f'[GraphConditioned2DHead] regression_head: '
              f'{self.regression_head.in_features} -> '
              f'{self.regression_head.out_features}  (J={J}, '
              f'expected total_params(J) = {self.output_dim_2d})')

    def forward(
        self,
        X_obs: torch.Tensor,
        T_obs: torch.Tensor,
        Y_obs: torch.Tensor,
        X_intv: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        T_intv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
