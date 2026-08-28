"""Graph-conditioned model with our 2D BarDistribution joint head.

Wraps UWYK's `PartialGraphConditionedInterventionalPFN` in
`partial_gcn_and_soft_attention` mode — the SAME architecture UWYK uses
for the Table 1 "Ancestral Info." result on IHDP/ACIC/CPS/PSID (verified
via reproduce-realcause-results branch REPRODUCE_REALCAUSE_RESULTS.md).

That architecture has:
  - Soft attention bias (learnable bias_edge / bias_no_edge for the
    {-1, 0, +1} partial-graph format)
  - Graph-convolutional encoder (GCN) with AdaLN modulation
  - Sample attention sinks

Our two departures from the parent:
  1. output_dim = total_params(J)  (K**2 + 9 + 4)  — for the 2D joint head
  2. Learned null_t_intv fills the query T_intv slot so the joint head
     predicts BOTH arms at once (no need for a hard-coded query arm)

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

from models.PartialGraphConditionedInterventionalPFN import (  # noqa: E402
    PartialGraphConditionedInterventionalPFN,
)

_REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)
from losses.BarDistribution2D import total_params  # noqa: E402


# Fixed kwargs for the "partial_gcn_and_soft_attention" mode — verbatim from
# UWYK's src/training/run.py:1125-1148 (branch reproduce-realcause-results).
# Do not change these — this is what UWYK's Table 1 checkpoint was trained
# with.
_GCN_SOFT_ATT_KWARGS = dict(
    use_attention_masking=True,
    use_gcn=True,
    use_adaln=True,
    use_soft_attention_bias=True,
    soft_bias_init=5.0,          # UWYK config API-compat default
    gcn_use_transpose=False,     # from best_model_config.yaml
    gcn_alpha_init=0.1,          # from best_model_config.yaml
)


class GraphConditioned2DHead(PartialGraphConditionedInterventionalPFN):
    """PartialGraphConditionedInterventionalPFN (partial_gcn_and_soft_attention)
    with our 2D BarDistribution head.

    Only two departures from the parent:
    1. output_dim = total_params(J)  (K**2 + 9 + 4)
    2. Learned null_t_intv fills the query T_intv slot at forward time
       so we can train a joint (Y_do0, Y_do1) predictor.

    Fixed kwargs (matching UWYK Table 1 checkpoint config):
        graph_conditioning_mode = 'partial_gcn_and_soft_attention'
        use_soft_attention_bias = True
        use_gcn = True
        use_adaln = True
        use_attention_masking = True
        gcn_use_transpose = False
        gcn_alpha_init = 0.1

    User-facing kwargs (still passed through):
        num_features, d_model, depth, heads_feat, heads_samp,
        dropout, hidden_mult, normalize_features,
        n_sample_attention_sink_rows, n_feature_attention_sink_cols
    """

    def __init__(self, *, J: int = 100, **kwargs):
        kwargs.setdefault('output_dim', total_params(J))
        # Apply UWYK's partial_gcn_and_soft_attention mode kwargs. These are
        # NOT overridable by the caller (fixed to match Table 1 checkpoint).
        for k, v in _GCN_SOFT_ATT_KWARGS.items():
            kwargs[k] = v
        super().__init__(**kwargs)
        self.J = J
        self.output_dim_2d = total_params(J)

        self.null_t_intv = nn.Parameter(torch.zeros(1, 1, 1))
        nn.init.normal_(self.null_t_intv, std=0.02)

        print(f'[GraphConditioned2DHead] mode=partial_gcn_and_soft_attention  '
              f'sink_rows={kwargs.get("n_sample_attention_sink_rows", 0)}  '
              f'sink_cols={kwargs.get("n_feature_attention_sink_cols", 0)}')
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
