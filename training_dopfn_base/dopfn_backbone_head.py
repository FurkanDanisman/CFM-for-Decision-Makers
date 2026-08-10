"""Wrap Do-PFN's PerFeatureTransformer with our 2D joint density head.

Verified against Do-PFN's actual pickle payload:

  PerFeatureTransformer(
    (encoder):         SequentialEncoder(NanHandling → VarFeatCount →
                        ColumnMarker → InputNorm → VarFeatCount →
                        LinearInputEncoderStep(Linear(170 -> 192)))
    (y_encoder):       SequentialEncoder(NanHandling →
                        LinearInputEncoderStep(Linear(2 -> 192)))
    (transformer_encoder): 12 x PerFeatureEncoderLayer with attention BOTH
                            between features (per-feature attention) AND
                            between items (per-example attention).
    (decoder_dict):    ModuleDict(standard=Sequential(Linear(192, 768) →
                        GELU → Linear(768, 100)))     # 100 bars, 1D BarDist
    (criterion):       FullSupportBarDistribution
  )
  ninp = 192

Key adaptation:
  - Y_context is the FACTUAL outcome only (Y under the assigned treatment T);
    DoPFN's y_encoder already takes (T, Y_factual) — Linear(2, 192) — so we
    reuse the pretrained encoder unchanged.
  - Only decoder_dict['standard'] is replaced with a 2D head that emits
    K^2 + 9 + 4 values. The model predicts the joint (Y_do0, Y_do1) density
    at query time; loss (2D NLL) is computed against paired targets.

Interface matches InterventionalPFN so train.py is nearly identical to
training/train_cfm_dopfn.py:

    forward(X_context, T_context, Y_context, X_query) -> {'predictions': ...}
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn


# ── Loader ─────────────────────────────────────────────────────────────────
def _load_dopfn_model(dopfn_root: str):
    """Load Do-PFN's PerFeatureTransformer via its own load_model helper."""
    _cwd = os.getcwd()
    if dopfn_root not in sys.path:
        sys.path.insert(0, dopfn_root)
    try:
        os.chdir(dopfn_root)
        from scripts.transformer_prediction_interface.model_builder import (
            load_model,
        )
        model, config = load_model(path='.', device='cpu', verbose=False)
    finally:
        os.chdir(_cwd)
    return model, config


# ── 2D head ────────────────────────────────────────────────────────────────
def _make_2d_decoder(d_model: int, K: int) -> nn.Module:
    """Same shape as DoPFN's default decoder: Linear(d, 4d) -> GELU -> Linear(4d, n_out)."""
    from losses.BarDistribution2D import total_params
    n_out = total_params(K)
    nhid = 4 * d_model
    return nn.Sequential(
        nn.Linear(d_model, nhid),
        nn.GELU(),
        nn.Linear(nhid, n_out),
    )


# ── Main wrapper ───────────────────────────────────────────────────────────
class DoPFNBackboneWith2DHead(nn.Module):
    """Do-PFN PerFeatureTransformer + 2D joint head.

    Same forward interface as models.InterventionalPFN:
        forward(X_context, T_context, Y_context, X_query) -> {'predictions'}

    Context = (X_obs, T_obs, Y_obs) where Y_obs is the factual outcome
    (Y observed under T_obs). Model predicts the joint (Y_do0, Y_do1) density
    at each query. Loss (2D NLL) is applied against the paired targets in the
    training loop.
    """

    def __init__(self, dopfn_root: str, K: int, head_only: bool = True):
        super().__init__()
        self.K = K
        self.head_only = head_only

        backbone, config = _load_dopfn_model(dopfn_root)
        self.backbone = backbone
        self.config = config

        d_model = int(getattr(backbone, 'ninp', 192))
        self.d_model = d_model

        # Swap the 1D decoder for our 2D head.
        if not hasattr(backbone, 'decoder_dict') or backbone.decoder_dict is None:
            raise RuntimeError(
                'DoPFN backbone has no decoder_dict — cannot install a 2D head')
        self.head_2d = _make_2d_decoder(d_model, K)
        backbone.decoder_dict['standard'] = self.head_2d

        # y_encoder is UNCHANGED: DoPFN's Linear(2, 192) already takes
        # (T, Y_factual) — exactly what we feed as context.

        if self.head_only:
            self._freeze_all_but_head()

    def _freeze_all_but_head(self):
        """Freeze everything except the new 2D head."""
        for name, p in self.backbone.named_parameters():
            if name.startswith('decoder_dict.standard.'):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def forward(self, X_context, T_context, Y_context, X_query):
        """
        Args (batch-first, matches models.InterventionalPFN)
        ----
        X_context (B, N, d)   real-valued covariates
        T_context (B, N, 1)   binary treatment for context units
        Y_context (B, N, 1)   FACTUAL outcome (Y under T_context)
        X_query   (B, M, d)   query covariates

        Returns
        -------
        {'predictions': (B, M, K^2 + 9 + 4)}
        """
        B, N, d = X_context.shape
        M = X_query.shape[1]

        # Sequence-first: DoPFN expects [seq, batch, ...].
        train_x = X_context.transpose(0, 1)                                  # (N, B, d)
        test_x  = X_query.transpose(0, 1)                                    # (M, B, d)

        # y_src = (T, Y_factual). Only context rows — DoPFN's _forward NaN-pads
        # query rows automatically when y.shape[1] == single_eval_pos
        # (model/transformer.py lines 832-844).
        y_train = torch.cat([T_context, Y_context], dim=-1)                  # (B, N, 2)
        y_src = y_train.transpose(0, 1)                                      # (N, B, 2)

        out_seq_first = self.backbone(
            train_x, y_src, test_x,
            only_return_standard_out=True,
        )                                                                     # (M, B, n_out)

        predictions = out_seq_first.transpose(0, 1).contiguous()             # (B, M, n_out)
        return {'predictions': predictions}
