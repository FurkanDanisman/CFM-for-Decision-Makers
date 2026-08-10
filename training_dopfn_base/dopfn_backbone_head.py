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
  ninp = 192   nhead = ?   nhid = 768

Key facts:
  - y_encoder already accepts 2-dim input `(T, Y_factual)`. We extend to
    3-dim `(T, Y_do0, Y_do1)` by rebuilding LinearInputEncoderStep with
    Linear(3, 192), warm-starting from the pretrained Linear(2, 192) so
    the first two columns are preserved.
  - decoder_dict['standard'] is replaced with our 2D head that outputs
    K^2 + 9 + 4 values.
  - Forward: DoPFN uses sequence-first tensors [seq_len, batch, ...]. We
    take batch-first inputs and transpose.

Reference:
  https://github.com/jr2021/Do-PFN/blob/main/model/transformer.py
    class PerFeatureTransformer (line 524)
    def forward   (line 693)
    def _forward  (line 753)
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn


# ── Loader ─────────────────────────────────────────────────────────────────
def _load_dopfn_model(dopfn_root: str):
    """Load Do-PFN's PerFeatureTransformer instance with its trained weights.

    Follows scripts/transformer_prediction_interface/model_builder.py::load_model:
      1. Unpickle the model INSTANCE from `artifacts/dopfn_model.pkl` — this
         is a full PerFeatureTransformer object (encoder + transformer +
         decoder), initialised but untrained.
      2. Load the Checkpoint's state_dict from
         `artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt` into it.
    """
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
    """Matches DoPFN's default decoder shape: Linear(d, nhid) -> GELU -> Linear(nhid, n_out)."""
    from losses.BarDistribution2D import total_params
    n_out = total_params(K)
    # Match DoPFN's default hidden width (4 * d_model = 768 for d_model=192).
    nhid = 4 * d_model
    return nn.Sequential(
        nn.Linear(d_model, nhid),
        nn.GELU(),
        nn.Linear(nhid, n_out),
    )


# ── Paired-Y encoder patch ─────────────────────────────────────────────────
def _extend_y_encoder_to_paired(y_encoder: nn.Module, d_model: int) -> nn.Module:
    """Replace y_encoder's LinearInputEncoderStep(Linear(2, d)) with Linear(3, d).

    Warm-starts by copying the pretrained 2-column weights into the first 2
    columns of the new 3-column matrix. The third column is zero-initialised
    so training the new "Y_counterfactual" slot starts as identity behaviour
    (adds nothing) and gradually learns.

    Structure of y_encoder (confirmed by pickle inspection):
        SequentialEncoder(
            (0): NanHandlingEncoderStep()
            (1): LinearInputEncoderStep(layer=Linear(2, d))
        )

    NanHandlingEncoderStep is dim-agnostic (fills NaN with 0 / a mask); it
    works unchanged for 3-dim input. Only the LinearInputEncoderStep's
    internal `layer` needs resizing.
    """
    # y_encoder is a SequentialEncoder whose ordered submodules we walk.
    # The pickle showed the 1st index is NanHandling and the 2nd is Linear.
    # Find the LinearInputEncoderStep robustly by class name, since the
    # class is defined in DoPFN's model/encoders.py under its own import.
    linear_step = None
    linear_step_name = None
    for name, sub in y_encoder.named_modules():
        if type(sub).__name__ == 'LinearInputEncoderStep':
            linear_step = sub
            linear_step_name = name
            break
    if linear_step is None:
        raise RuntimeError(
            'Could not find a LinearInputEncoderStep inside y_encoder; the '
            'pickle inspection expected one but the module tree does not '
            'contain it.')

    # LinearInputEncoderStep holds `.layer = Linear(2, d)` per the pickle.
    old_linear: nn.Linear = getattr(linear_step, 'layer')
    assert isinstance(old_linear, nn.Linear), \
        f'Expected .layer to be nn.Linear, got {type(old_linear)}'
    d_out = old_linear.out_features
    d_in_old = old_linear.in_features
    assert d_out == d_model, \
        f'Y-encoder Linear out_features {d_out} != d_model {d_model}'

    new_linear = nn.Linear(3, d_out, bias=old_linear.bias is not None)
    with torch.no_grad():
        # Copy weights and bias from pretrained (T, Y_factual) slots
        new_linear.weight[:, :d_in_old].copy_(old_linear.weight)
        if old_linear.bias is not None:
            new_linear.bias.copy_(old_linear.bias)
        # Zero-init the counterfactual-Y column so initial forward pass equals
        # the pretrained behaviour (Y_cf contributes 0).
        new_linear.weight[:, d_in_old:].zero_()

    linear_step.layer = new_linear
    return y_encoder


# ── Main wrapper ───────────────────────────────────────────────────────────
class DoPFNBackboneWith2DHead(nn.Module):
    """Do-PFN PerFeatureTransformer + 2D joint head + 3-dim (T, Y0, Y1) y_encoder."""

    def __init__(self, dopfn_root: str, K: int, head_only: bool = True):
        super().__init__()
        self.K = K
        self.head_only = head_only

        backbone, config = _load_dopfn_model(dopfn_root)
        self.backbone = backbone
        self.config = config

        d_model = int(getattr(backbone, 'ninp', 192))
        self.d_model = d_model

        # 1. Swap the 1D decoder for our 2D head.
        if not hasattr(backbone, 'decoder_dict') or backbone.decoder_dict is None:
            raise RuntimeError(
                'DoPFN backbone has no decoder_dict — cannot install a 2D head')
        self.head_2d = _make_2d_decoder(d_model, K)
        backbone.decoder_dict['standard'] = self.head_2d

        # 2. Extend y_encoder to accept (T, Y_do0, Y_do1).
        _extend_y_encoder_to_paired(backbone.y_encoder, d_model)

        if self.head_only:
            self._freeze_all_but_new_pieces()

    def _freeze_all_but_new_pieces(self):
        """Freeze everything except the new 2D head and the extended y_encoder
        LinearInputEncoderStep.layer (the only new/re-initialised parameter in
        y_encoder)."""
        # Names of parameters that must stay trainable
        trainable_prefixes = ('decoder_dict.standard.',)
        # Also unfreeze the specific Linear inside y_encoder that we resized
        for name, p in self.backbone.named_parameters():
            if any(name.startswith(pref) for pref in trainable_prefixes):
                p.requires_grad = True
            elif name.startswith('y_encoder.') and 'layer.' in name:
                # matches y_encoder.<step>.layer.weight / .bias  — our new
                # Linear(3, d_model)
                p.requires_grad = True
            else:
                p.requires_grad = False

    def forward(self, X_context, T_context, Y_context_pair, X_query):
        """
        Args (batch-first)
        ----
        X_context      (B, N, d)
        T_context      (B, N, 1)
        Y_context_pair (B, N, 2)   (Y_do0, Y_do1) per context example
        X_query        (B, M, d)

        Returns
        -------
        {'predictions': (B, M, K^2 + 9 + 4)}
        """
        B, N, d = X_context.shape
        M = X_query.shape[1]

        # DoPFN's PerFeatureTransformer expects sequence-first tensors.
        # forward(train_x, train_y, test_x, ...) concatenates x internally
        # (line 706 of model/transformer.py) via single_eval_pos = len(train_x).
        train_x = X_context.transpose(0, 1)                                  # (N, B, d)
        test_x  = X_query.transpose(0, 1)                                    # (M, B, d)

        # y_src: (T, Y_do0, Y_do1). Sequence-first, only context rows —
        # DoPFN's _forward NaN-pads query rows automatically (see
        # model/transformer.py lines 832-844: if y.shape[1] == single_eval_pos
        # it appends nan-rows for the test set; NanHandlingEncoderStep then
        # zero-fills them at encode time).
        y_train = torch.cat([T_context, Y_context_pair], dim=-1)             # (B, N, 3)
        y_src = y_train.transpose(0, 1)                                      # (N, B, 3)

        out_seq_first = self.backbone(
            train_x, y_src, test_x,
            only_return_standard_out=True,
        )                                                                     # (M, B, n_out)

        predictions = out_seq_first.transpose(0, 1).contiguous()             # (B, M, n_out)
        return {'predictions': predictions}
