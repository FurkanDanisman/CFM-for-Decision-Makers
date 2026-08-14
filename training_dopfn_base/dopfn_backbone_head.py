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
def _load_dopfn_architecture(dopfn_root: str):
    """Load Do-PFN's PerFeatureTransformer ARCHITECTURE ONLY — no trained weights.

    We're training from scratch: same architecture + same prior as DoPFN, only
    the head differs. So we unpickle the empty PerFeatureTransformer instance
    (dopfn_model.pkl) but skip loading Do-PFN's trained state_dict; then reset
    all parameters and apply DoPFN's zero-init pattern so training starts from
    a proper initialization identical to what DoPFN itself would have started
    from before pre-training.
    """
    _cwd = os.getcwd()
    # Do-PFN's model/*.py import a top-level `utils`, and so does UWYK. Whoever
    # loaded first owns sys.modules['utils'], so unpickling here after a UWYK
    # inference pass (the order context_sweep/run_one.py uses) picks up UWYK's
    # utils and dies on `from utils import print_once`. Evict the collidable
    # names for the duration of the load and put them back afterwards, so this
    # works regardless of what ran before us.
    _shadowed = {}
    for _name in list(sys.modules):
        if _name in ('utils', 'models', 'model') or _name.startswith(
                ('utils.', 'models.', 'model.')):
            _shadowed[_name] = sys.modules.pop(_name)
    # Force to the front — being merely present isn't enough if UWYK's src is
    # ahead of us in sys.path.
    while dopfn_root in sys.path:
        sys.path.remove(dopfn_root)
    sys.path.insert(0, dopfn_root)
    try:
        os.chdir(dopfn_root)
        import pickle as pkl
        # dopfn_model.pkl stores the PerFeatureTransformer INSTANCE (architecture
        # + Python object). We do NOT call `model.load_state_dict(checkpoint.state_dict)`.
        with open('artifacts/dopfn_model.pkl', 'rb') as f:
            model = pkl.load(f)
        # Also grab the config for reference.
        from scripts.transformer_prediction_interface.model_builder import Checkpoint
        ck = Checkpoint.load('artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt')
        config = ck.config
    finally:
        os.chdir(_cwd)
        # Drop Do-PFN's own utils/model entries and restore whatever we shadowed,
        # so a later UWYK pass still sees its own modules.
        for _name in list(sys.modules):
            if _name in ('utils', 'models', 'model') or _name.startswith(
                    ('utils.', 'models.', 'model.')):
                del sys.modules[_name]
        sys.modules.update(_shadowed)

    # Reset every parameter that has a reset_parameters() method — this
    # reinitialises all weights to their default init (Kaiming/Xavier/etc.
    # depending on layer type), guaranteeing we do NOT inherit any weights
    # that may have been saved in the pickle.
    for m in model.modules():
        if hasattr(m, 'reset_parameters') and callable(m.reset_parameters):
            try:
                m.reset_parameters()
            except Exception:
                pass                              # some modules have vestigial no-op reset_parameters

    # Apply Do-PFN's own zero-init pattern (see model/transformer.py::init_weights):
    # zeros the linear2/linear4 output projections and the attention out_proj
    # projections, so residual streams start neutral. This is what Do-PFN
    # itself does at initialisation.
    if hasattr(model, 'init_weights'):
        model.init_weights(zero_init=True)

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
    """Do-PFN PerFeatureTransformer + 2D joint head, trained FROM SCRATCH.

    Same architecture as Do-PFN (PerFeatureTransformer with the exact same
    config: 12 layers, ninp=192, per-feature attention, etc.) and same
    training prior (Do-PFN's SCM prior via PairedDoPFNDataset). Only the
    output head differs: 2D joint density decoder instead of 1D BarDist.
    This isolates the head-change as the sole difference between our model
    and Do-PFN for a controlled comparison.

    Interface matches InterventionalPFN:
        forward(X_context, T_context, Y_context, X_query) -> {'predictions'}

    NO checkpoint weights are loaded — training starts from a fresh
    initialisation matching Do-PFN's default init.
    """

    def __init__(self, dopfn_root: str, K: int):
        super().__init__()
        self.K = K

        backbone, config = _load_dopfn_architecture(dopfn_root)
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

        DoPFN convention (see benchmarks/methods/dopfn.py::dopfn_pipeline):
          - Treatment is passed as COLUMN 0 of X, not through y_encoder.
          - y_encoder Linear(2, 192) takes (Y_value, NaN-mask) — NanHandling
            step doubles the scalar Y into 2 channels; not (T, Y).
        We prepend T to X for the context; for the query we use a T=0
        placeholder (the joint head predicts both arms, so the query-side T
        is irrelevant to the output).
        """
        B, N, d = X_context.shape
        M = X_query.shape[1]

        # Prepend T as column 0 of X (DoPFN's convention).
        X_ctx_aug = torch.cat([T_context, X_context], dim=-1)                # (B, N, d+1)
        # Query rows: use T=0 placeholder (any constant would do; the
        # joint 2D head reads embeddings, not this per-query T).
        T_query_placeholder = torch.zeros(
            B, M, 1, dtype=T_context.dtype, device=T_context.device)
        X_qry_aug = torch.cat([T_query_placeholder, X_query], dim=-1)        # (B, M, d+1)

        # Sequence-first: DoPFN expects [seq, batch, ...].
        train_x = X_ctx_aug.transpose(0, 1)                                  # (N, B, d+1)
        test_x  = X_qry_aug.transpose(0, 1)                                  # (M, B, d+1)

        # y_src is the scalar Y_context. NanHandlingEncoderStep will
        # concatenate a mask channel internally, feeding Linear(2, 192).
        # Only context rows are provided; _forward NaN-pads query rows.
        y_src = Y_context.transpose(0, 1)                                    # (N, B, 1)

        out_seq_first = self.backbone(
            train_x, y_src, test_x,
            only_return_standard_out=True,
        )                                                                     # (M, B, n_out)

        predictions = out_seq_first.transpose(0, 1).contiguous()             # (B, M, n_out)
        return {'predictions': predictions}
