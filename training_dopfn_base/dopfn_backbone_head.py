"""Wrap Do-PFN's TabPFN TransformerModel with our 2D joint density head.

Goal: replace Do-PFN's default 1D BarDistribution decoder with a decoder
that emits `K^2 + 9 + 4` values per query, so the same backbone can be
trained (or fine-tuned) with our 2D loss.

Design (per training_dopfn_base/README.md):
  1. Load Do-PFN's `TransformerModel` from `artifacts/dopfn_model.pkl`.
  2. Swap `decoder_dict['standard']` for a new decoder that outputs
     K^2 + 9 + 4 values per query (matches losses/BarDistribution2D
     .total_params(K)).
  3. Adapt the Y-encoder path to accept paired outcomes (Y_do0, Y_do1)
     per context example (Option A in the README — concatenate along the
     Y-embed dim).

This file is a working SKELETON. The forward pass wiring depends on
Do-PFN's internal encoder/decoder interfaces; TODO markers below indicate
the exact places that need cluster-side inspection of Do-PFN's model
package to complete.
"""
from __future__ import annotations

import os
import pickle as pkl
from typing import Optional

import torch
import torch.nn as nn


def _load_dopfn_backbone(dopfn_root: str, checkpoint_relpath: str = 'artifacts/dopfn_model.pkl'):
    """Load Do-PFN's TransformerModel (backbone + default 1D decoder).

    Do-PFN's config uses a relative artifact path (see benchmarks/methods/dopfn.py),
    so we chdir into `dopfn_root` for the duration of the load.
    """
    _cwd = os.getcwd()
    try:
        os.chdir(dopfn_root)
        with open(checkpoint_relpath, 'rb') as f:
            payload = pkl.load(f)                              # TODO: verify the pickle's structure
        # Common conventions used by Do-PFN's loader (from
        # scripts/transformer_prediction_interface/model_builder.py::load_model):
        #   payload could be a dict with keys 'state_dict', 'config', 'criterion',
        #   or a Checkpoint object. Inspect on cluster and dispatch accordingly.
        return payload
    finally:
        os.chdir(_cwd)


class DoPFNBackboneWith2DHead(nn.Module):
    """Do-PFN TransformerModel with the 1D decoder replaced by a 2D one.

    Forward signature intentionally matches models.InterventionalPFN so
    training_dopfn_base/train.py can be a near-drop-in of
    training/train_cfm_dopfn.py:

        forward(X_context, T_context, Y_context_pair, X_query) -> dict
            X_context      (B, N, d)         real-valued covariates
            T_context      (B, N, 1)         binary treatment
            Y_context_pair (B, N, 2)         (Y_do0, Y_do1) per context example
            X_query        (B, M, d)         query covariates

        Returns:
            {'predictions': tensor of shape (B, M, K^2 + 9 + 4)}
    """

    def __init__(self,
                 dopfn_root: str,
                 K: int,
                 head_only: bool = True,
                 checkpoint_relpath: str = 'artifacts/dopfn_model.pkl',
                 ):
        super().__init__()
        self.K = K
        self.head_only = head_only

        payload = _load_dopfn_backbone(dopfn_root, checkpoint_relpath)
        # -----------------------------------------------------------------
        # TODO(cluster): unpack the payload and store the transformer model,
        # its encoder, its y_encoder, and the internal embed dim (`ninp`).
        #
        # Do-PFN's model_builder.py::load_model returns a `Checkpoint` with
        # a `.model` (TransformerModel). Its attributes we care about:
        #   backbone.encoder            (per-feature X encoder)
        #   backbone.y_encoder          (Y encoder — single scalar in)
        #   backbone.transformer_encoder (attention layers)
        #   backbone.decoder_dict       (ModuleDict with 'standard' -> 1D head)
        #   backbone.ninp               (embed dim used by decoders)
        #
        # Fill in the assignments below once the pickle format is confirmed:
        self._backbone = payload                              # TODO: extract .model / .state_dict
        self._ninp = getattr(payload, 'ninp', None)           # TODO: read from config
        # -----------------------------------------------------------------

        # New 2D output head. Mirrors Do-PFN's default `nn.Sequential(Linear,
        # GELU, Linear)` decoder shape (from transformer.py::make_decoder_dict)
        # but with output dim K^2 + 9 + 4.
        from losses.BarDistribution2D import total_params
        n_out = total_params(K)
        assert self._ninp is not None, (
            'set self._ninp above once you know the pickle layout')
        self.head_2d = nn.Sequential(
            nn.Linear(self._ninp, self._ninp),
            nn.GELU(),
            nn.Linear(self._ninp, n_out),
        )

        # Paired-Y adaptation (Option A from README). Do-PFN's y_encoder maps
        # a single scalar Y -> Y-embed of dim self._ninp. For paired (Y_do0,
        # Y_do1) input we compute two Y-embeds and combine them.
        # TODO(cluster): confirm Do-PFN's y_encoder input dim (scalar? vector?)
        # and adjust below. Cheapest baseline: two calls of the same encoder,
        # element-wise sum, followed by a learned projection so the backbone
        # sees an embedding shaped exactly like the original single-Y path.
        self.pair_y_proj = nn.Linear(2 * self._ninp, self._ninp)
        # Also encode T as a separate embedding.
        self.t_embed = nn.Embedding(2, self._ninp)

        if self.head_only:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze everything under self._backbone; only head + adapters train."""
        # TODO: iterate self._backbone.parameters() (after unpacking above)
        # and set requires_grad = False.
        pass

    def _embed_paired_y(self, Y_pair, T):
        """(B, N, 2) paired outcomes -> (B, N, ninp) Y-embed for the backbone.

        Uses Do-PFN's original y_encoder on each arm then combines.
        TODO: replace `getattr(...)` with the real y_encoder access once
        confirmed on cluster.
        """
        y_encoder = getattr(self._backbone, 'y_encoder', None)
        if y_encoder is None:
            raise RuntimeError(
                'Do-PFN backbone did not expose a y_encoder — update _embed_paired_y '
                'once the payload structure is known')
        e0 = y_encoder(Y_pair[..., 0:1])                      # (B, N, ninp)
        e1 = y_encoder(Y_pair[..., 1:2])                      # (B, N, ninp)
        combined = torch.cat([e0, e1], dim=-1)                # (B, N, 2*ninp)
        e = self.pair_y_proj(combined)                        # (B, N, ninp)
        # T-conditioning
        t_ix = T.squeeze(-1).long().clamp(0, 1)
        e = e + self.t_embed(t_ix)
        return e

    def forward(self, X_context, T_context, Y_context_pair, X_query):
        """
        Args
        ----
        X_context      (B, N, d) — real feature values per context example
        T_context      (B, N, 1) — binary treatment {0, 1}
        Y_context_pair (B, N, 2) — paired outcomes (Y_do0, Y_do1)
        X_query        (B, M, d) — feature values for the query examples

        Returns
        -------
        dict with:
            'predictions': (B, M, K^2 + 9 + 4)
        """
        # ────────────────────────────────────────────────────────────────
        # TODO(cluster): call Do-PFN's TransformerModel forward with:
        #   - context feature embeddings from self._backbone.encoder(X_context)
        #   - paired-Y embeddings from self._embed_paired_y(Y_context_pair, T_context)
        #   - query feature embeddings from self._backbone.encoder(X_query)
        # and receive the transformer's per-query embedding (shape (B, M, ninp))
        # BEFORE the default decoder is applied.
        #
        # Do-PFN's TransformerModel exposes this via a hook or by intercepting
        # decoder_dict output pre-projection; the simplest wiring is to
        # forward the model, then override decoder_dict['standard'] to be
        # `nn.Identity()` so it returns embeddings directly, then apply
        # self.head_2d to the returned query-embeddings.
        # ────────────────────────────────────────────────────────────────
        raise NotImplementedError(
            'Complete the forward-pass wiring after inspecting Do-PFN\'s '
            'TransformerModel.forward signature on the cluster. '
            'See training_dopfn_base/README.md ("ARCHITECTURE NOTES") for '
            'the paired-Y adaptation, and this file\'s TODOs for the '
            'decoder swap.')
