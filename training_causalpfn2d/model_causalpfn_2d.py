"""CausalPFN transformer body with our 2D BarDistribution joint head.

Wraps `causalpfn.models.model.TabDPTLongContextModel` (from the
``codex/add-training-code`` branch of vdblm/CausalPFN) and reproduces
`InContextModel`'s interface as closely as possible so it drops into
CausalPFN's own `trainer._train_step` — the only difference is that the
loss is our 2D joint-density loss instead of their single-arm HL-Gauss
cross-entropy.

Design:
  - Backbone constructed with ``nbins = total_params(J) = J**2 + 9 + 4``,
    so the head's last-nbins slice is exactly our 2D joint parameter
    vector. At J=25, nbins = 638, matching CausalPFN's default (500) in
    parameter budget.
  - Per-task pooled y-standardisation (mean & std of ``y_context``),
    applied to both ``E_y0_query`` and ``E_y1_query``. Bar-distribution
    edges are fitted globally at training start from a few warmup
    batches (see trainer). The pooled standardisation makes the target
    distribution stationary enough that fixed edges are sufficient.
  - The T column is filled with a learned ``null_t_intv`` scalar at
    query positions since we predict BOTH arms per query.

Forward signature matches `causalpfn.models.icl_model.InContextModel`:
    losses = model(X_context, t_context, y_context,
                   X_query, E_y0_query, E_y1_query, J, edges)
    # losses: (B,) per-task NLL, ready for valid_mask filtering in the
    # trainer. edges are external so trainer can fit them once from
    # warmup samples and share across steps.
"""
from __future__ import annotations
import os
import sys
from typing import Optional

import torch
import torch.nn as nn


def _wire_causalpfn_paths(causalpfn_root: Optional[str] = None):
    """Prepend the CausalPFN repo's ``src/`` to sys.path so
    ``from causalpfn.models.model import TabDPTLongContextModel`` resolves.
    Also prepends the shims dir so ``causalpfn/__init__.py`` can import
    ``faiss`` / ``huggingface_hub`` / ... without those packages actually
    being installed (Table 1 runner uses the same trick)."""
    causalpfn_root = causalpfn_root or os.environ.get(
        'CAUSALPFN_ROOT',
        '/scratch/furkanbd/rpfn_bench_kit/external/causalpfn',
    )
    shims = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          '..', 'benchmarks', 'uwyk_table1', 'shims'))
    for p in (shims, os.path.join(causalpfn_root, 'src'), causalpfn_root):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


_wire_causalpfn_paths()

from causalpfn.models.model import TabDPTLongContextModel  # noqa: E402

_REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)
from losses.BarDistribution2D import total_params, neg_log_prob_2d  # noqa: E402


class CausalPFN2DHead(nn.Module):
    """CausalPFN backbone + 2D joint head. Same interface as InContextModel."""

    def __init__(
        self,
        *,
        J: int = 25,
        num_features: int = 50,
        ninp: int = 256,
        nhid: int = 1024,
        nhead: int = 8,
        nlayers: int = 8,
        dropout: float = 0.0,
        n_out: int = 10,
    ):
        super().__init__()
        self.J = J
        self.num_features = num_features       # X features only (T column added inside)
        self.n_out = n_out
        self.nbins_2d = total_params(J)        # K**2 + 9 + 4 ; at J=25 this is 638

        # Backbone sees (T | X) as one flat vector, so its num_features = X + 1.
        self.backbone = TabDPTLongContextModel(
            dropout=dropout,
            n_out=n_out,
            nhead=nhead,
            nhid=nhid,
            ninp=ninp,
            nlayers=nlayers,
            num_features=num_features + 1,
            nbins=self.nbins_2d,
        )

        # Learned null token filling the T-column at query positions. Init near
        # zero (below the {0,1} training T-values); backprop is free to move it.
        self.null_t_intv = nn.Parameter(torch.zeros(1, 1, 1))
        nn.init.normal_(self.null_t_intv, std=0.02)

        print(
            f'[CausalPFN2DHead] backbone.head out_features = '
            f'{n_out + self.nbins_2d}  (using last {self.nbins_2d} as 2D-head logits, J={J})'
        )

    def _forward_logits(
        self,
        X_context: torch.Tensor,   # (B, N_ctx, F)
        t_context: torch.Tensor,   # (B, N_ctx)     in {0,1}
        y_context: torch.Tensor,   # (B, N_ctx)     standardised outside
        X_query:   torch.Tensor,   # (B, N_q,   F)
    ) -> torch.Tensor:
        """Returns joint-head logits at query positions, shape (B, N_q, nbins_2d)."""
        B, N_ctx, F = X_context.shape
        N_q = X_query.shape[1]
        assert F == self.num_features, (
            f'X_context has {F} features, model constructed for {self.num_features}'
        )

        if t_context.dim() == 3:
            t_context = t_context.squeeze(-1)
        t_ctx = t_context.unsqueeze(-1).float()                # (B, N_ctx, 1)
        t_null = self.null_t_intv.expand(B, N_q, 1).to(X_query.dtype)  # (B, N_q, 1)

        xt_ctx = torch.cat([t_ctx,  X_context], dim=-1)        # (B, N_ctx, F+1)
        xt_q   = torch.cat([t_null, X_query],   dim=-1)        # (B, N_q,   F+1)
        x_all  = torch.cat([xt_ctx, xt_q], dim=1)              # (B, N_ctx+N_q, F+1)

        # Backbone expects (S, B, F+1); y_src is context-only (N_ctx, B).
        x_src = x_all.transpose(0, 1).contiguous()             # (N_ctx+N_q, B, F+1)
        y_src = y_context.transpose(0, 1).contiguous()         # (N_ctx,      B)

        pred = self.backbone(x_src, y_src)                     # (N_q, B, n_out + nbins_2d)
        pred = pred.transpose(0, 1).contiguous()               # (B, N_q, n_out + nbins_2d)
        return pred[..., -self.nbins_2d:]                      # (B, N_q, nbins_2d)

    @staticmethod
    def _pooled_y_stats(y_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-task (mean, std) of the pooled y_context across both arms.

        Pooling both treated and control units (as opposed to CausalPFN's
        per-arm shift/scale) puts E_y0 and E_y1 on the SAME scale, which is
        what our joint head requires: CATE only makes sense when the two
        marginals live in a common outcome space.
        """
        mean = y_context.mean(dim=1, keepdim=True)             # (B, 1)
        std  = y_context.std(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1)
        return mean, std

    def forward(
        self,
        X_context: torch.Tensor,   # (B, N_ctx, F)
        t_context: torch.Tensor,   # (B, N_ctx)
        y_context: torch.Tensor,   # (B, N_ctx)  raw (unstandardised)
        X_query:   torch.Tensor,   # (B, N_q,   F)
        E_y0_query: torch.Tensor,  # (B, N_q)
        E_y1_query: torch.Tensor,  # (B, N_q)
        edges: torch.Tensor,       # (J+1,)  fitted once by the trainer
    ) -> torch.Tensor:
        """Returns per-task loss vector (B,) — trainer applies valid_mask + mean."""
        # Per-task pooled standardisation.
        y_mean, y_std = self._pooled_y_stats(y_context)
        y_context_std = (y_context - y_mean) / y_std           # (B, N_ctx)
        E_y0_std      = (E_y0_query - y_mean) / y_std          # (B, N_q)
        E_y1_std      = (E_y1_query - y_mean) / y_std

        logits = self._forward_logits(
            X_context, t_context, y_context_std, X_query,
        )                                                       # (B, N_q, nbins_2d)

        # neg_log_prob_2d takes (B, M, nbins) + edges. It internally averages
        # over M queries per task; returned as scalar. To get per-task
        # losses, unroll one task at a time via per-task masking would be
        # expensive — instead compute the raw per-(b, m) log-prob and reduce
        # over M only. We call the helper with a fake batch dim by unfolding.
        #
        # Simpler: neg_log_prob_2d returns .mean() over its input; call it
        # per task in a loop only if the trainer needs per-task losses
        # (which it does, for valid_mask). Batched call gives the SUM/N
        # average, not per-task. So we loop.
        B = logits.shape[0]
        losses = logits.new_zeros((B,))
        for b in range(B):
            losses[b] = neg_log_prob_2d(
                logits[b:b+1].float(),
                E_y0_std[b:b+1],
                E_y1_std[b:b+1],
                self.J,
                edges,
            )
        return losses
