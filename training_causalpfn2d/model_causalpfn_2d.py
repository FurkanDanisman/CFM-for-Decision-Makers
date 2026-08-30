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
        os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'external', 'causalpfn')),
    )
    shims = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          '..', 'benchmarks', 'uwyk_table1', 'shims'))
    for p in (shims, os.path.join(causalpfn_root, 'src'), causalpfn_root):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


_wire_causalpfn_paths()

# ── Monkey-patch: CausalPFN's clip_outliers mutates the `mask` bool tensor
# in-place (`mask &= ...`) after it's already been consumed by maskmean and
# maskstd, which retain references to the pre-mutation version for autograd.
# That triggers a "modified by an inplace operation" RuntimeError on
# backward. Replace with a version that rebinds `mask` out-of-place. Same
# numerical result, autograd-safe.
import causalpfn.models.model as _cpm  # noqa: E402

_maskmean = _cpm.maskmean
_maskstd  = _cpm.maskstd


def _clip_outliers_safe(data, eval_pos, n_sigma=4):
    assert data.dim() == 3, 'X must be T,B,H'
    X = data[:eval_pos] if eval_pos > 0 else data
    mask = ~torch.isnan(X)
    mean = _maskmean(X, mask, dim=0)
    cutoff = n_sigma * _maskstd(X, mask, dim=0)
    mask = mask & (cutoff >= torch.abs(X - mean))          # ← out-of-place
    cutoff = n_sigma * _maskstd(X, mask, dim=0)
    return torch.clip(data, mean - cutoff, mean + cutoff)


_cpm.clip_outliers = _clip_outliers_safe

from causalpfn.models.model import TabDPTLongContextModel  # noqa: E402

_REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)
from losses.BarDistribution2D import total_params, neg_log_prob_2d, neg_log_prob_2d_hlgauss  # noqa: E402


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
        y_scaling_mode: str = 'pooled_std',   # 'pooled_std' | 'uwyk_minmax'
        loss_type: str = 'density',            # 'density' | 'hlgauss'
        hlgauss_sigma: float = 0.2,            # only used when loss_type='hlgauss'
    ):
        super().__init__()
        self.J = J
        self.num_features = num_features       # X features only (T column added inside)
        self.n_out = n_out
        self.nbins_2d = total_params(J)        # K**2 + 9 + 4 ; at J=25 this is 638
        # Loss + scaling knobs (set via constructor from trainer's env vars).
        assert y_scaling_mode in ('pooled_std', 'uwyk_minmax'), y_scaling_mode
        assert loss_type in ('density', 'hlgauss'), loss_type
        self.y_scaling_mode = y_scaling_mode
        self.loss_type      = loss_type
        self.hlgauss_sigma  = float(hlgauss_sigma)

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

    @staticmethod
    def _uwyk_y_stats(y_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """UWYK-style per-task min/max stats.

        Returns (shift, scale) such that y_scaled = (y - shift) / scale
        lands each task's y_context in [-1, +1].
          shift = (y_min + y_max) / 2
          scale = (y_max - y_min) / 2
        Ties Y_do0 and Y_do1 to the same per-task shift/scale — required
        for the joint head (both axes on same scale so correlation and
        CATE are well-defined).
        """
        y_min = y_context.amin(dim=1, keepdim=True)              # (B, 1)
        y_max = y_context.amax(dim=1, keepdim=True)              # (B, 1)
        shift = 0.5 * (y_min + y_max)
        scale = (0.5 * (y_max - y_min)).clamp(min=1e-6)
        return shift, scale

    def forward(
        self,
        X_context: torch.Tensor,   # (B, N_ctx, F)
        t_context: torch.Tensor,   # (B, N_ctx)
        y_context: torch.Tensor,   # (B, N_ctx)  raw (unstandardised)
        X_query:   torch.Tensor,   # (B, N_q,   F)
        E_y0_query: torch.Tensor,  # (B, N_q)
        E_y1_query: torch.Tensor,  # (B, N_q)
        edges: torch.Tensor,       # (J+1,)  set by the trainer based on scaling mode
    ) -> torch.Tensor:
        """Returns per-task loss vector (B,) — trainer applies valid_mask + mean."""
        # Per-task Y-scaling. Two modes:
        #   pooled_std   → (y - pooled_mean) / pooled_std   ; edges [-10, +10]
        #   uwyk_minmax  → (y - shift) / scale to [-1, +1]  ; edges [-1, +1]
        if self.y_scaling_mode == 'uwyk_minmax':
            y_shift, y_scale = self._uwyk_y_stats(y_context)
        else:
            y_shift, y_scale = self._pooled_y_stats(y_context)
        y_context_std = (y_context - y_shift) / y_scale
        E_y0_std      = (E_y0_query - y_shift) / y_scale
        E_y1_std      = (E_y1_query - y_shift) / y_scale

        logits = self._forward_logits(
            X_context, t_context, y_context_std, X_query,
        )                                                       # (B, N_q, nbins_2d)

        # Loss selection:
        #   density → neg_log_prob_2d           (bar-distribution density loss)
        #   hlgauss → neg_log_prob_2d_hlgauss   (inner CE + tail densities)
        if self.loss_type == 'hlgauss':
            return neg_log_prob_2d_hlgauss(
                logits.float(), E_y0_std, E_y1_std, self.J, edges,
                sigma=self.hlgauss_sigma, reduce='per_task',
            )
        return neg_log_prob_2d(
            logits.float(), E_y0_std, E_y1_std, self.J, edges,
            reduce='per_task',
        )
