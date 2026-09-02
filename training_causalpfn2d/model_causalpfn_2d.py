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
from losses.BarDistribution2D import total_params, make_edges, neg_log_prob_2d, neg_log_prob_2d_hlgauss  # noqa: E402


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
        edge_lo: float | None = None,          # optional override for inner-region lo
        edge_hi: float | None = None,          # optional override for inner-region hi
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

        # ── STEP-CKPT / A1 PATCH ─────────────────────────────────────────────
        # Bin edges baked in as a buffer so the trainer never needs to pass
        # `edges`. Deterministic given (J, y_scaling_mode) by default:
        #   pooled_std  → grid [-10, +10]  (matches CausalPFN's vmin/vmax)
        #   uwyk_minmax → grid [-1, +1]    (targets already in [-1, +1])
        # Optional (edge_lo, edge_hi) override lets a caller tighten the inner
        # region (e.g. [-3, +3] under pooled_std) so the 9-region tail head
        # actually gets activated for the ~0.8% of training samples outside.
        if y_scaling_mode == 'pooled_std':
            _default_lo, _default_hi = -10.0, 10.0
        else:  # uwyk_minmax
            _default_lo, _default_hi = -1.0, 1.0
        _edge_lo = _default_lo if edge_lo is None else float(edge_lo)
        _edge_hi = _default_hi if edge_hi is None else float(edge_hi)
        self.register_buffer('edges', make_edges(J, y_min=_edge_lo, y_max=_edge_hi))
        self.edge_lo = _edge_lo
        self.edge_hi = _edge_hi

        # model_config: same shape as InContextModel.model_config so
        # Checkpoint callback can save+restore it verbatim and the resume
        # path in train.py can reconstruct us via hydra.
        self.model_config = {
            'model_type': 'cpfn2d',
            'model': {
                'J': J,
                'num_features': num_features,
                'ninp': ninp,
                'nhid': nhid,
                'nhead': nhead,
                'nlayers': nlayers,
                'dropout': dropout,
                'n_out': n_out,
                'nbins': self.nbins_2d,
                'edge_lo': _edge_lo,
                'edge_hi': _edge_hi,
            },
            'y_scaling_mode': y_scaling_mode,
            'loss_type':      loss_type,
            'hlgauss_sigma':  self.hlgauss_sigma,
        }

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
        # Alias for InContextModel-style code that does `model.model.state_dict()`.
        self.model = self.backbone

        # Learned null token filling the T-column at query positions. Init near
        # zero (below the {0,1} training T-values); backprop is free to move it.
        self.null_t_intv = nn.Parameter(torch.zeros(1, 1, 1))
        nn.init.normal_(self.null_t_intv, std=0.02)

        print(
            f'[CausalPFN2DHead] backbone.head out_features = '
            f'{n_out + self.nbins_2d}  (using last {self.nbins_2d} as 2D-head logits, J={J})'
        )

    def get_param_groups(self):
        """Mirror InContextModel.get_param_groups: backbone transformer gets
        weight_decay, everything else (head, null_t_intv) gets none.
        Matches CausalPFN's schedule-free AdamW convention."""
        return [
            {"params": self.backbone.transformer_encoder.parameters()},
            {
                "params": [
                    p for name, p in self.backbone.named_parameters()
                    if not name.startswith("transformer_encoder")
                ] + [self.null_t_intv],
                "weight_decay": 0.0,
            },
        ]

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

        # ── A1 PATCH ──  Pad/truncate X to self.num_features so the backbone's
        # input dim stays at (num_features + 1) regardless of what the DataLoader
        # produces. Mirrors InContextModel.prepare_input → pad_x behavior so
        # this class drops into CausalPFN's train.py unchanged.
        if F < self.num_features:
            pad = self.num_features - F
            X_context = torch.nn.functional.pad(X_context, (0, pad), value=0.0)
            X_query   = torch.nn.functional.pad(X_query,   (0, pad), value=0.0)
        elif F > self.num_features:
            X_context = X_context[..., : self.num_features]
            X_query   = X_query[...,   : self.num_features]

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
        edges: torch.Tensor | None = None,   # optional override; default self.edges
    ) -> torch.Tensor:
        """Returns per-task loss vector (B,) — trainer applies valid_mask + mean.

        The `edges` argument is optional and defaults to the buffer registered
        at __init__ time. This lets CausalPFN's trainer call us with the same
        (X_context, t_context, y_context, X_query, E_y0_query, E_y1_query)
        signature as InContextModel — no `edges` kwarg needed — while our
        historical trainer (which passed edges explicitly) still works
        untouched.
        """
        if edges is None:
            edges = self.edges

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
