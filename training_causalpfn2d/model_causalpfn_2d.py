"""CausalPFN transformer body with our 2D BarDistribution joint head.

Wraps `causalpfn.models.model.TabDPTLongContextModel` (from the
``codex/add-training-code`` branch of vdblm/CausalPFN) and:

  1. Sets the model's ``nbins = total_params(J) = J**2 + 9 + 4`` so the
     final linear projection emits our joint-head parameter budget. We
     read only the last ``nbins`` output dims — same slicing pattern
     CausalPFN itself uses (``logits[:, :, -nbins:]``).
  2. Adds a learned ``null_t_intv`` scalar that fills the T-column of
     ``X_intv`` at every query, since we predict BOTH potential
     outcomes per query rather than one arm conditional on T.

Forward:
    logits = model(X_obs, T_obs, Y_obs, X_intv)
    # logits: (B, M, J**2 + 9 + 4)   -- the joint-head parameter vector
    # per query, ready to be scored with losses.BarDistribution2D.neg_log_prob_2d.

CausalPFN's model applies its own X-standardisation and outlier clipping
inside its ``forward`` (see model.py:normalize_data / clip_outliers), so
the caller passes raw X here.
"""
from __future__ import annotations
import os
import sys
from typing import Optional

import torch
import torch.nn as nn


def _wire_causalpfn_paths(causalpfn_root: Optional[str] = None):
    """Prepend the CausalPFN repo's ``src/`` to sys.path so
    ``from causalpfn.models.model import TabDPTLongContextModel`` resolves."""
    causalpfn_root = causalpfn_root or os.environ.get(
        'CAUSALPFN_ROOT',
        '/scratch/furkanbd/rpfn_bench_kit/external/causalpfn',
    )
    for p in (os.path.join(causalpfn_root, 'src'), causalpfn_root):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


_wire_causalpfn_paths()

from causalpfn.models.model import TabDPTLongContextModel  # noqa: E402

_REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)
from losses.BarDistribution2D import total_params  # noqa: E402


class CausalPFN2DHead(nn.Module):
    """CausalPFN backbone + 2D joint head.

    Constructs a `TabDPTLongContextModel` with ``nbins = total_params(J)``
    so the head's last-nbins slice is exactly our joint-head parameter
    vector. Also owns a learned ``null_t_intv`` scalar that fills the
    T-column at query positions.

    Parameters
    ----------
    J : int
        Grid side length for the 2D bar distribution.
        `total_params(J) = J**2 + 9 + 4`.
    num_features : int
        Padded feature count seen by the model (X only; the T-column is
        concatenated internally, so the model's ``num_features`` is
        ``num_features + 1``).
    ninp, nhid, nhead, nlayers, dropout : passthrough to TabDPTLongContextModel.
    n_out : int
        CausalPFN's classification-head width; unused for us. Kept for
        compatibility with the base head layout ``n_out + nbins``.
    """

    def __init__(
        self,
        *,
        J: int = 100,
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
        self.num_features = num_features       # X features only (excluding T column)
        self.n_out = n_out
        self.nbins_2d = total_params(J)        # K**2 + 9 + 4

        # Model sees (T | X) as one flat vector, so backbone's num_features = X + 1.
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

        # Learned null token that fills the T-column at query positions.
        # Init near zero so at step 0 it looks like T=0 (in the raw
        # {0,1} T-value scale); backprop is free to move it.
        self.null_t_intv = nn.Parameter(torch.zeros(1, 1, 1))
        nn.init.normal_(self.null_t_intv, std=0.02)

        print(
            f'[CausalPFN2DHead] backbone.head out_features = '
            f'{n_out + self.nbins_2d}   (using last {self.nbins_2d} as 2D-head logits, J={J})'
        )

    def forward(
        self,
        X_obs: torch.Tensor,   # (B, N, F)
        T_obs: torch.Tensor,   # (B, N, 1) or (B, N)  in {0,1}
        Y_obs: torch.Tensor,   # (B, N)     scaled to [-1, 1]
        X_intv: torch.Tensor,  # (B, M, F)
    ) -> torch.Tensor:
        """Returns joint-head logits (B, M, J**2 + 9 + 4) at every query."""
        B, N, F = X_obs.shape
        M       = X_intv.shape[1]
        assert F == self.num_features, (
            f'X_obs has {F} features, model constructed for {self.num_features}'
        )

        if T_obs.dim() == 2:
            T_obs = T_obs.unsqueeze(-1)          # (B, N, 1)
        T_null = self.null_t_intv.expand(B, M, 1).to(X_intv.dtype)  # (B, M, 1)

        # CausalPFN convention: T is the FIRST column of the joint (T | X) tensor.
        xt_obs  = torch.cat([T_obs,  X_obs],  dim=-1)   # (B, N, F+1)
        xt_intv = torch.cat([T_null, X_intv], dim=-1)   # (B, M, F+1)
        x_all   = torch.cat([xt_obs, xt_intv], dim=1)   # (B, N+M, F+1)

        # Backbone expects seq-first (S, B, F+1) and returns (M, B, n_out+nbins).
        x_src = x_all.transpose(0, 1).contiguous()      # (N+M, B, F+1)
        y_src = Y_obs.transpose(0, 1).contiguous()      # (N,    B)   context only

        pred = self.backbone(x_src, y_src)              # (M, B, n_out + nbins_2d)
        pred = pred.transpose(0, 1).contiguous()        # (B, M, n_out + nbins_2d)
        logits = pred[..., -self.nbins_2d:]             # (B, M, nbins_2d)
        return logits
