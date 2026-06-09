"""
Transformer-based PFN for estimating the joint distribution of paired
potential outcomes (Y_do0, Y_do1).

Architecture mirrors UWYK's InterventionalPFN:
  - Context tokens: one token per (X_obs, T_obs, Y_obs) observation
  - Query tokens: one token per X_intv point, T slot filled with learned null token
  - Depth layers of sample-attention (context self-attention)
  - Depth layers of cross-attention (query attends to context)
  - Output head: query token → output_dim predictions

normalize_features=True applies a per-context quantile normalization on X
(uses 5th–95th percentile from context, same approach as UWYK's Preprocessor).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InterventionalPFN(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        depth: int = 4,
        heads_feat: int = 4,
        heads_samp: int = 4,
        dropout: float = 0.0,
        output_dim: int = 10013,
        normalize_features: bool = True,
        normalize_treatment: bool = False,
        use_treatment_in_query: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.normalize_features = normalize_features
        self.normalize_treatment = normalize_treatment
        self.use_treatment_in_query = use_treatment_in_query

        # Context encoder: (X, T, Y) → d_model
        self.context_proj = nn.Linear(num_features + 2, d_model)

        # Query encoder: (X, null_T, null_Y) → d_model
        self.query_proj = nn.Linear(num_features + 2, d_model)

        # Learned null tokens fill T and Y slots in query
        self.null_t = nn.Parameter(torch.zeros(1))
        self.null_y = nn.Parameter(torch.zeros(1))

        # Context self-attention (sample-wise, across N observations)
        self.ctx_layers = nn.ModuleList([
            _SampleAttentionBlock(d_model, heads_samp, dropout)
            for _ in range(depth)
        ])

        # Cross-attention: query → context
        self.cross_attn  = nn.ModuleList([
            nn.MultiheadAttention(d_model, heads_feat, dropout=dropout, batch_first=True)
            for _ in range(depth)
        ])
        self.cross_ffn   = nn.ModuleList([
            _FFN(d_model, dropout)
            for _ in range(depth)
        ])
        self.cross_norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(depth)])
        self.cross_norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(depth)])

        # Output head
        self.out_norm = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _quantile_normalize(self, X: Tensor, X_ref: Tensor) -> Tensor:
        """
        Normalize X using 5th–95th percentile from X_ref (context).
        X: (B, N, F), X_ref: (B, N_ref, F)
        Returns: (B, N, F) clamped to [-3, 3]
        """
        q05 = X_ref.float().quantile(0.05, dim=1)   # (B, F)
        q95 = X_ref.float().quantile(0.95, dim=1)   # (B, F)
        rng = (q95 - q05).unsqueeze(1).clamp(min=1e-6)
        return ((X.float() - q05.unsqueeze(1)) / rng).clamp(-3.0, 3.0)

    def forward(
        self,
        X_obs:  Tensor,              # (B, N, F)
        T_obs:  Tensor,              # (B, N, 1)
        Y_obs:  Tensor,              # (B, N)
        X_intv: Tensor,              # (B, M, F)
        T_intv: Tensor | None = None,  # ignored — use_treatment_in_query=False
    ) -> dict:
        B, N, F = X_obs.shape
        M = X_intv.shape[1]

        # Feature normalization using context statistics
        if self.normalize_features:
            X_obs_n  = self._quantile_normalize(X_obs, X_obs)
            X_intv_n = self._quantile_normalize(X_intv, X_obs)
        else:
            X_obs_n  = X_obs.float()
            X_intv_n = X_intv.float()

        # Context tokens: (X, T, Y) → (B, N, d_model)
        T_flat = T_obs.float()                           # (B, N, 1)
        Y_flat = Y_obs.float().unsqueeze(-1)             # (B, N, 1)
        ctx = self.context_proj(
            torch.cat([X_obs_n, T_flat, Y_flat], dim=-1)
        )  # (B, N, d_model)

        # Query tokens: (X, null_T, null_Y) → (B, M, d_model)
        null_t = self.null_t.expand(B, M, 1)
        null_y = self.null_y.expand(B, M, 1)
        qry = self.query_proj(
            torch.cat([X_intv_n, null_t, null_y], dim=-1)
        )  # (B, M, d_model)

        # Context self-attention
        for layer in self.ctx_layers:
            ctx = layer(ctx)

        # Cross-attention: query attends to context
        for i in range(len(self.cross_attn)):
            attn_out, _ = self.cross_attn[i](qry, ctx, ctx)
            qry = self.cross_norm1[i](qry + attn_out)
            qry = self.cross_norm2[i](qry + self.cross_ffn[i](qry))

        # Output
        preds = self.out_head(self.out_norm(qry))  # (B, M, output_dim)
        return {'predictions': preds}


class _SampleAttentionBlock(nn.Module):
    """Self-attention across the N sample dimension."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn  = _FFN(d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        return x


class _FFN(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
