"""
Interventional Prior-Data Fitted Network (PFN) for causal inference.

Body is UWYK's `src/models/InterventionalPFN.py` verbatim (MLP, InputMLP,
TwoWayBlock, _normalize_features, role embeddings, sinusoidal feature
positional encoding, optional attention sinks).

Only deviation from UWYK: the query has no T_intv. This was the agreed
design — "Query is X only — no T in query (model fills T slot with learned
null token)". `T_intv` defaults to `None`; when None the learned
`null_t_intv` parameter (shape (1,1,1)) is broadcast across queries to fill
the slot. Everything else — embedding, normalization, attention, output —
is UWYK's code unchanged.

Source mirrored: https://github.com/ArikReuter/Graphs4CausalFoundationModels
        /blob/main/src/models/InterventionalPFN.py
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Two-layer feed-forward with SwiGLU activation and dropout."""

    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = hidden_mult * dim
        self.fc1 = nn.Linear(dim, hidden)
        self.fc_gate = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x) * F.silu(self.fc_gate(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class InputMLP(nn.Module):
    """Generic 2-layer MLP with SwiGLU for input embeddings."""

    def __init__(self, in_dim: int, out_dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = hidden_mult * out_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc_gate = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x) * F.silu(self.fc_gate(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TwoWayBlock(nn.Module):
    """
    Alternating attention across features (columns) and samples (rows),
    followed by an MLP. Pre-layer norm; separate train (self-attn) and
    test (cross-attn to train) sample-attention layers.
    """

    def __init__(self, dim: int, heads_feat: int, heads_samp: int,
                 dropout: float = 0.0, hidden_mult: int = 4):
        super().__init__()
        self.feat_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads_feat,
            dropout=dropout, batch_first=True,
        )
        self.ln_feat = nn.LayerNorm(dim)

        self.samp_attn_train = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads_samp,
            dropout=dropout, batch_first=True,
        )
        self.ln_samp_train = nn.LayerNorm(dim)

        self.samp_attn_test = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads_samp,
            dropout=dropout, batch_first=True,
        )
        self.ln_samp_test = nn.LayerNorm(dim)

        self.mlp = MLP(dim, hidden_mult=hidden_mult, dropout=dropout)
        self.ln_mlp = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, N_train: int, N_test: int) -> torch.Tensor:
        B, S, F, D = x.shape
        assert S == N_train + N_test, f"Expected {N_train + N_test} samples, got {S}"

        # 1) Feature attention (within row)
        x_row = x.reshape(B * S, F, D)
        x_norm = self.ln_feat(x_row)
        x2, _ = self.feat_attn(x_norm, x_norm, x_norm, need_weights=False)
        x_row = x_row + self.drop(x2)
        x = x_row.reshape(B, S, F, D)

        # 2) Sample attention (within column) — train self-attn, test cross-attn
        x_col = x.permute(0, 2, 1, 3).contiguous().reshape(B * F, S, D)
        x_train = x_col[:, :N_train, :]
        x_test = x_col[:, N_train:, :]

        x_train_norm = self.ln_samp_train(x_train)
        x_train_attn, _ = self.samp_attn_train(
            x_train_norm, x_train_norm, x_train_norm, need_weights=False,
        )
        x_train = x_train + self.drop(x_train_attn)

        if N_test > 0:
            x_test_norm = self.ln_samp_test(x_test)
            x_train_norm_kv = self.ln_samp_test(x_train)
            x_test_attn, _ = self.samp_attn_test(
                x_test_norm, x_train_norm_kv, x_train_norm_kv, need_weights=False,
            )
            x_test = x_test + self.drop(x_test_attn)

        x_col = torch.cat([x_train, x_test], dim=1)
        x = x_col.reshape(B, F, S, D).permute(0, 2, 1, 3).contiguous()

        # 3) Position-wise MLP
        x_norm = self.ln_mlp(x)
        x2 = self.mlp(x_norm)
        x = x + self.drop(x2)
        return x


class InterventionalPFN(nn.Module):
    """
    PFN-like regressor for interventional causal data with two-way attention.

    Body identical to UWYK. Only deviation: query has no T_intv — when
    `T_intv` is not passed, a learned null parameter fills the slot
    (per the agreed "Query is X only — model fills T slot with learned null
    token" design).
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 256,
        depth: int = 8,
        heads_feat: int = 8,
        heads_samp: int = 8,
        dropout: float = 0.0,
        output_dim: int = 1,
        hidden_mult: int = 4,
        normalize_features: bool = True,
        use_same_row_mlp: bool = True,
        n_sample_attention_sink_rows: int = 0,
        n_feature_attention_sink_cols: int = 0,
        # ── deviation from UWYK ───────────────────────────────────────────
        # If True, T_intv is required at forward; identical to UWYK.
        # If False (default for our paired-outcome head), T_intv is filled
        # with `self.null_t_intv` — the learned null token.
        use_treatment_in_query: bool = False,
        # Backward-compat with previous wrapper (ignored, kept for kwarg parity)
        normalize_treatment: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.output_dim = output_dim
        self.normalize_features = normalize_features
        self.n_sample_attention_sink_rows = n_sample_attention_sink_rows
        self.n_feature_attention_sink_cols = n_feature_attention_sink_cols
        self.use_treatment_in_query = use_treatment_in_query

        # === Embedding MLPs ===
        if use_same_row_mlp:
            self.row_mlp_train = InputMLP(num_features + 1, d_model, hidden_mult, dropout)
            self.row_mlp_test = self.row_mlp_train
        else:
            self.row_mlp_train = InputMLP(num_features + 1, d_model, hidden_mult, dropout)
            self.row_mlp_test = InputMLP(num_features + 1, d_model, hidden_mult, dropout)

        self.cell_mlp = InputMLP(1, d_model, hidden_mult, dropout)
        self.label_mlp_train = InputMLP(1, d_model, hidden_mult, dropout)

        feat_pos = self._build_feature_positional(num_features + 2, d_model)  # L + 2 positions
        self.register_buffer("feature_positional", feat_pos.unsqueeze(0).unsqueeze(0), persistent=False)

        # Learnable scales for combining row vs cell embeddings
        self.row_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(2.0)))
        self.cell_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(2.0)))

        # === Attention sinks ===
        if n_sample_attention_sink_rows > 0:
            self.sink_rows_x = nn.Parameter(
                torch.zeros(1, n_sample_attention_sink_rows, num_features + 2, d_model)
            )
            nn.init.normal_(self.sink_rows_x, std=0.02)
            self.sink_rows_y = nn.Parameter(
                torch.zeros(1, n_sample_attention_sink_rows, d_model)
            )
            nn.init.normal_(self.sink_rows_y, std=0.02)
        else:
            self.sink_rows_x = None
            self.sink_rows_y = None

        if n_feature_attention_sink_cols > 0:
            self.sink_cols = nn.Parameter(
                torch.zeros(1, 1, n_feature_attention_sink_cols, d_model)
            )
            nn.init.normal_(self.sink_cols, std=0.02)
        else:
            self.sink_cols = None

        # Role embeddings
        self.obs_T_embed = self._create_role_embedding(1, 1, self.d_model)
        self.obs_label_embed = self._create_role_embedding(1, 1, self.d_model)
        self.obs_feature_embed = self._create_role_embedding(1, 1, self.d_model)
        self.intv_T_embed = self._create_role_embedding(1, 1, self.d_model)
        self.intv_label_embed = self._create_role_embedding(1, 1, self.d_model)
        self.intv_feature_embed = self._create_role_embedding(1, 1, self.d_model)

        # ── deviation from UWYK ─────────────────────────────────────────
        # Learned null token that fills the T_intv slot when the query has
        # no T (our paired-outcome design). One scalar, broadcast across
        # (B, M, 1). When use_treatment_in_query=True this stays untouched.
        self.null_t_intv = nn.Parameter(torch.zeros(1, 1, 1))

        # Two-way attention stack
        self.blocks = nn.ModuleList([
            TwoWayBlock(d_model, heads_feat, heads_samp, dropout=dropout, hidden_mult=hidden_mult)
            for _ in range(depth)
        ])

        self.regression_head = nn.Linear(d_model, output_dim)

    def _create_role_embedding(self, *shape, std=0.02):
        embed = nn.Parameter(torch.zeros(*shape))
        nn.init.normal_(embed, std=std)
        return embed

    @staticmethod
    def _build_feature_positional(num_tokens: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(num_tokens, dim)
        position = torch.arange(0, num_tokens, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    @staticmethod
    def _normalize_features(
        X_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-task uniform quantile transform + standardize, using X_train as support."""
        B, N, F = X_train.shape
        M = X_test.shape[1]

        X_train_sorted, _ = torch.sort(X_train, dim=1)

        def quantile_transform(X, X_sorted):
            B, S, F = X.shape
            B_s, N, F_s = X_sorted.shape
            assert B == B_s and F == F_s

            X_quantiles = torch.zeros_like(X)
            for b in range(B):
                for f in range(F):
                    sorted_vals = X_sorted[b, :, f]
                    vals = X[b, :, f]
                    ranks = torch.searchsorted(sorted_vals.contiguous(), vals.contiguous())
                    quantiles = ranks.float() / max(N - 1, 1)
                    quantiles = quantiles.clamp(0.0, 1.0)
                    X_quantiles[b, :, f] = quantiles
            return X_quantiles

        X_train_quantiles = quantile_transform(X_train, X_train_sorted)
        X_test_quantiles = quantile_transform(X_test, X_train_sorted)

        mean = X_train_quantiles.mean(dim=1, keepdim=True)
        std = X_train_quantiles.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        X_train_norm = (X_train_quantiles - mean) / std
        X_test_norm = (X_test_quantiles - mean) / std
        return X_train_norm, X_test_norm

    def _embed_features(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        row_mlp: nn.Module,
        is_intvn: bool,
    ) -> torch.Tensor:
        B, S, L = X.shape
        assert L == self.num_features
        assert T.shape == (B, S, 1)

        X_with_T = torch.cat([X, T], dim=2)  # (B, S, L+1)
        row_emb = row_mlp(X_with_T)           # (B, S, D)
        cell_emb = self.cell_mlp(X_with_T.unsqueeze(-1))  # (B, S, L+1, D)

        X_cells = cell_emb[:, :, :-1, :]
        T_cells = cell_emb[:, :, -1:, :]

        if is_intvn:
            X_cells = X_cells + self.intv_feature_embed.expand(B, S, L, -1)
            T_cells = T_cells + self.intv_T_embed.expand(B, S, 1, -1)
        else:
            X_cells = X_cells + self.obs_feature_embed.expand(B, S, L, -1)
            T_cells = T_cells + self.obs_T_embed.expand(B, S, 1, -1)

        cell_emb = torch.cat([X_cells, T_cells], dim=2)        # (B, S, L+1, D)
        row_exp = row_emb.unsqueeze(2).expand(-1, -1, L + 1, -1)
        feat_emb = self.row_scale * row_exp + self.cell_scale * cell_emb
        return feat_emb

    def _embed_labels(self, Y: torch.Tensor) -> torch.Tensor:
        if Y.dim() == 3:
            Y = Y.squeeze(-1)
        label_emb = self.label_mlp_train(Y.unsqueeze(-1))
        label_emb = label_emb + self.obs_label_embed.expand(Y.size(0), Y.size(1), self.d_model)
        return label_emb

    def forward(
        self,
        X_obs: torch.Tensor,
        T_obs: torch.Tensor,
        Y_obs: torch.Tensor,
        X_intv: torch.Tensor,
        T_intv: Optional[torch.Tensor] = None,   # ← optional in our design; UWYK requires it
    ) -> Dict[str, torch.Tensor]:
        B, N, L = X_obs.shape
        assert L == self.num_features, f"Expected {self.num_features} features, got {L}"
        M = X_intv.shape[1]

        if T_obs.dim() == 2:
            T_obs = T_obs.unsqueeze(-1)
        # ── deviation from UWYK ──────────────────────────────────────────
        # If no T_intv was passed, broadcast the learned null parameter.
        if T_intv is None:
            T_intv = self.null_t_intv.expand(B, M, 1)
        elif T_intv.dim() == 2:
            T_intv = T_intv.unsqueeze(-1)

        # === Normalize features ===
        if self.normalize_features:
            X_obs_with_T = torch.cat([X_obs, T_obs], dim=2)
            X_intv_with_T = torch.cat([X_intv, T_intv], dim=2)
            X_obs_norm, X_intv_norm = self._normalize_features(X_obs_with_T, X_intv_with_T)
            X_obs_norm, T_obs_norm = X_obs_norm[:, :, :L], X_obs_norm[:, :, L:L + 1]
            X_intv_norm, T_intv_norm = X_intv_norm[:, :, :L], X_intv_norm[:, :, L:L + 1]
        else:
            X_obs_norm, T_obs_norm = X_obs, T_obs
            X_intv_norm, T_intv_norm = X_intv, T_intv

        # === Embed features ===
        feat_obs = self._embed_features(X_obs_norm, T_obs_norm, self.row_mlp_train, is_intvn=False)
        feat_intv = self._embed_features(X_intv_norm, T_intv_norm, self.row_mlp_test, is_intvn=True)
        feat_all = torch.cat([feat_obs, feat_intv], dim=1)

        # === Embed labels ===
        lab_obs = self._embed_labels(Y_obs)
        lab_intv = self.intv_label_embed.expand(B, M, self.d_model)
        lab_all = torch.cat([lab_obs, lab_intv], dim=1)

        x = torch.cat([feat_all, lab_all.unsqueeze(2)], dim=2)  # (B, S, L+2, D)
        x = x + self.feature_positional

        # === Sinks ===
        if self.sink_rows_x is not None:
            sink_x = self.sink_rows_x.expand(B, -1, -1, -1)
            sink_x_features = sink_x[:, :, :-1, :]
            sink_y = self.sink_rows_y.expand(B, -1, -1).unsqueeze(2)
            sink_x = torch.cat([sink_x_features, sink_y], dim=2)
            x = torch.cat([sink_x, x], dim=1)

        n_sink_rows = self.n_sample_attention_sink_rows

        if self.sink_cols is not None:
            current_n_samples = x.shape[1]
            sink_c = self.sink_cols.expand(B, current_n_samples, -1, -1)
            x = torch.cat([sink_c, x], dim=2)

        n_sink_cols = self.n_feature_attention_sink_cols

        for blk in self.blocks:
            x = blk(x, N_train=n_sink_rows + N, N_test=M)

        # Readout: label column of test rows
        label_pos = n_sink_cols + self.num_features + 1
        test_start_idx = n_sink_rows + N
        h_intv = x[:, test_start_idx:, label_pos, :]
        predictions = self.regression_head(h_intv)

        if self.output_dim == 1:
            predictions = predictions.squeeze(-1)

        return {"predictions": predictions}
