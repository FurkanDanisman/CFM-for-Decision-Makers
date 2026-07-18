"""UWYK Ancestral Info pipeline.

Reproduces `dofm_full_conditioning.py` from UWYK's RealCauseEval repo exactly:
  1. Target-encode treatment: T ← mean(Y | T)
  2. Build full-graph adjacency (T→Y, X→T, X→Y all = 1; padded feats = -1)
  3. Call model.predict twice (T_intv_1 = encoded T=1, T_intv_0 = encoded T=0)
  4. CATE = y_pred_1 - y_pred_0 (inverse-transformed back to original y units)

Consumes a loaded `PreprocessingGraphConditionedPFN` wrapper — see
`benchmarks/run_one.py::_load_uwyk_model` for how to construct one.
"""
from __future__ import annotations
import numpy as np
import torch


def _to_np(a):
    if isinstance(a, torch.Tensor): return a.numpy()
    return np.asarray(a)


def build_ancestral_adjacency(model_n_features, n_real_features):
    """Full-graph adjacency: T→Y=1, X→T=1, X→Y=1 for real features; padded=-1."""
    adj = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
    T_idx = 0; Y_idx = 1; feature_offset = 2
    adj[T_idx, Y_idx] = 1.0
    for i in range(n_real_features):
        adj[feature_offset + i, T_idx] = 1.0
        adj[feature_offset + i, Y_idx] = 1.0
    for i in range(n_real_features, model_n_features):
        fi = feature_offset + i
        adj[fi, :] = -1.0; adj[:, fi] = -1.0; adj[fi, fi] = -1.0
    return adj


def uwyk_ancestral_pipeline(uwyk_model, cate_dataset):
    """Returns length-N cate predictions matching UWYK's Table 3 protocol."""
    X_train = _to_np(cate_dataset.X_train)
    t_train_orig = _to_np(cate_dataset.t_train)
    t_train_orig = t_train_orig.reshape(-1, 1) if t_train_orig.ndim == 1 else t_train_orig
    y_train_orig = _to_np(cate_dataset.y_train)
    y_train_orig = y_train_orig.reshape(-1, 1) if y_train_orig.ndim == 1 else y_train_orig
    X_test = _to_np(cate_dataset.X_test)
    y_train = y_train_orig

    n_test = X_test.shape[0]
    n_features_orig = X_train.shape[1]
    model_n_features = uwyk_model.model.num_features

    # UWYK's target encoding: T ← mean(Y | T)
    t_flat = t_train_orig.flatten()
    y_flat = y_train.flatten()
    mean_y_t0 = float(y_flat[t_flat == 0].mean())
    mean_y_t1 = float(y_flat[t_flat == 1].mean())
    t_train = np.where(t_train_orig == 0, mean_y_t0, mean_y_t1).astype(np.float32)

    uwyk_model.fit(X_train, t_train, y_train)

    n_real_features = min(n_features_orig, model_n_features)
    adjacency_matrix = build_ancestral_adjacency(model_n_features, n_real_features)

    T_intv_1 = np.full((n_test, 1), mean_y_t1, dtype=np.float32)
    y_pred_1 = uwyk_model.predict(
        X_obs=X_train, T_obs=t_train, Y_obs=y_train,
        X_intv=X_test, T_intv=T_intv_1,
        adjacency_matrix=adjacency_matrix,
        prediction_type="mean", inverse_transform=True,
    )

    T_intv_0 = np.full((n_test, 1), mean_y_t0, dtype=np.float32)
    y_pred_0 = uwyk_model.predict(
        X_obs=X_train, T_obs=t_train, Y_obs=y_train,
        X_intv=X_test, T_intv=T_intv_0,
        adjacency_matrix=adjacency_matrix,
        prediction_type="mean", inverse_transform=True,
    )
    return np.asarray(y_pred_1 - y_pred_0).reshape(-1)
