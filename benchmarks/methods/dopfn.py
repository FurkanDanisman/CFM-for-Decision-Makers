"""Do-PFN baseline pipeline.

Matches Do-PFN's own inference_example.py: the treatment goes as the first
column of the covariate matrix, then `fit(x, y)` + `predict_cate(x_test)`.
"""
from __future__ import annotations
import numpy as np
import torch


def _to_np(a):
    if isinstance(a, torch.Tensor): return a.numpy()
    return np.asarray(a)


def dopfn_pipeline(cate_dataset, DoPFNRegressor):
    """Returns length-N cate predictions on cate_dataset.X_test."""
    X_train = _to_np(cate_dataset.X_train).astype(np.float32)
    t_train = _to_np(cate_dataset.t_train).astype(np.float32).reshape(-1)
    y_train = _to_np(cate_dataset.y_train).astype(np.float32).reshape(-1)
    X_test  = _to_np(cate_dataset.X_test).astype(np.float32)

    # Do-PFN convention: treatment is the first covariate column
    x_tr = np.concatenate([t_train[:, None], X_train], axis=1)
    x_te = np.concatenate([np.zeros((X_test.shape[0], 1), dtype=np.float32), X_test], axis=1)

    reg = DoPFNRegressor()
    reg.fit(torch.tensor(x_tr), torch.tensor(y_train))
    cate = reg.predict_cate(torch.tensor(x_te))
    return np.asarray(cate).reshape(-1)
