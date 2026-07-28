"""Do-PFN baseline pipeline.

Matches Do-PFN's own inference_example.py: the treatment goes as the first
column of the covariate matrix, then `fit(x, y)` + `predict_cate(x_test)`.

sklearn-version shim: Do-PFN's transformer_prediction_interface/base.py
calls `sklearn.utils.check_array(..., ensure_all_finite=...)`. That kwarg
was only added in sklearn 1.6 (the old name is `force_all_finite`). But
Do-PFN's own semi-real pkls were pickled before sklearn added the
`missing_go_to_left` tree field (< 1.3), so on those datasets we have to
run against sklearn < 1.3. This shim renames the kwarg on the way in so
both requirements can coexist. It's a no-op on modern sklearn.
"""
from __future__ import annotations
import numpy as np
import torch

# Patch check_array (both the utils.validation location and the top-level
# alias) so Do-PFN can call it with `ensure_all_finite` on pre-1.6 sklearn.
def _install_check_array_shim():
    try:
        import inspect
        import sklearn.utils.validation as _v
        _sig = inspect.signature(_v.check_array)
        if 'ensure_all_finite' in _sig.parameters:
            return  # sklearn already supports it — no shim needed
        _orig = _v.check_array

        def _shim(*a, **kw):
            if 'ensure_all_finite' in kw:
                kw['force_all_finite'] = kw.pop('ensure_all_finite')
            return _orig(*a, **kw)

        _v.check_array = _shim
        import sklearn.utils
        if hasattr(sklearn.utils, 'check_array'):
            sklearn.utils.check_array = _shim
    except Exception:
        pass  # best-effort — the actual sklearn call will raise if it matters


_install_check_array_shim()


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
