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
# ALSO patches DoPFN's own module-local `check_array` reference if DoPFN's
# transformer_prediction_interface.base is already importable — because
# `from sklearn.utils.validation import check_array` at DoPFN's module top
# binds a separate reference that survives our patching of sklearn.
def _install_check_array_shim():
    try:
        import inspect
        import sklearn.utils.validation as _v
        _sig = inspect.signature(_v.check_array)
        needs_patch = 'ensure_all_finite' not in _sig.parameters
        _orig = _v.check_array

        def _shim(*a, **kw):
            if 'ensure_all_finite' in kw:
                kw['force_all_finite'] = kw.pop('ensure_all_finite')
            return _orig(*a, **kw)

        if needs_patch:
            _v.check_array = _shim
            import sklearn.utils
            if hasattr(sklearn.utils, 'check_array'):
                sklearn.utils.check_array = _shim
            # DoPFN's base.py does `from sklearn.utils.validation import check_array`
            # at module load; patch its local ref if the module is already loaded.
            import sys as _sys
            for _name in list(_sys.modules):
                if _name.endswith('.transformer_prediction_interface.base') \
                        or _name == 'scripts.transformer_prediction_interface.base':
                    _mod = _sys.modules[_name]
                    if hasattr(_mod, 'check_array'):
                        _mod.check_array = _shim
    except Exception:
        pass  # best-effort — the actual sklearn call will raise if it matters


_install_check_array_shim()


def _repatch_dopfn_check_array():
    """Call this AFTER importing DoPFN's DoPFNRegressor to catch late-bound
    check_array references. Idempotent + no-op if sklearn is >= 1.6."""
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
