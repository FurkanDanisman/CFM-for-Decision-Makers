"""Do-PFN baseline pipeline.

Matches Do-PFN's own inference_example.py: the treatment goes as the first
column of the covariate matrix, then `fit(x, y)` + `predict_cate(x_test)`.

sklearn-version shim: Do-PFN revisions use either `force_all_finite` or
`ensure_all_finite` in check_array. sklearn added the new name in 1.6 and
removed the old name in 1.8. Some semi-real dataset pickles also require
older sklearn. Translate only the unsupported spelling, using the installed
function's signature, so both old and new environments work.
"""
from __future__ import annotations
import numpy as np
import torch

def _install_check_array_shim(regressor=None):
    """Adapt both keyword spellings and refresh DoPFN's bound import aliases."""
    try:
        import sklearn.utils as utils
        import sklearn.utils.validation as _v
    except ImportError:
        return  # sklearn remains optional until a model is actually loaded.

    from functools import wraps
    import inspect
    import sys

    original = _v.check_array
    check_array = original
    if not getattr(original, '_dopfn_finite_keyword_compat', False):
        parameters = inspect.signature(original).parameters
        old, new = 'force_all_finite', 'ensure_all_finite'
        if (old in parameters) != (new in parameters):
            supported, unsupported = (new, old) if new in parameters else (old, new)

            @wraps(original)
            def check_array(*args, **kwargs):
                if unsupported in kwargs:
                    if supported in kwargs:
                        raise TypeError('Pass only one of force_all_finite and '
                                        'ensure_all_finite, not both')
                    kwargs[supported] = kwargs.pop(unsupported)
                return original(*args, **kwargs)

            check_array._dopfn_finite_keyword_compat = True

    _v.check_array = utils.check_array = check_array
    # Always refresh aliases, even when a wrapper was already installed.
    # DoPFN may have bound the original function before this call.
    for name, module in list(sys.modules.items()):
        if name.endswith('.transformer_prediction_interface.base'):
            if hasattr(module, 'check_array'):
                module.check_array = check_array

    # A class can outlive its sys.modules entry when the benchmark swaps
    # DoPFN/UWYK modules or loads pickled objects. Patch the global dictionaries
    # held by the actual training/query validation methods, not just the module
    # currently registered under their name. Both methods are inherited by
    # DoPFNRegressor, so looking at the subclass's own __dict__ is insufficient.
    if regressor is not None:
        for name in ('check_training_data', 'predict_common_setup'):
            method = getattr(regressor, name, None)
            function = getattr(method, '__func__', method)
            if function is not None:
                function = inspect.unwrap(function)
                namespace = getattr(function, '__globals__', {})
                if 'check_array' in namespace:
                    namespace['check_array'] = check_array
    return check_array


_install_check_array_shim()


def _repatch_dopfn_check_array(regressor=None):
    """Call this AFTER importing DoPFN's DoPFNRegressor to catch late-bound
    check_array references. Pass the actual class/instance when modules may
    have been evicted. Repeated calls reuse the same wrapper."""
    return _install_check_array_shim(regressor)


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
