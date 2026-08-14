"""Re-export l2_distance from l2_ihdp/l2.py so this package doesn't duplicate it.

Loaded by file path, not by name: eval_realization.py puts this directory on
sys.path ahead of l2_ihdp/, so a plain `from l2 import ...` here resolves back
to this very module (already in sys.modules, still executing) and raises a
circular-import ImportError.
"""
import importlib.util
import os

_impl_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'l2_ihdp', 'l2.py'))
_spec = importlib.util.spec_from_file_location('_l2_ihdp_impl', _impl_path)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

l2_distance = _impl.l2_distance                            # noqa: F401
resample_onto = _impl.resample_onto                        # noqa: F401
