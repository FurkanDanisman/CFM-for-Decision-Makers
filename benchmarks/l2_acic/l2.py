"""Re-export l2_distance from l2_ihdp/l2.py so this package doesn't duplicate it.

Loaded via importlib against an absolute path to sidestep the naming collision
with this file (both are named l2.py).
"""
import importlib.util as _iu
import os as _os

_ihdp_l2 = _os.path.abspath(_os.path.join(
    _os.path.dirname(__file__), '..', 'l2_ihdp', 'l2.py'))
_spec = _iu.spec_from_file_location('_l2_ihdp_l2', _ihdp_l2)
_mod = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

l2_distance = _mod.l2_distance                                # noqa: F401
