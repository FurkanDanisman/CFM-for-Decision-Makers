import os, sys
_ihdp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'l2_ihdp'))
if _ihdp not in sys.path: sys.path.insert(0, _ihdp)
from l2 import l2_distance                                # noqa: F401
