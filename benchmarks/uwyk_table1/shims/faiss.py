"""Stub faiss module for Table 1 reproduction.

`causalpfn/__init__.py` imports `causal_estimator`, which imports faiss at the
top level. Table 1's Predictive / DoFM (anc / noanc) rows do NOT actually use
faiss anywhere in their prediction pipeline (only IHDPDataset / CPS / PSID
loaders from `causalpfn.benchmarks`), so we only need the `import faiss` line
to succeed.

Any attribute access on this module returns a MagicMock so that class-level
constructions like `faiss.IndexFlatL2(...)` in causal_estimator.py don't
explode at import time. If the code path actually CALLS a faiss operation
at runtime, that call returns a MagicMock too — you'll see a downstream
error, at which point install the real faiss (`pip install faiss-cpu`) and
delete this shim dir from PYTHONPATH.
"""
from unittest.mock import MagicMock as _MagicMock


class _FaissStub:
    def __getattr__(self, name):
        # A fresh MagicMock per attribute access — supports both attribute
        # chains (faiss.IndexFlatL2) and calls (faiss.IndexFlatL2(64)).
        return _MagicMock(name=f'faiss.{name}')


import sys as _sys
_sys.modules[__name__] = _FaissStub()
