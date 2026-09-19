"""Fallback import finder for Table 1 reproduction.

`causalpfn/__init__.py` runs `from .causal_estimator import ATEEstimator,
CATEEstimator`, and `causal_estimator.py` imports a laundry list of ML libs
at top level (faiss, huggingface_hub, transformers, wandb, ...). Table 1's
Predictive / DoFM (anc / noanc) rows never actually invoke any of those —
they only use `causalpfn.benchmarks` for dataset loaders. Rather than stub
each one by hand as they surface, install a meta-path finder that stubs any
top-level module in a small denylist.

If a stubbed module is called at runtime, the call returns a MagicMock and
Python won't crash — you'll see a downstream error, which is your cue to
install the real package.

Enable by prepending this directory to PYTHONPATH — Python auto-imports
`sitecustomize` on interpreter start.
"""
from unittest.mock import MagicMock as _MagicMock
import os as _os
import sys as _sys
import types as _types


# UWYK's PreprocessingGraphConditionedPFN._pad_or_truncate_samples uses
# np.random.choice on numpy's GLOBAL RNG at eval time to pick 1000 rows out
# of the >1000-row training set (ACIC, CPS, PSID_unbal). If UWYK_SEED is set
# in the environment, seed the global numpy RNG at interpreter start so those
# picks are deterministic across runs. IHDP (n=672) and PSID_bal (~685 after
# balancing) never enter the truncation branch, so they are unaffected.
_uwyk_seed = _os.environ.get('UWYK_SEED')
if _uwyk_seed is not None:
    try:
        import numpy as _np
        _np.random.seed(int(_uwyk_seed))
    except Exception:
        pass


_STUB_TOPLEVEL = {
    'faiss',
    'huggingface_hub',
    'transformers',
    'wandb',
    'tensorboard',
    'tensorboardX',
    'accelerate',
    'safetensors',
    'peft',
    'bitsandbytes',
    'datasets',   # HF datasets, not our local shard datasets
    'evaluate',
    'diffusers',
}


class _StubModule(_types.ModuleType):
    def __getattr__(self, name):
        # Dunders stay absent, so the import system's own probes (__path__,
        # __file__, __wrapped__, ...) see a plain module, not a MagicMock.
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        # Any submodule access (huggingface_hub.hf_hub_download) → MagicMock
        m = _MagicMock(name=f'{self.__name__}.{name}')
        setattr(self, name, m)
        return m


class _StubFinder:
    """Meta-path finder AND loader, via find_spec (PEP 451).

    Python 3.12 stopped calling the legacy find_module/load_module on
    meta-path finders, so a finder with only those is silently skipped there
    and `import faiss` fails for real. find_spec works on every Python >= 3.4.
    """

    def find_spec(self, name, path=None, target=None):
        if not self.find_module(name, path):
            return None
        from importlib.machinery import ModuleSpec
        # is_package: `from name.sub import x` must work, as before.
        return ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        m = _StubModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass

    def find_module(self, name, path=None):
        # Exact match only — NOT name.split('.')[0]. A dotted name here means
        # some package (possibly a real, installed one) is importing a
        # submodule of itself, e.g. faiss-cpu's `from . import _gpu_build`
        # marker probe. If we matched on the top-level prefix, a genuinely
        # absent submodule of a denylisted-but-actually-installed package
        # would get silently handed a MagicMock stub instead of raising
        # ImportError, corrupting that package's own internal feature
        # detection (this broke real faiss-cpu: it mis-detected itself as a
        # CUDA-13 cuVS GPU build and crashed hunting for `nvidia.cu13`).
        # Only intercept bare top-level imports of denylisted packages.
        return self if name in _STUB_TOPLEVEL else None

    def load_module(self, name):
        if name in _sys.modules:
            return _sys.modules[name]
        m = _StubModule(name)
        m.__path__ = []  # mark as a package so `from name.sub import x` works
        _sys.modules[name] = m
        return m


# Register last-resort so real installs take precedence.
_sys.meta_path.append(_StubFinder())
