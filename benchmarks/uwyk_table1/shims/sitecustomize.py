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
import sys as _sys
import types as _types


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
        # Any submodule access (huggingface_hub.hf_hub_download) → MagicMock
        m = _MagicMock(name=f'{self.__name__}.{name}')
        setattr(self, name, m)
        return m


class _StubFinder:
    def find_module(self, name, path=None):
        top = name.split('.', 1)[0]
        return self if top in _STUB_TOPLEVEL else None

    def load_module(self, name):
        if name in _sys.modules:
            return _sys.modules[name]
        m = _StubModule(name)
        m.__path__ = []  # mark as a package so `from name.sub import x` works
        _sys.modules[name] = m
        return m


# Register last-resort so real installs take precedence.
_sys.meta_path.append(_StubFinder())
