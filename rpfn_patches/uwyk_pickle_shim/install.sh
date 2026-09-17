#!/usr/bin/env bash
# Make `utils_module` importable so spawned DataLoader workers can unpickle.
#
# THE BUG (upstream). src/utils/__init__.py loads src/utils.py via
#     spec = importlib.util.spec_from_file_location("utils_module", utils_py_path)
# but never registers the result in sys.modules. So FixedSampler.__module__ is
# "utils_module", pickle serialises it as utils_module.FixedSampler, and the
# spawned worker's `import utils_module` fails:
#     Can't pickle <class 'utils_module.FixedSampler'>:
#         import of module 'utils_module' failed
# run.py then reports "Multiprocessing validation failed" and falls back to
# single-threaded loading. With num_workers: 50 in the config -- set because,
# in their own comment, "sampling is not super fast" -- the GPU sits at 0%
# while one process samples SCMs and rejects NaNs.
#
# THE FIX. Add src/utils_module.py exposing the same names. Nothing existing is
# modified; a real module now answers `import utils_module`. Also note src/
# contains BOTH utils.py and a utils/ package, which is what forced the
# importlib workaround in the first place.
#
# Idempotent; refuses to clobber a pre-existing utils_module.py.
set -euo pipefail

UWYK_ROOT="${UWYK_ROOT:?UWYK_ROOT must point at the Graphs4CausalFoundationModels checkout}"
SRC="$UWYK_ROOT/src"
DST="$SRC/utils_module.py"
[ -d "$SRC" ] || { echo "FATAL: not found: $SRC" >&2; exit 1; }
[ -f "$SRC/utils.py" ] || { echo "FATAL: not found: $SRC/utils.py" >&2; exit 1; }

if [ -f "$DST" ]; then
    if grep -q "UWYK PICKLE SHIM" "$DST"; then
        echo "[install] already present: $DST"; exit 0
    fi
    echo "FATAL: $DST exists and is not ours — refusing to overwrite" >&2; exit 1
fi

cat > "$DST" <<'PYEOF'
"""UWYK PICKLE SHIM — added, not modified.

src/utils/__init__.py loads src/utils.py under the name "utils_module" via
importlib but never puts it in sys.modules. Classes defined there therefore
carry __module__ == "utils_module", pickle refers to them by that name, and a
spawned DataLoader worker cannot import it -- so multiprocessing validation
fails and data loading silently drops to a single process.

Providing a real module of that name makes those references resolve. Loading
src/utils.py by path (rather than `from utils import *`) is deliberate: src/
holds both utils.py and a utils/ package, and the package shadows the module.
"""
import importlib.util as _ilu
import os as _os

_src = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "utils.py")
_spec = _ilu.spec_from_file_location("utils_module", _src)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

globals().update({_k: _v for _k, _v in vars(_mod).items() if not _k.startswith("__")})

__all__ = [_k for _k in globals() if not _k.startswith("_")]
PYEOF

# Remove the shim if anything below fails, so a botched install leaves no
# half-working file behind for the next run to trip over.
trap 'rc=$?; [ $rc -ne 0 ] && rm -f "$DST" && echo "[install] verification failed — removed $DST" >&2; exit $rc' EXIT

python3 -c "import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec'); print('[install] syntax OK')" "$DST"
cd "$SRC" && python3 -c "
import utils_module
need = ['FixedSampler','TorchDistributionSampler','CategoricalSampler','DiscreteUniformSampler']
missing = [n for n in need if not hasattr(utils_module, n)]
assert not missing, f'shim missing: {missing}'
import pickle
s = utils_module.FixedSampler.__module__
print(f'[install] utils_module imports; FixedSampler.__module__={s}')
print('[install] verified: the names workers need are resolvable')
"
trap - EXIT
echo "[install] done: $DST"
