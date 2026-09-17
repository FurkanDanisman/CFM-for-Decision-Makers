#!/usr/bin/env bash
# Make `utils_module` importable AND identical to the object upstream creates,
# so spawned DataLoader workers can unpickle.
#
# THE BUG (upstream). src/utils/__init__.py loads src/utils.py with
#     spec = importlib.util.spec_from_file_location("utils_module", utils_py_path)
#     utils_module = importlib.util.module_from_spec(spec)
# and never registers it in sys.modules. FixedSampler.__module__ is therefore
# "utils_module", pickle refers to it by that name, and the spawned worker
# cannot import it. run.py reports "Multiprocessing validation failed" and
# silently drops to single-process data loading -- with num_workers: 50 in the
# config (set because, upstream's words, "sampling is not super fast"), the GPU
# then sits at 0% while one process samples SCMs.
#
# WHY V1 WAS NOT ENOUGH. Providing a module that re-loads utils.py fixes the
# import but creates a SECOND set of class objects, so pickle reports
#     Can't pickle <class 'utils_module.FixedSampler'>:
#         it's not the same object as utils_module.FixedSampler
# Identity matters, not just the name. This version imports the `utils` package
# and re-exports the very module object its __init__ built (bound at module
# level as utils.utils_module), so the classes ARE the same objects.
#
# The fix ADDS src/utils_module.py; nothing existing is modified. Note src/
# holds both utils.py and a utils/ package -- that shadowing is what forced the
# importlib workaround upstream in the first place.
#
# Idempotent, upgrades V1 in place, refuses to clobber a foreign file, and
# removes the shim if verification fails.
set -euo pipefail

UWYK_ROOT="${UWYK_ROOT:?UWYK_ROOT must point at the Graphs4CausalFoundationModels checkout}"
SRC="$UWYK_ROOT/src"
DST="$SRC/utils_module.py"
[ -d "$SRC" ] || { echo "FATAL: not found: $SRC" >&2; exit 1; }
[ -f "$SRC/utils.py" ] || { echo "FATAL: not found: $SRC/utils.py" >&2; exit 1; }

if [ -f "$DST" ]; then
    if grep -q "UWYK PICKLE SHIM V2" "$DST"; then
        echo "[install] already present (V2): $DST"; exit 0
    elif grep -q "UWYK PICKLE SHIM" "$DST"; then
        echo "[install] upgrading V1 -> V2"
    else
        echo "FATAL: $DST exists and is not ours — refusing to overwrite" >&2; exit 1
    fi
fi

trap 'rc=$?; [ $rc -ne 0 ] && rm -f "$DST" && echo "[install] verification failed — removed $DST" >&2; exit $rc' EXIT

cat > "$DST" <<'PYEOF'
"""UWYK PICKLE SHIM V2 — added, not modified.

src/utils/__init__.py loads src/utils.py under the name "utils_module" via
importlib but never registers it in sys.modules, so classes defined there carry
__module__ == "utils_module" and a spawned DataLoader worker cannot import it.

Re-exporting the SAME module object that __init__ built (it binds it at module
level as `utils.utils_module`) is what makes pickle work: the name has to
resolve to the identical class objects, not merely to equivalent ones. Loading
utils.py a second time here would produce distinct classes and pickle would
report "it's not the same object as utils_module.FixedSampler".
"""
_mod = None
try:
    import utils as _pkg  # executes src/utils/__init__.py
    _mod = getattr(_pkg, "utils_module", None)
except Exception:  # pragma: no cover - fall through to the path load
    _mod = None

if _mod is None:
    # Fallback only: upstream changed and no longer exposes the object. This
    # restores importability but NOT identity, so pickling may still fail.
    import importlib.util as _ilu
    import os as _os

    _src = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "utils.py")
    _spec = _ilu.spec_from_file_location("utils_module", _src)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

globals().update({_k: _v for _k, _v in vars(_mod).items() if not _k.startswith("__")})

__all__ = [_k for _k in globals() if not _k.startswith("_")]
PYEOF

python3 -c "import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec'); print('[install] syntax OK')" "$DST"

cd "$SRC" && python3 -c "
import pickle, utils, utils_module
need = ['FixedSampler','TorchDistributionSampler','CategoricalSampler','DiscreteUniformSampler']
missing = [n for n in need if not hasattr(utils_module, n)]
assert not missing, f'shim missing: {missing}'

# IDENTITY is the point -- name resolution alone left V1 broken.
same = [n for n in need if getattr(utils_module, n) is getattr(utils, n)]
assert len(same) == len(need), f'not identical to utils.*: {set(need) - set(same)}'

# the exact round trip validate_multiprocessing_compatibility performs
blob = pickle.dumps(utils.FixedSampler(1.0))
back = pickle.loads(blob)
assert isinstance(back, utils.FixedSampler) and back.value == 1.0
print(f'[install] FixedSampler.__module__={utils.FixedSampler.__module__}')
print(f'[install] verified: identical objects, pickle round-trip OK ({len(blob)} bytes)')
"
trap - EXIT
echo "[install] done: $DST"
