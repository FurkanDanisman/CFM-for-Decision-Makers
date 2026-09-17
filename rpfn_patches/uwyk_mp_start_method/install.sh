#!/usr/bin/env bash
# Make UWYK's DataLoader start method configurable, defaulting to fork.
#
# THE SYMPTOM. Training reaches ~step 10 and hangs forever: GPU pinned at 100%,
# CPU load ~7/32, step counter frozen (10 -> 10 over 600 s), and
#     ps -eo args | grep run.py | wc -l   ->  1
# i.e. NO worker processes alive, while dmesg repeats "Core dump ... failed".
# The workers segfault; the main process then blocks on an empty prefetch queue
# while CUDA spins. Step ~10 is simply where the prefetch buffer runs dry.
#
# THE CAUSE. run.py forces spawn:
#     mp.set_start_method('spawn', force=True)
# Under spawn each worker re-imports the package, re-initialises CUDA (~518 MiB
# of context each) and rebuilds the dataset -- including the XGBoost/SCM prior
# objects. That is what crashes. The comment there says spawn avoids
# "fork-related segmentation faults", but empirically spawn is what segfaults
# on this stack.
#
# THE EVIDENCE. training_graph2d/train_graph_2d.py runs the SAME
# PartialGraphConditionedInterventionalPFN model with num_workers=8, sets no
# start method at all (so Linux default = fork), and has never shown this. Fork
# inherits the parent's memory instead of rebuilding it, so there is nothing to
# re-import, no per-worker CUDA context, and nothing to re-pickle.
#
# Env: UWYK_MP_START_METHOD = fork (default) | spawn | forkserver
#
# Idempotent; backs up to run.py.beforemp.
set -euo pipefail

UWYK_ROOT="${UWYK_ROOT:?UWYK_ROOT must point at the Graphs4CausalFoundationModels checkout}"
DST="$UWYK_ROOT/src/training/run.py"
[ -f "$DST" ] || { echo "FATAL: not found: $DST" >&2; exit 1; }

if grep -q "UWYK_MP_START_METHOD" "$DST"; then
    echo "[install] already patched: $DST"; exit 0
fi

cp -n "$DST" "$DST.beforemp" 2>/dev/null || true
echo "[install] backed up → $DST.beforemp"

python3 - "$DST" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()

OLD = """# Force spawn method to avoid fork-related segmentation faults
# This is especially important on HPC clusters and when using CUDA
try:
    mp.set_start_method('spawn', force=True)
    print(f"[MULTIPROCESSING] Set start method to 'spawn' for stability")
except RuntimeError as e:
    print(f"[MULTIPROCESSING] Start method already set: {e}")
"""
NEW = '''# Start method, overridable. Default is fork, NOT spawn.
#
# Spawn makes every DataLoader worker re-import the package, re-initialise CUDA
# and rebuild the dataset (XGBoost/SCM prior objects included). On this stack
# those workers segfault: the run reaches ~step 10 -- where the prefetch buffer
# empties -- and then hangs forever with no workers alive, the GPU spinning at
# 100%, and "Core dump ... failed" in dmesg.
#
# training_graph2d runs the same model with 8 workers and no start method set
# (Linux default: fork) and does not exhibit this. Fork inherits the parent's
# memory, so there is nothing to re-import, no per-worker CUDA context, and
# nothing to re-pickle.
import os as _os
_mp_method = _os.environ.get("UWYK_MP_START_METHOD", "fork")
try:
    mp.set_start_method(_mp_method, force=True)
    print(f"[MULTIPROCESSING] Set start method to {_mp_method!r}")
except RuntimeError as e:
    print(f"[MULTIPROCESSING] Start method already set: {e}")
'''
assert OLD in s, "start-method block not found — upstream changed?"
s = s.replace(OLD, NEW, 1)
open(p, "w").write(s)
print("[install] patched run.py: start method now UWYK_MP_START_METHOD (default fork)")
PYEOF

python3 -c "import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec'); print('[install] syntax OK')" "$DST"
echo "[install] done. Default is fork; override with UWYK_MP_START_METHOD=spawn"
