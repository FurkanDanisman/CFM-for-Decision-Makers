#!/usr/bin/env bash
# Make UWYK's training progress observable.
#
# THE PROBLEM. trainer.py prints a Step line only when
#     self.global_step <= 10 or self.global_step % max(1, self.max_steps // 10) == 0
# With max_steps=50000 that is steps 1..10, then NOTHING until step 5000. The
# config's log_every_n_steps is never read by the trainer -- it is a dead key.
# save_every is also 5000, so the checkpoint directory is empty for the same
# window. At ~1.8 s/step a healthy run is therefore indistinguishable from a
# hung one for ~2.5 hours, in both the log and on disk. That ambiguity has
# already cost several cancelled runs.
#
# THE FIX. Log every UWYK_LOG_EVERY steps (default 50, matching
# training_dopfn_repro's --log-every). The original conditions are kept, so
# nothing that used to print stops printing.
#
# Env: UWYK_LOG_EVERY = 50 (default), 0 disables the extra logging.
#
# Idempotent; backs up to trainer.py.beforelogevery.
set -euo pipefail

UWYK_ROOT="${UWYK_ROOT:?UWYK_ROOT must point at the Graphs4CausalFoundationModels checkout}"
DST="$UWYK_ROOT/src/training/trainer.py"
[ -f "$DST" ] || { echo "FATAL: not found: $DST" >&2; exit 1; }

if grep -q "UWYK_LOG_EVERY" "$DST"; then
    echo "[install] already patched: $DST"; exit 0
fi

cp -n "$DST" "$DST.beforelogevery" 2>/dev/null || true
echo "[install] backed up → $DST.beforelogevery"

python3 - "$DST" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()

OLD = """                if self.global_step <= 10 or self.global_step % max(1, self.max_steps // 10) == 0:
"""
NEW = """                # ── UWYK_LOG_EVERY ── upstream printed steps 1..10 and then
                # nothing until max_steps//10 (= 5000 at max_steps=50000), while
                # save_every is also 5000 -- so a healthy run and a hung one look
                # identical in both the log and on disk for ~2.5 hours. Log on a
                # fixed interval as well. Original conditions are preserved.
                _log_every = int(os.environ.get("UWYK_LOG_EVERY", "50"))
                if (self.global_step <= 10
                        or (_log_every > 0 and self.global_step % _log_every == 0)
                        or self.global_step % max(1, self.max_steps // 10) == 0):
"""
assert OLD in s, "logging condition not found — upstream changed?"
assert s.count(OLD) == 1, "logging condition matched more than once"
s = s.replace(OLD, NEW, 1)
open(p, "w").write(s)
print("[install] patched trainer.py: progress now logged every UWYK_LOG_EVERY steps")
PYEOF

python3 -c "import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec'); print('[install] syntax OK')" "$DST"
echo "[install] done. Default: a Step line every 50 steps (UWYK_LOG_EVERY to change)."
