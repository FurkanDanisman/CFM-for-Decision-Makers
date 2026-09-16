#!/bin/bash
#SBATCH --job-name=uwyk-numbars
#SBATCH --output=logs_uwyk_numbars/train_%j.out
#SBATCH --error=logs_uwyk_numbars/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --requeue

# Train UWYK with a different BarDistribution bin count, using THEIR code
# unmodified. The only difference from the released model is num_bars.
#
# num_bars is a CONFIG value, not code, so nothing in the upstream tree is
# touched. This script copies
#     experiments/checkpoints/full_conditioned_model/config.yaml
# (the config that produced the released full-conditioning checkpoint) and
# rewrites exactly one line, then DIFFS the two and aborts unless precisely one
# line differs. That check is the point: it is what makes "only J changed" a
# verified claim rather than an assumption.
#
# Usage (tamIA):
#   UWYK_ROOT=$SCRATCH/rpfn_bench_kit/external/g4cfm NUM_BARS=32 \
#     sbatch --account=aip-rgrosse --gres=gpu:h100:4 \
#            training_uwyk_variants/submit_train_uwyk_numbars.sbatch
#
# Env:
#   UWYK_ROOT    (required) the Graphs4CausalFoundationModels checkout
#   NUM_BARS     bin count to train at            (default 32; released = 1000)
#   VENV         venv to activate                 (default $UWYK_ROOT/venv)
#   NUM_WORKERS  override dataloader workers      (default: leave config alone)
#                Their config asks for 50; this allocation has 32 CPUs, so
#                oversubscription is expected. Setting this changes a SECOND
#                line, so it is off by default to keep the diff at one.
set -euo pipefail
mkdir -p logs_uwyk_numbars

UWYK_ROOT="${UWYK_ROOT:?UWYK_ROOT required — the g4cfm / Graphs4CausalFoundationModels checkout}"
NUM_BARS="${NUM_BARS:-32}"
VENV="${VENV:-$UWYK_ROOT/venv}"
SRC_CFG="$UWYK_ROOT/experiments/checkpoints/full_conditioned_model/config.yaml"

[ -f "$SRC_CFG" ] || { echo "FATAL: released config not found: $SRC_CFG" >&2; exit 1; }
[ -f "$VENV/bin/activate" ] || { echo "FATAL: venv not found: $VENV" >&2; exit 1; }

OUT_CFG="$UWYK_ROOT/configs_numbars/config_numbars_${NUM_BARS}.yaml"
mkdir -p "$(dirname "$OUT_CFG")"

# ── the one-line edit ────────────────────────────────────────────────────────
# Targeted: rewrite the `value:` line that FOLLOWS the `num_bars:` key. A blind
# s/value: 1000/value: N/ would also hit max_fit_items and max_fit_batches,
# which are both 1000 and must not move.
python3 - "$SRC_CFG" "$OUT_CFG" "$NUM_BARS" <<'PYEOF'
import re, sys
src, dst, n = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(src).read().splitlines(keepends=True)
out, hits, armed = [], 0, False
for ln in lines:
    if armed and re.match(r'^\s*value:', ln):
        out.append(re.sub(r'(^\s*value:\s*)\S+', rf'\g<1>{n}', ln)); hits += 1; armed = False
        continue
    if re.match(r'^\s*num_bars:', ln):
        armed = True
    out.append(ln)
assert hits == 1, f'expected exactly one num_bars value line, rewrote {hits}'
open(dst, 'w').writelines(out)
print(f'[config] num_bars -> {n}')
PYEOF

# ── prove it is one line ─────────────────────────────────────────────────────
NDIFF=$(diff "$SRC_CFG" "$OUT_CFG" | grep -c '^[<>]' || true)
echo "[config] changed lines (diff < and > combined): $NDIFF"
diff "$SRC_CFG" "$OUT_CFG" || true
if [ "$NDIFF" -ne 2 ]; then
    echo "FATAL: expected exactly one changed line (one '<' + one '>'), got $NDIFF" >&2
    exit 1
fi

if [ -n "${NUM_WORKERS:-}" ]; then
    echo "[config] NOTE: also overriding num_workers=$NUM_WORKERS (diff is no longer one line)"
    python3 - "$OUT_CFG" "$NUM_WORKERS" <<'PYEOF'
import re, sys
p, n = sys.argv[1], sys.argv[2]
lines = open(p).read().splitlines(keepends=True)
out, armed = [], False
for ln in lines:
    if armed and re.match(r'^\s*value:', ln):
        out.append(re.sub(r'(^\s*value:\s*)\S+', rf'\g<1>{n}', ln)); armed = False; continue
    if re.match(r'^\s*num_workers:', ln):
        armed = True
    out.append(ln)
open(p, 'w').writelines(out)
PYEOF
fi

source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
# Compute nodes have no outbound network; wandb would block on init.
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] UWYK num_bars=$NUM_BARS"
echo "  UWYK_ROOT=$UWYK_ROOT"
echo "  CONFIG=$OUT_CFG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# checkpoint_dir and save_dir in their config are RELATIVE ("./checkpoints",
# "./experiments/..."), so the working directory decides where output lands.
cd "$UWYK_ROOT"
python -u src/training/run.py --config "$OUT_CFG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
