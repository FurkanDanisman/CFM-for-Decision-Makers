#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=cfm-train-debug
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/train_debug_%j.out
#SBATCH --error=logs/train_debug_%j.err

# DEBUG variant of submit_train.sh — instruments the `none_dealloc` C abort that
# kills a DataLoader worker ~500 steps (~55-66 min) into the real run.
#
# This is NOT a training run; do not chain it or resume from it. It exists only
# to make the crash self-explaining. Differences from submit_train.sh:
#   - 2h wall clock (the crash reproduces well inside this window).
#   - DATA_REFCOUNT=1  → logs sys.getrefcount(None) deltas per __getitem__ phase
#                        (scm_search / deepcopy_intervene / obs_test_propagate /
#                        propagate_paired / ancestor_matrix) plus a NET line.
#                        The phase that nets a NEGATIVE delta is the over-decref
#                        of None that eventually aborts the process.
#   - STREAM_WORKERS=8 → kept at the production value so this faithfully
#                        reproduces the real crash. The per-sample [REFCOUNT]
#                        line is tagged worker=N, so filter one worker when
#                        reading (the leak is uniform across workers). NOTE:
#                        fewer workers would NOT slow the crash down — the abort
#                        is driven by per-process sample count, and the workers
#                        are mostly idle here (compute-bound at ~7s/step).
#   - RESUME=0         → always start fresh from step 1 so the drift is observed
#                        from the beginning (the real runs never reached a
#                        checkpoint anyway — they died at ~step 500 < 5000).
#   - separate train_debug_%j.{out,err} filenames so debug logs don't mix with
#                        production train_%j.* logs.
#
# What to look for afterwards (filter to one worker — the leak is uniform):
#   grep '\[REFCOUNT\]' logs/train_debug_<jobid>.err | grep 'worker=0' \
#       | sed -E 's/.*phase=([a-z_]+):.*delta=([+-][0-9]+).*/\1 \2/' | sort | uniq -c
#   → the phase printed with a consistent delta=-N is the culprit. If only the
#     NET line drifts (no single phase stands out), add finer tracker.mark()
#     calls inside the retry loop of _generate_one.

set -e
PROJ_DIR="${PROJ_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJ_DIR"
mkdir -p logs checkpoints

export UWYK_SRC="${UWYK_SRC:-$PROJ_DIR/../g4cfm/src}"
if [ ! -d "$UWYK_SRC" ]; then
    echo "ERROR: UWYK_SRC not found at $UWYK_SRC"
    echo "Clone it:  git clone --depth 1 https://github.com/ArikReuter/Graphs4CausalFoundationModels.git \"$(dirname "$UWYK_SRC")\""
    exit 1
fi

module load python/3.11 2>/dev/null || true
source .venv/bin/activate

echo "=== Node info ==="
hostname
date
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "================="

# ── Model (UWYK Appendix G) — identical to submit_train.sh ───
export J=100
export D_MODEL=256
export DEPTH=8
export HEADS=8
export HIDDEN_MULT=4
export DROPOUT=0.0

# ── Optimizer (UWYK Appendix G) ─────────────────────────────
export LR=1e-4
export WEIGHT_DECAY=1e-5
export WARMUP_FRAC=0.1
export MIN_LR_RATIO=0.1
export GRAD_CLIP=1.0

# ── Training ────────────────────────────────────────────────
export N_STEPS=50000
export MICROBATCH=4
export GRAD_ACCUM=8
export N_CONTEXT_TRAIN=1000
export N_QUERY_TRAIN=250

# ── Precision ───────────────────────────────────────────────
export USE_BF16=1
export USE_CHECKPOINT=1

# ── Streaming data ──────────────────────────────────────────
# Kept at the production value of 8 so this faithfully reproduces the real
# crash. See header for why fewer workers would NOT slow the abort down.
export STREAM_WORKERS=8
export STREAM_SEED=42
export STREAM_WARMUP=4

# ── Data-pipeline diagnostics ───────────────────────────────
export DATA_REFCOUNT=1     # the new None-refcount probe (the key one here)
export DATA_BREADCRUMB=1    # last (idx,seed) per worker, survives a hard C abort
export DATA_VALIDATE=1      # flag NaN/Inf / target scaling blow-ups per sample

# ── Checkpoints ─────────────────────────────────────────────
export CHECKPOINT_DIR="$PROJ_DIR/checkpoints"
export CHECKPOINT_EVERY=5000
export RESUME=0             # debug run: always start fresh from step 1

# ── Logging ─────────────────────────────────────────────────
export LOG_EVERY=100

time python -u training/train_cfm.py
