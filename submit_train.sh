#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=cfm-train
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/lukez/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers/logs/train_%j.out
#SBATCH --error=/home/lukez/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers/logs/train_%j.err

# Full UWYK-scale training run on 1 H100.
#
# 20-hour SLURM chunks. Estimated total wall-clock ~53 hours (single-process
# data loading), so submit this script 3 times in sequence: each invocation
# runs until the 20h limit, saves a checkpoint, exits cleanly; the next
# invocation picks up at the latest checkpoint thanks to RESUME=1. Sequence:
#
#   FIRST=$(sbatch --parsable submit_train.sh)
#   for i in 2 3; do
#       FIRST=$(sbatch --parsable --dependency=afterany:$FIRST submit_train.sh)
#   done
#
# Or just `sbatch submit_train.sh` after each finishes if you want manual
# control.
#
# Config: UWYK Appendix G defaults + our 2D BarDistribution head.
#   - d_model=256, depth=8, heads=8           → ~8M backbone params
#   - Adam(lr=1e-4, wd=1e-5), cosine + 10% warmup
#   - effective batch 32 (microbatch 8 × grad accum 4)
#   - N=1000 context rows, M=250 queries per task
#   - bf16 mixed precision via torch.autocast
#   - 50,000 steps
#   - checkpoints every 5,000 steps to $SCRATCH/CFM-for-Decision-Makers/checkpoints/

set -e
PROJ_DIR="/home/lukez/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers"
cd "$PROJ_DIR"
mkdir -p logs checkpoints

export UWYK_SRC="$PROJ_DIR/g4cfm/src"
if [ ! -d "$UWYK_SRC" ]; then
    echo "ERROR: UWYK_SRC not found at $UWYK_SRC"
    echo "Clone it:  git clone --depth 1 https://github.com/ArikReuter/Graphs4CausalFoundationModels.git \"$PROJ_DIR/g4cfm\""
    exit 1
fi

# venv was built by uv against the module python/3.11 (3.11.5) and is
# self-contained (include-system-site-packages=false), so all deps live inside
# .venv. Only the base python module is needed; scipy-stack would be ignored.
module load python/3.11
source .venv/bin/activate

echo "=== Node info ==="
hostname
date
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "================="

# ── Model (UWYK Appendix G) ─────────────────────────────────
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
# Halved from UWYK's 8 to fit the bigger 2D-head model in 80GB.
# Effective batch stays at 32 (UWYK App. G value) via GRAD_ACCUM=8.
export MICROBATCH=4
export GRAD_ACCUM=8
export N_CONTEXT_TRAIN=1000
export N_QUERY_TRAIN=250

# ── Precision ───────────────────────────────────────────────
export USE_BF16=1
export USE_CHECKPOINT=1

# ── Streaming data ──────────────────────────────────────────
export STREAM_WORKERS=8
export STREAM_SEED=42
export STREAM_WARMUP=4

# ── Checkpoints ─────────────────────────────────────────────
export CHECKPOINT_DIR="$PROJ_DIR/checkpoints"
export CHECKPOINT_EVERY=5000
export RESUME=1

# ── Logging ─────────────────────────────────────────────────
export LOG_EVERY=100

time python -u train_cfm.py
