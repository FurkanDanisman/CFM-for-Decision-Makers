#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=cfm-train
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Full UWYK-scale training run on 1 H100.
#
# Resumes from $SCRATCH/CFM-for-Decision-Makers/checkpoints/step_*.pt if any
# exist, so you can resubmit this script after a time-out and it picks up
# where it left off.
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
cd $SCRATCH/CFM-for-Decision-Makers
mkdir -p logs checkpoints

# UWYK source (must be cloned from a login node — compute nodes have no internet)
export UWYK_SRC=$SCRATCH/g4cfm/src
if [ ! -d "$UWYK_SRC" ]; then
    echo "ERROR: UWYK_SRC not found at $UWYK_SRC"
    echo "Clone it on the login node:"
    echo "    cd \$SCRATCH && git clone --depth 1 https://github.com/ArikReuter/Graphs4CausalFoundationModels.git g4cfm"
    exit 1
fi

module purge
module load python/3.11
module load scipy-stack
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
export MICROBATCH=8
export GRAD_ACCUM=4
export N_CONTEXT_TRAIN=1000
export N_QUERY_TRAIN=250

# ── Precision ───────────────────────────────────────────────
export USE_BF16=1

# ── Streaming data ──────────────────────────────────────────
export STREAM_WORKERS=8
export STREAM_SEED=42
export STREAM_WARMUP=4

# ── Checkpoints ─────────────────────────────────────────────
export CHECKPOINT_DIR=$SCRATCH/CFM-for-Decision-Makers/checkpoints
export CHECKPOINT_EVERY=5000
export RESUME=1

# ── Logging ─────────────────────────────────────────────────
export LOG_EVERY=100

time python train_cfm.py
