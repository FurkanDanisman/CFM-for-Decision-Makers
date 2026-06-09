#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=cfm-scaled
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/scaled_%j.out
#SBATCH --error=logs/scaled_%j.err

# Short scaled-up test: UWYK Appendix G model dims + real config, but only
# 500 steps. Catches OOM, throughput, and noise behaviour with real grad
# accumulation BEFORE committing to the 72-hour run.

set -e
cd $SCRATCH/CFM-for-Decision-Makers
mkdir -p logs

export UWYK_SRC=$SCRATCH/g4cfm/src
if [ ! -d "$UWYK_SRC" ]; then
    echo "ERROR: UWYK_SRC not found at $UWYK_SRC"
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

# ── UWYK Appendix G config ──────────────────────────────────
export J=100
export D_MODEL=256
export DEPTH=8
export HEADS=8
export LR=1e-4
export WEIGHT_DECAY=1e-5
export WARMUP_FRAC=0.1
# Halved from UWYK's 8 to fit the bigger 2D-head model in 80GB.
# Effective batch stays at 32 (UWYK App. G value) via GRAD_ACCUM=8.
export MICROBATCH=4
export GRAD_ACCUM=8
export N_CONTEXT_TRAIN=1000
export N_QUERY_TRAIN=250
export USE_BF16=1
export USE_CHECKPOINT=1
export STREAM_WORKERS=8

# ── Short run, no checkpoint resume ────────────────────────
export N_STEPS=500
export LOG_EVERY=20
export CHECKPOINT_DIR=$SCRATCH/CFM-for-Decision-Makers/checkpoints_scaled
export CHECKPOINT_EVERY=500
export RESUME=0

time python train_cfm.py
