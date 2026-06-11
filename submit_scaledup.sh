#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=cfm-scaled
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/lukez/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers/logs/scaled_%j.out
#SBATCH --error=/home/lukez/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers/logs/scaled_%j.err

# Short scaled-up test: UWYK Appendix G model dims + real config, but only
# 500 steps. Catches OOM, throughput, and noise behaviour with real grad
# accumulation BEFORE committing to the 72-hour run.

set -e
PROJ_DIR="/home/lukez/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers"
cd "$PROJ_DIR"
mkdir -p logs

export UWYK_SRC="$PROJ_DIR/g4cfm/src"
if [ ! -d "$UWYK_SRC" ]; then
    echo "ERROR: UWYK_SRC not found at $UWYK_SRC"
    exit 1
fi

# venv was built by uv against the module python/3.11 (3.11.5) and is
# self-contained (include-system-site-packages=false), so all deps — torch,
# numpy, scipy, clarabel — live inside .venv. Only the base python module is
# needed; scipy-stack would be ignored by the venv.
module load python/3.11
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
export CHECKPOINT_DIR="$PROJ_DIR/checkpoints_scaled"
export CHECKPOINT_EVERY=500
export RESUME=0

time python -u train_cfm.py
