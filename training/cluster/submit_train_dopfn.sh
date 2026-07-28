#!/bin/bash
#SBATCH --account=def-rgrosse
#SBATCH --job-name=cfm-train-dopfn
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/train_dopfn_%j.out
#SBATCH --error=logs/train_dopfn_%j.err

# Full-scale training run for the Do-PFN prior variant on 1 H100 (Nibi).
#
# Twin of submit_train.sh (the UWYK-prior trainer). Model / optimizer /
# schedule / precision / effective batch are all UWYK Appendix G — only the
# data source differs (Do-PFN's SCM prior instead of UWYK's).
#
# 20-hour SLURM chunks with RESUME=1. Submit N times in sequence — each
# invocation runs until the wall-clock limit, saves a checkpoint (via the
# SIGTERM handler), exits cleanly; the next invocation picks up from the
# latest checkpoint in $CHECKPOINT_DIR. Chain them with:
#
#   FIRST=$(sbatch --parsable training/cluster/submit_train_dopfn.sh)
#   for i in 2 3; do
#       FIRST=$(sbatch --parsable --dependency=afterany:$FIRST \
#               training/cluster/submit_train_dopfn.sh)
#   done
#
# Config: UWYK Appendix G + our 2D BarDistribution head + Do-PFN prior.
#   - d_model=256, depth=8, heads=8
#   - Adam(lr=1e-4, wd=1e-5), cosine + 10% warmup
#   - effective batch 32 (microbatch 4 × grad accum 8)
#   - N=1000 context rows, M=250 queries per task
#   - Do-PFN emits NUM_FEATURES=10 per task (no pad-to-50)
#   - bf16 mixed precision via torch.autocast
#   - 50,000 steps
#   - checkpoints every 5,000 steps to $PROJ_DIR/checkpoints_dopfn/

set -e
PROJ_DIR="${PROJ_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$PROJ_DIR"
mkdir -p logs

# ── Environment ─────────────────────────────────────────────────────────
module load python/3.11

VENV_DIR="${VENV_DIR:-$PROJ_DIR/.venv}"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: venv not found at $VENV_DIR (override with VENV_DIR=…)"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# Do-PFN source — must contain priors/playground_scm/…
export DOPFN_SRC="${DOPFN_SRC:-$PROJ_DIR/Do-PFN}"
if [ ! -d "$DOPFN_SRC/priors/playground_scm" ]; then
    echo "ERROR: DOPFN_SRC=$DOPFN_SRC does not contain priors/playground_scm/"
    exit 1
fi

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
# MICROBATCH halved from UWYK's 8 to fit the 2D-head model in 80 GB;
# effective batch stays at 32 (App. G) via GRAD_ACCUM=8.
export MICROBATCH=4
export GRAD_ACCUM=8
export N_CONTEXT_TRAIN=1000
export N_QUERY_TRAIN=250

# ── Do-PFN prior ────────────────────────────────────────────
export NUM_FEATURES=10    # fixed by prior, no padding
export N_TRAIN=1000       # per-task SCM sample sizes (>= N_CONTEXT_TRAIN)
export N_TEST=500         #                            (>= N_QUERY_TRAIN)

# ── Precision ───────────────────────────────────────────────
export USE_BF16=1
export USE_CHECKPOINT=1

# ── Streaming data ──────────────────────────────────────────
export STREAM_WORKERS=8
export STREAM_SEED=42
export STREAM_WARMUP=4

# ── Checkpoints ─────────────────────────────────────────────
export CHECKPOINT_DIR="$PROJ_DIR/checkpoints_dopfn"
export CHECKPOINT_EVERY=5000
export RESUME=1

# ── Logging ─────────────────────────────────────────────────
export LOG_EVERY=100

time python -u training/train_cfm_dopfn.py
