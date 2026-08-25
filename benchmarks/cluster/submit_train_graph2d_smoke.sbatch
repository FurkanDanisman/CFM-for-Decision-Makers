#!/bin/bash
#SBATCH --job-name=rpfn-graph2d-smoke
#SBATCH --output=logs_graph2d_smoke/train_%j.out
#SBATCH --error=logs_graph2d_smoke/train_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Fast smoke test for the graph-conditioned 2D-head trainer. Exercises the
# ACTUAL training loop (streaming loader, model forward, joint loss,
# backward, grad clip, LR schedule, checkpoint save + resume path) on an
# L40S so we catch bugs before burning H100 time.
#
# Downsized settings:
#   - J=10           head is 10*10+13 = 113 outputs (fits in seconds/step)
#   - microbatch=2 × grad_accum=1
#   - depth=2, d_model=64
#   - 50 steps, checkpoint every 25 → we see one save + one resume
#   - workers=2
# Expected wall time: ~5-15 min on L40S.

set -euo pipefail
mkdir -p logs_graph2d_smoke

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
for _ in 1 2 3; do
    [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
    DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
done
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
[ ! -d "$REPO/training_graph2d" ] && REPO="$DEPLOY_ROOT"
UWYK_ROOT="${UWYK_ROOT:-$DEPLOY_ROOT/external/uwyk}"

# Tiny config so a step takes ≤1s
export J=10
export NUM_FEATURES=50
export D_MODEL=64
export DEPTH=2
export HEADS=4
export DROPOUT=0.0
export HIDDEN_MULT=2

export LR=1e-4
export WEIGHT_DECAY=1e-5
export WARMUP_FRAC=0.1
export MIN_LR_RATIO=0.1
export GRAD_CLIP=1.0

export N_STEPS=50
export MICROBATCH=2
export GRAD_ACCUM=1
export N_CONTEXT_TRAIN=200
export N_QUERY_TRAIN=50

export USE_BF16=1
export USE_CHECKPOINT=0            # small model — no need for act ckpt

export STREAM_WORKERS=2
export STREAM_SEED=1
export STREAM_WARMUP=2

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO/checkpoints_graph2d_smoke}"
export CHECKPOINT_EVERY=25
export RESUME=1

export LOG_EVERY=5
export LOSS_WARN_THRESH=1e6

export UWYK_ROOT
export UWYK_SRC="$UWYK_ROOT/src"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] SMOKE  job=$SLURM_JOB_ID  repo=$REPO"
echo "  CHECKPOINT_DIR=$CHECKPOINT_DIR   N_STEPS=$N_STEPS   J=$J"

# Wipe the smoke checkpoint dir at the start so we always test cold-start
# (a leftover from a previous smoke run would silently trigger the RESUME
# path and hide cold-start bugs).
rm -rf "$CHECKPOINT_DIR"

cd "$REPO"

echo
echo "── PHASE 1: cold-start, N_STEPS=25 (produces one checkpoint) ──"
N_STEPS=25 python -u -m training_graph2d.train_graph_2d

echo
echo "── PHASE 2: resume from step 25, N_STEPS=50 (exercises resume path) ──"
N_STEPS=50 python -u -m training_graph2d.train_graph_2d

echo
echo "── PHASE 3: assert checkpoints landed ──"
ls -la "$CHECKPOINT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] SMOKE DONE"
