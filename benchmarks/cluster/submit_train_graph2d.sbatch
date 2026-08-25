#!/bin/bash
#SBATCH --job-name=rpfn-graph2d
#SBATCH --output=logs_graph2d/train_%j.out
#SBATCH --error=logs_graph2d/train_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

# Graph-conditioned 2D-head training. 20h wall clock. 1×L40S GPU. Requeues on
# preemption; the trainer's SIGTERM handler saves a checkpoint 120s before
# SLURM sends SIGKILL and RESUME=1 picks up from the latest checkpoint on
# resubmit.

set -euo pipefail
mkdir -p logs_graph2d

# ── Cluster paths ─────────────────────────────────────────────────────────
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
# If we were submitted from inside R-PFN, DEPLOY_ROOT will point at R-PFN
# but the venv/external siblings live one level up. Climb up until we find
# the venv, capping at 3 hops so a genuinely missing venv still errors out.
for _ in 1 2 3; do
    [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
    DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
done
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
[ ! -d "$REPO/training_graph2d" ] && REPO="$DEPLOY_ROOT"
UWYK_ROOT="${UWYK_ROOT:-$DEPLOY_ROOT/external/uwyk}"

# ── Training config (env-var → src/training_graph2d/train_graph_2d.py) ──
# Values MUST mirror training_graph2d/configs/graph2d_20k.yaml.

export J=100
export NUM_FEATURES=50

export D_MODEL=256
export DEPTH=8
export HEADS=8
export DROPOUT=0.0
export HIDDEN_MULT=4

export LR=1e-4
export WEIGHT_DECAY=1e-5
export WARMUP_FRAC=0.1
export MIN_LR_RATIO=0.1
export GRAD_CLIP=1.0

export N_STEPS=20000
export MICROBATCH=8
export GRAD_ACCUM=4
export N_CONTEXT_TRAIN=1000
export N_QUERY_TRAIN=250

export USE_BF16=1
export USE_CHECKPOINT=1

export STREAM_WORKERS=8
export STREAM_SEED=42
export STREAM_WARMUP=4

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO/checkpoints_graph2d}"
export CHECKPOINT_EVERY=2000
export RESUME=1

export LOG_EVERY=100
export LOSS_WARN_THRESH=1e3

# ── Env for imports ──────────────────────────────────────────────────────
export UWYK_ROOT
export UWYK_SRC="$UWYK_ROOT/src"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

# On wall-clock timeout SLURM sends SIGTERM to this batch shell 120s before
# SIGKILL (--signal=B:SIGTERM@120 above). We forward SIGTERM to the trainer
# so its Python handler saves step_{step}_interrupt.pt, wait for it to flush,
# then requeue the job. RESUME=1 picks the latest checkpoint on the next run.
CHILD=""
_on_term() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SIGTERM — forwarding to trainer pid=$CHILD"
    [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] requeueing $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap _on_term SIGTERM

echo "[$(date '+%Y-%m-%d %H:%M:%S')] job=$SLURM_JOB_ID  repo=$REPO  uwyk=$UWYK_ROOT"
echo "  CHECKPOINT_DIR=$CHECKPOINT_DIR   N_STEPS=$N_STEPS"

cd "$REPO"
python -u -m training_graph2d.train_graph_2d &
CHILD=$!
wait "$CHILD"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
