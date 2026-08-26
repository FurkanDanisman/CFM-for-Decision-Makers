#!/bin/bash
#SBATCH --job-name=rpfn-cpfn2d
#SBATCH --output=logs_causalpfn2d/train_%j.out
#SBATCH --error=logs_causalpfn2d/train_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

# CausalPFN 2D-head training. 20h wall clock. 1×H100 GPU. Requeues on
# preemption; SIGTERM handler saves an interrupt checkpoint 120s before
# SLURM sends SIGKILL. RESUME=1 picks up from latest checkpoint on
# the next submit.
#
# Requires the CausalPFN repo (codex/add-training-code branch) checked out
# under $DEPLOY_ROOT/external/causalpfn. The trainer imports the model
# body from causalpfn.models.model.TabDPTLongContextModel.

set -euo pipefail
mkdir -p logs_causalpfn2d

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    if   [ -f "$(dirname "$DEPLOY_ROOT")/venv/bin/activate" ]; then
        DEPLOY_ROOT="$(dirname "$DEPLOY_ROOT")"
    elif [ -f "$(dirname "$(dirname "$DEPLOY_ROOT")")/venv/bin/activate" ]; then
        DEPLOY_ROOT="$(dirname "$(dirname "$DEPLOY_ROOT")")"
    fi
fi
echo "[bootstrap] DEPLOY_ROOT=$DEPLOY_ROOT  PWD=$PWD"
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
[ ! -d "$REPO/training_causalpfn2d" ] && REPO="$DEPLOY_ROOT"
export CAUSALPFN_ROOT="${CAUSALPFN_ROOT:-$DEPLOY_ROOT/external/causalpfn}"
export UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk/src}"     # PairedInterventionalDataset needs UWYK's SCMSampler

# LR-schedule horizon (see comment in submit_train_graph2d.sbatch about the
# 1M mistake — never decouple this from the intended training length).
export J=100
export NUM_FEATURES=50

export NINP=256
export NHID_FACTOR=4
export NHEAD=8
export NLAYERS=8
export DROPOUT=0.0
export N_OUT=10

export LR=1e-4
export WEIGHT_DECAY=1e-5
export WARMUP_FRAC=0.1
export MIN_LR_RATIO=0.1
export GRAD_CLIP=1.0

export N_STEPS=50000
export MICROBATCH=2
export GRAD_ACCUM=16
export N_CONTEXT_TRAIN=1000
export N_QUERY_TRAIN=250

export USE_BF16=1

export STREAM_WORKERS=8
export STREAM_SEED=42
export STREAM_WARMUP=4

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO/checkpoints_causalpfn2d}"
export CHECKPOINT_EVERY=2000
export RESUME=1

export LOG_EVERY=100
export LOSS_WARN_THRESH=1e3

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

# Forward SIGTERM to python so the trainer's interrupt-checkpoint handler
# fires, then requeue for the next 20h window.
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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] job=$SLURM_JOB_ID  repo=$REPO  causalpfn=$CAUSALPFN_ROOT"
echo "  CHECKPOINT_DIR=$CHECKPOINT_DIR   N_STEPS=$N_STEPS"

cd "$REPO"
python -u -m training_causalpfn2d.train_causalpfn_2d &
CHILD=$!
wait "$CHILD"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
