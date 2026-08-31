#!/bin/bash
#SBATCH --job-name=cpfn2d-j100-rand
#SBATCH --output=logs_causalpfn2d_l40s/train_%j.out
#SBATCH --error=logs_causalpfn2d_l40s/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

# Exp 6: cpfn2d J=100, HL-Gauss σ=0.1024 (σ/bin_width = 0.512) — redo of
# hlg02 with the correct sigma. Previous hlg02 used σ=0.2 (σ/bw = 1.0)
# which we discovered is a different smoothing regime than CausalPFN's
# native 0.51 ratio. Random-init 2D head (same as hlg02).

set -euo pipefail
mkdir -p logs_causalpfn2d_l40s

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export CAUSALPFN_ROOT="${CAUSALPFN_ROOT:-$DEPLOY_ROOT/external/causalpfn}"
export UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk/src}"

export WARMSTART_LOCAL="${WARMSTART_LOCAL:-$DEPLOY_ROOT/warmstart/tabdpt_long_context.ckpt}"
if [ ! -f "$WARMSTART_LOCAL" ]; then
    echo "FATAL: WARMSTART_LOCAL not found at $WARMSTART_LOCAL." >&2
    exit 1
fi

export J=100
export NUM_FEATURES=100
export NINP=384
export NHID=768
export NHEAD=6
export NLAYERS=20
export DROPOUT=0.0
export N_OUT=10
export LR=5e-4
export WEIGHT_DECAY=0.05
export WARMUP_STEPS=1000
export BETA1=0.98
export BETA2=0.999
export GRAD_CLIP=1.0
export N_STEPS=262144
export MICROBATCH=32
export GRAD_ACCUM=8
export MIN_TRAIN_SPLIT=0.333
export MAX_TRAIN_SPLIT=0.666
export OVERLAP_THRESHOLD=0.01
export N_SAMPLES_PER_TASK=2048
export MAX_N_COVARIATES=98
export USE_BF16=1
export STREAM_WORKERS="${STREAM_WORKERS:-32}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
export STREAM_WARMUP="${STREAM_WARMUP:-16}"
export COMPILE="${COMPILE:-1}"
export COMPILE_MODE="${COMPILE_MODE:-backbone}"
export PROFILE_STEPS=3

export Y_SCALING_MODE=pooled_std
export LOSS_TYPE=hlgauss
export HLGAUSS_SIGMA=0.1024     # 0.512 * (20 / 100) = 0.1024
export HEAD_WARMSTART_MODE=random

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO/checkpoints_cpfn2d_j100_random}"
export CHECKPOINT_EVERY=2000
export RESUME=1
export LOG_EVERY=100
export LOSS_WARN_THRESH=1e3

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

CHILD=""
_on_term() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SIGTERM — forwarding to trainer pid=$CHILD"
    [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap _on_term SIGTERM

echo "[$(date '+%Y-%m-%d %H:%M:%S')] cpfn2d Exp 6: J=100 σ=0.1024 head=random CKPT=$CHECKPOINT_DIR"

cd "$REPO"
python -u -m training_causalpfn2d.train_causalpfn_2d &
CHILD=$!
wait "$CHILD"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
