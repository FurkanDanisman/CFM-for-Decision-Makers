#!/bin/bash
#SBATCH --job-name=rpfn-cpfn2d-ddp
#SBATCH --output=logs_causalpfn2d/train_ddp_%j.out
#SBATCH --error=logs_causalpfn2d/train_ddp_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

# 4-GPU DDP training that mirrors CausalPFN's setup (4-GPU DDP was
# their causal-phase config on 1 H100 node — verified from the paper
# Section D.2 and their src/causalpfn/training/trainer.py). Effective
# batch across 4 ranks = MICROBATCH × GRAD_ACCUM × 4 = 32 × 8 × 4 = 1024
# per gradient update — matches CausalPFN.
#
# Launch pattern:
#   `srun torchrun --standalone --nproc_per_node=4 -m training_causalpfn2d.
#    train_causalpfn_2d`
#
# The trainer auto-detects DDP via LOCAL_RANK/RANK/WORLD_SIZE env vars
# torchrun sets. When those aren't present (single-GPU sbatch), it runs
# as before — no code duplication.

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
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
[ ! -d "$REPO/training_causalpfn2d" ] && REPO="$DEPLOY_ROOT"
export CAUSALPFN_ROOT="${CAUSALPFN_ROOT:-$DEPLOY_ROOT/external/causalpfn}"
export UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk/src}"

# WARMSTART IS MANDATORY. See submit_train_causalpfn2d.sbatch for the
# rationale — training from scratch invalidates the CausalPFN comparison.
export WARMSTART_LOCAL="${WARMSTART_LOCAL:-$DEPLOY_ROOT/warmstart/tabdpt_long_context.ckpt}"
if [ ! -f "$WARMSTART_LOCAL" ]; then
    echo "FATAL: WARMSTART_LOCAL not found at $WARMSTART_LOCAL." >&2
    echo "       Refusing to run — training from scratch would invalidate the CausalPFN comparison." >&2
    exit 1
fi
echo "[warmstart-preflight] found ckpt at $WARMSTART_LOCAL"

# Same knobs as single-GPU sbatch. Per-rank batch matches CausalPFN's;
# world_size=4 gives effective batch = 1024 (paper spec).
export J=32
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

# Per-rank workers. Slurm allocates 32 CPUs to the srun task; torchrun
# then forks 4 rank processes internally that share those CPUs. Setting
# STREAM_WORKERS=8 per rank gives 8×4 = 32 workers total = 1 worker per
# CPU (no oversubscription). CausalPFN's conf/train.yaml sets
# num_workers=32 per rank, but that assumes each rank has its own 32 CPUs
# (e.g. a 128-CPU node). Here Slurm gives us 32 total, so 8 per rank
# matches the CPU budget. If you bump --cpus-per-task=128 above, also
# bump STREAM_WORKERS to 32 to match CausalPFN verbatim.
export STREAM_WORKERS="${STREAM_WORKERS:-8}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
export STREAM_WARMUP="${STREAM_WARMUP:-16}"

# Compile mode. Full-compile hit an Inductor tiling assert on the joint
# 2D loss (job 5038595) that eventually deadlocked. Backbone-only is
# proven safe. Full compile is faster if it works but risky. Default:
# backbone. Override with COMPILE_MODE=full to try.
export COMPILE="${COMPILE:-1}"
export COMPILE_MODE="${COMPILE_MODE:-backbone}"

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO/checkpoints_causalpfn2d_ddp}"
export CHECKPOINT_EVERY=2000
export RESUME=1

export LOG_EVERY=100
export LOSS_WARN_THRESH=1e3
export PROFILE_STEPS=3

source "$DEPLOY_ROOT/venv/bin/activate"
# CRITICAL: OMP=SLURM_CPUS_PER_TASK caused thread oversubscription in
# DataLoader workers. See submit_train_causalpfn2d.sbatch for the full
# analysis. Same fix applies to DDP.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

# NCCL environment. Use a network interface torchrun expects. If the
# cluster uses SLURM's built-in NCCL topology, defaults are fine.
export MASTER_ADDR="${MASTER_ADDR:-$(hostname)}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# Forward SIGTERM to torchrun so ranks flush their interrupt checkpoint
# then requeue for the next 20h window.
CHILD=""
_on_term() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SIGTERM — forwarding to torchrun pid=$CHILD"
    [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] requeueing $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap _on_term SIGTERM

echo "[$(date '+%Y-%m-%d %H:%M:%S')] job=$SLURM_JOB_ID  repo=$REPO  causalpfn=$CAUSALPFN_ROOT"
echo "  CHECKPOINT_DIR=$CHECKPOINT_DIR   N_STEPS=$N_STEPS"
echo "  DDP: nproc_per_node=4  MASTER=$MASTER_ADDR:$MASTER_PORT"

cd "$REPO"
# `srun` inside a 1-task allocation launches torchrun as a single process
# on the compute node. torchrun then spawns 4 rank processes, each grabbing
# its assigned GPU via LOCAL_RANK.
srun --unbuffered torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    --max_restarts=0 \
    -m training_causalpfn2d.train_causalpfn_2d &
CHILD=$!
wait "$CHILD"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
