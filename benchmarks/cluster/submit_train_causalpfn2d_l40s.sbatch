#!/bin/bash
#SBATCH --job-name=rpfn-cpfn2d-l40s
#SBATCH --output=logs_causalpfn2d_l40s/train_%j.out
#SBATCH --error=logs_causalpfn2d_l40s/train_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:l40s:1
# Same worker allocation as the H100 sbatch. The data-loader is CPU-bound
# (BackdoorDGPMetaDataset does SCM sampling per task), so under-provisioning
# CPU here starves the GPU regardless of which GPU we're using.
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

# L40s fallback for CausalPFN 2D-head training. Identical trainer config to
# submit_train_causalpfn2d.sbatch; only the GPU class, log dir, and
# checkpoint dir differ. Purpose: unblock queue wait when all H100
# partitions (gpubase_h100_b1..b5) are fully packed. L40s throughput
# is ~2x slower per step than H100, so the 262144-step run projects to
# ~4 days on L40s vs ~2 days on H100 — worth it when H100 queue is >24h.
#
# Checkpoint dir is separate (checkpoints_causalpfn2d_l40s) so this run
# does not clobber the H100 run's progress if both start.

set -euo pipefail
mkdir -p logs_causalpfn2d_l40s

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

# WARMSTART IS MANDATORY. Every comparison against CausalPFN's published
# Stage-1 (TabDPT long-context) baseline requires that we start from the
# same checkpoint. Silently training from scratch invalidates every
# downstream number. If the file is missing, abort the job before any GPU
# time is charged. Download once on the login node with:
#   PYTHONPATH= python -c "from huggingface_hub import hf_hub_download; \
#     print(hf_hub_download('vdblm/causalpfn','tabdpt_long_context.ckpt'))"
#   cp <printed-path> $DEPLOY_ROOT/warmstart/tabdpt_long_context.ckpt
export WARMSTART_LOCAL="${WARMSTART_LOCAL:-$DEPLOY_ROOT/warmstart/tabdpt_long_context.ckpt}"
if [ ! -f "$WARMSTART_LOCAL" ]; then
    echo "FATAL: WARMSTART_LOCAL not found at $WARMSTART_LOCAL." >&2
    echo "       Refusing to run — training from scratch would invalidate the CausalPFN comparison." >&2
    echo "       Download vdblm/causalpfn/tabdpt_long_context.ckpt on the login node first." >&2
    exit 1
fi
echo "[warmstart-preflight] found ckpt at $WARMSTART_LOCAL"

# ---------------------------------------------------------------------------
# Model / trainer knobs: identical to the H100 sbatch. Do not diverge —
# both runs must produce comparable checkpoints.
# ---------------------------------------------------------------------------
export J=32                          # K^2 + 13 = 1037 ≈ CausalPFN's nbins=1024
export NUM_FEATURES=100

# Model: conf/model/tabdpt_long_context.yaml
export NINP=384
export NHID=768
export NHEAD=6
export NLAYERS=20
export DROPOUT=0.0
export N_OUT=10

# Optimizer: conf/optimizer/schedulefree_adamw.yaml
export LR=5e-4
export WEIGHT_DECAY=0.05
export WARMUP_STEPS=1000              # absolute steps, not fraction
export BETA1=0.98
export BETA2=0.999
export GRAD_CLIP=1.0

# Trainer: conf/trainer/default.yaml   (2048 epochs × 128 updates = 262144 steps;
# effective batch = batch_size × num_agg = 32 × 8 = 256)
export N_STEPS=262144
export MICROBATCH=32
export GRAD_ACCUM=8
export MIN_TRAIN_SPLIT=0.333
export MAX_TRAIN_SPLIT=0.666
export OVERLAP_THRESHOLD=0.01

# Meta-dataset: conf/meta_dataset/synthetic_backdoor.yaml
export N_SAMPLES_PER_TASK=2048
export MAX_N_COVARIATES=98

export USE_BF16=1

# Worker count matches CausalPFN's conf/train.yaml verbatim: num_workers=32,
# prefetch_factor=4. The earlier "8 workers per rank in DDP" argument was
# wrong — CausalPFN's DataLoader uses num_workers=32 PER RANK, not 8. Job
# 5038775's 32-worker thrash was a shared-node artefact; on L40s we
# request 32 CPUs so 1 worker per CPU is fine.
export STREAM_WORKERS="${STREAM_WORKERS:-32}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
export STREAM_WARMUP="${STREAM_WARMUP:-16}"

# torch.compile: backbone-only. Full-model compile hit an Inductor tiling
# assertion on the 2D loss (job 5032910). Backbone-only compiles the
# 20-layer transformer and leaves the loss eager. Set COMPILE_MODE=full
# to retry, COMPILE_MODE=off to disable.
export COMPILE="${COMPILE:-1}"
export COMPILE_MODE="${COMPILE_MODE:-backbone}"

# Profile fwd/bwd/data-load timings for the first N steps so we can verify
# the L40s run is not stalling and is within ~2x of the H100 wall-clock.
# Set to 0 in steady-state runs.
export PROFILE_STEPS=3

# NOTE: separate checkpoint dir from H100 run so both can co-exist. If you
# want to migrate progress between runs, copy the checkpoint manually rather
# than pointing both jobs at the same dir.
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO/checkpoints_causalpfn2d_l40s}"
export CHECKPOINT_EVERY=2000
export RESUME=1

export LOG_EVERY=100
export LOSS_WARN_THRESH=1e3

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
# Shims for faiss / huggingface_hub / ... that causalpfn/__init__.py
# tries to import unconditionally. Must be on PYTHONPATH BEFORE python
# starts so sitecustomize.py auto-loads.
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

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
echo "  GPU class: L40s   worker count: $STREAM_WORKERS"

cd "$REPO"
python -u -m training_causalpfn2d.train_causalpfn_2d &
CHILD=$!
wait "$CHILD"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
