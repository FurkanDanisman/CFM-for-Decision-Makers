#!/bin/bash
#SBATCH --job-name=rpfn-cpfn2d
#SBATCH --output=logs_causalpfn2d/train_%j.out
#SBATCH --error=logs_causalpfn2d/train_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:h100:1
# Paper trainer uses num_workers=32; we match. BackdoorDGPMetaDataset does
# CPU-bound SCM sampling per task, so an under-provisioned CPU quota
# starves the GPU. Prior run at cpus=8 was ~4x slower than paper on data
# loading alone.
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
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

# LR-schedule horizon (see comment in submit_train_graph2d.sbatch about the
# 1M mistake — never decouple this from the intended training length).
# Every knob mirrors conf/{trainer,optimizer,meta_dataset,model}/*.yaml on
# CausalPFN's codex/add-training-code branch. Only difference from their
# published run: the head is our 2D joint (K**2 + 13 = 1037 ≈ their
# nbins=1024) and the loss is neg_log_prob_2d instead of HL-Gauss CE.
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

# Match CausalPFN's conf/train.yaml verbatim: num_workers=32,
# prefetch_factor=4. The prior "8 workers per rank in DDP" reasoning was
# wrong — CausalPFN's DataLoader is num_workers=32 per rank. Job 5038775's
# thrash was because the H100 node was shared with other jobs; on a
# dedicated allocation (this sbatch requests --cpus-per-task=32) 32
# workers × 1 CPU each fits exactly.
export STREAM_WORKERS="${STREAM_WORKERS:-32}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
# BarDist edges are computed from STREAM_WARMUP batches before training
# starts. Independent of DataLoader prefetch_factor.
export STREAM_WARMUP="${STREAM_WARMUP:-16}"
# torch.compile: backbone-only mode. Full-model compile hit an
# Inductor tiling assertion (job 5032910) triggered by the loss's
# data-dependent scatter writes in neg_log_prob_2d. Backbone-only
# compiles the 20-layer transformer (where the compute lives) and
# leaves the loss eager — no compile-hostile patterns in the compiled
# graph.  Set COMPILE_MODE=full to retry full compile, COMPILE_MODE=off
# to disable.
export COMPILE="${COMPILE:-1}"
# Honor COMPILE_MODE passed via --export= if set; else default to backbone.
# Options: 'backbone' (safe, compiles only the transformer body),
#          'full' (compiles model + loss end-to-end — now safe after the
#                  scatter/.item() rewrite of neg_log_prob_2d),
#          'off' (no compile at all).
export COMPILE_MODE="${COMPILE_MODE:-backbone}"

# One-time diagnostic: profile per-microbatch fwd/bwd/data-load timings
# for the first N steps so we can see where wall-clock actually goes.
# Set to 0 in steady-state runs.
export PROFILE_STEPS=3

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO/checkpoints_causalpfn2d}"
export CHECKPOINT_EVERY=2000
export RESUME=1

export LOG_EVERY=100
export LOSS_WARN_THRESH=1e3

source "$DEPLOY_ROOT/venv/bin/activate"
# CRITICAL: OMP=SLURM_CPUS_PER_TASK caused thread oversubscription in
# DataLoader workers (32 workers × 32 OMP threads = 1024-thread thrash,
# load average 500+, training stalled). NumPy's SCM sampling doesn't
# benefit from >1 BLAS thread anyway. See L40s sbatch for full analysis.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
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

cd "$REPO"
python -u -m training_causalpfn2d.train_causalpfn_2d &
CHILD=$!
wait "$CHILD"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
