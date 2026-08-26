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

# Match CausalPFN's num_workers=32 exactly. Requires cpus-per-task=32
# above; if you reduce the CPU quota, drop this in lockstep.
export STREAM_WORKERS=32
export STREAM_WARMUP=4
export COMPILE=1                       # torch.compile(model, dynamic=True)

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
