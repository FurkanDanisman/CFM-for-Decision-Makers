#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=cfm-train-dopfn-backbone
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/train_dopfn_backbone_%j.out
#SBATCH --error=logs/train_dopfn_backbone_%j.err

# From-scratch training run for R-PFN with Do-PFN's architecture.
#
# Uses Do-PFN's PerFeatureTransformer architecture (per-feature attention,
# unlimited feature count) + Do-PFN's SCM prior + our 2D joint head. NO
# pre-trained weights are loaded — training starts from a fresh init. This
# isolates the head-change as the sole architectural/training difference
# vs. Do-PFN, matching the paper's "only change is the head" narrative.
#
# 20-hour SLURM chunk pattern with RESUME=1:
#
#   FIRST=$(sbatch --parsable training_dopfn_base/cluster/submit_train.sh)
#   for i in 2 3; do
#       FIRST=$(sbatch --parsable --dependency=afterany:$FIRST \
#               training_dopfn_base/cluster/submit_train.sh)
#   done

set -e
# Match the working submit_ihdp_l2.sbatch layout: DEPLOY_ROOT is the parent of
# R-PFN/ (contains venv/, external/, R-PFN/, checkpoints/...).
DEPLOY_ROOT="${DEPLOY_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
[ ! -d "$REPO/training_dopfn_base" ] && REPO="$DEPLOY_ROOT"
cd "$DEPLOY_ROOT"
mkdir -p logs

# ── Environment ─────────────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-$DEPLOY_ROOT/venv}"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    if [ -f "$DEPLOY_ROOT/.venv/bin/activate" ]; then
        VENV_DIR="$DEPLOY_ROOT/.venv"
    else
        echo "ERROR: venv not found at $VENV_DIR (override with VENV_DIR=…)"
        exit 1
    fi
fi
source "$VENV_DIR/bin/activate"

# Do-PFN source — must contain artifacts/dopfn_model.pkl AND
# priors/playground_scm/ (the SCM prior used by the streaming dataset).
export DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
export DOPFN_SRC="${DOPFN_SRC:-$DOPFN_ROOT}"
if [ ! -f "$DOPFN_ROOT/artifacts/dopfn_model.pkl" ]; then
    echo "ERROR: DOPFN_ROOT=$DOPFN_ROOT does not contain artifacts/dopfn_model.pkl"
    exit 1
fi
if [ ! -d "$DOPFN_SRC/priors/playground_scm" ]; then
    echo "ERROR: DOPFN_SRC=$DOPFN_SRC does not contain priors/playground_scm/"
    exit 1
fi

echo "=== Node info ==="
hostname
date
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "================="
echo "DOPFN_ROOT=$DOPFN_ROOT"
echo "Training FROM SCRATCH (no pre-trained weights)"

# ── Head / grid ─────────────────────────────────────────────
# CRITICAL: keep this overridable via env var. A previous version
# hardcoded `export J=100` here, which silently overrode any
# `J=10 sbatch ...` submissions and produced the "J=10 scam"
# (200K checkpoint mislabelled as J=10 but actually J=100).
# See memory: feedback_verify_experimental_facts.md
export J="${J:-100}"
# NUM_FEATURES here is what the STREAMING PRIOR emits; the DoPFN backbone
# itself accepts any number of features (per-feature attention).
export NUM_FEATURES="${NUM_FEATURES:-10}"

# ── Optimizer ───────────────────────────────────────────────
export LR="${LR:-1e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
export WARMUP_FRAC="${WARMUP_FRAC:-0.1}"
export MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
export GRAD_CLIP="${GRAD_CLIP:-1.0}"

# ── Training schedule ───────────────────────────────────────
export N_STEPS="${N_STEPS:-50000}"
export MICROBATCH="${MICROBATCH:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export N_CONTEXT_TRAIN="${N_CONTEXT_TRAIN:-1000}"
export N_QUERY_TRAIN="${N_QUERY_TRAIN:-250}"

# ── Data ────────────────────────────────────────────────────
export N_TRAIN="${N_TRAIN:-1000}"
export N_TEST="${N_TEST:-500}"

# ── Precision ───────────────────────────────────────────────
export USE_BF16="${USE_BF16:-1}"

# ── Streaming ───────────────────────────────────────────────
export STREAM_WORKERS="${STREAM_WORKERS:-8}"
export STREAM_SEED="${STREAM_SEED:-42}"
export STREAM_WARMUP="${STREAM_WARMUP:-4}"

# ── Checkpoints ─────────────────────────────────────────────
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$DEPLOY_ROOT/checkpoints_dopfn_backbone}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
export RESUME="${RESUME:-1}"

# ── Logging ─────────────────────────────────────────────────
export LOG_EVERY="${LOG_EVERY:-100}"

time python -u "$REPO/training_dopfn_base/train.py"
