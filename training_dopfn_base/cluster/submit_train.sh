#!/bin/bash
#SBATCH --account=def-rgrosse
#SBATCH --job-name=cfm-train-dopfn-backbone
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/train_dopfn_backbone_%j.out
#SBATCH --error=logs/train_dopfn_backbone_%j.err

# Full-scale training run for the DoPFN-backbone R-PFN.
#
# Differs from submit_train_dopfn.sh (the InterventionalPFN-with-DoPFN-prior
# trainer) in ONE thing: the model backbone. Instead of UWYK's
# InterventionalPFN with a hardcoded num_features cap, this loads Do-PFN's
# TabPFN transformer (per-feature attention, no cap) and replaces its 1D
# BarDistribution decoder with our 2D joint head.
#
# Same 20-hour SLURM chunking + RESUME=1 pattern:
#
#   FIRST=$(sbatch --parsable training_dopfn_base/cluster/submit_train.sh)
#   for i in 2 3; do
#       FIRST=$(sbatch --parsable --dependency=afterany:$FIRST \
#               training_dopfn_base/cluster/submit_train.sh)
#   done
#
# By default this trains HEAD-ONLY (freezes the DoPFN transformer). Set
# HEAD_ONLY=0 to unfreeze for full fine-tune (~2-3x wall clock).

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

# Do-PFN source — must contain artifacts/dopfn_model.pkl AND
# priors/playground_scm/ (the SCM prior used by the streaming dataset).
export DOPFN_ROOT="${DOPFN_ROOT:-$PROJ_DIR/Do-PFN}"
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
echo "HEAD_ONLY=${HEAD_ONLY:-1}"

# ── Head / grid ─────────────────────────────────────────────
export J=100
# NUM_FEATURES here is what the STREAMING PRIOR emits; the DoPFN backbone
# itself accepts any number of features (per-feature attention).
export NUM_FEATURES="${NUM_FEATURES:-10}"

# ── Training mode ───────────────────────────────────────────
export HEAD_ONLY="${HEAD_ONLY:-1}"    # 1 = freeze backbone; 0 = full fine-tune

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
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJ_DIR/checkpoints_dopfn_backbone}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
export RESUME="${RESUME:-1}"

# ── Logging ─────────────────────────────────────────────────
export LOG_EVERY="${LOG_EVERY:-100}"

time python -u training_dopfn_base/train.py
