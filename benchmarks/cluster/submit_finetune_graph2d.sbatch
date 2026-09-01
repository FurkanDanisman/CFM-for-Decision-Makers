#!/bin/bash
#SBATCH --job-name=g2d-ft
#SBATCH --output=logs_graph2d/finetune_%j.out
#SBATCH --error=logs_graph2d/finetune_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

# Fine-tune graph2d from step_50000 with corrected data pipeline:
#   1. xgboost_prob probabilities [1.0, 0, 0, 0] -> [0.9, 0.1, 0.01, 0.001]
#      (bakes ~10% XGBoost-mechanism tasks into training data — matches
#      reproduce-realcause-results best_model_config.yaml)
#   2. _clip_outliers wired at outlier_quantile=0.99 in _standardize path
#      (matches their preprocessing_config.remove_outliers=true)
#
# RESUME=1 picks up from checkpoints_graph2d/step_50000.pt; extended
# N_STEPS=55000 gives 5000 fine-tune steps under the corrected data
# pipeline. Cosine LR is at min (1e-5) at step_50000 so the extension
# is naturally in fine-tune regime.

set -euo pipefail
mkdir -p logs_graph2d

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
for _ in 1 2 3; do
    [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
    DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
done
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
[ ! -d "$REPO/training_graph2d" ] && REPO="$DEPLOY_ROOT"
UWYK_ROOT="${UWYK_ROOT:-$DEPLOY_ROOT/external/uwyk}"

# ── Architecture (unchanged from initial training) ──
export J=32
export NUM_FEATURES=50
export D_MODEL=256
export DEPTH=6
export HEADS=4
export N_SAMPLE_SINK_ROWS=10
export N_FEATURE_SINK_COLS=0
export DROPOUT=0.0
export HIDDEN_MULT=4

# ── Optim (unchanged) ──
export LR=1e-4
export WEIGHT_DECAY=1e-5
export WARMUP_FRAC=0.1
export MIN_LR_RATIO=0.1
export GRAD_CLIP=1.0

# ── FINE-TUNE: extend N_STEPS from 50000 to 60000 (10000 additional).
# With XGB probabilities boosted to [0.5, 0.4, 0.08, 0.02] (~50% XGB
# tasks per step), 10k fine-tune steps ≈ ~650 effective XGB task-units,
# matching the ~610 units the reproduce branch's original config gave
# over its 50k-step training.
export N_STEPS=60000

export MICROBATCH=2
export GRAD_ACCUM=16
export N_CONTEXT_TRAIN=1000
export N_QUERY_TRAIN=250
export USE_BF16=1
export USE_CHECKPOINT=0
export STREAM_WORKERS=8
export STREAM_SEED=42
export STREAM_WARMUP=4

# ── Point at existing checkpoint dir; RESUME=1 loads step_50000 ──
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO/checkpoints_graph2d}"
export CHECKPOINT_EVERY=1000       # more frequent during fine-tune
export RESUME=1

export LOG_EVERY=100
export LOSS_WARN_THRESH=1e3

export UWYK_ROOT
export UWYK_SRC="$UWYK_ROOT/src"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

CHILD=""
_on_term() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SIGTERM — forwarding to trainer pid=$CHILD"
    [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap _on_term SIGTERM

echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINE-TUNE graph2d  N_STEPS=$N_STEPS  CKPT_DIR=$CHECKPOINT_DIR"
echo "  fixes: xgboost_prob=[0.9,0.1,0.01,0.001] + _clip_outliers wired at q=0.99"

cd "$REPO"
python -u -m training_graph2d.train_graph_2d &
CHILD=$!
wait "$CHILD"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
