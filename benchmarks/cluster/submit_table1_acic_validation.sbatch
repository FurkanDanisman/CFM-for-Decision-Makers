#!/bin/bash
#SBATCH --job-name=t1-acic-valid
#SBATCH --output=logs_table1_acic_valid/task_%A_%a.out
#SBATCH --error=logs_table1_acic_valid/task_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-3

# Table-1 ACIC validation:
#
#   task 0 : UWYK  predictive -- predmodel_Slearner_full_context.py
#   task 1 : UWYK  no-anc     -- dofm_no_clustering, adjacency all-zero
#   task 2 : UWYK  anc        -- dofm_no_clustering, full graph
#   task 3 : ours  fn=50      -- ours_fn50_no_clustering.py
#
# All three tasks use the SAME inference protocol:
#   - single forward pass over the full training context (no clustering)
#   - per-task X standardisation (mean/std from X_train)
#   - per-task Y scaling to [-1, 1]
#   - inverse-scale CATE back to raw Y units
#
# The only true differences are the model architecture (K+4 vs K^2+13
# joint head) and, because our checkpoint was trained on binary T while
# UWYK's was trained on target-encoded T, the T convention at inference.
# See run_uwyk_table1.sh + ours_fn50_no_clustering.py headers for the
# per-model contracts.

set -euo pipefail
mkdir -p logs_table1_acic_valid

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
export UWYK_ROOT="${UWYK_ROOT:-$DEPLOY_ROOT/external/uwyk}"
export CAUSALPFN_ROOT="${CAUSALPFN_ROOT:-$DEPLOY_ROOT/external/causalpfn}"
export UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$UWYK_ROOT/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
export CKPT_FILE="${CKPT_FILE:-best_model.pt}"
export CONFIG_FILE="${CONFIG_FILE:-best_model_config.yaml}"
export OURS_CKPT="${OURS_CKPT:-$REPO/checkpoints/step_50000_final.pt}"
export TABLE1_OUT_ROOT="${TABLE1_OUT_ROOT:-$DEPLOY_ROOT/results_table1_acic_valid}"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] task=$SLURM_ARRAY_TASK_ID"

case "$SLURM_ARRAY_TASK_ID" in
    0)
        echo "== UWYK predictive on ACIC =="
        bash "$REPO/benchmarks/uwyk_table1/run_uwyk_table1.sh" \
            ACIC predictive "acic_valid_${SLURM_ARRAY_JOB_ID}"
        ;;
    1)
        echo "== UWYK no-anc on ACIC =="
        bash "$REPO/benchmarks/uwyk_table1/run_uwyk_table1.sh" \
            ACIC noanc "acic_valid_${SLURM_ARRAY_JOB_ID}"
        ;;
    2)
        echo "== UWYK anc on ACIC =="
        bash "$REPO/benchmarks/uwyk_table1/run_uwyk_table1.sh" \
            ACIC anc "acic_valid_${SLURM_ARRAY_JOB_ID}"
        ;;
    3)
        echo "== ours fn=50 on ACIC (predictive-mirror) =="
        [ -f "$OURS_CKPT" ] || { echo "FATAL: fn=50 ckpt not at $OURS_CKPT" >&2; exit 1; }
        python -u "$REPO/benchmarks/uwyk_table1/ours_fn50_predictive_style.py" \
            --dataset ACIC --model ours_fn50 \
            --exp_name "table1_ours_fn50_acic_${SLURM_ARRAY_JOB_ID}"
        ;;
    *)
        echo "invalid array index $SLURM_ARRAY_TASK_ID" >&2; exit 2
        ;;
esac

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done task=$SLURM_ARRAY_TASK_ID"
