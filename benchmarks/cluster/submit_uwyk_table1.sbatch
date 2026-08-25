#!/bin/bash
#SBATCH --job-name=uwyk-table1
#SBATCH --output=logs_uwyk_table1/task_%A_%a.out
#SBATCH --error=logs_uwyk_table1/task_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-8

# Reproduce UWYK's Table 1 — three rows × three datasets, run as a Slurm
# array (all 9 tasks execute in parallel, one GPU each):
#
#   task 0  predictive  × CPS
#   task 1  predictive  × PSID
#   task 2  predictive  × IHDP
#   task 3  noanc       × CPS
#   task 4  noanc       × PSID
#   task 5  noanc       × IHDP
#   task 6  anc         × CPS
#   task 7  anc         × PSID
#   task 8  anc         × IHDP
#
# Add ACIC by expanding --array=0-11 and appending to CASES.
#
# Submit:  sbatch benchmarks/cluster/submit_uwyk_table1.sbatch

set -euo pipefail
mkdir -p logs_uwyk_table1

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
[ ! -d "$REPO/benchmarks" ] && REPO="$DEPLOY_ROOT"
UWYK_ROOT="${UWYK_ROOT:-$DEPLOY_ROOT/external/uwyk}"

CASES=(
    "predictive CPS"
    "predictive PSID"
    "predictive IHDP"
    "noanc      CPS"
    "noanc      PSID"
    "noanc      IHDP"
    "anc        CPS"
    "anc        PSID"
    "anc        IHDP"
)
CASE="${CASES[$SLURM_ARRAY_TASK_ID]}"
MODE=$(echo    "$CASE" | awk '{print $1}')
DATASET=$(echo "$CASE" | awk '{print $2}')

echo "[$(date '+%Y-%m-%d %H:%M:%S')] task=$SLURM_ARRAY_TASK_ID  mode=$MODE  dataset=$DATASET"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

export DEPLOY_ROOT UWYK_ROOT
# Full-conditioning (anc/noanc) checkpoint
export UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$UWYK_ROOT/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
export CKPT_FILE="${CKPT_FILE:-best_model.pt}"
export CONFIG_FILE="${CONFIG_FILE:-best_model_config.yaml}"
# Predictive (S-learner) checkpoint — falls back to the unconditional model we have
export PRED_CKPT_DIR="${PRED_CKPT_DIR:-$UWYK_ROOT/experiments/checkpoints/no_graph_conditioning/unconditional}"
export PRED_CKPT_FILE="${PRED_CKPT_FILE:-best_model.pt}"
export PRED_CONFIG_FILE="${PRED_CONFIG_FILE:-best_model_config.yaml}"

bash "$REPO/benchmarks/uwyk_table1/run_uwyk_table1.sh" \
    "$DATASET" "$MODE" "arr${SLURM_ARRAY_JOB_ID}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
