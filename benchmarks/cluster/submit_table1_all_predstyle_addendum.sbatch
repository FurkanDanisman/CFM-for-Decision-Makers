#!/bin/bash
#SBATCH --job-name=t1-all-fn50p
#SBATCH --output=logs_table1_all/predstyle_%A_%a.out
#SBATCH --error=logs_table1_all/predstyle_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-3
#
# Addendum to submit_table1_all_datasets.sbatch: runs ONLY the fn=50
# pred-mirror row for each of the four datasets. Use this when the
# main all-datasets array run only covers 4 methods (predictive / noanc
# / anc / fn50 null-t) and you want to add the 5th row without cancelling
# or resubmitting the main array.
#
#   0 : IHDP        fn=50 pred-mirror
#   1 : CPS         fn=50 pred-mirror
#   2 : PSID_unbal  fn=50 pred-mirror
#   3 : PSID_bal    fn=50 pred-mirror   (uses same subsample as UWYK PSID-bal
#                                         once ours_fn50_predictive_style.py
#                                         gains a --psid-balanced flag; for
#                                         now runs unbalanced — see note below)

set -euo pipefail
mkdir -p logs_table1_all

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
export UWYK_REPRO="${UWYK_REPRO:-$DEPLOY_ROOT/external/uwyk_reproduce}"
export CAUSALPFN_ROOT="${CAUSALPFN_ROOT:-$UWYK_REPRO/RealCauseEval/CausalPFN}"
export OURS_CKPT="${OURS_CKPT:-$REPO/checkpoints/step_50000_final.pt}"
# Route pred-mirror outputs to the same TABLE1_OUT_ROOT the main
# all-datasets sbatch uses so the aggregator finds both variants side-by-side.
export TABLE1_OUT_ROOT="${TABLE1_OUT_ROOT:-$DEPLOY_ROOT/results_table1_all}"

[ -f "$OURS_CKPT" ] || { echo "FATAL: fn=50 ckpt missing at $OURS_CKPT" >&2; exit 1; }
[ -d "$CAUSALPFN_ROOT/benchmarks" ] || { echo "FATAL: CausalPFN benchmarks not at $CAUSALPFN_ROOT" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

# Exp-name scheme MUST match what aggregate_all_datasets.py's fn50p entry
# expects: table1_all_<DATASET_LOG>_fn50p_<JOB>. That way the aggregator
# picks up this addendum job as the fn50p source for each dataset via
# JOB_ID_OURS_PRED (or a wildcarded newest-match).
JOB="${SLURM_ARRAY_JOB_ID}"
TASK="${SLURM_ARRAY_TASK_ID}"

case "$TASK" in
    0) DATASET_LOG="IHDP";       DATASET_ARG="IHDP" ;;
    1) DATASET_LOG="CPS";        DATASET_ARG="CPS"  ;;
    2) DATASET_LOG="PSID_unbal"; DATASET_ARG="PSID" ;;
    3) DATASET_LOG="PSID_bal";   DATASET_ARG="PSID" ;;
esac

EXP="table1_all_${DATASET_LOG}_fn50p_${JOB}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] task=$TASK  ds=$DATASET_LOG  exp=$EXP"

python -u "$REPO/benchmarks/uwyk_table1/ours_fn50_predictive_style.py" \
    --dataset "$DATASET_ARG" --model ours_fn50p --exp_name "$EXP"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done task=$TASK"
