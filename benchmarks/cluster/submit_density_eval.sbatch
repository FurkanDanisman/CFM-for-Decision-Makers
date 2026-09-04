#!/bin/bash
#SBATCH --job-name=scm-cs-density
#SBATCH --output=logs_scm_case_studies_density/dens_%A_%a.out
#SBATCH --error=logs_scm_case_studies_density/dens_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-5
#
# 6-task array — one per DoPFN synthetic case study.
#   0: Observed_Confounder                1: Observed_Mediator
#   2: Observed_Mediator_and_Confounder   3: Unobserved_Confounder
#   4: Frontdoor_Criterion                5: Backdoor_Criterion
#
# Env vars:
#   MODEL              which model tag the shards belong to
#   SHARD_ROOT         parent dir of per-case shard subdirs (e.g.
#                      $DEPLOY_ROOT/results/scm_case_studies_rho02_reruns/$MODEL)
#   OUT_ROOT           results_density_eval_rho02/$MODEL
#   DOPFN_DATA_ROOT    /path/to/prior_sampling_rho02  (parent of the case dirs)
#   DOPFN_ROOT         used by the unpickle shim for InterventionalDataset

set -euo pipefail
mkdir -p logs_scm_case_studies_density

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export MODEL="${MODEL:?MODEL required}"
export SHARD_ROOT="${SHARD_ROOT:?SHARD_ROOT required (per-model)}"
export OUT_ROOT="${OUT_ROOT:?OUT_ROOT required (per-model)}"
export DOPFN_DATA_ROOT="${DOPFN_DATA_ROOT:?DOPFN_DATA_ROOT required (parent of prior_sampling_rho02/<Case>)}"
export DOPFN_ROOT="${DOPFN_ROOT:-${DOPFN_DATA_ROOT%/data/prior_sampling_rho02}}"

case "$SLURM_ARRAY_TASK_ID" in
    0) DATASET=Observed_Confounder              ;;
    1) DATASET=Observed_Mediator                ;;
    2) DATASET=Observed_Mediator_and_Confounder ;;
    3) DATASET=Unobserved_Confounder            ;;
    4) DATASET=Frontdoor_Criterion              ;;
    5) DATASET=Backdoor_Criterion               ;;
    *) echo "invalid task id: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

SHARD_DIR="$SHARD_ROOT/$DATASET"
OUT_DIR="$OUT_ROOT"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims:$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] density-eval MODEL=$MODEL DATASET=$DATASET"
echo "  SHARD_DIR=$SHARD_DIR"
echo "  OUT_DIR=$OUT_DIR"
echo "  DOPFN_DATA_ROOT=$DOPFN_DATA_ROOT"

python -u "$REPO/benchmarks/eval_scm_case_studies/density_eval.py" \
    --model            "$MODEL" \
    --dataset          "$DATASET" \
    --shard-dir        "$SHARD_DIR" \
    --out-dir          "$OUT_DIR" \
    --dopfn-data-root  "$DOPFN_DATA_ROOT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
