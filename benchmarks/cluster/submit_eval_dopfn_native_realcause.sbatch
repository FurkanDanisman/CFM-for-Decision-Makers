#!/bin/bash
#SBATCH --job-name=dopfn-native-rc
#SBATCH --output=logs_dopfn_native_rc/eval_%A_%a.out
#SBATCH --error=logs_dopfn_native_rc/eval_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4

# 5-task array — one per RealCause dataset.
#   0: IHDP  1: ACIC  2: CPS  3: PSID  4: PSID_bal
#
# Env vars:
#   OUT_ROOT     per-dataset outputs land at OUT_ROOT/<DATASET>
#   DENSITY_DUMP 1 to save density npz keys (default 0)
#   DOPFN_ROOT   path to dopfn_upstream repo root (default: DEPLOY_ROOT/external/dopfn)

set -euo pipefail
mkdir -p logs_dopfn_native_rc

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
export OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
export DENSITY_DUMP="${DENSITY_DUMP:-0}"

case "$SLURM_ARRAY_TASK_ID" in
    0) DATASET=IHDP     ;;
    1) DATASET=ACIC     ;;
    2) DATASET=CPS      ;;
    3) DATASET=PSID     ;;
    4) DATASET=PSID_bal ;;
    *) echo "invalid task id: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
export DATASET
export OUT="$OUT_ROOT/$DATASET"
mkdir -p "$OUT"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] native DoPFN  DATASET=$DATASET  OUT=$OUT  DENSITY_DUMP=$DENSITY_DUMP"
python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DATASET"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
