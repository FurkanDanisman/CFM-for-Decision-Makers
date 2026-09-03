#!/bin/bash
#SBATCH --job-name=cpfn1d-rc
#SBATCH --output=logs_cpfn1d_realcause/eval_%A_%a.out
#SBATCH --error=logs_cpfn1d_realcause/eval_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4

# 5-task array — one per RealCause dataset.
#   0: IHDP  1: ACIC  2: CPS  3: PSID  4: PSID_bal
# Pure ICL (no faiss, no CATEEstimator).

set -euo pipefail
mkdir -p logs_cpfn1d_realcause

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
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
export CKPT="${CKPT:?CKPT required — path to 1D CausalPFN ckpt}"
export OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
export STD_MODE="${STD_MODE:-per_arm}"   # per_arm | pooled | log | log_per_arm | log_winsor

[ -f "$CKPT" ] || { echo "FATAL: ckpt not found: $CKPT" >&2; exit 1; }

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

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] cpfn 1D RealCause  dataset=$DATASET  OUT=$OUT  STD_MODE=$STD_MODE"
python -u "$REPO/benchmarks/eval_causalpfn2d/eval_causalpfn_v0_realcause.py" --dataset "$DATASET"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
