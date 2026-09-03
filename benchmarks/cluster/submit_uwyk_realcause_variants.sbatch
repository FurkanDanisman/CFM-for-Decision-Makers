#!/bin/bash
#SBATCH --job-name=uwyk-rc-variant
#SBATCH --output=logs_uwyk_realcause_variants/eval_%A_%a.out
#SBATCH --error=logs_uwyk_realcause_variants/eval_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4

# 5-task array — one per RealCause dataset. ANC_VARIANT passed via env.
#   0: IHDP  1: ACIC  2: CPS  3: PSID  4: PSID_bal
set -euo pipefail
mkdir -p logs_uwyk_realcause_variants

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
export UWYK_SRC="${UWYK_SRC:-$DEPLOY_ROOT/external/uwyk_reproduce/src}"
export UWYK_CKPT="${UWYK_CKPT:?UWYK_CKPT required}"
export UWYK_CONFIG="${UWYK_CONFIG:?UWYK_CONFIG required}"
export ANC_VARIANT="${ANC_VARIANT:?ANC_VARIANT required — v3a|v3b|v3c|v3d|v6a|noanc}"
export OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"

case "$SLURM_ARRAY_TASK_ID" in
    0) DATASET=IHDP     ;;
    1) DATASET=ACIC     ;;
    2) DATASET=CPS      ;;
    3) DATASET=PSID     ;;
    4) DATASET=PSID_bal ;;
    *) exit 2 ;;
esac
export DATASET
export OUT="$OUT_ROOT/$DATASET"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%F %T')] UWYK RealCause  DATASET=$DATASET  ANC_VARIANT=$ANC_VARIANT"
echo "  UWYK_CKPT=$UWYK_CKPT"
echo "  OUT=$OUT"
python -u "$REPO/benchmarks/eval_scm_case_studies/eval_uwyk_realcause.py" --dataset "$DATASET"
echo "[$(date '+%F %T')] done"
