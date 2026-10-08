#!/bin/bash
#SBATCH --job-name=examext-C
#SBATCH --account=def-rgrosse
#SBATCH --output=logs_exam_ext/C_%A_%a.out
#SBATCH --error=logs_exam_ext/C_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Experiment C of the exam extension (exp_C.py) with 1000 data sets per level: one array task
# per condition (0 = in-prior, 1..54 = g x lambda x target), both case studies, r = 0..999.
#
#   sbatch --array=0-54 comprehensive_exam_extension/cluster/exp_C.sbatch
#   (condition count: python exp_C.py --list-conditions)
#
# Env vars:
#   OUT_ROOT   output dir (default: $SCRATCH/exam_ext/exp_C_1000); one sub-folder per task
#   DOPFN_ROOT upstream Do-PFN checkout holding artifacts/ (default: $DEPLOY_ROOT/external/dopfn);
#              its weights are linked into the extension's dopfn/artifacts if missing.
# A finished task is skipped on resubmit (exp_C.py skips conditions already in its CSV).
set -euo pipefail
mkdir -p logs_exam_ext

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found under $DEPLOY_ROOT" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
EXT="$REPO/comprehensive_exam_extension"
OUT_ROOT="${OUT_ROOT:-$SCRATCH/exam_ext/exp_C_1000}"
DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

ART="$EXT/dopfn/artifacts"
mkdir -p "$ART"
for f in dopfn_config.pkl dopfn_model.pkl model_submitit_0ccc_id_171b69db_epoch_-1.cpkt; do
    [ -e "$ART/$f" ] || ln -s "$DOPFN_ROOT/artifacts/$f" "$ART/$f"
    [ -e "$ART/$f" ] || { echo "FATAL: $ART/$f missing (DOPFN_ROOT=$DOPFN_ROOT)" >&2; exit 1; }
done

echo "[$(date '+%F %T')] exp_C condition=$TASK out=$OUT_ROOT/task_$TASK"
python -u "$EXT/exp_C.py" --n 200 --n-real 1000 --condition-index "$TASK" --out "$OUT_ROOT/task_$TASK"
echo "[$(date '+%F %T')] done"
