#!/bin/bash
#SBATCH --job-name=examext-D
#SBATCH --account=def-rgrosse
#SBATCH --output=logs_exam_ext/D_%A_%a.out
#SBATCH --error=logs_exam_ext/D_%A_%a.err
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Experiment D of the exam extension (exp_D.py), one array task per (case, setting, SCM)
# for PART=conv or per (g, target, SCM) for PART=rate. CPU only.
#
#   PART=conv sbatch --array=0-159 comprehensive_exam_extension/cluster/exp_D.sbatch
#   PART=rate sbatch --array=0-59  comprehensive_exam_extension/cluster/exp_D.sbatch
#   (task counts: python exp_D.py --part conv|rate --list-tasks)
#
# Env vars:
#   PART       conv | rate (required)
#   OUT_ROOT   output dir (default: $SCRATCH/exam_ext/exp_D)
#   DOPFN_ROOT upstream Do-PFN checkout holding artifacts/ (default: $DEPLOY_ROOT/external/dopfn);
#              its weights are linked into the extension's dopfn/artifacts if missing.
# Finished tasks are skipped (the npz exists), so a resubmit only redoes what is missing.
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
PART="${PART:?set PART=conv or PART=rate}"
OUT_ROOT="${OUT_ROOT:-$SCRATCH/exam_ext/exp_D}"
DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

ART="$EXT/dopfn/artifacts"
mkdir -p "$ART"
for f in dopfn_config.pkl dopfn_model.pkl model_submitit_0ccc_id_171b69db_epoch_-1.cpkt; do
    [ -e "$ART/$f" ] || ln -s "$DOPFN_ROOT/artifacts/$f" "$ART/$f"
    [ -e "$ART/$f" ] || { echo "FATAL: $ART/$f missing (DOPFN_ROOT=$DOPFN_ROOT)" >&2; exit 1; }
done

echo "[$(date '+%F %T')] exp_D part=$PART task=${SLURM_ARRAY_TASK_ID:-0} out=$OUT_ROOT"
python -u "$EXT/exp_D.py" --part "$PART" --task-id "${SLURM_ARRAY_TASK_ID:-0}" --out "$OUT_ROOT"
echo "[$(date '+%F %T')] done"
