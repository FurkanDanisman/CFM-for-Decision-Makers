#!/bin/bash
#SBATCH --job-name=examext-C
#SBATCH --account=def-rgrosse
#SBATCH --output=logs_exam_ext/C_%x_%j.out
#SBATCH --error=logs_exam_ext/C_%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

# Experiment C of the exam extension (exp_C.py) with 1000 data sets per level. ONE job per
# new function g (no arrays): all 18 conditions of that g (6 lambda x 3 targets) run stacked in
# the job, both case studies, r = 0..999. The in-prior baseline runs inside the G=sin job.
#
#   G=sin   sbatch --account=aip-rgrosse --job-name=examext-C-sin   .../cluster/exp_C.sbatch
#   G=sign  sbatch --account=aip-rgrosse --job-name=examext-C-sign  .../cluster/exp_C.sbatch
#   G=cubic sbatch --account=aip-rgrosse --job-name=examext-C-cubic .../cluster/exp_C.sbatch
#   G=expsq sbatch --account=aip-rgrosse --job-name=examext-C-expsq .../cluster/exp_C.sbatch   # exp(-x^2/2)
#   G=sin3  sbatch --account=aip-rgrosse --job-name=examext-C-sin3  .../cluster/exp_C.sbatch   # sin(3x)
#
# Env vars:
#   G          sin | sign | cubic | expsq | sin3 (required)
#   OUT_ROOT   output dir (default: $SCRATCH/exam_ext/exp_C_1000); results in $OUT_ROOT/$G
#   DOPFN_ROOT upstream Do-PFN checkout holding artifacts/ (default: $DEPLOY_ROOT/external/dopfn);
#              its weights are linked into the extension's dopfn/artifacts if missing.
# Resumable: exp_C.py skips (condition, case) pairs already in its CSV, so if a job hits the
# time limit, submitting the same command again continues where it stopped.
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
G="${G:?set G=sin, sign, cubic, expsq or sin3}"
BASELINE_FLAG=""
[ "$G" = "sin" ] || BASELINE_FLAG="--no-baseline"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

ART="$EXT/dopfn/artifacts"
mkdir -p "$ART"
for f in dopfn_config.pkl dopfn_model.pkl model_submitit_0ccc_id_171b69db_epoch_-1.cpkt; do
    [ -e "$ART/$f" ] || ln -s "$DOPFN_ROOT/artifacts/$f" "$ART/$f"
    [ -e "$ART/$f" ] || { echo "FATAL: $ART/$f missing (DOPFN_ROOT=$DOPFN_ROOT)" >&2; exit 1; }
done

echo "[$(date '+%F %T')] exp_C g=$G out=$OUT_ROOT/$G"
python -u "$EXT/exp_C.py" --n 200 --n-real 1000 --g-funcs "$G" $BASELINE_FLAG --out "$OUT_ROOT/$G"
echo "[$(date '+%F %T')] done"
