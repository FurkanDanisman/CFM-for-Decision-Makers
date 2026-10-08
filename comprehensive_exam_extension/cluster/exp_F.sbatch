#!/bin/bash
#SBATCH --job-name=examext-F
#SBATCH --account=def-rgrosse
#SBATCH --output=logs_exam_ext/F_%x_%j.out
#SBATCH --error=logs_exam_ext/F_%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

# Experiment F of the exam extension (exp_F.py, real-data properties) with 1000 data sets per
# level. ONE job per dial (no arrays): all levels of that dial run stacked in the job, both case
# studies, r = 0..999. The in-prior baseline runs inside the D=treat job.
#
#   D=treat sbatch --account=aip-rgrosse --job-name=examext-F-treat .../cluster/exp_F.sbatch   # treated fraction
#   D=covk  sbatch --account=aip-rgrosse --job-name=examext-F-covk  .../cluster/exp_F.sbatch   # categorical confounder
#   D=irrel sbatch --account=aip-rgrosse --job-name=examext-F-irrel .../cluster/exp_F.sbatch   # irrelevant covariates
#   D=outk  sbatch --account=aip-rgrosse --job-name=examext-F-outk  .../cluster/exp_F.sbatch   # categorical outcome
#
# Env vars:
#   D          treat | covk | irrel | outk (required)
#   OUT_ROOT   output dir (default: $SCRATCH/exam_ext/exp_F_1000); results in $OUT_ROOT/$D
#   DOPFN_ROOT upstream Do-PFN checkout holding artifacts/ (default: $DEPLOY_ROOT/external/dopfn);
#              its weights are linked into the extension's dopfn/artifacts if missing.
# Resumable: exp_F.py skips (condition, case) pairs already in its CSV, so if a job hits the
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
OUT_ROOT="${OUT_ROOT:-$SCRATCH/exam_ext/exp_F_1000}"
DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
D="${D:?set D=treat, covk, irrel or outk}"
BASELINE_FLAG=""
[ "$D" = "treat" ] || BASELINE_FLAG="--no-baseline"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

ART="$EXT/dopfn/artifacts"
mkdir -p "$ART"
for f in dopfn_config.pkl dopfn_model.pkl model_submitit_0ccc_id_171b69db_epoch_-1.cpkt; do
    [ -e "$ART/$f" ] || ln -s "$DOPFN_ROOT/artifacts/$f" "$ART/$f"
    [ -e "$ART/$f" ] || { echo "FATAL: $ART/$f missing (DOPFN_ROOT=$DOPFN_ROOT)" >&2; exit 1; }
done

echo "[$(date '+%F %T')] exp_F dial=$D out=$OUT_ROOT/$D"
python -u "$EXT/exp_F.py" --n 200 --n-real 1000 --dial "$D" $BASELINE_FLAG --out "$OUT_ROOT/$D"
echo "[$(date '+%F %T')] done"
