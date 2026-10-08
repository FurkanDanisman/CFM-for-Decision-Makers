#!/bin/bash
#SBATCH --job-name=examext-app
#SBATCH --account=def-rgrosse
#SBATCH --output=logs_exam_ext/APP_%x_%j.out
#SBATCH --error=logs_exam_ext/APP_%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

# Appendix: the other four case studies with Do-PFN, 100 data sets per level, every experiment
# of the main study: B (shape paths, scaled and unscaled, noise on all nodes and outcome only),
# C (5 functions x 6 lambda, outcome equation only) and F (treated fraction, categorical
# covariates, irrelevant covariates, categorical outcome). ONE job per case study (no arrays);
# all conditions stacked. The in-prior condition runs once, inside B scaled.
#
#   CASE=Observed_Mediator                sbatch --account=aip-rgrosse --job-name=examext-app-OM  .../cluster/exp_appendix.sbatch
#   CASE=Observed_Mediator_and_Confounder sbatch --account=aip-rgrosse --job-name=examext-app-OMC .../cluster/exp_appendix.sbatch
#   CASE=Backdoor_Criterion               sbatch --account=aip-rgrosse --job-name=examext-app-BD  .../cluster/exp_appendix.sbatch
#   CASE=Frontdoor_Criterion              sbatch --account=aip-rgrosse --job-name=examext-app-FD  .../cluster/exp_appendix.sbatch
#
# Env vars:
#   CASE       case study (required)
#   OUT_ROOT   output dir (default: $SCRATCH/exam_ext/appendix_100); results in $OUT_ROOT/$CASE/<part>
#   DOPFN_ROOT upstream Do-PFN checkout holding artifacts/ (default: $DEPLOY_ROOT/external/dopfn).
# Resumable: every script skips (condition, case) pairs already in its CSV, so if a job hits the
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
OUT_ROOT="${OUT_ROOT:-$SCRATCH/exam_ext/appendix_100}"
DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
CASE="${CASE:?set CASE=Observed_Mediator, Observed_Mediator_and_Confounder, Backdoor_Criterion or Frontdoor_Criterion}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

ART="$EXT/dopfn/artifacts"
mkdir -p "$ART"
for f in dopfn_config.pkl dopfn_model.pkl model_submitit_0ccc_id_171b69db_epoch_-1.cpkt; do
    [ -e "$ART/$f" ] || ln -s "$DOPFN_ROOT/artifacts/$f" "$ART/$f"
    [ -e "$ART/$f" ] || { echo "FATAL: $ART/$f missing (DOPFN_ROOT=$DOPFN_ROOT)" >&2; exit 1; }
done

O="$OUT_ROOT/$CASE"
COMMON="--n 200 --n-real 100 --cases $CASE"
echo "[$(date '+%F %T')] appendix case=$CASE out=$O"
python -u "$EXT/exp_B_shapes.py" $COMMON --scale scaled --out "$O/B_scaled"
python -u "$EXT/exp_B_shapes.py" $COMMON --scale unscaled --no-baseline --out "$O/B_unscaled"
python -u "$EXT/exp_C.py" $COMMON --g-funcs sin sign cubic expsq sin3 --targets outcome --no-baseline --out "$O/C"
for D in treat covk irrel outk; do
    python -u "$EXT/exp_F.py" $COMMON --dial "$D" --no-baseline --out "$O/F_$D"
done
echo "[$(date '+%F %T')] done"
