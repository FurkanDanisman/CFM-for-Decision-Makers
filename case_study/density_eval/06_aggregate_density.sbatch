#!/bin/bash
#SBATCH --job-name=dens-report
#SBATCH --output=logs_density/report_%j.out
#SBATCH --error=logs_density/report_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
# CPU ONLY -- no --gres. Aggregation re-scores every realization (barycenter +
# metrics); there is no model forward pass, so a GPU would sit idle and the job
# would just queue longer.
#
#   export DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit
#   sbatch case_study/density_eval/06_aggregate_density.sbatch
set -euo pipefail
mkdir -p logs_density

DEPLOY_ROOT="${DEPLOY_ROOT:?DEPLOY_ROOT required}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
RES="${RES:-$DEPLOY_ROOT/results_case_study/dvar}"
DATA="${DATA:-$DEPLOY_ROOT/case_study_data/d_variation}"
OUT="${OUT:-$RES/combined_cen3_density.csv}"
SHIFTS="${SHIFTS:-shift0 shift+2 shift-2}"
LABEL="${LABEL:-cen3}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-8}}"

source "${VENV:-$DEPLOY_ROOT/venv}/bin/activate"
export PYTHONUNBUFFERED=1
# one BLAS thread per worker: the parallelism is across cells, and letting each
# worker also spawn 32 BLAS threads oversubscribes the node badly.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[report] root=$RES  shifts=[$SHIFTS] label=$LABEL  jobs=$JOBS"
python -u "$REPO/case_study/density_eval/dsweep_density_report.py" \
    --root "$RES" --data-root "$DATA" \
    --combine-shifts $SHIFTS --combine-label "$LABEL" \
    --jobs "$JOBS" ${ONLY_D:+--only-d $ONLY_D} \
    --out "$OUT"
echo "[report] done -> $OUT"
