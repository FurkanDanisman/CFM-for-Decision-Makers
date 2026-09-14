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
# tail-quadrature resolution for the tauC densities; 1024 is 5x faster
# than the eval default of 4096 for 8e-4 of mass, which the scorer
# renormalises away. Raise if you need the tails resolved harder.
export TAUC_N_Y0="${TAUC_N_Y0:-1024}"
# queries scored per realization. The per-query density rebuild IS the
# cost for the tauC models; coverage averages over shifts x realizations
# x queries, so cutting queries is the cheapest precision trade there is.
export MAX_QUERIES="${MAX_QUERIES:-20}"
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[report] root=$RES  shifts=[$SHIFTS] label=$LABEL  jobs=$JOBS  TAUC_N_Y0=$TAUC_N_Y0  MAX_QUERIES=$MAX_QUERIES"
[ -n "${ONLY_N:-}" ] && echo "[report] contexts restricted to: $ONLY_N"
[ -n "${ONLY_D:-}" ] && echo "[report] d restricted to: $ONLY_D"
[ -n "${MODELS:-}" ] && echo "[report] models: $MODELS"
[ -n "${CASES:-}" ]  && echo "[report] cases: $CASES"
: # keep set -e happy when the greps above are false
python -u "$REPO/case_study/density_eval/dsweep_density_report.py" \
    --root "$RES" --data-root "$DATA" \
    --combine-shifts $SHIFTS --combine-label "$LABEL" \
    --jobs "$JOBS" ${ONLY_D:+--only-d $ONLY_D} ${ONLY_N:+--only-n $ONLY_N} \
    ${MODELS:+--models $MODELS} ${CASES:+--cases $CASES} \
    --out "$OUT"
echo "[report] done -> $OUT"
