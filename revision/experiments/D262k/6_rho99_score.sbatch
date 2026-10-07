#!/bin/bash
#SBATCH --job-name=d262-score99
#SBATCH --output=logs_hist/d262_99_%j.out
#SBATCH --error=logs_hist/d262_99_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
# Stage 6 (after stage 4 has finished): hist scores and point estimates of the 262k model on rho > 0.99.
#   cd $KIT && sbatch --account=aip-rgrosse R-PFN/revision/experiments/D262k/6_rho99_score.sbatch
set -uo pipefail
mkdir -p logs_hist
KIT="${KIT:-$PWD}"; source "$KIT/R-PFN/revision/experiments/D262k/common.sh"
source "$KIT/venv/bin/activate"
export CDM_INTERVAL=hist PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
B="$SCRATCH/cmech_dumps_rho99/$NAME"
[ -d "$B/N1000" ] || { echo "MISSING $B/N1000 (run stage 4 first)"; exit 1; }
for n in 5 10 20 30 40 50; do
  o="$HIST/perreal/rho99_$NAME/raw__cmech_n${n}__-.npz"
  [ -s "$o" ] && { echo "skip $o"; continue; }; mkdir -p "$(dirname "$o")"
  python -u "$REPO/benchmarks/coverage_by_realization.py" --root "$B" --nodes "$n" --subset total --context 1000 \
    --tau-smoother none --dump-per-real "$o" > "${o%.npz}.md" 2> "${o%.npz}.err" || echo "ERROR ${o%.npz}.err"
done
python -u "$REPO/benchmarks/cmech_point_total.py" --dumps "$SCRATCH/cmech_dumps_rho99" --data "$SCRATCH/cmech_data_rho99" \
  --nodes 5 10 20 30 40 50 --csv "$SCRATCH/cm_point_rho99_d262.csv"
echo "done"
