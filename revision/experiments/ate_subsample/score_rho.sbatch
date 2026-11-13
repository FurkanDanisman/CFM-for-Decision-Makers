#!/bin/bash
#SBATCH --job-name=ate-score-rho
#SBATCH --output=logs_ate/%x_%j.out
#SBATCH --error=logs_ate/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
# Score one benchmark for every model of models_rho.sh: coverage and interval score of the OT ATE density, as is
# and with each variant's in-prior s, per cell (bench_rho.py). Writes $SCRATCH/ate_rho/<bench>.csv. CPU only.
#   ATE_BENCH=rc|cs|cm sbatch --account=aip-rgrosse R-PFN/revision/experiments/ate_subsample/score_rho.sbatch
set -uo pipefail
mkdir -p logs_ate
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; HERE="$REPO/revision/experiments/ate_subsample"
source "$KIT/venv/bin/activate"; source "$HERE/models_rho.sh"
export PYTHONUNBUFFERED=1 ATOM_GRID_MAX=4096 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
B="${ATE_BENCH:?ATE_BENCH = rc | cs | cm}"
D="$SCRATCH/ate_${B}_dumps"; OUTD="$SCRATCH/ate_rho"; mkdir -p "$OUTD"
CSV="$OUTD/$B.csv"; rm -f "$CSV" "$CSV".*.npz
OPT=(); [ "$B" = cm ] && OPT=(--cmech-data "$SCRATCH/cmech_data_v2")
for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r name tags coup <<<"$spec"
  [ -d "$D/$name" ] || { echo "$name: no dumps in $D"; continue; }
  echo "===== [$(date '+%F %T')] $B $name"
  python "$HERE/bench_rho.py" --root "$D/$name" --bench "$B" --model "$name" --tags "$tags" --couplings "$coup" \
    --inprior "$SCRATCH/ate_bary" --out "$CSV" "${OPT[@]}" | grep -v '^  '
done
echo "[$(date '+%F %T')] wrote $CSV ($(($(wc -l <"$CSV") - 1)) rows)"
