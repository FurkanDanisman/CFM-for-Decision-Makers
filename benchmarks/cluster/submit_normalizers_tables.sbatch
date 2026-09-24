#!/bin/bash
#SBATCH --job-name=norm-tab
#SBATCH --output=logs_norm/nt_%j.out
#SBATCH --error=logs_norm/nt_%j.err
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#
# sd(Y) normalisers + final tables, for the main results and the rho>0.99 cells.
# Reads truth arrays and existing dumps -- no inference, no GPU.
#
#   sbatch --account=def-zhijing R-PFN/benchmarks/cluster/submit_normalizers_tables.sbatch
#
# Optional: ACIC_FILES (default 10 -- causallib ships only zymu_1..10),
#           SKIP_RHO99=1, SKIP_MAIN=1

set -uo pipefail
mkdir -p logs_norm
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SCRATCH="${SCRATCH:-/scratch/$USER}"
CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
ACIC_FILES="${ACIC_FILES:-10}"

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
LN="$REPO/benchmarks/length_normalizers.py"
FT="$REPO/benchmarks/final_table.py"

if [ "${SKIP_MAIN:-0}" != 1 ]; then
    echo "==== [$(date '+%F %T')] MAIN normalisers"
    python -u "$LN" --scratch "$SCRATCH" --causalpfn "$CAUSALPFN" \
        --acic-files "$ACIC_FILES" --out "$SCRATCH/normalizers.json"
    echo "==== [$(date '+%F %T')] MAIN table"
    python -u "$FT" --perreal "$SCRATCH/perreal" --scratch "$SCRATCH" \
        --normalizers "$SCRATCH/normalizers.json" --out "$SCRATCH/final_table.md"
fi

if [ "${SKIP_RHO99:-0}" != 1 ] && [ -d "$SCRATCH/rho99view" ]; then
    # length_normalizers hardcodes the dir name cmech_data_v2, so the rho>0.99
    # cells are reached through a symlink of that name inside the view dir.
    ln -sfn "$SCRATCH/cmech_data_rho99" "$SCRATCH/rho99view/cmech_data_v2"
    echo "==== [$(date '+%F %T')] RHO99 normalisers"
    python -u "$LN" --scratch "$SCRATCH/rho99view" \
        --out "$SCRATCH/normalizers_rho99.json"
    echo "==== [$(date '+%F %T')] RHO99 table"
    python -u "$FT" --perreal "$SCRATCH/perreal_rho99" \
        --scratch "$SCRATCH/rho99view" \
        --normalizers "$SCRATCH/normalizers_rho99.json" \
        --out "$SCRATCH/final_table_rho99.md"
fi

echo "[$(date '+%F %T')] done"
