#!/bin/bash
#SBATCH --job-name=d6-plot-ext
#SBATCH --output=logs_d6_plot_extremes/task_%j.out
#SBATCH --error=logs_d6_plot_extremes/task_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1

set -euo pipefail
mkdir -p logs_d6_plot_extremes

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
[ ! -d "$REPO/benchmarks" ] && REPO="$DEPLOY_ROOT"
DOPFN="${DOPFN:-$DEPLOY_ROOT/external/dopfn}"
CHECKPOINT_DOPFN_BB="${CHECKPOINT_DOPFN_BB:-$DEPLOY_ROOT/checkpoints_dopfn_backbone_realj10/step_150000.pt}"
N_SEEDS="${N_SEEDS:-15}"
N_TEST="${N_TEST:-25}"
N_CONTEXT="${N_CONTEXT:-200}"
MALC_B="${MALC_B:-1000}"
MALC_MAX_K="${MALC_MAX_K:-1}"
TAU_BASE="${TAU_BASE:-5.0}"
TAU_SCALE="${TAU_SCALE:-0.5}"
Y_SCALE="${Y_SCALE:-0.5}"
OUT="${OUT:-$DEPLOY_ROOT/d6_extremes}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] d=6 extremes  n_seeds=$N_SEEDS n_test=$N_TEST malc_B=$MALC_B"
echo "  OUT=$OUT"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python "$REPO/benchmarks/empirical_tests/d6_plot_extremes.py" \
    --repo               "$REPO" \
    --dopfn              "$DOPFN" \
    --checkpoint-dopfn-bb "$CHECKPOINT_DOPFN_BB" \
    --N                  "$N_CONTEXT" \
    --n-test             "$N_TEST" \
    --n-seeds            "$N_SEEDS" \
    --malc-B             "$MALC_B" \
    --malc-max-K         "$MALC_MAX_K" \
    --tau-base           "$TAU_BASE" \
    --tau-scale          "$TAU_SCALE" \
    --y-scale            "$Y_SCALE" \
    --out                "$OUT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
