#!/bin/bash
#SBATCH --job-name=d6-per-bin-l2
#SBATCH --output=logs_d6_per_bin_l2/task_%j.out
#SBATCH --error=logs_d6_per_bin_l2/task_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1

# d=6 synthetic per-bin probability L2 for Do-PFN + BB (B=1000, 2D-marg).
# Identical recipe to l2_ihdp/l2_per_bin_prob.py — only the dataset differs.

set -euo pipefail
mkdir -p logs_d6_per_bin_l2

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
N_EVAL="${N_EVAL:-200}"
D="${D:-6}"
DEGREE="${DEGREE:-3}"
SIGMA_EPS="${SIGMA_EPS:-1.0}"
RHO="${RHO:-0.0}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] d=$D N=$N_CONTEXT n_test=$N_TEST n_seeds=$N_SEEDS"
echo "  REPO=$REPO  DOPFN=$DOPFN  CKPT=$CHECKPOINT_DOPFN_BB"
echo "  malc_B=$MALC_B  K=$MALC_MAX_K"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python "$REPO/benchmarks/empirical_tests/d6_per_bin_l2.py" \
    --repo               "$REPO" \
    --dopfn              "$DOPFN" \
    --checkpoint-dopfn-bb "$CHECKPOINT_DOPFN_BB" \
    --d                  "$D" \
    --N                  "$N_CONTEXT" \
    --n-test             "$N_TEST" \
    --n-seeds            "$N_SEEDS" \
    --rho                "$RHO" \
    --sigma-eps          "$SIGMA_EPS" \
    --degree             "$DEGREE" \
    --malc-B             "$MALC_B" \
    --malc-max-K         "$MALC_MAX_K" \
    --n-eval             "$N_EVAL"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE"
