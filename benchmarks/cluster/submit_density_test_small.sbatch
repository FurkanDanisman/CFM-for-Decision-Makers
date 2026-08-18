#!/bin/bash
#SBATCH --job-name=density-small
#SBATCH --account=aip-rgrosse
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/density_small_%j.out
#SBATCH --error=logs/density_small_%j.err

# Controlled density-L2 test: polynomial SCM at fixed d, N, ρ; known
# analytic Gaussian truth. Compares Do-PFN vs DoPFN-bb only. Env vars:
#   D          (default 3)   polynomial SCM x_dim
#   N          (default 200) context size per SCM
#   N_SEEDS    (default 15)  number of SCM seeds
#   RHO        (default 0.0) true DGP ρ between Y0/Y1 residuals
#   N_TEST     (default 25)  test queries per SCM

set -euo pipefail
: "${DEPLOY_ROOT:?DEPLOY_ROOT must be set}"
: "${D:=3}"
: "${N:=200}"
: "${N_SEEDS:=15}"
: "${RHO:=0.0}"
: "${N_TEST:=25}"

cd "$DEPLOY_ROOT/R-PFN"
if [[ -f .venv/bin/activate ]]; then . .venv/bin/activate
elif [[ -f venv/bin/activate ]]; then . venv/bin/activate; fi

echo "[$(date)] d=$D N=$N n_seeds=$N_SEEDS rho=$RHO"
python benchmarks/empirical_tests/density_test_small.py \
    --repo "$DEPLOY_ROOT/R-PFN" \
    --dopfn "$DEPLOY_ROOT/external/dopfn" \
    --checkpoint-dopfn-bb "$DEPLOY_ROOT/checkpoints_dopfn_backbone_realj10/step_150000.pt" \
    --d "$D" --N "$N" --n-seeds "$N_SEEDS" --rho "$RHO" --n-test "$N_TEST"

echo "[$(date)] DONE"
