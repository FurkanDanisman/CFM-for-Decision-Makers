#!/bin/bash
#SBATCH --job-name=diag-repro-acic
#SBATCH --output=logs_diag_reproduce_acic/eval_%j.out
#SBATCH --error=logs_diag_reproduce_acic/eval_%j.err
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Diagnostic: their reproduce-branch 1D-head checkpoint on ACIC with our
# adjacency matrices. Should reproduce their table (anc=2.79, noanc=3.47).

set -euo pipefail
mkdir -p logs_diag_reproduce_acic

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export UWYK_REPRO="${UWYK_REPRO:-$DEPLOY_ROOT/external/uwyk_reproduce}"
export CAUSALPFN="${CAUSALPFN:-$UWYK_REPRO/RealCauseEval/CausalPFN}"
export CKPT="${CKPT:-$UWYK_REPRO/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0/best_model.pt}"
export CONFIG="${CONFIG:-$UWYK_REPRO/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0/best_model_config.yaml}"
export OUT="${OUT:-$DEPLOY_ROOT/results_diag_reproduce_acic}"

[ -f "$CKPT" ] || { echo "FATAL: their ckpt not found at $CKPT — git lfs pull?" >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "FATAL: their config not found at $CONFIG" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] diagnostic: their ckpt on ACIC via our adjacency"
python -u "$REPO/benchmarks/eval_graph2d/diag_reproduce_ckpt_acic.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
