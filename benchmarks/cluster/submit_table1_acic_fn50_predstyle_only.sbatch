#!/bin/bash
#SBATCH --job-name=t1-acic-fn50p
#SBATCH --output=logs_table1_acic_repro/fn50_predstyle_%j.out
#SBATCH --error=logs_table1_acic_repro/fn50_predstyle_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# fn=50 on ACIC using the predictive-mirror inference protocol
# (two-pass with target-encoded T at query, OOD for our checkpoint).
# Everything else identical to the null-t variant.

set -euo pipefail
mkdir -p logs_table1_acic_repro

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    if   [ -f "$(dirname "$DEPLOY_ROOT")/venv/bin/activate" ]; then
        DEPLOY_ROOT="$(dirname "$DEPLOY_ROOT")"
    elif [ -f "$(dirname "$(dirname "$DEPLOY_ROOT")")/venv/bin/activate" ]; then
        DEPLOY_ROOT="$(dirname "$(dirname "$DEPLOY_ROOT")")"
    fi
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export UWYK_REPRO="${UWYK_REPRO:-$DEPLOY_ROOT/external/uwyk_reproduce}"
export CAUSALPFN_ROOT="${CAUSALPFN_ROOT:-$UWYK_REPRO/RealCauseEval/CausalPFN}"
export OURS_CKPT="${OURS_CKPT:-$REPO/checkpoints/step_50000_final.pt}"
export TABLE1_OUT_ROOT="${TABLE1_OUT_ROOT:-$DEPLOY_ROOT/results_table1_ours_fn50}"

[ -f "$OURS_CKPT" ] || { echo "FATAL: fn=50 ckpt not at $OURS_CKPT" >&2; exit 1; }
[ -d "$CAUSALPFN_ROOT/benchmarks" ] || { echo "FATAL: CausalPFN benchmarks not at $CAUSALPFN_ROOT" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] fn=50 (pred-mirror) on ACIC"
python -u "$REPO/benchmarks/uwyk_table1/ours_fn50_predictive_style.py" \
    --dataset ACIC --model ours_fn50_predstyle \
    --exp_name "table1_repro_ours_fn50_predstyle_${SLURM_JOB_ID}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
