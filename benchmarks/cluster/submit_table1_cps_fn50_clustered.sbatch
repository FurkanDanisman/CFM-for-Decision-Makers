#!/bin/bash
#SBATCH --job-name=t1-cps-fn50c
#SBATCH --output=logs_table1_all/cps_fn50_clustered_%j.out
#SBATCH --error=logs_table1_all/cps_fn50_clustered_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#
# CPS-only fn=50 with UWYK-style cluster-and-average inference.
# Should close the ~7% CPS gap vs UWYK Predictive by keeping every
# per-cluster forward inside our model's ~1000-row training regime
# (14559 rows -> 15 clusters -> 15 forwards -> mean of 15 CATE
# predictions per test query).

set -euo pipefail
mkdir -p logs_table1_all

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
export TABLE1_OUT_ROOT="${TABLE1_OUT_ROOT:-$DEPLOY_ROOT/results_table1_all}"

[ -f "$OURS_CKPT" ] || { echo "FATAL: fn=50 ckpt missing at $OURS_CKPT" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

JOB="${SLURM_JOB_ID}"
# Use a distinct method tag "fn50c" so the aggregator can pick this up as
# a separate row alongside the plain "fn50" (null-t, single-pass) result.
EXP="table1_all_CPS_fn50c_${JOB}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching  ds=CPS  variant=fn50 clustered  exp=$EXP"
python -u "$REPO/benchmarks/uwyk_table1/ours_fn50_no_clustering.py" \
    --dataset CPS --model ours_fn50c --exp_name "$EXP" \
    --cluster --max-n-train 1000
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
