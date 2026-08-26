#!/bin/bash
#SBATCH --job-name=t1-cps-fn50s
#SBATCH --output=logs_table1_all/cps_fn50_subsample_%j.out
#SBATCH --error=logs_table1_all/cps_fn50_subsample_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#
# CPS-only fn=50 with bagged random-subsample inference. Replacement for
# the KMeans-clustered variant, which didn't help (12727 vs single-pass
# 12688) because KMeans creates narrow-X clusters whose per-cluster CATE
# doesn't generalize to test queries outside that X domain.
#
# --subsample keeps the marginal X distribution intact in every 1000-row
# subset, so each per-subset CATE is unbiased. Averaging 20 such subsets
# reduces variance without introducing bias, and every forward stays
# inside our model's trained ~1000-row regime.
#
# --n-repeats 20  ->  20 forwards per realization. At ~0.5 s/forward on
# H100 and 100 realizations, total wall clock ≈ 15-20 min.

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
# Distinct method tag "fn50s" so the aggregator can pick this up as a
# separate row alongside the KMeans-clustered "fn50c" and single-pass
# "fn50" results.
EXP="table1_all_CPS_fn50s_${JOB}"

N_REPEATS="${N_REPEATS:-20}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching  ds=CPS  variant=fn50 subsample  " \
     "n_repeats=$N_REPEATS  max_n_train=1000  exp=$EXP"
python -u "$REPO/benchmarks/uwyk_table1/ours_fn50_no_clustering.py" \
    --dataset CPS --model ours_fn50s --exp_name "$EXP" \
    --subsample --n-repeats "$N_REPEATS" --max-n-train 1000
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
