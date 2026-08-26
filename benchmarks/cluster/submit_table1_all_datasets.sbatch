#!/bin/bash
#SBATCH --job-name=t1-all-4way
#SBATCH --output=logs_table1_all/task_%A_%a.out
#SBATCH --error=logs_table1_all/task_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-15%8
#
# 4-way ACIC/IHDP/CPS/PSID(unbal)/PSID(bal) Table-1 reproduction.
# 16-task array: 4 datasets x 4 methods each.
#
# NOTE: array is 16 tasks but ACIC is already done — skip it here.
# Layout:
#   0-3   : IHDP        (predictive, noanc, anc, fn=50)
#   4-7   : CPS         (predictive [clustered variant], noanc, anc, fn=50)
#   8-11  : PSID unbal  (predictive [clustered], noanc, anc, fn=50)
#   12-15 : PSID bal    (predictive [balanced], noanc [balanced], anc [balanced], fn=50 [balanced])
#
# Concurrency capped at %8 so we don't over-request cluster GPUs.

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
export CAUSALPFN_ROOT="$UWYK_REPRO/RealCauseEval/CausalPFN"
export OURS_CKPT="${OURS_CKPT:-$REPO/checkpoints/step_50000_final.pt}"
export TABLE1_OUT_ROOT="${TABLE1_OUT_ROOT:-$DEPLOY_ROOT/results_table1_all}"

[ -d "$UWYK_REPRO/RealCauseEval/CausalPFN/benchmarks" ] || { echo "FATAL: pinned CausalPFN missing at $CAUSALPFN_ROOT" >&2; exit 1; }
[ -f "$OURS_CKPT" ]                                     || { echo "FATAL: fn=50 ckpt missing at $OURS_CKPT" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
# Shim + all-paths so both UWYK's scripts (from RealCauseEval/) and our
# fn=50 script (from R-PFN/) find their imports.
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims:$UWYK_REPRO:$UWYK_REPRO/RealCauseEval:$CAUSALPFN_ROOT:$CAUSALPFN_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

JOB="${SLURM_ARRAY_JOB_ID}"
TASK="${SLURM_ARRAY_TASK_ID}"
DS_IDX=$(( TASK / 4 ))
METHOD_IDX=$(( TASK % 4 ))

case "$DS_IDX" in
    0) DATASET_LOG="IHDP"       ;;
    1) DATASET_LOG="CPS"        ;;
    2) DATASET_LOG="PSID_unbal" ;;
    3) DATASET_LOG="PSID_bal"   ;;
esac
case "$METHOD_IDX" in
    0) METHOD="predictive" ;;
    1) METHOD="noanc"      ;;
    2) METHOD="anc"        ;;
    3) METHOD="fn50"       ;;
esac
EXP="table1_all_${DATASET_LOG}_${METHOD}_${JOB}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] task=$TASK  ds=$DATASET_LOG  method=$METHOD  exp=$EXP"

# All UWYK Predictive/DoFM script invocations chdir into RealCauseEval so
# their `from run_baselines...` relative imports resolve and their output
# lands under RealCauseEval/results/<exp>/.
cd "$UWYK_REPRO/RealCauseEval"

run_uwyk_pred() {
    # $1 = which predictive script; $2 = dataset arg
    python -u "run_baselines/$1" \
        --dataset "$2" --model slearner --exp_name "$EXP"
}
run_dofm_noclust() {
    # $1 = dataset arg; $2 = graph_mode
    python -u run_baselines/dofm_no_clustering.py \
        --dataset "$1" --model dofm_noclust \
        --exp_name "$EXP" --graph_mode "$2"
}
run_dofm_psid_bal() {
    # $1 = graph_mode
    python -u run_baselines/dofm_psid_balanced.py \
        --dataset PSID --model dofm_psid_bal \
        --exp_name "$EXP" --graph_mode "$1"
}
run_ours_fn50() {
    # $1 = dataset arg; $2 = "" or "--psid-balanced"
    python -u "$REPO/benchmarks/uwyk_table1/ours_fn50_no_clustering.py" \
        --dataset "$1" --model ours_fn50 --exp_name "$EXP" $2
}

case "$DS_IDX-$METHOD_IDX" in
    # ── IHDP ─────────────────────────────
    0-0) run_uwyk_pred    predmodel_Slearner_full_context.py    IHDP ;;
    0-1) run_dofm_noclust IHDP all_unknown ;;
    0-2) run_dofm_noclust IHDP full_graph ;;
    0-3) run_ours_fn50    IHDP "" ;;
    # ── CPS ──────────────────────────────
    1-0) run_uwyk_pred    predmodel_Slearner.py                 CPS  ;;
    1-1) run_dofm_noclust CPS all_unknown ;;
    1-2) run_dofm_noclust CPS full_graph ;;
    1-3) run_ours_fn50    CPS "" ;;
    # ── PSID unbal ───────────────────────
    2-0) run_uwyk_pred    predmodel_Slearner.py                 PSID ;;
    2-1) run_dofm_noclust PSID all_unknown ;;
    2-2) run_dofm_noclust PSID full_graph ;;
    2-3) run_ours_fn50    PSID "" ;;
    # ── PSID bal ─────────────────────────
    3-0) run_uwyk_pred    predmodel_Slearner_psid_balanced.py   PSID ;;
    3-1) run_dofm_psid_bal all_unknown ;;
    3-2) run_dofm_psid_bal full_graph  ;;
    3-3) run_ours_fn50    PSID --psid-balanced ;;
    *) echo "invalid ($DS_IDX, $METHOD_IDX)" >&2; exit 2 ;;
esac

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done task=$TASK"
