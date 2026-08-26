#!/bin/bash
#SBATCH --job-name=t1-acic-repro
#SBATCH --output=logs_table1_acic_repro/task_%A_%a.out
#SBATCH --error=logs_table1_acic_repro/task_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-3
#
# ACIC Table-1 reproduction using UWYK's OWN scripts from their
# reproduce-realcause-results branch, with THEIR pinned CausalPFN.
# Zero patching, zero sed injection. Only the fn=50 task uses our own
# code (since it's our checkpoint).
#
# Prerequisites (one-time, from login node):
#
#   cd /scratch/furkanbd/rpfn_bench_kit
#
#   # UWYK repo on the exact reproduction branch, with LFS ckpts
#   [ -d external/uwyk_reproduce ] || {
#       git clone -b reproduce-realcause-results \
#           https://github.com/ArikReuter/Graphs4CausalFoundationModels.git \
#           external/uwyk_reproduce
#       git -C external/uwyk_reproduce lfs pull
#   }
#
#   # CausalPFN pinned to the commit their guide names (a3de754)
#   [ -d external/uwyk_reproduce/RealCauseEval/CausalPFN ] || {
#       git clone https://github.com/vdblm/CausalPFN.git \
#           external/uwyk_reproduce/RealCauseEval/CausalPFN
#       git -C external/uwyk_reproduce/RealCauseEval/CausalPFN checkout a3de754
#   }
#
#   sbatch R-PFN/benchmarks/cluster/submit_table1_acic_repro.sbatch
#
# Row structure (matches paper Table 1):
#   task 0 : UWYK Predictive  -- predmodel_Slearner_full_context.py    (paper 3.14)
#   task 1 : UWYK No-anc      -- dofm_no_clustering.py all_unknown     (paper 3.47)
#   task 2 : UWYK Anc         -- dofm_no_clustering.py full_graph      (paper 2.79)
#   task 3 : ours fn=50       -- ours_fn50_predictive_style.py         (no paper ref)

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

# Sanity-check the two prerequisites BEFORE launching any expensive work.
[ -f "$UWYK_REPRO/RealCauseEval/run_baselines/dofm_no_clustering.py" ] || {
    echo "FATAL: reproduce-branch UWYK not at $UWYK_REPRO." >&2
    echo "  See top of this sbatch for the two-line clone." >&2
    exit 1
}
[ -d "$UWYK_REPRO/RealCauseEval/CausalPFN" ] || {
    echo "FATAL: pinned CausalPFN not at $UWYK_REPRO/RealCauseEval/CausalPFN." >&2
    echo "  See top of this sbatch." >&2
    exit 1
}
PRED_CKPT="$UWYK_REPRO/experiments/checkpoints/predictive_slearner/best_model.pt"
DOFM_CKPT="$UWYK_REPRO/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0/best_model.pt"
[ -f "$PRED_CKPT" ] || { echo "FATAL: predictive ckpt missing at $PRED_CKPT (git lfs pull?)" >&2; exit 1; }
[ -f "$DOFM_CKPT" ] || { echo "FATAL: DoFM ckpt missing at $DOFM_CKPT (git lfs pull?)" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

# Everything on PYTHONPATH so their scripts' `from src...` and `from CausalPFN.benchmarks...`
# resolve without any patching. Shims dir first so causalpfn/__init__'s
# faiss import doesn't crash the interpreter at startup.
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims:$UWYK_REPRO:$UWYK_REPRO/RealCauseEval:$UWYK_REPRO/RealCauseEval/CausalPFN:$UWYK_REPRO/RealCauseEval/CausalPFN/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] task=$SLURM_ARRAY_TASK_ID"
echo "  UWYK_REPRO = $UWYK_REPRO"
echo "  PYTHONPATH = $PYTHONPATH"

# Their scripts write to <cwd>/results/... -- so cd into RealCauseEval and let
# outputs go to $UWYK_REPRO/RealCauseEval/results/<exp_name>/*
cd "$UWYK_REPRO/RealCauseEval"

case "$SLURM_ARRAY_TASK_ID" in
    0)
        echo "== UWYK Predictive on ACIC =="
        python -u run_baselines/predmodel_Slearner_full_context.py \
            --dataset ACIC --model slearner_full \
            --exp_name "table1_repro_predictive_${SLURM_ARRAY_JOB_ID}"
        ;;
    1)
        echo "== UWYK No-anc on ACIC =="
        python -u run_baselines/dofm_no_clustering.py \
            --dataset ACIC --model dofm_noclust \
            --exp_name "table1_repro_noanc_${SLURM_ARRAY_JOB_ID}" \
            --graph_mode all_unknown
        ;;
    2)
        echo "== UWYK Anc on ACIC =="
        python -u run_baselines/dofm_no_clustering.py \
            --dataset ACIC --model dofm_noclust \
            --exp_name "table1_repro_anc_${SLURM_ARRAY_JOB_ID}" \
            --graph_mode full_graph
        ;;
    3)
        echo "== ours fn=50 on ACIC (null_t_intv, single pass, in-distribution) =="
        OURS_CKPT="${OURS_CKPT:-$REPO/checkpoints/step_50000_final.pt}"
        [ -f "$OURS_CKPT" ] || { echo "FATAL: fn=50 ckpt not at $OURS_CKPT" >&2; exit 1; }
        # Use the pinned CausalPFN benchmarks so the ACIC realizations are
        # byte-identical to what the three UWYK rows above just consumed.
        export CAUSALPFN_ROOT="$UWYK_REPRO/RealCauseEval/CausalPFN"
        export OURS_CKPT
        export TABLE1_OUT_ROOT="${TABLE1_OUT_ROOT:-$DEPLOY_ROOT/results_table1_ours_fn50}"
        python -u "$REPO/benchmarks/uwyk_table1/ours_fn50_no_clustering.py" \
            --dataset ACIC --model ours_fn50 \
            --exp_name "table1_repro_ours_fn50_${SLURM_ARRAY_JOB_ID}"
        ;;
    *)
        echo "invalid array index" >&2; exit 2
        ;;
esac

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done task=$SLURM_ARRAY_TASK_ID"
