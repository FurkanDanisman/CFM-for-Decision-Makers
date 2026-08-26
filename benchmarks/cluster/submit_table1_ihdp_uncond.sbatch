#!/bin/bash
#SBATCH --job-name=t1-ihdp-uncond
#SBATCH --output=logs_table1_all/ihdp_uncond_%j.out
#SBATCH --error=logs_table1_all/ihdp_uncond_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#
# UWYK "no-graph-conditioning" baseline on IHDP.
#
# Loads the reproduce-branch checkpoint at
#   experiments/checkpoints/no_graph_conditioning/unconditional/best_model.pt
# which is architecturally identical to No-Anc / Anc but was trained with
# hide_fraction_matrix = 1.0 (adjacency always fully hidden). This is the
# paper's Figures 3/4 "baseline that does not support graph conditioning
# but is otherwise trained identically."
#
# Query at inference with --graph_mode all_unknown so the runtime adjacency
# matches the training regime: every edge marked unknown.
#
# This gives fn=50 a true head-only ablation: same body architecture family,
# same training prior, no useful graph info at either train or eval — the
# only remaining difference from fn=50 is our 2D joint head vs their
# single-arm HL-Gauss CE head.

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

# Ensure the unconditional checkpoint is actually pulled from LFS.
# `git lfs ls-files` showed it's tracked (4be3fb0816 -> no_graph_conditioning/
# unconditional/best_model.pt) but the actual weight file may not be pulled
# on the cluster yet.
CKPT="$UWYK_REPRO/experiments/checkpoints/no_graph_conditioning/unconditional/best_model.pt"
# CFG defaults to the shipped config, but the shipped one has
# test_feature_mask_fraction: 1.0 which zeros out ALL X_intv features at
# inference — making the model predict near-zero CATE (eps_ATE ≈ 1.0).
# Override with PATCHED_CFG (a copy of the config with the mask set to
# 0.0) to get meaningful predictions.
CFG="${PATCHED_CFG:-$UWYK_REPRO/experiments/checkpoints/no_graph_conditioning/unconditional/best_model_config.yaml}"
if [ ! -f "$CKPT" ] || [ "$(wc -c < "$CKPT" 2>/dev/null || echo 0)" -lt 1000 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Fetching LFS weight for no_graph_conditioning/unconditional/…"
    (cd "$UWYK_REPRO" && git lfs pull --include="experiments/checkpoints/no_graph_conditioning/**/*.pt")
fi
[ -f "$CFG"  ] || { echo "FATAL: config missing at $CFG" >&2; exit 1; }
[ -f "$CKPT" ] || { echo "FATAL: checkpoint still missing after LFS pull at $CKPT" >&2; exit 1; }
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ckpt size = $(du -h "$CKPT" | cut -f1)"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
# Shim faiss/huggingface_hub so causalpfn/__init__.py imports don't blow up
# on the cluster where those aren't installed.
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims:$UWYK_REPRO/src${PYTHONPATH:+:$PYTHONPATH}"

JOB="${SLURM_JOB_ID}"
# Distinct method tag "uncond" so aggregator can pick this up as a separate
# row alongside predictive/noanc/anc/fn50/fn50p. Output goes to the standard
# UWYK results dir: $UWYK_REPRO/RealCauseEval/results/<exp_name>/*
EXP="table1_all_IHDP_uncond_${JOB}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching UWYK unconditional (no-graph-conditioning) on IHDP"
echo "  ckpt=$CKPT"
echo "  cfg=$CFG"
echo "  graph_mode=all_unknown  (matches training's hide_fraction=1.0)"
echo "  exp_name=$EXP"

cd "$UWYK_REPRO/RealCauseEval"
python -u run_baselines/dofm_no_clustering.py \
    --dataset IHDP --model uncond --exp_name "$EXP" \
    --checkpoint_path "$CKPT" \
    --config_path "$CFG" \
    --graph_mode all_unknown

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done"
