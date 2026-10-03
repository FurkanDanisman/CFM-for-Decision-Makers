#!/bin/bash
#SBATCH --job-name=e2-eval
#SBATCH --output=logs_e2/eval_%a.out
#SBATCH --error=logs_e2/eval_%a.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-59%6
# E2 densities: the paper's per-model harnesses, unchanged, on the E2 worlds (DENSITY_DUMP=1).
# Task id: model = id / 10, shard = id % 10 (100 worlds per shard).
# Submit AFTER gen_e2 has finished (no dependency jobs):
#   E2_DATA=$SCRATCH/e2/data E2_DUMPS=$SCRATCH/e2/dumps sbatch R-PFN/revision/experiments/E2_in_prior/eval_e2.sbatch
set -euo pipefail
mkdir -p logs_e2
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="$DEPLOY_ROOT/R-PFN"
CK="$REPO/Required_checkpoints"
export CAUSALPFN="$DEPLOY_ROOT/external/causalpfn" DOPFN_ROOT="$DEPLOY_ROOT/external/dopfn"
UWYK="$DEPLOY_ROOT/external/uwyk_reproduce"
UWYK_CKPT_DIR="$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"

MODELS=(dopfn_native dopfn_joint2d uwyk1d graph2d cpfn1d cpfn2d)
PRIORS=(dopfn        dopfn         uwyk_1d uwyk_2d cpfn_1d cpfn_2d)
I=$(( SLURM_ARRAY_TASK_ID / 10 )); SHARD=$(( SLURM_ARRAY_TASK_ID % 10 ))
MODEL=${MODELS[$I]}
export UWYK_FIG34_DATA="${E2_DATA:?}/${PRIORS[$I]}/shard$SHARD"
export DATASET=CMECH_n5                       # all 100 queries of every world
export OUT="${E2_DUMPS:?}/$MODEL/shard$SHARD" DENSITY_DUMP=1
mkdir -p "$OUT"
source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
echo "[$(date '+%F %T')] model=$MODEL data=$UWYK_FIG34_DATA out=$OUT"

case "$MODEL" in
  dopfn_native)
    python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DATASET" ;;
  dopfn_joint2d)
    DOPFN_CKPT="$CK/dopfn_repro_joint2d_step150000.pt" \
      python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" --dataset "$DATASET" ;;
  uwyk1d)
    python -u "$REPO/realcause_eval/Table1/uwyk1d.py" --dataset "$DATASET" --outdir "$OUT" \
      --uwyk-repro "$UWYK" --repo "$REPO" --causalpfn "$CAUSALPFN" \
      --ckpt "$UWYK_CKPT_DIR/best_model.pt" --config "$UWYK_CKPT_DIR/best_model_config.yaml" \
      --anc-mode v3ab_only --t-encoding target --eval-max-context 1000 --eval-context-seed 43 ;;
  graph2d)
    python -u "$REPO/realcause_eval/Table1/graph2d.py" --dataset "$DATASET" --outdir "$OUT" \
      --density-dump 1 --ckpt "$CK/graph2d_step_50000.pt" \
      --repo "$REPO" --causalpfn "$CAUSALPFN" --uwyk "$UWYK" \
      --anc-mode v3ab_only --density-primary-mode noanc --eval-max-context 1000 --eval-context-seed 43 ;;
  cpfn1d)                                     # per-arm y scaling (its training convention)
    CKPT="$CK/cpfn1d_j1024_headrand_step_50000.pt" EVAL_MAX_CONTEXT= EVAL_CONTEXT_SEED=1 \
    STD_MODE=per_arm COMPILE=0 SKIP_EM=1 \
      python -u "$REPO/case_study/density_eval/eval_cpfn1d_perarm.py" --dataset "$DATASET" ;;
  cpfn2d)
    python -u "$REPO/realcause_eval/Table1/cpfn2d.py" --dataset "$DATASET" --outdir "$OUT" \
      --density-dump 1 --ckpt "$CK/cpfn2d_j32_eta0_y01_step50000.pt" \
      --repo "$REPO" --causalpfn "$CAUSALPFN" --std-mode pooled ;;
esac
echo "[$(date '+%F %T')] done"
