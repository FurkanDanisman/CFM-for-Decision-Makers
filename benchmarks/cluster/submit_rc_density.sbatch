#!/bin/bash
#SBATCH --job-name=rc-density
#SBATCH --output=logs_rc_density/rc_%A_%a.out
#SBATCH --error=logs_rc_density/rc_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-29
#
# RealCause density dumps for the CATE/ATE calibration tables.
#
#   6 models x 5 datasets = 30 tasks.
#   models    dopfn_native dopfn_bb uwyk1d graph2d cpfn1d cpfn2d
#   datasets  IHDP ACIC CPS PSID PSID_bal
#
#   task -> model = id/5 ;  dataset = id%5
#       0 dopfn_native   5 dopfn_bb   10 uwyk1d
#      15 graph2d       20 cpfn1d     25 cpfn2d
#
# Writes <OUT_ROOT>/<model>/<DATASET>/ — the layout
# UWYK_Fig3_4/cate_density_metrics.py --dataset expects.
#
# Score with:
#   for t in cate ate; do
#     python $REPO/UWYK_Fig3_4/cate_density_metrics.py --root $OUT_ROOT \
#         --dataset IHDP ACIC CPS PSID PSID_bal --target $t
#   done
#
# Required:  OUT_ROOT
# Submit:
#   cd $DEPLOY_ROOT && OUT_ROOT=$SCRATCH/rc_dens \
#       sbatch $REPO/benchmarks/cluster/submit_rc_density.sbatch

set -uo pipefail
mkdir -p logs_rc_density

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
export DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
export UWYK="${UWYK:-$DEPLOY_ROOT/external/uwyk_reproduce}"
export OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"

CK_DIR="${CK_DIR:-$REPO/Required_checkpoints}"
CKPT_CPFN1D="${CKPT_CPFN1D:-$CK_DIR/cpfn1d_j1024_headrand_step_50000.pt}"
CKPT_CPFN2D="${CKPT_CPFN2D:-$CK_DIR/cpfn2d_j32_random_step_50000.pt}"
CKPT_GRAPH2D="${CKPT_GRAPH2D:-$CK_DIR/graph2d_step_50000.pt}"
CKPT_DOPFN_BB="${CKPT_DOPFN_BB:-$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt}"
UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"

MODELS=(dopfn_native dopfn_bb uwyk1d graph2d cpfn1d cpfn2d)
DATASETS=(IHDP ACIC CPS PSID PSID_bal)
MODEL="${MODELS[$(( SLURM_ARRAY_TASK_ID / 5 ))]}"
DATASET="${DATASETS[$(( SLURM_ARRAY_TASK_ID % 5 ))]}"

export DATASET
export OUT="$OUT_ROOT/$MODEL/$DATASET"
mkdir -p "$OUT"
export DENSITY_DUMP=1

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
# causalpfn/__init__ pulls faiss, transformers, wandb at import time purely to
# reach its dataset loaders. This shim stubs them; without it every RealCause
# task dies on `ModuleNotFoundError: No module named 'faiss'`.
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%F %T')] task=$SLURM_ARRAY_TASK_ID  model=$MODEL  dataset=$DATASET"
echo "  OUT=$OUT"

case "$MODEL" in
  dopfn_native)
    python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" \
        --dataset "$DATASET"
    ;;
  dopfn_bb)
    python -u "$REPO/benchmarks/l2_ihdp/eval_dopfn_bb_raw.py" \
        --dataset "${DATASET/PSID_bal/PSIDbal}" --out "$OUT/summary.npz" \
        --repo "$REPO" --dopfn "$DOPFN_ROOT" --causalpfn "$CAUSALPFN" \
        --checkpoint "$CKPT_DOPFN_BB"
    ;;
  uwyk1d)
    export CKPT="$UWYK_CKPT_DIR/best_model.pt"
    export CONFIG="$UWYK_CKPT_DIR/best_model_config.yaml"
    python -u "$REPO/realcause_eval/Table1/uwyk1d.py" \
        --dataset "$DATASET" --outdir "$OUT" \
        --uwyk-repro "$UWYK" --repo "$REPO" --causalpfn "$CAUSALPFN" \
        --anc-mode v3ab_only
    ;;
  graph2d)
    python -u "$REPO/realcause_eval/Table1/graph2d.py" \
        --dataset "$DATASET" --outdir "$OUT" --ckpt "$CKPT_GRAPH2D" \
        --repo "$REPO" --causalpfn "$CAUSALPFN" --uwyk "$UWYK" \
        --anc-mode v3ab_only --density-dump 1
    ;;
  cpfn1d)
    # pooled, not the per_arm its point estimates use: the harness asserts the
    # dumped density mean reproduces cate_raw, which needs both arms to share
    # y_shift/y_scale. Same split as in the ComplexMech run — footnote it.
    python -u "$REPO/realcause_eval/Table1/cpfn1d.py" \
        --dataset "$DATASET" --outdir "$OUT" --ckpt "$CKPT_CPFN1D" \
        --repo "$REPO" --causalpfn "$CAUSALPFN" \
        --std-mode pooled --density-dump 1
    ;;
  cpfn2d)
    python -u "$REPO/realcause_eval/Table1/cpfn2d.py" \
        --dataset "$DATASET" --outdir "$OUT" --ckpt "$CKPT_CPFN2D" \
        --repo "$REPO" --causalpfn "$CAUSALPFN" \
        --std-mode pooled --density-dump 1
    ;;
  *) echo "unknown model: $MODEL" >&2; exit 2 ;;
esac

echo "[$(date '+%F %T')] done  model=$MODEL  dataset=$DATASET"
