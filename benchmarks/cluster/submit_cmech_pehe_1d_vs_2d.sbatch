#!/bin/bash
#SBATCH --job-name=cmech-1d2d
#SBATCH --output=logs_cmech_1d2d/eval_%A_%a.out
#SBATCH --error=logs_cmech_1d2d/eval_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-59
#
# 1D-vs-2D head comparison on the UWYK ComplexMech (Fig-4 prior) PEHE benchmark.
#
#   6 models x 5 node counts x 2 query subsets = 60 tasks.
#
#   models      dopfn_native | dopfn_bb   (1D vs 2D)
#               uwyk1d       | graph2d    (1D vs 2D)
#               cpfn1d       | cpfn2d     (1D vs 2D)
#
#   Task id -> model: id/10 indexes MODELS, so the first task of each model is
#       0 dopfn_native   10 dopfn_bb   20 uwyk1d
#      30 graph2d        40 cpfn1d     50 cpfn2d
#   and within a model, (id%10)/2 indexes nodes (5,20,30,40,50) while id%2
#   picks the subset (0=nonzero, 1=zero).
#   node counts 5, 20, 30, 40, 50
#   subsets     nonzero (true tau != 0)  |  zero (true tau == 0 exactly)
#
# The two subsets are disjoint, so the pooled PEHE is recoverable exactly:
#   pehe_all^2 = (n0*pehe_zero^2 + n1*pehe_nonzero^2) / (n0 + n1)
# aggregate_cmech_1d_vs_2d.py does that for you.
#
# PREREQUISITE — generate the data once (CPU only, ~3 min, no GPU needed):
#   python $REPO/UWYK_Fig3_4/generate_pehe_benchmark.py \
#       --prior complexmech --nodes 5 20 30 40 50 --regimes path_TY \
#       --hide-fractions 0.0 --n-realizations 100 \
#       --out-dir $REPO/UWYK_Fig3_4/data
#   (needs UWYK_SRC / UWYK_ROOT; see UWYK_Fig3_4/README.md)
#
# Required env:
#   OUT_ROOT          results land at OUT_ROOT/<model>/<DATASET>/
# Optional env:
#   UWYK_FIG34_DATA   benchmark data root (default $REPO/UWYK_Fig3_4/data)
#   CKPT_*            per-model checkpoint overrides (defaults below)
#
# Submit:
#   cd $DEPLOY_ROOT && OUT_ROOT=$SCRATCH/cmech_1d2d \
#       sbatch $DEPLOY_ROOT/R-PFN/benchmarks/cluster/submit_cmech_pehe_1d_vs_2d.sbatch

set -euo pipefail
mkdir -p logs_cmech_1d2d

# ── deploy-root discovery (venv, repo) ───────────────────────────────────────
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

# Where the generated benchmark npzs live. Read by benchmarks/uwyk_fig34_dataset.py.
export UWYK_FIG34_DATA="${UWYK_FIG34_DATA:-$REPO/UWYK_Fig3_4/data}"
[ -d "$UWYK_FIG34_DATA/complexmech" ] || {
    echo "FATAL: no benchmark data at $UWYK_FIG34_DATA/complexmech" >&2
    echo "       run UWYK_Fig3_4/generate_pehe_benchmark.py first (see header)" >&2
    exit 1; }

# ── per-model checkpoints (override via env) ─────────────────────────────────
CKPT_CPFN1D="${CKPT_CPFN1D:-/scratch/furkanbd/rpfn_bench_kit/causalpfn_j1024_headrand_A1_h100/step_checkpoints/run/step_0050000.pt}"
CKPT_CPFN2D="${CKPT_CPFN2D:-/scratch/furkanbd/rpfn_bench_kit/cpfn2d_j32_random_A1_h100/step_checkpoints/run/step_0040000.pt}"
CKPT_GRAPH2D="${CKPT_GRAPH2D:-$REPO/Required_checkpoints/graph2d_step_50000.pt}"
CKPT_DOPFN_BB="${CKPT_DOPFN_BB:-$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt}"
UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"

# ── task dispatch: model x nodes x subset ────────────────────────────────────
MODELS=(dopfn_native dopfn_bb uwyk1d graph2d cpfn1d cpfn2d)
NODES=(5 20 30 40 50)
SUBSETS=(nonzero zero)

N_NODES=${#NODES[@]}; N_SUB=${#SUBSETS[@]}
PER_MODEL=$(( N_NODES * N_SUB ))
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / PER_MODEL ))
REM=$(( SLURM_ARRAY_TASK_ID % PER_MODEL ))
NODE_IDX=$(( REM / N_SUB ))
SUB_IDX=$(( REM % N_SUB ))

MODEL="${MODELS[$MODEL_IDX]}"
NODE="${NODES[$NODE_IDX]}"
SUBSET="${SUBSETS[$SUB_IDX]}"
DATASET="CMECH_n${NODE}_${SUBSET}"

export DATASET
export OUT="$OUT_ROOT/$MODEL/$DATASET"
mkdir -p "$OUT"

# ── env / activation ─────────────────────────────────────────────────────────
source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%F %T')] task=$SLURM_ARRAY_TASK_ID  model=$MODEL  dataset=$DATASET"
echo "  OUT=$OUT"
echo "  UWYK_FIG34_DATA=$UWYK_FIG34_DATA"

# Context cap: these datasets ship 1000 training rows, which is already the cap
# used for the RealCause runs, so no subsampling happens. Kept explicit so the
# comparison is on the same footing as Table 1.
CTX=1000
SEED=43

case "$MODEL" in
  dopfn_native)
    python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" \
        --dataset "$DATASET"
    ;;
  dopfn_bb)
    [ -f "$CKPT_DOPFN_BB" ] || { echo "FATAL: ckpt $CKPT_DOPFN_BB" >&2; exit 1; }
    # NB: dopfn_bb writes ONE npz of per-realization arrays, not per-r files.
    python -u "$REPO/benchmarks/l2_ihdp/eval_dopfn_bb_raw.py" \
        --dataset "$DATASET" --out "$OUT/summary.npz" \
        --repo "$REPO" --dopfn "$DOPFN_ROOT" --causalpfn "$CAUSALPFN" \
        --checkpoint "$CKPT_DOPFN_BB" \
        --y-scaling "${Y_SCALING:-min_max}" --std-target "${STD_TARGET:-0.3}"
    ;;
  uwyk1d)
    export CKPT="$UWYK_CKPT_DIR/best_model.pt"
    export CONFIG="$UWYK_CKPT_DIR/best_model_config.yaml"
    python -u "$REPO/realcause_eval/Table1/uwyk1d.py" \
        --dataset "$DATASET" --outdir "$OUT" \
        --uwyk-repro "$UWYK" --repo "$REPO" --causalpfn "$CAUSALPFN" \
        --eval-max-context "$CTX" --eval-context-seed "$SEED"
    ;;
  graph2d)
    [ -f "$CKPT_GRAPH2D" ] || { echo "FATAL: ckpt $CKPT_GRAPH2D" >&2; exit 1; }
    python -u "$REPO/realcause_eval/Table1/graph2d.py" \
        --dataset "$DATASET" --outdir "$OUT" --ckpt "$CKPT_GRAPH2D" \
        --repo "$REPO" --causalpfn "$CAUSALPFN" --uwyk "$UWYK" \
        --eval-max-context "$CTX" --eval-context-seed "$SEED"
    ;;
  cpfn1d)
    [ -f "$CKPT_CPFN1D" ] || { echo "FATAL: ckpt $CKPT_CPFN1D" >&2; exit 1; }
    python -u "$REPO/realcause_eval/Table1/cpfn1d.py" \
        --dataset "$DATASET" --outdir "$OUT" --ckpt "$CKPT_CPFN1D" \
        --repo "$REPO" --causalpfn "$CAUSALPFN"
    ;;
  cpfn2d)
    [ -f "$CKPT_CPFN2D" ] || { echo "FATAL: ckpt $CKPT_CPFN2D" >&2; exit 1; }
    python -u "$REPO/realcause_eval/Table1/cpfn2d.py" \
        --dataset "$DATASET" --outdir "$OUT" --ckpt "$CKPT_CPFN2D" \
        --repo "$REPO" --causalpfn "$CAUSALPFN"
    ;;
  *) echo "unknown model: $MODEL" >&2; exit 2 ;;
esac

echo "[$(date '+%F %T')] done  model=$MODEL  dataset=$DATASET"
