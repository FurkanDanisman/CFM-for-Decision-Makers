#!/bin/bash
#SBATCH --job-name=rc-dens-uni
#SBATCH --output=logs_rc_dens_uni/dens_%A_%a.out
#SBATCH --error=logs_rc_dens_uni/dens_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-29
#
# RealCause density dumps on the SAME path as ComplexMech and the case
# studies: every harness writes the cpfn schema itself at its own native
# resolution with DENSITY_DUMP=1, and UWYK_Fig3_4/cate_density_metrics.py
# scores all of them for coverage95 / length / IS_0.05.
#
# Why this exists: RealCause currently goes through a DIFFERENT scorer --
# cate_ci_from_density.py / compute_malc_ci_cell.py with MALC smoothing, and
# submit_realcause_density_metrics.sbatch reporting NLL / L2 / KL. Those are
# not the numbers the other two benchmarks report, so RealCause rows could not
# be placed beside them. This array makes the three comparable.
#
#   30 tasks = 6 models x 5 datasets.  task/5 -> model, task%5 -> dataset.
#     0 dopfn_native   5 dopfn_bb   10 uwyk1d
#    15 graph2d       20 cpfn1d     25 cpfn2d_pooled
#
#   uwyk1d and graph2d run ANC_MODE=v3ab_only (noanc + v3a + v3b in one pass),
#   matching ComplexMech. NOT case_family -- that needs a per-case DAG, which
#   RealCause datasets do not have.
#
# Output: $OUT_ROOT/<model>/<DATASET>/  -- the layout
# cate_density_metrics.py --dataset expects, with the METHODS subdir names.
#
# Required env: OUT_ROOT
#
#   cd $DEPLOY_ROOT && OUT_ROOT=$SCRATCH/rc_dens_uni \
#     sbatch $DEPLOY_ROOT/R-PFN/benchmarks/cluster/submit_realcause_density_unified.sbatch

set -euo pipefail
mkdir -p logs_rc_dens_uni

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
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"

CKPT_CPFN1D="${CKPT_CPFN1D:-/scratch/furkanbd/rpfn_bench_kit/causalpfn_j1024_headrand_A1_h100/step_checkpoints/run/step_0050000.pt}"
CKPT_CPFN2D="${CKPT_CPFN2D:-/scratch/furkanbd/rpfn_bench_kit/cpfn2d_j32_random_A1_h100/step_checkpoints/run/step_0050000.pt}"
CKPT_GRAPH2D="${CKPT_GRAPH2D:-$REPO/Required_checkpoints/graph2d_step_50000.pt}"
CKPT_DOPFN_BB="${CKPT_DOPFN_BB:-$REPO/Required_checkpoints/dopfn_bb_j10_step_150000.pt}"
UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"

MODELS=(dopfn_native dopfn_bb uwyk1d graph2d cpfn1d cpfn2d_pooled)
SETS=(IHDP ACIC CPS PSID PSID_bal)
MODEL="${MODELS[$(( SLURM_ARRAY_TASK_ID / 5 ))]}"
DATASET="${SETS[$(( SLURM_ARRAY_TASK_ID % 5 ))]}"

export DENSITY_DUMP=1
export SKIP_EM="${SKIP_EM:-1}"
export COMPILE="${COMPILE:-0}"
export DATASET
OUT="$OUT_ROOT/$MODEL/$DATASET"
mkdir -p "$OUT"
export OUT

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%F %T')] task=$SLURM_ARRAY_TASK_ID model=$MODEL dataset=$DATASET"
echo "  OUT=$OUT"

case "$MODEL" in
  dopfn_native)
    python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" \
        --dataset "$DATASET"
    ;;
  dopfn_bb)
    # eval_dopfn_bb_raw.py spells the balanced PSID variant 'PSIDbal' (no
    # underscore) in its --dataset choices, unlike every other harness here.
    # Passing PSID_bal aborts with "invalid choice" before any work happens,
    # which is how that one cell came back empty while the other four were
    # fine. submit_realcause_2d_density_dump.sbatch carries the same
    # translation as DS_ON_DISK.
    DS_ON_DISK="$DATASET"
    [ "$DATASET" = "PSID_bal" ] && DS_ON_DISK=PSIDbal
    python -u "$REPO/benchmarks/l2_ihdp/eval_dopfn_bb_raw.py" \
        --dataset "$DS_ON_DISK" --out "$OUT/summary.npz" \
        --repo "$REPO" --dopfn "$DOPFN_ROOT" --causalpfn "$CAUSALPFN" \
        --checkpoint "$CKPT_DOPFN_BB" \
        --y-scaling "${Y_SCALING:-min_max}" --std-target "${STD_TARGET:-0.3}"
    ;;
  uwyk1d)
    # Overridable so a UWYK VARIANT (e.g. the binarized head) can be run
    # without inventing a fake checkpoint directory around it.
    export CKPT="${CKPT:-$UWYK_CKPT_DIR/best_model.pt}"
    export CONFIG="${CONFIG:-$UWYK_CKPT_DIR/best_model_config.yaml}"
    export ANC_MODE=v3ab_only
    python -u "$REPO/realcause_eval/Table1/uwyk1d.py" \
        --dataset "$DATASET" --outdir "$OUT" \
        --uwyk-repro "$UWYK" --repo "$REPO" --causalpfn "$CAUSALPFN" \
        --anc-mode v3ab_only --t-encoding "${UWYK_T_ENCODING:-target}"
    ;;
  graph2d)
    python -u "$REPO/realcause_eval/Table1/graph2d.py" \
        --dataset "$DATASET" --outdir "$OUT" --density-dump 1 \
        --ckpt "$CKPT_GRAPH2D" --repo "$REPO" \
        --causalpfn "$CAUSALPFN" --uwyk "$UWYK" \
        --anc-mode v3ab_only --density-primary-mode noanc
    ;;
  cpfn1d)
    # pooled, not per_arm: the dump asserts both arms share y_shift/y_scale so
    # the shift cancels in tau. Same constraint ComplexMech runs under.
    python -u "$REPO/realcause_eval/Table1/cpfn1d.py" \
        --dataset "$DATASET" --outdir "$OUT" --density-dump 1 \
        --ckpt "$CKPT_CPFN1D" --repo "$REPO" --causalpfn "$CAUSALPFN" \
        --std-mode pooled
    ;;
  cpfn2d_pooled)
    python -u "$REPO/realcause_eval/Table1/cpfn2d.py" \
        --dataset "$DATASET" --outdir "$OUT" --density-dump 1 \
        --ckpt "$CKPT_CPFN2D" --repo "$REPO" --causalpfn "$CAUSALPFN" \
        --std-mode pooled
    ;;
esac

echo "[$(date '+%F %T')] done  $MODEL $DATASET"
