#!/bin/bash
#SBATCH --job-name=cs-density
#SBATCH --output=logs_cs_density/eval_%A_%a.out
#SBATCH --error=logs_cs_density/eval_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-41
#
# CATE/ATE interval calibration on the 6 DoPFN SCM case studies.
#
#   7 model-runs x 6 case studies = 42 tasks, DENSITY_DUMP=1 throughout.
#   Produces the per-query predictive densities that
#   UWYK_Fig3_4/cate_density_metrics.py turns into coverage / length / IS_0.05.
#
#   models  dopfn_native | dopfn_bb        (1D vs 2D)
#           uwyk1d       | graph2d         (1D vs 2D)
#           cpfn1d       | cpfn2d_pooled   (1D vs 2D)
#           cpfn2d_log                     (point estimates only — see below)
#
#   Task id -> model: id/6 indexes MODELS, id%6 indexes CASES.
#        0 dopfn_native    6 dopfn_bb     12 uwyk1d    18 graph2d
#       24 cpfn1d         30 cpfn2d_pooled  36 cpfn2d_log
#
#   uwyk1d and graph2d run ANC_MODE=v3ab_only, emitting noanc + v3a + v3b in
#   one pass — matching the ComplexMech array, so the scorer's METHODS table
#   (which asks for the `noanc` and `v3a` tags) serves both benchmarks.
#
#   This is the case-study twin of submit_cmech_pehe_1d_vs_2d.sbatch. The
#   differences are only in the data source: case studies come from the
#   ORIGINAL DoPFN pkls under $DOPFN_DATA_ROOT, so there is no node-count /
#   context / subset grid and no UWYK_FIG34_MAX_CONTEXT.
#
# Output layout (what the scorer's --dataset mode expects):
#   $OUT_ROOT/<model>/<CASE>/*.npz
#
# Required env:
#   OUT_ROOT          results root
# Optional env:
#   DOPFN_DATA_ROOT   direct parent of the case-study dirs (prior_sampling/)
#   CKPT_*            per-model checkpoint overrides (defaults below)
#   DENSITY_DUMP      default 1 here (the whole point of this array)
#
# Submit:
#   cd $DEPLOY_ROOT && OUT_ROOT=$SCRATCH/cs_density \
#       sbatch $DEPLOY_ROOT/R-PFN/benchmarks/cluster/submit_cs_density_1d_vs_2d.sbatch

set -euo pipefail
mkdir -p logs_cs_density

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
export UWYK="${UWYK:-$DEPLOY_ROOT/external/uwyk_reproduce}"
export OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"

# ── case-study data: the ORIGINAL DoPFN pkls, not any regen ──────────────────
# DOPFN_DATA_ROOT must be the DIRECT parent of Observed_Confounder/ etc.
export DOPFN_DATA_ROOT="${DOPFN_DATA_ROOT:-$DEPLOY_ROOT/external/dopfn/data/prior_sampling}"
# DOPFN_ROOT = repo root; needed to unpickle the pkls (they carry DoPFN's
# `datasets.InterventionalDataset` class refs).
export DOPFN_ROOT="${DOPFN_ROOT:-${DOPFN_DATA_ROOT%/data/prior_sampling}}"
[ -d "$DOPFN_DATA_ROOT/Observed_Confounder" ] || {
    echo "FATAL: no case-study pkls under $DOPFN_DATA_ROOT" >&2; exit 1; }
# Case studies need raw mean CATE only — skip EM everywhere (large speedup).
export SKIP_EM="${SKIP_EM:-1}"

# ── per-model checkpoints (override via env) ─────────────────────────────────
CKPT_CPFN1D="${CKPT_CPFN1D:-/scratch/furkanbd/rpfn_bench_kit/causalpfn_j1024_headrand_A1_h100/step_checkpoints/run/step_0050000.pt}"
CKPT_CPFN2D="${CKPT_CPFN2D:-/scratch/furkanbd/rpfn_bench_kit/cpfn2d_j32_random_A1_h100/step_checkpoints/run/step_0050000.pt}"
CKPT_GRAPH2D="${CKPT_GRAPH2D:-$REPO/Required_checkpoints/graph2d_step_50000.pt}"
CKPT_DOPFN_BB="${CKPT_DOPFN_BB:-$DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt}"
UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"

# ── task dispatch: model x case study ────────────────────────────────────────
MODELS=(dopfn_native dopfn_bb uwyk1d graph2d cpfn1d cpfn2d_pooled cpfn2d_log)
CASES=(Observed_Confounder
       Observed_Mediator
       Observed_Mediator_and_Confounder
       Unobserved_Confounder
       Frontdoor_Criterion
       Backdoor_Criterion)

N_CASES=${#CASES[@]}
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / N_CASES ))
CASE_IDX=$(( SLURM_ARRAY_TASK_ID % N_CASES ))
MODEL="${MODELS[$MODEL_IDX]}"
DATASET="${CASES[$CASE_IDX]}"

# DENSITY_DUMP=1 makes every harness write p_y0/p_y1/p_joint per query.
# uwyk1d, dopfn_native and dopfn_bb read it from the environment; graph2d /
# cpfn1d / cpfn2d go through wrappers that set it explicitly and would
# otherwise clobber it to 0, so those get it as a flag below.
export DENSITY_DUMP="${DENSITY_DUMP:-1}"

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

echo "[$(date '+%F %T')] task=$SLURM_ARRAY_TASK_ID  model=$MODEL  case=$DATASET"
echo "  OUT=$OUT   DENSITY_DUMP=$DENSITY_DUMP"
echo "  DOPFN_DATA_ROOT=$DOPFN_DATA_ROOT"

case "$MODEL" in
  dopfn_native)
    # native DoPFNRegressor — reads DATASET from the environment.
    python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" \
        --dataset "$DATASET"
    ;;
  dopfn_bb)
    [ -f "$CKPT_DOPFN_BB" ] || { echo "FATAL: ckpt $CKPT_DOPFN_BB" >&2; exit 1; }
    # NB: dopfn_bb writes density_r<###>.npz beside summary.npz, not per-r files.
    python -u "$REPO/benchmarks/l2_ihdp/eval_dopfn_bb_raw.py" \
        --dataset "$DATASET" --out "$OUT/summary.npz" \
        --repo "$REPO" --dopfn "$DOPFN_ROOT" --causalpfn "$CAUSALPFN" \
        --checkpoint "$CKPT_DOPFN_BB" \
        --y-scaling "${Y_SCALING:-min_max}" --std-target "${STD_TARGET:-0.3}"
    ;;
  uwyk1d)
    export CKPT="$UWYK_CKPT_DIR/best_model.pt"
    export CONFIG="$UWYK_CKPT_DIR/best_model_config.yaml"
    export ANC_MODE=v3ab_only
    python -u "$REPO/benchmarks/eval_graph2d/eval_uwyk1d_realcause.py"
    ;;
  graph2d)
    [ -f "$CKPT_GRAPH2D" ] || { echo "FATAL: ckpt $CKPT_GRAPH2D" >&2; exit 1; }
    export CKPT="$CKPT_GRAPH2D"
    export ANC_MODE=v3ab_only
    export DENSITY_PRIMARY_MODE="${DENSITY_PRIMARY_MODE:-noanc}"
    python -u "$REPO/benchmarks/eval_graph2d/eval_graph2d_realcause.py" \
        --dataset "$DATASET"
    ;;
  cpfn1d)
    [ -f "$CKPT_CPFN1D" ] || { echo "FATAL: ckpt $CKPT_CPFN1D" >&2; exit 1; }
    # eval_causalpfn_v0_realcause.py hard-asserts that the dumped density mean
    # reproduces cate_raw, which only holds when both arms share y_shift /
    # y_scale — i.e. STD_MODE=pooled. A density run under the per_arm default
    # raises and the task dies. NB this makes the density run's cpfn1d point
    # estimates pooled-scaled, not per_arm: worth a footnote if both are shown.
    export CKPT="$CKPT_CPFN1D"
    export STD_MODE="${STD_MODE:-pooled}"
    export COMPILE="${COMPILE:-0}"
    python -u "$REPO/benchmarks/eval_causalpfn2d/eval_causalpfn_v0_realcause.py" \
        --dataset "$DATASET"
    ;;
  cpfn2d_pooled|cpfn2d_log)
    # cpfn2d_log cannot emit a CATE density: under a log target transform
    # tau = Y1 - Y0 in RAW space is not a linear function of the scaled
    # difference, so one (y_shift, y_scale) pair cannot reverse it and the
    # diagonal projection would be wrong. eval_cpfn2d_realcause.py asserts
    # exactly this. Skip the cell rather than fail the task; point estimates
    # are unaffected, so cpfn2d_log still appears in the PEHE/ATE tables.
    if [ "$DENSITY_DUMP" = "1" ] && [ "$MODEL" = "cpfn2d_log" ]; then
        echo "[skip] cpfn2d_log emits no density (log transform is not affine" \
             "in tau); nothing to do for DENSITY_DUMP=1."
        exit 0
    fi
    [ -f "$CKPT_CPFN2D" ] || { echo "FATAL: ckpt $CKPT_CPFN2D" >&2; exit 1; }
    export CKPT="$CKPT_CPFN2D"
    case "$MODEL" in
      cpfn2d_pooled) export STD_MODE=pooled ;;
      cpfn2d_log)    export STD_MODE=log ;;
    esac
    export COMPILE="${COMPILE:-0}"
    python -u "$REPO/benchmarks/eval_causalpfn2d/eval_cpfn2d_realcause.py" \
        --dataset "$DATASET"
    ;;
  *) echo "unknown model: $MODEL" >&2; exit 2 ;;
esac

echo "[$(date '+%F %T')] done  model=$MODEL  case=$DATASET"
