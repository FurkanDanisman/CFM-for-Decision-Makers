#!/usr/bin/env bash
# Run one row × one dataset of UWYK's Table 1.
#
# Rows:
#   predictive  — S-learner on an unconditional model
#                 script: run_baselines/predmodel_Slearner_full_context.py
#   noanc       — full-conditioning model with adjacency zeroed (--all_unknown)
#                 script: run_baselines/dofm_full_conditioning.py --all_unknown
#   anc         — full-conditioning model with the true adjacency
#                 script: run_baselines/dofm_full_conditioning.py
#
# Both dofm rows share ONE checkpoint; only the --all_unknown flag differs.
#
# Usage:
#   ./run_uwyk_table1.sh <DATASET> <MODE> [<CKPT_TAG>]
#
# Env overrides:
#   DEPLOY_ROOT / UWYK_ROOT / CAUSALPFN_ROOT
#   UWYK_CKPT_DIR / CKPT_FILE / CONFIG_FILE
#   PRED_CKPT_DIR / PRED_CKPT_FILE / PRED_CONFIG_FILE
#   TASK_ID    unique suffix for patched files (default: SLURM_ARRAY_TASK_ID or $$)

set -euo pipefail

DATASET="${1:?dataset required (IHDP|ACIC|CPS|PSID|all)}"
MODE="${2:?mode required (predictive|noanc|anc)}"
CKPT_TAG="${3:-$(date +%Y%m%d_%H%M%S)}"

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
UWYK_ROOT="${UWYK_ROOT:-$DEPLOY_ROOT/external/uwyk}"
CAUSALPFN_ROOT="${CAUSALPFN_ROOT:-$DEPLOY_ROOT/external/causalpfn}"

UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$UWYK_ROOT/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
CKPT_FILE="${CKPT_FILE:-best_model.pt}"
CONFIG_FILE="${CONFIG_FILE:-best_model_config.yaml}"

# Predictive checkpoint. The reproduce-realcause-results branch shipped the
# real one (via git LFS) at experiments/checkpoints/predictive_slearner/;
# our previous fallback (no_graph_conditioning/unconditional) gave a ~17%
# gap on IHDP because it's a different, unconditional model. Point at the
# real predictive_slearner if you've pulled the reproduce-realcause-results
# branch + LFS; otherwise fall back to the old ckpt for smoke tests.
PRED_CKPT_DIR="${PRED_CKPT_DIR:-$UWYK_ROOT/experiments/checkpoints/predictive_slearner}"
PRED_CKPT_FILE="${PRED_CKPT_FILE:-best_model.pt}"
PRED_CONFIG_FILE="${PRED_CONFIG_FILE:-best_model_config.yaml}"

# ── Per-task uniqueness (avoid clobbering when tasks run in parallel) ────
TASK_ID="${TASK_ID:-${SLURM_ARRAY_TASK_ID:-$$}}"
TAG="__patched_t${TASK_ID}"
INSTALL_DIR="$UWYK_ROOT/RealCauseEval/run_baselines"

# Escape UWYK_ROOT for sed
ESC_ROOT=$(printf '%s' "$UWYK_ROOT" | sed 's|/|\\/|g')

patch_and_stage() {
    local src_basename="$1"; local dst_basename="$2"
    local src="$INSTALL_DIR/$src_basename"
    local dst="$INSTALL_DIR/$dst_basename"
    cp "$src" "$dst"
    sed -i "s/<REPO_ROOT>/$ESC_ROOT/g" "$dst"
}

EVAL_PATCHED="eval${TAG}.py"
DOFM_PATCHED="dofm_full_conditioning${TAG}.py"
PRED_PATCHED="predmodel_Slearner${TAG}.py"

patch_and_stage "eval.py"                             "$EVAL_PATCHED"
patch_and_stage "dofm_full_conditioning.py"           "$DOFM_PATCHED"
patch_and_stage "predmodel_Slearner_full_context.py"  "$PRED_PATCHED"

# Point the two entry scripts at OUR patched eval instead of run_baselines.eval
EVAL_MOD="run_baselines.${EVAL_PATCHED%.py}"
sed -i "s/from run_baselines.eval import/from $EVAL_MOD import/g" \
    "$INSTALL_DIR/$DOFM_PATCHED" "$INSTALL_DIR/$PRED_PATCHED"

# Disable the wrapper's default KMeans clustering. UWYK's reproduce-realcause
# branch's dofm_no_clustering.py shows the fix: pass use_clustering=False to
# PreprocessingGraphConditionedPFN. Their default (use_clustering=True)
# clusters CPS's 14559 rows into 15 chunks and produces the ~2x PEHE
# inflation on CPS + ~1.7x on PSID(unbal) that we spent last week chasing.
sed -i 's/verbose=True,$/verbose=True, use_clustering=False,/' \
    "$INSTALL_DIR/$DOFM_PATCHED"

# UWYK's eval.py imports `from CausalPFN.benchmarks import …`, treating
# CausalPFN as the repo dir with `benchmarks/` as a top-level submodule.
# On our cluster that layout is:
#   $CAUSALPFN_ROOT/benchmarks/{ihdp.py, realcause.py, ...}
#   $CAUSALPFN_ROOT/src/causalpfn/…       (the pip package — NOT what upstream wants)
# We rewrite the import to grab `benchmarks` directly from the repo root.
sed -i 's/from CausalPFN\.benchmarks import/from benchmarks import/g' \
    "$INSTALL_DIR/$EVAL_PATCHED"

# ── PYTHONPATH so upstream imports resolve ─────────────────────────────
# - shims/                     → sitecustomize denylist (safety; not required
#                                once we bypass causalpfn/__init__.py, but
#                                harmless and future-proofs against causalpfn
#                                deps sneaking into `benchmarks/`)
# - $UWYK_ROOT                 → 'src.models.*' and 'run_baselines.*'
# - $UWYK_ROOT/RealCauseEval   → 'run_baselines.eval__patched_*'
# - $CAUSALPFN_ROOT            → top-level `benchmarks` package
SHIM_DIR="$(dirname "$(readlink -f "$0")")/shims"
# CAUSALPFN_ROOT gives us the top-level `benchmarks/` package; CAUSALPFN_ROOT/src
# gives us the `causalpfn` package that `benchmarks/__init__.py` imports from.
export PYTHONPATH="$SHIM_DIR:$UWYK_ROOT:$UWYK_ROOT/RealCauseEval:$CAUSALPFN_ROOT:$CAUSALPFN_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL_NAME="uwyk_table1_${MODE}"
EXP_NAME="table1_${MODE}_${CKPT_TAG}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching Table 1 row"
echo "  dataset=$DATASET  mode=$MODE  ckpt_tag=$CKPT_TAG  task_id=$TASK_ID"
echo "  PYTHONPATH=$PYTHONPATH"

cd "$UWYK_ROOT/RealCauseEval"

case "$MODE" in
    predictive)
        CKPT="$PRED_CKPT_DIR/$PRED_CKPT_FILE"
        CFG="$PRED_CKPT_DIR/$PRED_CONFIG_FILE"
        [ -f "$CKPT" ] || { echo "ERROR: $CKPT missing" >&2; ls "$PRED_CKPT_DIR" >&2 || true; exit 1; }
        [ -f "$CFG"  ] || { echo "ERROR: $CFG missing"  >&2; ls "$PRED_CKPT_DIR" >&2 || true; exit 1; }
        echo "  checkpoint: $CKPT"
        echo "  config:     $CFG"

        # Replace the two hardcoded upstream paths inside the patched Predictive
        # script with our real ones (no ordering / race on shared files).
        ESC_CKPT=$(printf '%s' "$CKPT" | sed 's|[\/&]|\\&|g')
        ESC_CFG=$( printf '%s' "$CFG"  | sed 's|[\/&]|\\&|g')
        sed -i "s|<REPO_ROOT_REPLACED>/experiments/FirstTests/checkpoints/simple_pfn_16691166.0_tabpfn_benchmark/step_55000.pt|$ESC_CKPT|" \
            "$INSTALL_DIR/$PRED_PATCHED" 2>/dev/null || true
        sed -i "s|$ESC_ROOT/experiments/FirstTests/checkpoints/simple_pfn_16691166.0_tabpfn_benchmark/step_55000.pt|$ESC_CKPT|" \
            "$INSTALL_DIR/$PRED_PATCHED"
        sed -i "s|$ESC_ROOT/experiments/FirstTests/checkpoints/simple_pfn_16691166.0_tabpfn_benchmark/basic_16691166.0.yaml|$ESC_CFG|" \
            "$INSTALL_DIR/$PRED_PATCHED"

        python -u "run_baselines/$PRED_PATCHED" \
            --dataset "$DATASET" --model "$MODEL_NAME" --exp_name "$EXP_NAME"
        ;;

    noanc|anc)
        CKPT="$UWYK_CKPT_DIR/$CKPT_FILE"
        CFG="$UWYK_CKPT_DIR/$CONFIG_FILE"
        [ -f "$CKPT" ] || { echo "ERROR: $CKPT missing" >&2; ls "$UWYK_CKPT_DIR" >&2 || true; exit 1; }
        [ -f "$CFG"  ] || { echo "ERROR: $CFG missing"  >&2; ls "$UWYK_CKPT_DIR" >&2 || true; exit 1; }
        echo "  checkpoint: $CKPT"
        echo "  config:     $CFG"

        FLAG=""
        [ "$MODE" = "noanc" ] && FLAG="--all_unknown"

        python -u "run_baselines/$DOFM_PATCHED" \
            --dataset         "$DATASET" \
            --model           "$MODEL_NAME" \
            --exp_name        "$EXP_NAME" \
            --checkpoint_path "$CKPT" \
            --config_path     "$CFG" \
            $FLAG
        ;;

    *)
        echo "ERROR: MODE must be predictive | noanc | anc (got: $MODE)" >&2
        exit 2
        ;;
esac

# Best-effort cleanup so we don't leave patched siblings lying around forever.
rm -f "$INSTALL_DIR/$EVAL_PATCHED" "$INSTALL_DIR/$DOFM_PATCHED" "$INSTALL_DIR/$PRED_PATCHED"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done — results in $UWYK_ROOT/RealCauseEval/results/$EXP_NAME"
