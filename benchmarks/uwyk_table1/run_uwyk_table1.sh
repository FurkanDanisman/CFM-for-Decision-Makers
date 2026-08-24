#!/usr/bin/env bash
# Run one row × one dataset of UWYK's Table 1.
#
# Three rows (all from ArikReuter/Graphs4CausalFoundationModels, RealCauseEval/):
#
#   predictive  — S-learner on an unconditional model
#                 script: run_baselines/predmodel_Slearner_full_context.py
#                 ckpt:   simple_pfn_16691166.0_tabpfn_benchmark/step_55000.pt
#                         (upstream default; NOT in the public release — we
#                         instead point this at no_graph_conditioning/unconditional/
#                         best_model.pt if it can be loaded by SimplePFNSklearn.
#                         If the load fails, use MODE=noanc as a fallback for
#                         the "unconditional" row — same underlying idea.)
#
#   noanc       — full-conditioning model with adjacency all-zero (--all_unknown)
#                 script: run_baselines/dofm_full_conditioning.py --all_unknown
#                 ckpt:   full_conditioned_model/…16773252.0/best_model.pt
#
#   anc         — full-conditioning model with the true adjacency
#                 script: run_baselines/dofm_full_conditioning.py
#                 ckpt:   full_conditioned_model/…16773252.0/best_model.pt
#
# Both dofm rows share ONE checkpoint; only the --all_unknown flag differs.
#
# Usage:
#   ./run_uwyk_table1.sh <DATASET> <MODE> [<CKPT_TAG>]
#     DATASET   IHDP | ACIC | CPS | PSID | all
#     MODE      predictive | noanc | anc
#     CKPT_TAG  optional label added to exp_name (default: yyyymmdd_HHMMSS)
#
# Env overrides:
#   DEPLOY_ROOT / UWYK_ROOT
#   UWYK_CKPT_DIR        graph-conditioned ckpt dir (default full_conditioning_16773252.0)
#   CKPT_FILE            default: best_model.pt
#   CONFIG_FILE          default: config.yaml
#   PRED_CKPT_DIR        Predictive-row ckpt dir (default no_graph_conditioning/unconditional)
#   PRED_CKPT_FILE       default: best_model.pt
#   PRED_CONFIG_FILE     default: config.yaml

set -euo pipefail

DATASET="${1:?dataset required (IHDP|ACIC|CPS|PSID|all)}"
MODE="${2:?mode required (predictive|noanc|anc)}"
CKPT_TAG="${3:-$(date +%Y%m%d_%H%M%S)}"

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
UWYK_ROOT="${UWYK_ROOT:-$DEPLOY_ROOT/external/uwyk}"

UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$UWYK_ROOT/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
CKPT_FILE="${CKPT_FILE:-best_model.pt}"
CONFIG_FILE="${CONFIG_FILE:-config.yaml}"

PRED_CKPT_DIR="${PRED_CKPT_DIR:-$UWYK_ROOT/experiments/checkpoints/no_graph_conditioning/unconditional}"
PRED_CKPT_FILE="${PRED_CKPT_FILE:-best_model.pt}"
PRED_CONFIG_FILE="${PRED_CONFIG_FILE:-config.yaml}"

# Patch <REPO_ROOT> placeholders in the two upstream scripts and stage
# alongside their real siblings so relative imports still resolve.
INSTALL_DIR="$UWYK_ROOT/RealCauseEval/run_baselines"
ESC_ROOT=$(printf '%s' "$UWYK_ROOT" | sed 's|/|\\/|g')

patch_and_stage() {
    local src="$1"; local dst="$2"
    cp "$src" "$dst"
    sed -i.bak "s/<REPO_ROOT>/$ESC_ROOT/g" "$dst"
}

patch_and_stage "$INSTALL_DIR/eval.py"                        "$INSTALL_DIR/eval__patched.py"
patch_and_stage "$INSTALL_DIR/dofm_full_conditioning.py"      "$INSTALL_DIR/dofm_full_conditioning__patched.py"
patch_and_stage "$INSTALL_DIR/predmodel_Slearner_full_context.py" "$INSTALL_DIR/predmodel_Slearner__patched.py"

# Rewrite eval import in the patched scripts so they see the patched eval
sed -i.bak 's/from run_baselines.eval import/from run_baselines.eval__patched import/g' \
    "$INSTALL_DIR/dofm_full_conditioning__patched.py" \
    "$INSTALL_DIR/predmodel_Slearner__patched.py"

MODEL_NAME="uwyk_table1_${MODE}"
EXP_NAME="table1_${MODE}_${CKPT_TAG}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching Table 1 row"
echo "  dataset=$DATASET  mode=$MODE  ckpt_tag=$CKPT_TAG"

cd "$UWYK_ROOT/RealCauseEval"

case "$MODE" in
    predictive)
        CKPT="$PRED_CKPT_DIR/$PRED_CKPT_FILE"
        CFG="$PRED_CKPT_DIR/$PRED_CONFIG_FILE"
        [ -f "$CKPT" ] || { echo "ERROR: $CKPT missing" >&2; ls "$PRED_CKPT_DIR" >&2 || true; exit 1; }
        [ -f "$CFG"  ] || { echo "ERROR: $CFG missing"  >&2; ls "$PRED_CKPT_DIR" >&2 || true; exit 1; }
        echo "  checkpoint: $CKPT"
        echo "  config:     $CFG"
        # The upstream S-learner script HARDCODES checkpoint/config paths (no
        # CLI). We monkeypatch by exporting env vars the patched sed captures.
        # Simplest: sed-substitute the two paths in the patched script.
        ESC_CKPT=$(printf '%s' "$CKPT" | sed 's|/|\\/|g')
        ESC_CFG=$( printf '%s' "$CFG"  | sed 's|/|\\/|g')
        sed -i.bak2 "s|\"$ESC_ROOT/experiments/FirstTests/checkpoints/simple_pfn_16691166.0_tabpfn_benchmark/step_55000.pt\"|\"$CKPT\"|" \
            "$INSTALL_DIR/predmodel_Slearner__patched.py"
        sed -i.bak3 "s|\"$ESC_ROOT/experiments/FirstTests/checkpoints/simple_pfn_16691166.0_tabpfn_benchmark/basic_16691166.0.yaml\"|\"$CFG\"|" \
            "$INSTALL_DIR/predmodel_Slearner__patched.py"

        python -u run_baselines/predmodel_Slearner__patched.py \
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

        python -u run_baselines/dofm_full_conditioning__patched.py \
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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done — results in $UWYK_ROOT/RealCauseEval/results/$EXP_NAME"
