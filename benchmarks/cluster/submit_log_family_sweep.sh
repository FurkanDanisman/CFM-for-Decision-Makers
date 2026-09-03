#!/bin/bash
# Focused sweep: 5 configs × up to 3 checkpoints.
#
# Configs (per user):
#   1) log                 STD_MODE=log         K_NN=0
#   2) log + per_arm       STD_MODE=log_per_arm K_NN=0
#   3) log + winsor        STD_MODE=log_winsor  K_NN=0
#   4) log + k-NN          STD_MODE=log         K_NN=200
#   5) quantile-normal     STD_MODE=quantile    K_NN=0
#
# Every job also reports EM-mean alongside raw/full — 2D reports raw + full + em,
# 1D reports raw + em (no full — 1D has no 9-region mixture).
#
# Usage:
#   CKPT_WIDE=/path/to/wide/step_XXXXX.pt \
#   CKPT_TIGHT=/path/to/tight/step_XXXXX.pt \
#   CKPT_1D=/path/to/1d/step_XXXXX.pt \
#   MAX_REAL=10 \
#   bash submit_log_family_sweep.sh
#
# Any of CKPT_WIDE, CKPT_TIGHT, CKPT_1D may be blank — that model tier is skipped.
set -euo pipefail

CKPT_WIDE="${CKPT_WIDE:-}"
CKPT_TIGHT="${CKPT_TIGHT:-}"
CKPT_1D="${CKPT_1D:-}"
MAX_REAL="${MAX_REAL:-}"

SB2D="$(cd "$(dirname "$0")" && pwd)/submit_eval_cpfn2d_ihdp_stdmodes.sbatch"
SB1D="$(cd "$(dirname "$0")" && pwd)/submit_eval_causalpfn_1d_stdmodes.sbatch"

# (STD_MODE, K_NN, label)
CONFIGS=(
  "log:0:log"
  "log_per_arm:0:log+per_arm"
  "log_winsor:0:log+winsor"
  "log:200:log+kNN200"
  "quantile:0:quantile"
)

submit_2d() {
    local ckpt="$1" tag="$2"
    [ -z "$ckpt" ] && return 0
    for cfg in "${CONFIGS[@]}"; do
        IFS=':' read -r MODE KNN LABEL <<< "$cfg"
        echo "→ [2D:$tag]  $LABEL   ($(basename $(dirname $(dirname "$ckpt")))/$(basename "$ckpt"))"
        sbatch --export=ALL,CKPT="$ckpt",STD_MODE="$MODE",K_NN="$KNN",MAX_REAL="$MAX_REAL" "$SB2D"
    done
}

submit_1d() {
    local ckpt="$1"
    [ -z "$ckpt" ] && return 0
    for cfg in "${CONFIGS[@]}"; do
        IFS=':' read -r MODE KNN LABEL <<< "$cfg"
        echo "→ [1D]  $LABEL   ($(basename "$ckpt"))"
        sbatch --export=ALL,CKPT="$ckpt",STD_MODE="$MODE",K_NN="$KNN",MAX_REAL="$MAX_REAL" "$SB1D"
    done
}

submit_2d "$CKPT_WIDE"  "wide"
submit_2d "$CKPT_TIGHT" "tight"
submit_1d "$CKPT_1D"

NTIER=0
[ -n "$CKPT_WIDE" ]  && NTIER=$((NTIER+1))
[ -n "$CKPT_TIGHT" ] && NTIER=$((NTIER+1))
[ -n "$CKPT_1D" ]    && NTIER=$((NTIER+1))
echo "─── submitted ${#CONFIGS[@]} × $NTIER = $((${#CONFIGS[@]} * NTIER)) jobs."
