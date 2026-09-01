#!/bin/bash
# Install step-checkpoint patches into the CausalPFN deploy tree.
#
# Usage:
#   DEPLOY_ROOT=/path/to/deploy ./install.sh
# or (auto-detect from PWD):
#   ./install.sh
#
# Idempotent: original files are backed up as *.beforestepckpt on first run;
# re-running just re-copies the patched files.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -d "$DEPLOY_ROOT/external/causalpfn" ]; then
    # try climbing
    for _ in 1 2 3; do
        [ -d "$DEPLOY_ROOT/external/causalpfn" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
[ -d "$CAUSALPFN" ] || { echo "FATAL: causalpfn deploy tree not found at $CAUSALPFN" >&2; exit 1; }

TRAINER_DST="$CAUSALPFN/src/causalpfn/training/trainer.py"
CHECKPOINT_DST="$CAUSALPFN/src/causalpfn/training/callbacks/checkpoint.py"
TRAIN_ENTRY_DST="$CAUSALPFN/train.py"

for dst in "$TRAINER_DST" "$CHECKPOINT_DST" "$TRAIN_ENTRY_DST"; do
    [ -f "$dst" ] || { echo "FATAL: $dst does not exist" >&2; exit 1; }
    if [ ! -f "${dst}.beforestepckpt" ]; then
        echo "[install] backing up $dst → ${dst}.beforestepckpt"
        cp "$dst" "${dst}.beforestepckpt"
    fi
done

echo "[install] copying patched trainer.py    → $TRAINER_DST"
cp "$HERE/trainer.py" "$TRAINER_DST"

echo "[install] copying patched checkpoint.py → $CHECKPOINT_DST"
cp "$HERE/checkpoint.py" "$CHECKPOINT_DST"

echo "[install] copying patched train.py      → $TRAIN_ENTRY_DST"
cp "$HERE/train.py" "$TRAIN_ENTRY_DST"

echo "[install] OK. To revert:"
echo "  cp ${TRAINER_DST}.beforestepckpt    ${TRAINER_DST}"
echo "  cp ${CHECKPOINT_DST}.beforestepckpt ${CHECKPOINT_DST}"
echo "  cp ${TRAIN_ENTRY_DST}.beforestepckpt ${TRAIN_ENTRY_DST}"
