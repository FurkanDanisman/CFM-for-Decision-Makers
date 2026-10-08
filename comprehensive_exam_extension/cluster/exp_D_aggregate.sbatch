#!/bin/bash
#SBATCH --job-name=examext-Dagg
#SBATCH --account=def-rgrosse
#SBATCH --output=logs_exam_ext/Dagg_%j.out
#SBATCH --error=logs_exam_ext/Dagg_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Aggregate Experiment D (exp_D_aggregate.py) after both arrays have finished.
# Writes small CSVs and PNGs next to the npz files in OUT_ROOT.
#   sbatch comprehensive_exam_extension/cluster/exp_D_aggregate.sbatch
set -euo pipefail
mkdir -p logs_exam_ext

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found under $DEPLOY_ROOT" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
OUT_ROOT="${OUT_ROOT:-$SCRATCH/exam_ext/exp_D}"
source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1

echo "conv npz: $(find "$OUT_ROOT/conv" -name 'k*.npz' 2>/dev/null | wc -l) / 160"
echo "rate npz: $(find "$OUT_ROOT/rate" -name 'k*.npz' 2>/dev/null | wc -l) / 60"
python -u "$REPO/comprehensive_exam_extension/exp_D_aggregate.py" --res "$OUT_ROOT"
ls -la "$OUT_ROOT"/*.csv "$OUT_ROOT"/*.png
