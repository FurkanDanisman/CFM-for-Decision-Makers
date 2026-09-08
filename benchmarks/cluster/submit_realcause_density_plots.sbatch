#!/bin/bash
#SBATCH --job-name=rc-density-plots
#SBATCH --output=logs_rc_density_metrics/plots_%A_%a.out
#SBATCH --error=logs_rc_density_metrics/plots_%A_%a.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-1

# 2-task array: 0=IHDP, 1=ACIC. Writes plot_densities_<DATASET>.png to
# $OUT_2D. Each PNG: 6 methods × N queries subplots showing truth vs
# estimated p(τ|x_q). Only IHDP+ACIC (analytic Gaussian truth).

set -euo pipefail
mkdir -p logs_rc_density_metrics

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
OUT_1D="${OUT_1D:?OUT_1D required}"
OUT_2D="${OUT_2D:?OUT_2D required}"
CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
REALIZATION="${REALIZATION:-0}"
N_QUERIES="${N_QUERIES:-6}"

case "$SLURM_ARRAY_TASK_ID" in
    0) DATASET=IHDP ;;
    1) DATASET=ACIC ;;
    *) echo "invalid SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
OUT_PNG="$OUT_2D/plot_densities_${DATASET}_r${REALIZATION}.png"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] $DATASET r=$REALIZATION → $OUT_PNG"
python -u "$REPO/realcause_eval/plot_density_examples.py" \
    --dataset     "$DATASET" \
    --root-1d     "$OUT_1D" \
    --root-2d     "$OUT_2D" \
    --causalpfn   "$CAUSALPFN" \
    --repo        "$REPO" \
    --realization "$REALIZATION" \
    --n-queries   "$N_QUERIES" \
    --out         "$OUT_PNG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done $DATASET"
