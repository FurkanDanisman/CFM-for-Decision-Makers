#!/bin/bash
#SBATCH --job-name=rc-density-metrics
#SBATCH --output=logs_rc_density_metrics/eval_%A_%a.out
#SBATCH --error=logs_rc_density_metrics/eval_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-1

# CPU-only 2-task array — one per Gaussian-truth dataset: IHDP, ACIC.
# Computes NLL / L2 / KL_fwd / KL_rev of each method's per-query CATE
# density against the analytic Gaussian truth (Y_do_t | x ~ N(μ_t, σ²)).
#
#   task 0: IHDP  (100 realizations × 75 queries)
#   task 1: ACIC  (10 realizations × ~200 queries)
#
# CPS / PSID / PSID_bal need a 100-CSV MC-stack truth pipeline that's not
# implemented yet — those datasets are skipped by the script.
#
# Env vars:
#   OUT_1D       required — root of 1D density dumps (…/results_1d_density)
#   OUT_2D       required — root of 2D density dumps (…/results_rc_2d_density)
#   CAUSALPFN    required — CausalPFN repo root
#   T            optional (default 4001)  — τ-grid resolution

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
T="${T:-4001}"

case "$SLURM_ARRAY_TASK_ID" in
    0) DATASET=IHDP ;;
    1) DATASET=ACIC ;;
    *) echo "invalid SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1

OUT_MD="$OUT_2D/density_metrics_${DATASET}.md"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] $DATASET  T=$T  → $OUT_MD"

python -u "$REPO/realcause_eval/eval_density_metrics.py" \
    --dataset   "$DATASET" \
    --root-1d   "$OUT_1D" \
    --root-2d   "$OUT_2D" \
    --causalpfn "$CAUSALPFN" \
    --repo      "$REPO" \
    --T         "$T" \
    --out-md    "$OUT_MD"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done  $DATASET"
