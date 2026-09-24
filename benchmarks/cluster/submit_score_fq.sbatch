#!/bin/bash
#SBATCH --job-name=fq-score
#SBATCH --output=logs_fq/score_%j.out
#SBATCH --error=logs_fq/score_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#
# Score the fixed-query CI coverage for both settings, as a CPU job.
#
# Serial, this is ~13000 np.load calls on shared storage -- latency-bound, minutes
# to tens of minutes, and on a login node it is also everyone else's problem. With
# 16 workers it is a few minutes. Safe to rerun while the dumps are still going;
# each model's replicate count is reported so partial reads as partial.
#
#   sbatch --account=def-zhijing R-PFN/benchmarks/cluster/submit_score_fq.sbatch
#
# Env: same as score_fq.sh (ROOT_A/ROOT_B, A/B, CASE, CTX, QUERY, OUT_DIR)

set -uo pipefail
mkdir -p logs_fq

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }
# shellcheck disable=SC1091
source "$DEPLOY_ROOT/venv/bin/activate"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export CUDA_VISIBLE_DEVICES=

cd "$DEPLOY_ROOT"
echo "[$(date '+%F %T')] scoring with ${SLURM_CPUS_PER_TASK:-1} workers"
bash "$REPO/benchmarks/cluster/score_fq.sh"
echo "[$(date '+%F %T')] done"
