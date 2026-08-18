#!/bin/bash
#SBATCH --job-name=dump-margs-r0
#SBATCH --account=aip-rgrosse
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/dump_margs_%j.out
#SBATCH --error=logs/dump_margs_%j.err

# Small helper: run dump_marginals_for_R.py inside a proper job so the
# model forward + IHDP truth generation doesn't happen on the login node.
# Writes a JSON blob under benchmarks/l2_ihdp/ that can then be git-added
# for local R-vs-Python MALC comparison.

set -euo pipefail
: "${DEPLOY_ROOT:?DEPLOY_ROOT must be set}"
: "${REALIZATION:=0}"
: "${OUT_JSON:=benchmarks/l2_ihdp/ihdp_marginals_r${REALIZATION}.json}"

cd "$DEPLOY_ROOT/R-PFN"
# Match the venv activation the other sbatches use
if [[ -f .venv/bin/activate ]]; then
    . .venv/bin/activate
elif [[ -f venv/bin/activate ]]; then
    . venv/bin/activate
fi

echo "[$(date)] dump r=${REALIZATION} -> ${OUT_JSON}"
python benchmarks/l2_ihdp/dump_marginals_for_R.py \
    --repo "$DEPLOY_ROOT/R-PFN" \
    --dopfn "$DEPLOY_ROOT/external/dopfn" \
    --causalpfn "$DEPLOY_ROOT/external/causalpfn" \
    --checkpoint-dopfn-bb "$DEPLOY_ROOT/checkpoints_dopfn_backbone_realj10/step_150000.pt" \
    --realization "$REALIZATION" \
    --out "$OUT_JSON"

echo "[$(date)] DONE  size=$(du -h "$OUT_JSON" | cut -f1)"
