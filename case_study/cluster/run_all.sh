#!/bin/bash
# End-to-end case-study evaluation on the cluster:
#   1. generate the npz (CPU job)
#   2. submit the model × context × case eval sweep (GPU array jobs; waits on 1)
#   3. print the aggregate command to run once the sweep finishes
#
# Usage (on Killarney, from your deploy root):
#   export DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit
#   cd $DEPLOY_ROOT
#   bash $DEPLOY_ROOT/R-PFN/case_study/cluster/run_all.sh
#
# Env vars:
#   CATE_SHIFT  effect shift beta baked into the data. Default 2.
#   CONTEXTS    context sizes. Default "200 500 1000".
#   MODELS      model subset (see 02_submit_eval.sh). Default all.
#   DATA_ROOT / SWEEP  override the default scratch locations.
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:?DEPLOY_ROOT required (e.g. \$SCRATCH/rpfn_bench_kit)}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
CATE_SHIFT="${CATE_SHIFT:-2}"
CONTEXTS="${CONTEXTS:-200 500 1000}"
_tag="$(printf '%+g' "$CATE_SHIFT")"
DATA_ROOT="${DATA_ROOT:-$DEPLOY_ROOT/case_study_data/shift${_tag}}"
SWEEP="${SWEEP:-$DEPLOY_ROOT/results_case_study/shift${_tag}}"

cd "$DEPLOY_ROOT"     # logs_*/ are created relative to the submit dir

echo "[run_all] beta=$CATE_SHIFT  contexts=[$CONTEXTS]"
echo "          DATA_ROOT=$DATA_ROOT"
echo "          SWEEP=$SWEEP"

# 1. generate (CPU) — capture the jobid so eval can depend on it.
GEN_JID=$(DATA_ROOT="$DATA_ROOT" CATE_SHIFT="$CATE_SHIFT" CONTEXTS="$CONTEXTS" \
    sbatch --parsable "$REPO/case_study/cluster/01_generate.sbatch")
echo "[run_all] submitted generate job $GEN_JID"

# 2. eval sweep — each array job waits for the generate job to succeed.
DEP="$GEN_JID" CONTEXTS="$CONTEXTS" MODELS="${MODELS:-}" \
    bash "$REPO/case_study/cluster/02_submit_eval.sh" "$DATA_ROOT" "$SWEEP"

# 3. aggregate (run after the sweep completes).
echo
echo "[run_all] when the sweep finishes, aggregate with:"
echo "  python $REPO/case_study/cluster/03_aggregate.py \\"
echo "      --sweep $SWEEP --contexts $CONTEXTS --out $SWEEP/summary.csv"
