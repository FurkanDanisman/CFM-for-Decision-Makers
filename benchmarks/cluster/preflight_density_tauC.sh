#!/bin/bash
# Preflight for submit_density_tauC.sbatch — run on a Killarney LOGIN node.
#
# The sbatch script defaults every path to Luke's tree
# (/home/lukez/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers) and exits
# FATAL on the first missing input. This checks the same inputs, but reports
# ALL of them at once so you fix everything in one pass instead of one
# resubmission per missing file.
#
# Usage:
#   export DEPLOY_ROOT=/scratch/furkanbd/rpfn_bench_kit
#   export REPO=$DEPLOY_ROOT/R-PFN
#   MODEL_FAMILY=all bash $REPO/benchmarks/cluster/preflight_density_tauC.sh

DEPLOY_ROOT="${DEPLOY_ROOT:-$SCRATCH/rpfn_bench_kit}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
MODEL_FAMILY="${MODEL_FAMILY:-all}"

UWYK="${UWYK:-$DEPLOY_ROOT/external/uwyk}"
DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
VENV="${VENV:-$DEPLOY_ROOT/venv}"
CKPT="${CKPT:-$REPO/Required_checkpoints/graph2d_step_50000.pt}"
DOPFN_JOINT_CKPT="${DOPFN_JOINT_CKPT:-$REPO/Required_checkpoints/dopfn_bb_j10_step_150000.pt}"
UWYK_CKPT_DIR="$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"
UWYK_CKPT="${UWYK_CKPT:-$UWYK_CKPT_DIR/best_model.pt}"
UWYK_CFG="${UWYK_CFG:-$UWYK_CKPT_DIR/best_model_config.yaml}"
ACIC_CACHE_DIR="${ACIC_CACHE_DIR:-$REPO/data/acic_cache}"

fail=0
chk() {  # chk <label> <path> [dir]
    local kind="${3:-f}"
    if [ "$kind" = d ] && [ -d "$2" ]; then printf '  OK    %-22s %s\n' "$1" "$2"
    elif [ "$kind" = f ] && [ -f "$2" ]; then
        printf '  OK    %-22s %s  (%s)\n' "$1" "$2" "$(du -h "$2" 2>/dev/null | cut -f1)"
    else printf '  MISS  %-22s %s\n' "$1" "$2"; fail=$((fail+1)); fi
}

echo "MODEL_FAMILY=$MODEL_FAMILY"
echo "DEPLOY_ROOT=$DEPLOY_ROOT"
echo "REPO=$REPO"
echo
echo "--- environment ---"
chk venv-activate  "$VENV/bin/activate"
chk repo           "$REPO" d
chk eval-script    "$REPO/benchmarks/eval_graph2d/eval_density_tauC.py"
chk shims          "$REPO/benchmarks/uwyk_table1/shims" d
chk acic-x.csv     "$ACIC_CACHE_DIR/x.csv"

if [ "$MODEL_FAMILY" = uwyk ] || [ "$MODEL_FAMILY" = all ]; then
    echo "--- uwyk / joint-2d ---"
    chk graph2d-ckpt "$CKPT"
    chk uwyk-ckpt    "$UWYK_CKPT"
    chk uwyk-cfg     "$UWYK_CFG"
fi
if [ "$MODEL_FAMILY" = dopfn ] || [ "$MODEL_FAMILY" = all ]; then
    echo "--- dopfn ---"
    chk dopfn-root       "$DOPFN_ROOT" d
    chk dopfn-joint-ckpt "$DOPFN_JOINT_CKPT"
    chk dopfn-model.pkl  "$DOPFN_ROOT/artifacts/dopfn_model.pkl"
    chk dopfn-config.pkl "$DOPFN_ROOT/artifacts/dopfn_config.pkl"
    chk dopfn-backbone   "$DOPFN_ROOT/artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt"
fi

echo
echo "--- slurm ---"
sacctmgr -nP show assoc user="$USER" format=account,partition 2>/dev/null \
    | sed 's/^/  /' | head || echo "  (sacctmgr unavailable)"

echo
if [ "$fail" -eq 0 ]; then
    echo "PREFLIGHT PASS — safe to submit."
else
    echo "PREFLIGHT FAIL — $fail missing input(s) above; sbatch would exit FATAL on the first."
fi
exit $fail
