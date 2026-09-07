#!/bin/bash
#SBATCH --job-name=uwyk-repro-noanc
#SBATCH --output=logs_uwyk_reproduce/noanc_%A_%a.out
#SBATCH --error=logs_uwyk_reproduce/noanc_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4

# Reproduces paper Table 3 "UWYK No Ancestral Info." row for all 5 datasets by
# running UWYK's own reproduce-branch scripts UNMODIFIED. Per their doc
# (REPRODUCE_REALCAUSE_RESULTS.md, section 3):
#
#   IHDP  : dofm_no_clustering.py --dataset IHDP  --graph_mode all_unknown
#   ACIC  : dofm_no_clustering.py --dataset ACIC  --graph_mode all_unknown
#   CPS   : dofm_no_clustering.py --dataset CPS   --graph_mode all_unknown
#   PSID  : dofm_no_clustering.py --dataset PSID  --graph_mode all_unknown
#   PSIDb : dofm_psid_balanced.py  --dataset PSID  --graph_mode all_unknown
#
# Both scripts default to the checkpoint at
#   experiments/checkpoints/full_conditioned_model/
#     final_earlytest_full_conditioning_16773252.0/best_model.pt
# so no --checkpoint_path/--config_path needed — pass none, use their default,
# stay byte-identical to their reproduce recipe.
#
# 5-task array — 0:IHDP  1:ACIC  2:CPS  3:PSID  4:PSID_bal.
# Per-realization pkls land at $UWYK_REPRO/results/$EXP_NAME/.
# EXP_NAME defaults per-dataset (see below) so aggregate_results.py picks
# them up automatically.
#
# Submit:  sbatch benchmarks/cluster/submit_uwyk_reproduce_noanc.sbatch
# Then :   see aggregate_uwyk_reproduce.sh below

set -euo pipefail
mkdir -p logs_uwyk_reproduce

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found under $DEPLOY_ROOT" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
UWYK_REPRO="${UWYK_REPRO:-$DEPLOY_ROOT/external/uwyk_reproduce}"

[ -d "$UWYK_REPRO" ] || { echo "FATAL: UWYK_REPRO not found: $UWYK_REPRO" >&2; exit 1; }
[ -f "$UWYK_REPRO/RealCauseEval/run_baselines/dofm_no_clustering.py" ] || { echo "FATAL: dofm_no_clustering.py missing" >&2; exit 1; }
[ -f "$UWYK_REPRO/RealCauseEval/run_baselines/dofm_psid_balanced.py" ] || { echo "FATAL: dofm_psid_balanced.py missing" >&2; exit 1; }
[ -d "$UWYK_REPRO/RealCauseEval/CausalPFN" ] || { echo "FATAL: RealCauseEval/CausalPFN not vendored (needed for benchmark loaders)" >&2; exit 1; }

# Their default checkpoint — sanity-check it exists before we launch anything.
CKPT_DIR="$UWYK_REPRO/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"
[ -f "$CKPT_DIR/best_model.pt" ]        || { echo "FATAL: their default ckpt missing at $CKPT_DIR/best_model.pt — git lfs pull?" >&2; exit 1; }
[ -f "$CKPT_DIR/best_model_config.yaml" ] || { echo "FATAL: their default config missing at $CKPT_DIR/best_model_config.yaml" >&2; exit 1; }

case "$SLURM_ARRAY_TASK_ID" in
    0) SCRIPT=dofm_no_clustering.py  DATASET=IHDP     MODEL=dofm_noclust        EXP=dofm_noclust_all_unknown        ;;
    1) SCRIPT=dofm_no_clustering.py  DATASET=ACIC     MODEL=dofm_noclust        EXP=dofm_noclust_all_unknown        ;;
    2) SCRIPT=dofm_no_clustering.py  DATASET=CPS      MODEL=dofm_noclust        EXP=dofm_noclust_all_unknown        ;;
    3) SCRIPT=dofm_no_clustering.py  DATASET=PSID     MODEL=dofm_noclust        EXP=dofm_noclust_all_unknown        ;;
    4) SCRIPT=dofm_psid_balanced.py  DATASET=PSID     MODEL=dofm_psid_balanced  EXP=dofm_psid_balanced_all_unknown  ;;
    *) echo "invalid task id: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

# eval.py does `from CausalPFN.benchmarks import ...`; the scripts themselves
# sys.path.insert() RealCauseEval/, so CausalPFN (vendored inside) resolves.
# Keep the shim dir on PYTHONPATH so stray optional imports (faiss etc.) get
# stubbed rather than failing.
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] UWYK unmodified reproduce  script=$SCRIPT  DATASET=$DATASET  EXP=$EXP"
echo "  UWYK_REPRO=$UWYK_REPRO"
echo "  results dir will be: $UWYK_REPRO/results/$EXP/"

cd "$UWYK_REPRO"
python -u "RealCauseEval/run_baselines/$SCRIPT" \
    --dataset    "$DATASET" \
    --model      "$MODEL" \
    --exp_name   "$EXP" \
    --graph_mode all_unknown

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done — $DATASET → $UWYK_REPRO/results/$EXP/"
