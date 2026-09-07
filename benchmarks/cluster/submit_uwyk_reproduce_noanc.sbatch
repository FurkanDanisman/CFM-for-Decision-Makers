#!/bin/bash
#SBATCH --job-name=uwyk-repro-noanc
#SBATCH --output=logs_uwyk_reproduce/noanc_%A_%a.out
#SBATCH --error=logs_uwyk_reproduce/noanc_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4

# Reproduces the "UWYK No Ancestral Info." row of Table 3 by running UWYK's
# reproduce-branch scripts UNMODIFIED. All 5 datasets write into the SAME
# results folder so summarizing needs only one path.
#
#   0 IHDP     → dofm_no_clustering.py  --dataset IHDP  --graph_mode all_unknown
#   1 ACIC     → dofm_no_clustering.py  --dataset ACIC  --graph_mode all_unknown
#   2 CPS      → dofm_no_clustering.py  --dataset CPS   --graph_mode all_unknown
#   3 PSID     → dofm_no_clustering.py  --dataset PSID  --graph_mode all_unknown
#   4 PSID_bal → dofm_psid_balanced.py  --dataset PSID  --graph_mode all_unknown
#
# Both scripts use their default checkpoint (best_model.pt in the
# full_conditioned_model dir). Pkls land at
#   $UWYK_REPRO/RealCauseEval/results/$EXP_NAME/{dofm_noclust,dofm_psid_balanced}_<DATASET>_<r>
# then:
#   python R-PFN/realcause_eval/uwyk.py summary \
#          --uwyk-repro $DEPLOY_ROOT/external/uwyk_reproduce
# prints the 5-column table.

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
EXP_NAME="${EXP_NAME:-uwyk_realcause}"
SEED="${SEED:-43}"   # seeds numpy so ACIC/CPS/PSID_unbal 1000-row subsample is deterministic

[ -d "$UWYK_REPRO" ] || { echo "FATAL: UWYK_REPRO not found: $UWYK_REPRO" >&2; exit 1; }
[ -f "$UWYK_REPRO/RealCauseEval/run_baselines/dofm_no_clustering.py" ] || { echo "FATAL: dofm_no_clustering.py missing" >&2; exit 1; }
[ -f "$UWYK_REPRO/RealCauseEval/run_baselines/dofm_psid_balanced.py" ] || { echo "FATAL: dofm_psid_balanced.py missing" >&2; exit 1; }
[ -d "$UWYK_REPRO/RealCauseEval/CausalPFN" ] || { echo "FATAL: RealCauseEval/CausalPFN not vendored" >&2; exit 1; }

CKPT_DIR="$UWYK_REPRO/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"
[ -f "$CKPT_DIR/best_model.pt" ]          || { echo "FATAL: default ckpt missing at $CKPT_DIR/best_model.pt — git lfs pull?" >&2; exit 1; }
[ -f "$CKPT_DIR/best_model_config.yaml" ] || { echo "FATAL: default config missing at $CKPT_DIR/best_model_config.yaml" >&2; exit 1; }

case "$SLURM_ARRAY_TASK_ID" in
    0) DATASET=IHDP     ;;
    1) DATASET=ACIC     ;;
    2) DATASET=CPS      ;;
    3) DATASET=PSID     ;;
    4) DATASET=PSID_bal ;;
    *) echo "invalid task id: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] UWYK unmodified reproduce  DATASET=$DATASET  EXP_NAME=$EXP_NAME"
echo "  UWYK_REPRO=$UWYK_REPRO"
echo "  results dir: $UWYK_REPRO/RealCauseEval/results/$EXP_NAME/"

python -u "$REPO/realcause_eval/uwyk.py" run \
    --dataset    "$DATASET" \
    --uwyk-repro "$UWYK_REPRO" \
    --exp-name   "$EXP_NAME" \
    --repo       "$REPO" \
    --seed       "$SEED"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
