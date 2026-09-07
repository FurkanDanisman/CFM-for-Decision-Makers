#!/bin/bash
#SBATCH --job-name=uwyk-repro-psidbal
#SBATCH --output=logs_uwyk_reproduce/psidbal_%j.out
#SBATCH --error=logs_uwyk_reproduce/psidbal_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Runs UWYK's dofm_psid_balanced.py from the reproduce-branch UNMODIFIED,
# so PSID_bal must match paper Table 3 (No-Anc row) exactly (21896).
#
# No wrapper, no our-side preprocessing — just their script, their
# PreprocessingGraphConditionedPFN, their eval loop.
#
# Submit:  sbatch benchmarks/cluster/submit_uwyk_reproduce_psidbal.sbatch

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
CKPT_DIR="$UWYK_REPRO/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"
CKPT="${CKPT:-$CKPT_DIR/best_model.pt}"
CONFIG="${CONFIG:-$CKPT_DIR/best_model_config.yaml}"

[ -d "$UWYK_REPRO" ]                        || { echo "FATAL: UWYK_REPRO not found: $UWYK_REPRO" >&2; exit 1; }
[ -f "$UWYK_REPRO/RealCauseEval/run_baselines/dofm_psid_balanced.py" ] || { echo "FATAL: dofm_psid_balanced.py not found" >&2; exit 1; }
[ -f "$CKPT" ]                              || { echo "FATAL: ckpt not found: $CKPT — git lfs pull?" >&2; exit 1; }
[ -f "$CONFIG" ]                            || { echo "FATAL: config not found: $CONFIG" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

# eval.py does `from CausalPFN.benchmarks import ...` and the script inserts
# $UWYK_REPRO/RealCauseEval onto sys.path; CausalPFN is vendored there. Also
# keep the shim dir so any lazy imports (faiss etc.) get stubbed out.
export PYTHONPATH="$UWYK_REPRO/RealCauseEval:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

# UWYK writes per-realization pkl files to $UWYK_REPRO/results/$EXP_NAME/
# and prints the summary at the end. --graph_mode all_unknown → No-Anc row.
EXP_NAME="${EXP_NAME:-uwyk_noanc_psidbal_$(date +%s)}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] UWYK unmodified reproduce  PSID_bal (No-Anc)"
echo "  UWYK_REPRO=$UWYK_REPRO"
echo "  CKPT=$CKPT"
echo "  CONFIG=$CONFIG"
echo "  EXP_NAME=$EXP_NAME"
echo "  results dir: $UWYK_REPRO/results/$EXP_NAME/"

cd "$UWYK_REPRO"
python -u RealCauseEval/run_baselines/dofm_psid_balanced.py \
    --dataset PSID \
    --model   uwyk_noanc_psidbal \
    --exp_name "$EXP_NAME" \
    --checkpoint_path "$CKPT" \
    --config_path "$CONFIG" \
    --graph_mode all_unknown

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done — EXP_NAME=$EXP_NAME"
