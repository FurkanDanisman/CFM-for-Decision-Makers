#!/bin/bash
#SBATCH --job-name=examext-E
#SBATCH --account=def-rgrosse
#SBATCH --output=logs_exam_ext/E_%A_%a.out
#SBATCH --error=logs_exam_ext/E_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Experiment E of the exam extension (exp_E.py): S-learner (TabPFN v2) and causal forest DML
# on the exact data sets of Experiments B and C. One array task per (experiment, condition, case).
#
#   1) once:  SETUP=1 sbatch comprehensive_exam_extension/cluster/exp_E.sbatch
#             installs tabpfn + econml into the venv (aborts if torch would change) and
#             downloads the TabPFN v2 regressor weights into TABPFN_MODEL_CACHE_DIR.
#   2) then:  sbatch --array=0-165 comprehensive_exam_extension/cluster/exp_E.sbatch
#             (task count: python exp_E.py --list-tasks)
#
# Env vars:
#   OUT_ROOT                output dir (default: $SCRATCH/exam_ext/exp_E); small CSVs only
#   TABPFN_MODEL_CACHE_DIR  TabPFN weights (default: $SCRATCH/tabpfn_cache)
# Finished tasks are skipped (the CSV exists), so a resubmit only redoes what is missing.
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
OUT_ROOT="${OUT_ROOT:-$SCRATCH/exam_ext/exp_E}"
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-$SCRATCH/tabpfn_cache}"
source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [ "${SETUP:-0}" = "1" ]; then
    echo "[$(date '+%F %T')] setup: dry run"
    plan="$(pip install --dry-run tabpfn econml 2>&1 | grep -i 'would install' || true)"
    echo "$plan"
    if echo "$plan" | grep -q -E '(^| )torch-'; then
        echo "FATAL: installing tabpfn/econml would change torch; not installing" >&2; exit 1
    fi
    pip install tabpfn econml
    mkdir -p "$TABPFN_MODEL_CACHE_DIR"
    python - <<'EOF'
import numpy as np
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
reg = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu")
X = np.random.randn(50, 2); reg.fit(X, X[:, 0]); print("TabPFN v2 ok:", reg.predict(X[:2]))
import econml; print("econml", econml.__version__)
EOF
    ls -la "$TABPFN_MODEL_CACHE_DIR"
    exit 0
fi

echo "[$(date '+%F %T')] exp_E task=${SLURM_ARRAY_TASK_ID:-0} out=$OUT_ROOT"
python -u "$REPO/comprehensive_exam_extension/exp_E.py" --task-id "${SLURM_ARRAY_TASK_ID:-0}" --out "$OUT_ROOT"
echo "[$(date '+%F %T')] done"
