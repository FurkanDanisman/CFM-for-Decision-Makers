#!/bin/bash
#SBATCH --job-name=examext-M
#SBATCH --account=def-rgrosse
#SBATCH --output=logs_exam_ext/M_%x_%j.out
#SBATCH --error=logs_exam_ext/M_%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G

# Model comparison on the Observed Confounder case (exp_M.py): the paper's CATE baselines on the
# exact data sets of Experiments B, C and F, 100 data sets per condition. ONE job per model group
# (no arrays); all 76 conditions stacked, data sets of a condition spread over the job's CPUs.
#
#   GROUP=st  sbatch --account=aip-rgrosse --job-name=examext-M-st  .../cluster/exp_M.sbatch   # S-, T-learner (TabPFN)
#   GROUP=x   sbatch --account=aip-rgrosse --job-name=examext-M-x   .../cluster/exp_M.sbatch   # X-learner (TabPFN)
#   GROUP=dml sbatch --account=aip-rgrosse --job-name=examext-M-dml .../cluster/exp_M.sbatch   # DML (TabPFN)
#   GROUP=cf  sbatch --account=aip-rgrosse --job-name=examext-M-cf  .../cluster/exp_M.sbatch   # Causal forest (DML)
#   GROUP=nn  sbatch --account=aip-rgrosse --job-name=examext-M-nn  .../cluster/exp_M.sbatch   # TARNet, DragonNet
#
# Runs in its own venv ($DEPLOY_ROOT/venv_baselines: numpy<2, econml, tabpfn, torch) so the
# Do-PFN venv is never touched. One-time setup on the login node (compute nodes have no
# internet): create that venv and download the TabPFN v2 weights into $DEPLOY_ROOT/tabpfn_cache.
# Env vars:
#   GROUP      st | x | dml | cf | nn (required)
#   OUT_ROOT   output dir (default: $SCRATCH/exam_ext/exp_M_100); results in $OUT_ROOT/$GROUP
#   VENV       baseline venv (default: $DEPLOY_ROOT/venv_baselines)
# Resumable: exp_M.py skips conditions already in its CSV, so if a job hits the time limit,
# submitting the same command again continues where it stopped.
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
EXT="$REPO/comprehensive_exam_extension"
OUT_ROOT="${OUT_ROOT:-$SCRATCH/exam_ext/exp_M_100}"
GROUP="${GROUP:?set GROUP=st, x, dml, cf or nn}"

VENV="${VENV:-$DEPLOY_ROOT/venv_baselines}"
[ -f "$VENV/bin/activate" ] || { echo "FATAL: baseline venv $VENV missing" >&2; exit 1; }
source "$VENV/bin/activate"
# one thread per worker process; TabPFN weights from the pre-downloaded cache, never the network
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TABPFN_MODEL_CACHE_DIR="$DEPLOY_ROOT/tabpfn_cache" HF_HUB_OFFLINE=1

echo "[$(date '+%F %T')] exp_M group=$GROUP out=$OUT_ROOT/$GROUP"
python -u "$EXT/exp_M.py" --group "$GROUP" --n 200 --n-real 100 --workers "${SLURM_CPUS_PER_TASK:-8}" --out "$OUT_ROOT"
echo "[$(date '+%F %T')] done"
