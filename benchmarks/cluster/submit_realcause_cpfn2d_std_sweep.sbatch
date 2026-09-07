#!/bin/bash
#SBATCH --job-name=cpfn2d-std-sweep
#SBATCH --output=logs_cpfn2d_std_sweep/eval_%A_%a.out
#SBATCH --error=logs_cpfn2d_std_sweep/eval_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-39

# 40-task array — cpfn2d × 8 std_target values × 5 RealCause datasets.
#
#   sigma_idx = task / 5   →  {0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1}
#   ds_idx    = task % 5   →  0=IHDP 1=ACIC 2=CPS 3=PSID 4=PSID_bal
#
# All runs use --std-mode std_target (pooled shrink), writing to
#     $OUT_ROOT/cpfn2d_std_<sigma>/<DATASET>/<DATASET>_r<###>.npz
#
# Reminder on cpfn2d's head:
#   edges [-10, +10]  J=32  bin_width=0.625
#   → σ < 0.625 collapses all mass into ~1 bin (0.1 will look degenerate)
#   → σ > ~3.3 starts clipping tails at the edges (2.1 still safe)
#   → σ ≈ 1.0 = CausalPFN training convention (baseline)

set -euo pipefail
mkdir -p logs_cpfn2d_std_sweep

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required — per-sigma dirs land under it}"
CKPT_CPFN2D="${CKPT_CPFN2D:-/scratch/furkanbd/rpfn_bench_kit/cpfn2d_j32_random_A1_h100/step_checkpoints/run/step_0050000.pt}"

SIGMAS=(0.1 0.3 0.6 0.9 1.2 1.5 1.8 2.1)
DATASETS=(IHDP ACIC CPS PSID PSID_bal)

SIGMA_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
DS_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))
SIGMA="${SIGMAS[$SIGMA_IDX]}"
DATASET="${DATASETS[$DS_IDX]}"

OUT="$OUT_ROOT/cpfn2d_std_${SIGMA}/$DATASET"
mkdir -p "$OUT"

[ -f "$CKPT_CPFN2D" ] || { echo "FATAL: CKPT_CPFN2D not found: $CKPT_CPFN2D" >&2; exit 1; }

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] cpfn2d  σ=$SIGMA  DATASET=$DATASET  OUT=$OUT"

python -u "$REPO/realcause_eval/cpfn2d.py" \
    --dataset "$DATASET" --outdir "$OUT" \
    --ckpt "$CKPT_CPFN2D" --repo "$REPO" --causalpfn "$CAUSALPFN" \
    --std-mode std_target --std-target "$SIGMA"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done  σ=$SIGMA  DATASET=$DATASET"
