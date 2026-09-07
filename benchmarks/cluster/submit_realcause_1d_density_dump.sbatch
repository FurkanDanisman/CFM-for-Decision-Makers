#!/bin/bash
#SBATCH --job-name=rc-1d-density
#SBATCH --output=logs_rc_1d_density/eval_%A_%a.out
#SBATCH --error=logs_rc_1d_density/eval_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-14

# 15-task array — 3 1D methods × 5 RealCause datasets, all with DENSITY_DUMP=1.
# Point CATE stays identical to the mega-sbatch (density mean = point mean by
# construction under independence); the density is an ADD-ON so we can compute
# 95% CI, coverage, and length.
#
#   method_idx = task / 5   → 0=cpfn1d 1=dopfn 2=uwyk1d
#   ds_idx     = task % 5   → 0=IHDP 1=ACIC 2=CPS 3=PSID 4=PSID_bal
#
# Writes to $OUT_ROOT/<method>/<DATASET>/<D>_r<###>.npz with the extra keys
# edges / p_y0_scaled / p_y1_scaled / y_shift / y_scale / true_cate_per_query.
#
# After the run:
#   python R-PFN/realcause_eval/cate_ci_from_density.py --out-root $OUT_ROOT

set -euo pipefail
mkdir -p logs_rc_1d_density

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
DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
UWYK_REPRO="${UWYK_REPRO:-$DEPLOY_ROOT/external/uwyk_reproduce}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required — per-method dirs land under it}"

CKPT_CPFN1D="${CKPT_CPFN1D:-/scratch/furkanbd/rpfn_bench_kit/causalpfn_j1024_headrand_A1_h100/step_checkpoints/run/step_0050000.pt}"

METHOD_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
DS_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))
case "$METHOD_IDX" in
    0) METHOD=cpfn1d ;;
    1) METHOD=dopfn  ;;
    2) METHOD=uwyk1d ;;
    *) echo "invalid METHOD_IDX=$METHOD_IDX" >&2; exit 2 ;;
esac
case "$DS_IDX" in
    0) DATASET=IHDP     ;;
    1) DATASET=ACIC     ;;
    2) DATASET=CPS      ;;
    3) DATASET=PSID     ;;
    4) DATASET=PSID_bal ;;
    *) echo "invalid DS_IDX=$DS_IDX" >&2; exit 2 ;;
esac
OUT="$OUT_ROOT/$METHOD/$DATASET"
mkdir -p "$OUT"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
export DENSITY_DUMP=1   # key knob for this array

echo "[$(date '+%Y-%m-%d %H:%M:%S')] $METHOD  $DATASET  OUT=$OUT  DENSITY_DUMP=1"

case "$METHOD" in
    cpfn1d)
        [ -f "$CKPT_CPFN1D" ] || { echo "FATAL: CKPT_CPFN1D not found: $CKPT_CPFN1D" >&2; exit 1; }
        python -u "$REPO/realcause_eval/cpfn1d.py" \
            --dataset "$DATASET" --outdir "$OUT" \
            --ckpt "$CKPT_CPFN1D" --repo "$REPO" --causalpfn "$CAUSALPFN" \
            --density-dump 1
        ;;
    dopfn)
        [ -d "$DOPFN_ROOT" ] || { echo "FATAL: DOPFN_ROOT not found: $DOPFN_ROOT" >&2; exit 1; }
        DS_DP="$DATASET"; [ "$DATASET" = "PSID_bal" ] && DS_DP="PSIDbal"
        python -u "$REPO/realcause_eval/do_pfn.py" \
            --dataset "$DS_DP" --outdir "$OUT" \
            --dopfn "$DOPFN_ROOT" --causalpfn "$CAUSALPFN"
        ;;
    uwyk1d)
        [ -d "$UWYK_REPRO" ] || { echo "FATAL: UWYK_REPRO not found: $UWYK_REPRO" >&2; exit 1; }
        python -u "$REPO/realcause_eval/uwyk1d.py" \
            --dataset "$DATASET" --outdir "$OUT" \
            --uwyk-repro "$UWYK_REPRO" --repo "$REPO" \
            --causalpfn "$CAUSALPFN" --anc-mode v3b_only
        ;;
esac

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done  $METHOD  $DATASET"
