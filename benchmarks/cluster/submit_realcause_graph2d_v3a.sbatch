#!/bin/bash
#SBATCH --job-name=g2d-v3a-rc
#SBATCH --output=logs_graph2d_v3a_rc/eval_%A_%a.out
#SBATCH --error=logs_graph2d_v3a_rc/eval_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4
#
# graph2d on RealCause with ANC_MODE=v3ab_only -- ONE pass yielding, per
# realization, BOTH the point estimates (PEHE / ATE) and the densities for
# noanc + v3a + v3b.
#
#   0 IHDP   1 ACIC   2 CPS   3 PSID   4 PSID_bal
#
# The existing submit_realcause_2d_density_dump.sbatch uses --anc-mode
# v3b_only, which emits v3b + noanc only -- there is no v3a in it. v3ab_only
# adds v3a for 3 forward passes per realization instead of 2, and the dump
# writes per-mode keys (p_y0_scaled_<mode>, p_joint_scaled_<mode>, ...), so
# all three modes' densities land in the same npz and the scorer selects by
# tag. NOT v3_family: that also carries v3d, whose +1 diagonal UWYK's PAM
# validator rejects.
#
# Output: $OUT_ROOT/graph2d/<DATASET>/<DATASET>_r<###>.npz
# which is the layout UWYK_Fig3_4/cate_density_metrics.py --dataset expects,
# with METHODS entries ("graph2d-noanc","graph2d","noanc") and
# ("graph2d-v3a","graph2d","v3a").
#
# Required env:  OUT_ROOT
# Optional:      CKPT_GRAPH2D, DENSITY_PRIMARY_MODE (default v3a)
#
#   cd $DEPLOY_ROOT && OUT_ROOT=$SCRATCH/rc_graph2d_v3a \
#     sbatch $DEPLOY_ROOT/R-PFN/benchmarks/cluster/submit_realcause_graph2d_v3a.sbatch

set -euo pipefail
mkdir -p logs_graph2d_v3a_rc

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
UWYK_REPRO="${UWYK_REPRO:-${UWYK:-$DEPLOY_ROOT/external/uwyk_reproduce}}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
CKPT_GRAPH2D="${CKPT_GRAPH2D:-$REPO/Required_checkpoints/graph2d_step_50000.pt}"
DENSITY_PRIMARY_MODE="${DENSITY_PRIMARY_MODE:-v3a}"

[ -f "$CKPT_GRAPH2D" ] || { echo "FATAL: ckpt not found: $CKPT_GRAPH2D" >&2; exit 1; }

case "$SLURM_ARRAY_TASK_ID" in
    0) DATASET=IHDP     ;;
    1) DATASET=ACIC     ;;
    2) DATASET=CPS      ;;
    3) DATASET=PSID     ;;
    4) DATASET=PSID_bal ;;
    *) echo "invalid task id: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
OUT="$OUT_ROOT/graph2d/$DATASET"
mkdir -p "$OUT"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%F %T')] graph2d v3ab_only  DATASET=$DATASET"
echo "  CKPT=$CKPT_GRAPH2D"
echo "  OUT=$OUT   primary_mode=$DENSITY_PRIMARY_MODE"

python -u "$REPO/realcause_eval/Table1/graph2d.py" \
    --dataset "$DATASET" --outdir "$OUT" \
    --ckpt "$CKPT_GRAPH2D" --repo "$REPO" \
    --causalpfn "$CAUSALPFN" --uwyk "$UWYK_REPRO" \
    --anc-mode v3ab_only --density-primary-mode "$DENSITY_PRIMARY_MODE" \
    --density-dump 1

echo "[$(date '+%F %T')] done  $DATASET"
