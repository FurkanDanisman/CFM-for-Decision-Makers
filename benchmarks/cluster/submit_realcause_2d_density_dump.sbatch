#!/bin/bash
#SBATCH --job-name=rc-2d-density
#SBATCH --output=logs_rc_2d_density/eval_%A_%a.out
#SBATCH --error=logs_rc_2d_density/eval_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-14

# 15-task array — 3 2D methods × 5 RealCause datasets, all with DENSITY_DUMP=1.
# Point CATE stays identical to the mega-sbatch (density mean = point mean by
# construction for 2D joint heads — E[τ] = E[Y1] − E[Y0] under the joint's
# marginals); the density is an ADD-ON so we can compute 95% CI, coverage,
# and length via anti-diagonal projection of p(Y0, Y1) (no independence
# assumption needed — this is the strictly-more-informative 2D path).
#
#   method_idx = task / 5   → 0=cpfn2d 1=graph2d 2=dopfnbb
#   ds_idx     = task % 5   → 0=IHDP 1=ACIC 2=CPS 3=PSID 4=PSID_bal
#
# Writes to $OUT_ROOT/<method>/<DATASET>/... with schema:
#   cpfn2d/graph2d:  <DATASET>_r<###>.npz with p_joint_scaled, p_y{0,1}_scaled,
#                    edges, y_shift, y_scale, true_cate_per_query + pehe_raw/err_raw.
#   dopfnbb:         density_r<###>.npz per realization + summary.npz aggregate
#                    with pehe[] / eps_ate[]. Uses PSIDbal (legacy naming).
#
# After the run:
#   python R-PFN/realcause_eval/cate_ci_from_density.py --out-root $OUT_ROOT \
#       --methods cpfn2d graph2d dopfnbb --graph2d-tag noanc

set -euo pipefail
mkdir -p logs_rc_2d_density

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

# Checkpoints — same defaults as submit_realcause_all_methods.sbatch so the
# CI numbers are computed from the identical density the mega-sbatch used.
CKPT_CPFN2D="${CKPT_CPFN2D:-/scratch/furkanbd/rpfn_bench_kit/cpfn2d_j32_random_A1_h100/step_checkpoints/run/step_0050000.pt}"
CKPT_GRAPH2D="${CKPT_GRAPH2D:-$REPO/Required_checkpoints/graph2d_step_50000.pt}"
CKPT_DOPFNBB="${CKPT_DOPFNBB:-$REPO/Required_checkpoints/dopfn_bb_j10_step_150000.pt}"

METHOD_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
DS_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))
case "$METHOD_IDX" in
    0) METHOD=cpfn2d  ;;
    1) METHOD=graph2d ;;
    2) METHOD=dopfnbb ;;
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

# dopfn_bb legacy uses PSIDbal (no underscore) as its dataset key AND its
# on-disk dir name. Aggregator knows this via _DATASET_DIR_ALIASES.
DS_ON_DISK="$DATASET"
[ "$METHOD" = "dopfnbb" ] && [ "$DATASET" = "PSID_bal" ] && DS_ON_DISK="PSIDbal"

OUT="$OUT_ROOT/$METHOD/$DS_ON_DISK"
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
    cpfn2d)
        [ -f "$CKPT_CPFN2D" ] || { echo "FATAL: CKPT_CPFN2D not found: $CKPT_CPFN2D" >&2; exit 1; }
        # --std-mode pooled: density dump saves ONE (y_shift, y_scale) pair,
        # so per_arm (cpfn2d's default) would break the density-vs-point sanity
        # — the stored point CATE would un-scale with per-arm y0sc/y1sc while
        # the aggregator can only apply one scale factor. Same fix as cpfn1d.
        python -u "$REPO/realcause_eval/cpfn2d.py" \
            --dataset "$DATASET" --outdir "$OUT" \
            --ckpt "$CKPT_CPFN2D" --repo "$REPO" --causalpfn "$CAUSALPFN" \
            --std-mode pooled --density-dump 1
        ;;
    graph2d)
        [ -f "$CKPT_GRAPH2D" ] || { echo "FATAL: CKPT_GRAPH2D not found: $CKPT_GRAPH2D" >&2; exit 1; }
        # Point-CATE row we report is noanc (matches paper Table 3), so the
        # density we dump must be the noanc-mode density. --anc-mode v3b_only
        # keeps both v3b + noanc point columns in the NPZ so debug / v3b tag
        # switching still works via --graph2d-tag.
        python -u "$REPO/realcause_eval/graph2d.py" \
            --dataset "$DATASET" --outdir "$OUT" \
            --ckpt "$CKPT_GRAPH2D" --repo "$REPO" \
            --causalpfn "$CAUSALPFN" --uwyk "$UWYK_REPRO" \
            --anc-mode v3b_only --density-primary-mode noanc \
            --density-dump 1
        ;;
    dopfnbb)
        [ -f "$CKPT_DOPFNBB" ] || { echo "FATAL: CKPT_DOPFNBB not found: $CKPT_DOPFNBB" >&2; exit 1; }
        # do_pfn_bb.py wrapper always sets DENSITY_DUMP=1 internally.
        python -u "$REPO/realcause_eval/do_pfn_bb.py" \
            --dataset "$DS_ON_DISK" --outdir "$OUT" \
            --ckpt "$CKPT_DOPFNBB" --repo "$REPO" \
            --dopfn "$DOPFN_ROOT" --causalpfn "$CAUSALPFN"
        ;;
esac

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done  $METHOD  $DATASET"
