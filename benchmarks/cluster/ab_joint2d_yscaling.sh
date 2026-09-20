#!/bin/bash
# A/B the y-scaling for dopfn_repro_joint2d. One job per arm, RealCause/IHDP.
#
# WHY: training_dopfn_repro trains with y_space='zscore_ctx' (batch.py:64, the
# default; README: "Settled empirically: +2.60 +- 0.52 nats better than raw"),
# and fit_grid_edges_2d builds the 2D grid over targets IN THAT SPACE. The bb
# eval decodes with --y-scaling min_max (y_center=mid, y_scale=(max-min)/2), so
# the joint's edges and the data are in different spaces. That inflates tau's
# scale, which is what the smoke run shows: worst PEHE (6.83), worst RealCause
# IS (41.96), and case-study coverage 0.526 against a nominal 0.95.
#
# _compute_y_scale(scheme='std') returns (mean, std/std_target), so
# std_target=1.0 is exact z-score -- and it is already called with y_train, i.e.
# context-only statistics, matching zscore_ctx. The sbatch default std_target=0.3
# would give 0.3*z, which is also wrong.
#
# This does NOT assume the diagnosis is right: it runs both arms and lets the
# numbers decide. If min_max wins, the hypothesis is dead and the joint2d problem
# is elsewhere.
#
#   bash R-PFN/benchmarks/cluster/ab_joint2d_yscaling.sh           # submit both
#   bash R-PFN/benchmarks/cluster/ab_joint2d_yscaling.sh --compare # after they land

set -uo pipefail
# Paths are DERIVED, not hardcoded, so this runs on any cluster: the script lives
# at <kit>/R-PFN/benchmarks/cluster/, so the kit root is three levels up. SCRATCH
# must be set by the environment -- guessing a per-cluster scratch path is how a
# run silently writes to the wrong filesystem.
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_KIT_DEFAULT="$(cd "$_SELF_DIR/../../.." && pwd)"
KIT="${KIT:-$_KIT_DEFAULT}"
REPO="${REPO:-$KIT/R-PFN}"
CK="${CK:-$REPO/Required_checkpoints}"
SC="${SCRATCH:?SCRATCH must be set}"
AB="${AB:-$SC/ab_joint2d}"
RC_SB="$REPO/benchmarks/cluster/submit_realcause_density_unified.sbatch"
CKPT="$CK/dopfn_repro_joint2d_bb.pt"
CD="$REPO/UWYK_Fig3_4/cate_density_metrics.py"
PT="$REPO/realcause_eval/point_raw_em.py"

# arm | Y_SCALING | STD_TARGET
ARMS=(
  "minmax|min_max|0.3"     # what it ran under; the current baseline
  "zscore|std|1.0"         # exact context z-score, matching y_space=zscore_ctx
)

cd "$KIT" || exit 1
if [ "${1:-}" = "--compare" ]; then
    source "$KIT/venv/bin/activate"
    for a in "${ARMS[@]}"; do
        IFS='|' read -r arm ys st <<<"$a"
        root="$AB/$arm"
        n=$(find "$root" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
        echo "=== $arm  (y-scaling=$ys std_target=$st)   npz=$n"
        [ "$n" = 0 ] && { echo "    no dumps"; continue; }
        python -u "$PT" --root "$root" --dataset IHDP --modes raw 2>&1 \
          | grep -E '^\| *dopfn' | sed 's/^/    PEHE  /'
        python -u "$CD" --root "$root" --dataset IHDP --target cate \
               --tau-smoother none 2>&1 \
          | awk -F'|' 'NF>8 { f=$3; gsub(/ /,"",f)
              if (f ~ /^[0-9]+$/ && f+0>0) {
                m=$2;c=$5;l=$6;i=$7
                gsub(/^ +| +$/,"",m);gsub(/^ +| +$/,"",c)
                gsub(/^ +| +$/,"",l);gsub(/^ +| +$/,"",i)
                printf("    CALIB %-12s cov=%-8s len=%-18s IS=%s\n",m,c,l,i) } }'
    done
    echo
    echo "Lower PEHE and IS is better; coverage should sit near 0.95."
    exit 0
fi

for a in "${ARMS[@]}"; do
    IFS='|' read -r arm ys st <<<"$a"
    echo "--- submitting arm=$arm  Y_SCALING=$ys  STD_TARGET=$st"
    CKPT_DOPFN_BB="$CKPT" OUT_ROOT="$AB/$arm" DENSITY_DUMP=1 \
    Y_SCALING="$ys" STD_TARGET="$st" \
        sbatch --array=5 --job-name="ab-$arm" "$RC_SB"
    sleep 2
done
echo
echo "when both finish:  bash $REPO/benchmarks/cluster/ab_joint2d_yscaling.sh --compare"
