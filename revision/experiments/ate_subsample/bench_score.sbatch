#!/bin/bash
#SBATCH --job-name=ate-score
#SBATCH --output=logs_ate/%x_%j.out
#SBATCH --error=logs_ate/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
# Score the dumps of rc_dumps.sbatch / cs_cm_dumps.sbatch: coverage of the true ATE by the OT barycenter
# as is and scaled by the model's in-prior conformal s (ate_bary, "s on all 500 datasets"). CPU only.
#   cd $KIT && ATE_BENCH=cs sbatch --account=aip-rgrosse R-PFN/revision/experiments/ate_subsample/bench_score.sbatch
#   (ATE_BENCH = rc | cs | cm; cs pools by case, cm by node count; per-cell lines are in the log)
set -uo pipefail
mkdir -p logs_ate
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; HERE="$REPO/revision/experiments/ate_subsample"
source "$KIT/venv/bin/activate"
export PYTHONUNBUFFERED=1 ATOM_GRID_MAX=4096
BENCH="${ATE_BENCH:?ATE_BENCH = rc | cs | cm}"
case "$BENCH" in
  rc) D="${ATE_RC_DUMPS:-$SCRATCH/ate_rc_dumps}"; OPT=() ;;
  cs) D="${ATE_CS_DUMPS:-$SCRATCH/ate_cs_dumps}"; OPT=(--pool 4) ;;            # shift/d/ctx/model/CASE
  cm) D="${ATE_CM_DUMPS:-$SCRATCH/ate_cm_dumps}"; OPT=(--pool 2 --cmech-data "$SCRATCH/cmech_data_v2") ;;
  *) echo "unknown bench $BENCH"; exit 1 ;;
esac
OUTF="$D/coverage_$BENCH.txt"; : > "$OUTF"
for pair in dopfn_native:0.795 dopfn_joint2d_262k:0.888 cpfn1d:0.097 cpfn2d:0.140; do
  M="${pair%%:*}"; S="${pair##*:}"
  [ -d "$D/$M" ] || { echo "$M: no dumps" | tee -a "$OUTF"; continue; }
  python "$HERE/bench_bary_coverage.py" --root "$D/$M" --s "$S" --label "$M" "${OPT[@]}" > "$D/$M.score.log"
  grep -E '^[^ ]|^  (skip|\[|pooled)' "$D/$M.score.log" | tee -a "$OUTF"   # header, skips, [pools], pooled
done
