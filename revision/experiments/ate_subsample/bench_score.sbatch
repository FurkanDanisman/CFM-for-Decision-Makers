#!/bin/bash
#SBATCH --job-name=ate-score
#SBATCH --output=logs_ate/%x_%j.out
#SBATCH --error=logs_ate/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
# Score the dumps of rc_dumps.sbatch / cs_cm_dumps.sbatch: coverage of the true ATE by the OT barycenter
# as is and scaled by the in-prior conformal s of the same model and tag (read from
# $SCRATCH/ate_bary/<model[-tag]>/coverage.txt). CPU only.
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
# Variants: model[:tag]. s = the in-prior conformal scale of the same variant ("s on all 500 datasets").
for v in dopfn_native dopfn_joint2d_262k cpfn1d cpfn2d uwyk1d:noanc uwyk1d:v3a graph2d:noanc graph2d:v3a; do
  M="${v%%:*}"; T=""; [ "$v" != "$M" ] && T="${v##*:}"
  NAME="$M${T:+-$T}"
  S=$(sed -n 's/.*s on all [0-9]* datasets: \([0-9.]*\).*/\1/p' "${ATE_RESULTS:-$SCRATCH/ate_bary}/$NAME/coverage.txt" 2>/dev/null)
  [ -n "$S" ] || { echo "$NAME: no in-prior s yet" | tee -a "$OUTF"; continue; }
  [ -d "$D/$M" ] || { echo "$NAME: no dumps" | tee -a "$OUTF"; continue; }
  python "$HERE/bench_bary_coverage.py" --root "$D/$M" --s "$S" --label "$NAME" ${T:+--tag "$T"} "${OPT[@]}" > "$D/$NAME.score.log"
  grep -E '^[^ ]|^  (skip|\[|pooled)' "$D/$NAME.score.log" | tee -a "$OUTF"   # header, skips, [pools], pooled
done
