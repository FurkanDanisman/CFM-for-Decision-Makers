#!/bin/bash
#SBATCH --job-name=ate-rc-score
#SBATCH --output=logs_ate/%x_%j.out
#SBATCH --error=logs_ate/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
# Score the RealCause dumps of rc_dumps.sbatch: coverage of the true ATE by the OT barycenter as is and
# scaled by the model's in-prior conformal s (ate_bary results, "s on all 500 datasets"). CPU only.
#   cd $KIT && sbatch --account=aip-rgrosse R-PFN/revision/experiments/ate_subsample/rc_score.sbatch
set -uo pipefail
mkdir -p logs_ate
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; HERE="$REPO/revision/experiments/ate_subsample"
source "$KIT/venv/bin/activate"
export PYTHONUNBUFFERED=1 ATOM_GRID_MAX=4096
OUTF="${ATE_RC_DUMPS:-$SCRATCH/ate_rc_dumps}/coverage_rc.txt"; : > "$OUTF"
for pair in dopfn_native:0.795 dopfn_joint2d_262k:0.888 cpfn1d:0.097 cpfn2d:0.140; do
  M="${pair%%:*}"; S="${pair##*:}"
  python "$HERE/bench_bary_coverage.py" --root "${ATE_RC_DUMPS:-$SCRATCH/ate_rc_dumps}/$M" --s "$S" --label "$M" \
    | tee -a "$OUTF"
done
