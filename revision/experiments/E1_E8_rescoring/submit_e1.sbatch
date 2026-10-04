#!/bin/bash
#SBATCH --job-name=e1-rescore
#SBATCH --output=logs_e1/e1_%j.out
#SBATCH --error=logs_e1/e1_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#
# Revision E1: IHDP and ACIC rescored against realized delta = y1 - y0, raw and
# MALC, for the published models. CPU only; reads the existing dumps.
#
#   cd $KIT && sbatch --account=aip-rgrosse R-PFN/revision/experiments/E1_E8_rescoring/submit_e1.sbatch
#
# Output: $OUT/truth/<DS>_r###.npz and $OUT/perreal/<label>/<stage>__<DS>__-.npz,
# the same per-realization layout as $SCRATCH/perreal, read by e8_tables.py.
# MALC settings match submit_full_table.sbatch (B=1000, K=1, n_tau=4001).
set -uo pipefail
mkdir -p logs_e1
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
OUT="${OUT:-$SCRATCH/revision_e1}"
MALC_B="${MALC_B:-1000}"; MALC_K="${MALC_K:-1}"; N_TAU="${N_TAU:-4001}"
NW="${SLURM_CPUS_PER_TASK:-64}"
source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"
CBR="$REPO/benchmarks/coverage_by_realization.py"

# label | RealCause dump root | published methods in it
ROOTS=(
  "orig|$SCRATCH/rc_dens_uni|dopfn_native uwyk1d-noanc uwyk1d-v3a graph2d-noanc graph2d-v3a cpfn1d"
  "joint2d|$SCRATCH/dumps_all/dopfn_repro_joint2d/rc|dopfn_native"
  "eta0|$SCRATCH/rc_dens_eta0|cpfn2d"
)

python -u "$REPO/revision/experiments/E1_E8_rescoring/build_delta_truth.py" \
    --causalpfn "$CAUSALPFN" --out "$OUT/truth" || { echo "FATAL: truth build failed"; exit 1; }

FAILED=0
for row in "${ROOTS[@]}"; do
  IFS='|' read -r lbl root meths <<<"$row"
  [ -d "$root" ] || { echo "MISSING root $root"; FAILED=1; continue; }
  mkdir -p "$OUT/perreal/$lbl"
  for ds in IHDP ACIC; do
    for S in raw malc; do
      o="$OUT/perreal/$lbl/${S}__${ds}__-.npz"
      [ -s "$o" ] && { echo "skip (done) $o"; continue; }
      SM=(--tau-smoother none)
      [ "$S" = malc ] && SM=(--tau-smoother malc --malc-B "$MALC_B" --malc-K "$MALC_K" \
                             --n-tau "$N_TAU" --malc-workers "$NW")
      echo "[$(date '+%F %T')] $lbl $ds $S"
      python -u "$CBR" --root "$root" --dataset "$ds" --methods $meths "${SM[@]}" \
          --truth-dir "$OUT/truth" --dump-per-real "$o" \
          > "$OUT/perreal/$lbl/${S}__${ds}.md" 2> "$OUT/perreal/$lbl/${S}__${ds}.err" \
          || { echo "  ERROR: see $OUT/perreal/$lbl/${S}__${ds}.err"; FAILED=1; }
      cat "$OUT/perreal/$lbl/${S}__${ds}.md"
    done
  done
done
echo "[$(date '+%F %T')] done -> $OUT  (failed=$FAILED)"
exit $FAILED
