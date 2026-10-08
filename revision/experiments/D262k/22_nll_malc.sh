#!/bin/bash
# Stage 22: Table 22 (MALC-T, B=$MALC_B), same dumps as Table 21 (tab:density-nll) NLLs from OUR RealCause density dumps, CPU only, raw outcome units.
# Scorer: benchmarks/eval_graph2d/nll_from_dumps.py (collaborator's density_common objects; see its docstring).
# Inputs (defaults inside the scorer): $SCRATCH/rc_dens_uni/{dopfn_native,uwyk1d,cpfn1d},
#   $SCRATCH/dumps_all/dopfn_repro_joint2d_262k_t1/rc/dopfn_native, $SCRATCH/rc_nll2d/{graph2d,cpfn2d_pooled}.
#   cd $KIT && sbatch --account=aip-rgrosse --time=3:00:00 --cpus-per-task=16 --mem=64G -o logs_hist/t1_nll_malc_%j.out \
#        --wrap "bash R-PFN/revision/experiments/D262k/22_nll_malc.sh"
set -uo pipefail
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; cd "$KIT"; source venv/bin/activate
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/eval_graph2d:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
export CAUSALPFN="${CAUSALPFN:-$KIT/external/causalpfn}"
export ACIC_CACHE_DIR="${ACIC_CACHE_DIR:-$REPO/data/acic_cache}"
# Compute nodes have no internet: ACIC truth must come from the cache (zymu_1..10.csv).
for i in $(seq 1 10); do [ -s "$ACIC_CACHE_DIR/zymu_$i.csv" ] || {
  echo "FATAL: $ACIC_CACHE_DIR/zymu_$i.csv missing. On a login node:"
  echo "  mkdir -p $ACIC_CACHE_DIR && for i in \$(seq 1 10); do curl -sSfL https://raw.githubusercontent.com/BiomedSciAI/causallib/master/causallib/datasets/data/acic_challenge_2016/zymu_\$i.csv -o $ACIC_CACHE_DIR/zymu_\$i.csv; done"
  exit 1; }; done
O="$REPO/revision/results/d262t1"; mkdir -p "$O"; MALC_B="${MALC_B:-100}"; MD="$O/nll_malc_summary.md"
W="${SLURM_CPUS_PER_TASK:-16}"
printf '# Table 22 NLL (MALC-T) from our dumps (raw outcome units)\n\nScorer: benchmarks/eval_graph2d/nll_from_dumps.py, %s\n\n' "$(date -u +%F)" > "$MD"
fail=0
for ds in IHDP ACIC; do
  python -u "$REPO/benchmarks/eval_graph2d/nll_from_dumps.py" --dataset "$ds" --workers "$W" \
    --malc-b "$MALC_B" --out-csv "$O/nll_malc_$ds.csv" --out-md "$MD" || { echo "FAILED $ds"; fail=1; }
done
[ "$fail" = 0 ] || exit 1
cd "$REPO" && git add revision/results/d262t1/nll_malc_* && git commit -q -m "NLL with MALC-T from our dumps" \
  && git pull -q --rebase && git push -q && echo pushed
