#!/bin/bash
# Finish the tau-C MALC-T run: re-dump Do-PFN from scratch, then smooth
# everything that is still missing.
#
#     bash benchmarks/cluster/submit_density_tauC_chain.sh
#
# ── WHERE THINGS STAND (checked 2026-09-23) ───────────────────────────────
#
#   shard                  raw dumps      MALC-T
#   5312882  UWYK v3a      100/10         DONE
#   5312884  UWYK noanc    100/10         DONE
#   5571187  CausalPFN     100/10 *       missing
#   60508900 Do-PFN        90/8           missing
#
#   * repaired by submit_density_tauC_repair.sbatch on 2026-09-22; its probe
#     passed and IHDP r000-r009 were back-filled.
#
# ── WHY DO-PFN IS A FRESH RUN AND NOT A REPAIR ────────────────────────────
# The repair's reproduction probe REFUSED to back-fill Do-PFN, correctly.
# Measured on job 5629355, recomputing IHDP r000 and ACIC r002:
#
#     dopfn_pred0                9.07e-04   FAIL     <- pretrained library model
#     dopfn_pred1                7.53e-04   FAIL     <- pretrained library model
#     dopfn_repro_1d_J10_pred0   7.77e-07   ok
#     dopfn_repro_1d_J100_pred0  1.07e-06   ok
#     dopfn_repro_joint2d_logits 9.76e-07   ok
#
# The three checkpoints we own reproduce at ~1e-6, i.e. hardware noise. Only
# `native` -- the Do-PFN LIBRARY model, recorded as dopfn_sources[0]=='library'
# with no version or hash to pin -- drifted, by ~1e-3. That is far below a
# different model (O(1)) and far above the noise floor: the package moved.
#
# Back-filling would have put two library builds inside one row. So the whole
# family is re-dumped into a fresh OUT_ROOT instead, and every Do-PFN
# realization comes from one environment. The old 60508900 shard is left
# untouched for comparison.
#
# ── WHAT GETS SUBMITTED ───────────────────────────────────────────────────
#   1. raw Do-PFN    array 0-11, 100 IHDP + 10 ACIC, into $DOPFN_FRESH.
#                    Needs checkpoints, the Do-PFN package and the datasets.
#   2. malcT Do-PFN  --dependency=afterok on (1)
#   3. malcT CausalPFN   no dependency -- its dumps are already complete
#
# The dependency is per-job, not per-array-task. That mattered last time: the
# repair array was gated as a whole, so Do-PFN's failing tasks blocked
# CausalPFN's MALC-T even though CausalPFN's own repair had succeeded. Here
# CausalPFN is simply not gated on Do-PFN at all.
#
# DRY RUN -- print what would be submitted, submit nothing:
#     CHAIN_DRY_RUN=1 bash benchmarks/cluster/submit_density_tauC_chain.sh
#
# Overrides work as for the individual jobs, e.g.
#     MALC_B=100 bash benchmarks/cluster/submit_density_tauC_chain.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

# Point every job at THIS checkout. The sbatches each default REPO to a
# hardcoded cluster path, and the shards record three different roots (the
# families were dumped on different machines), so pinning it here is what keeps
# one submission internally consistent.
export REPO="${REPO:-$PWD}"

RAW="$REPO/benchmarks/cluster/submit_density_tauC.sbatch"
MALCT="$REPO/benchmarks/cluster/submit_density_tauC_malcT.sbatch"
RESULTS="${RESULTS:-$REPO/results_density_tauC}"
MALCT_OUT="${MALCT_OUT:-$REPO/results_density_tauC_malcT}"

# Fresh Do-PFN destination. A FIXED name, not the array job id the raw sbatch
# would otherwise pick, because the dependent MALC job has to be told where to
# look at submission time -- before that job id exists.
DOPFN_FRESH="${DOPFN_FRESH:-$RESULTS/dopfn_refresh}"
CPFN_SHARD="${CPFN_SHARD:-$RESULTS/5571187}"

DRY="${CHAIN_DRY_RUN:-0}"

for f in "$RAW" "$MALCT"; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
[ -d "$CPFN_SHARD/IHDP/predictions" ] || {
    echo "FATAL: no CausalPFN dumps at $CPFN_SHARD" >&2; exit 1; }

# A fresh run must start empty. eval_density_tauC.py skips realizations whose
# shard exists, so a leftover directory from an aborted attempt would be
# silently kept -- which is the exact failure this re-run exists to avoid.
if [ -e "$DOPFN_FRESH" ] && [ "${DOPFN_FRESH_REUSE:-0}" != "1" ]; then
    echo "FATAL: $DOPFN_FRESH already exists." >&2
    echo "  A fresh Do-PFN run must start from an empty directory, or the" >&2
    echo "  resume logic keeps whatever is in there." >&2
    echo "  Remove it, set DOPFN_FRESH to another path, or pass" >&2
    echo "  DOPFN_FRESH_REUSE=1 to deliberately resume an interrupted run." >&2
    exit 1
fi

submit() {   # submit <description> <args...>
    local desc="$1"; shift
    if [ "$DRY" = "1" ]; then
        echo "[dry-run] $desc" >&2
        echo "          sbatch $*" >&2
        echo "DRYRUN$RANDOM"
    else
        sbatch --parsable "$@"
    fi
}

echo "Do-PFN fresh dumps -> $DOPFN_FRESH"
echo

RAW_ID=$(MODEL_FAMILY=dopfn OUT_ROOT="$DOPFN_FRESH" \
         DOPFN_ROOT="${DOPFN_ROOT:-${DOPFN:-$REPO/Do-PFN}}" \
         submit "raw Do-PFN, array 0-11" "$RAW")
echo "raw   dopfn_refresh   $RAW_ID   (array 0-11, needs GPU-tier data + checkpoints)"

MALC_DOPFN=$(DUMPS_ROOT="$DOPFN_FRESH" OUT_ROOT="$MALCT_OUT/dopfn_refresh" \
             submit "malcT Do-PFN" --dependency=afterok:"$RAW_ID" "$MALCT")
echo "malcT dopfn_refresh   $MALC_DOPFN   after raw $RAW_ID"

MALC_CPFN=$(DUMPS_ROOT="$CPFN_SHARD" OUT_ROOT="$MALCT_OUT/$(basename "$CPFN_SHARD")" \
            submit "malcT CausalPFN" "$MALCT")
echo "malcT $(basename "$CPFN_SHARD")        $MALC_CPFN   (no dependency)"

if [ "$DRY" = "1" ]; then
    echo; echo "CHAIN_DRY_RUN=1 -- nothing submitted."; exit 0
fi

cat <<EOF

Submitted. Watch with:  squeue -u \$USER

The raw Do-PFN job is the long pole: its ACIC tasks are 5 realizations x ~481
queries against a 3h cap, and Do-PFN pays per model (4 of them). If a task hits
the wall clock it is resumable -- re-submit the same array with
DOPFN_FRESH_REUSE=1 and it picks up where it stopped. WIDEN THE ARRAY rather
than raising --time: on this cluster --time routes the partition, and >3h lands
in a slower bucket.

If the raw job fails, the gated MALC job stays in DependencyNeverSatisfied:
  scancel $MALC_DOPFN

When everything finishes, all four families are smoothed at 100/10:
  for S in dopfn_refresh $(basename "$CPFN_SHARD") 5312882 5312884; do
    python benchmarks/eval_graph2d/summarize_density_tauC.py $MALCT_OUT/\$S
  done
EOF
