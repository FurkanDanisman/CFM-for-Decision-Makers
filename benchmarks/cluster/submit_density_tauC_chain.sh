#!/bin/bash
# Submit the repair, then the MALC-T arms, in one go. Nothing to do in between.
#
#     bash benchmarks/cluster/submit_density_tauC_chain.sh
#
# Slurm holds the MALC jobs until the repair finishes successfully, so this is
# fire-and-forget: the gap-filling and the smoothing are one submission.
#
# WHAT GETS SUBMITTED
#
#   repair     array 0-2, fills Do-PFN IHDP 80-94 + ACIC 0-1 and CausalPFN
#              IHDP 0-9 into the existing shards. Needs checkpoints and data.
#   malcT x4   one per shard, array 0-1 (IHDP, ACIC). Reads dumps only.
#
# THE DEPENDENCY IS NOT UNIFORM, ON PURPOSE. Only two shards are being
# repaired, so only their MALC jobs wait:
#
#   60508900  Do-PFN      --dependency=afterok:<repair>
#   5571187   CausalPFN   --dependency=afterok:<repair>
#   5312882   UWYK v3a    no dependency -- already 100/10, nothing to wait for
#   5312884   UWYK noanc  no dependency -- likewise
#
# Gating the UWYK pair on a repair they do not need would idle them behind it
# and, worse, strand them if the repair fails.
#
# afterok GATES ON EXIT STATUS, which is why the repair job asserts its own
# realization counts before exiting. A task that filled half its gap and exited
# 0 would otherwise let the MALC arms run on a still-uneven set -- the exact
# problem this chain exists to fix.
#
# IF THE REPAIR FAILS, the dependent jobs sit in DependencyNeverSatisfied
# rather than erroring out. They do NOT run, which is the point, but on most
# clusters they also do not disappear:
#
#     scancel <the two pending malcT job ids printed below>
#
# Read the repair log first. A FAILED reproduction probe means that family must
# be re-run whole into a fresh OUT_ROOT -- do not re-submit the repair to force
# past it.
#
# DRY RUN -- probes only, writes nothing, submits no MALC jobs:
#     REPAIR_DRY_RUN=1 bash benchmarks/cluster/submit_density_tauC_chain.sh
#
# Override any of these the same way you would for the individual jobs, e.g.
#     MALC_B=100 bash benchmarks/cluster/submit_density_tauC_chain.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

# Point every job at THIS checkout. The three sbatches each default REPO to a
# hardcoded cluster path, and the shards themselves record three different
# roots (the families were dumped on different machines), so pinning it here is
# what keeps one submission internally consistent.
export REPO="${REPO:-$PWD}"

REPAIR="$REPO/benchmarks/cluster/submit_density_tauC_repair.sbatch"
MALCT="$REPO/benchmarks/cluster/submit_density_tauC_malcT.sbatch"
RESULTS="${RESULTS:-$REPO/results_density_tauC}"

# Absolute: the sbatches check for the dump directory BEFORE they cd to $REPO.
export DOPFN_SHARD="${DOPFN_SHARD:-$RESULTS/60508900}"
export CPFN_SHARD="${CPFN_SHARD:-$RESULTS/5571187}"
UWYK_V3A_SHARD="${UWYK_V3A_SHARD:-$RESULTS/5312882}"
UWYK_NOANC_SHARD="${UWYK_NOANC_SHARD:-$RESULTS/5312884}"

for f in "$REPAIR" "$MALCT"; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
for d in "$DOPFN_SHARD" "$CPFN_SHARD" "$UWYK_V3A_SHARD" "$UWYK_NOANC_SHARD"; do
    [ -d "$d" ] || { echo "FATAL: no shard at $d" >&2; exit 1; }
done

REPAIR_ID=$(sbatch --parsable "$REPAIR")
echo "repair            $REPAIR_ID   (array 0-2)"

if [ "${REPAIR_DRY_RUN:-0}" = "1" ]; then
    echo
    echo "REPAIR_DRY_RUN=1 -- probes only. No MALC jobs submitted, nothing"
    echo "written to the shards. Re-run without REPAIR_DRY_RUN once the probes"
    echo "pass to submit the real chain."
    exit 0
fi

# OUT_ROOT is pinned per shard rather than left to default. The MALC sbatch
# names its output after its OWN array job id, so four submissions would land in
# four directories you then have to map back to shards by hand. Naming them
# after the source shard keeps raw / J / T lined up per family.
MALCT_OUT="${MALCT_OUT:-$REPO/results_density_tauC_malcT}"

# Repaired shards: wait for the whole repair array to succeed.
GATED=()
for shard in "$DOPFN_SHARD" "$CPFN_SHARD"; do
    id=$(DUMPS_ROOT="$shard" OUT_ROOT="$MALCT_OUT/$(basename "$shard")" \
         sbatch --parsable --dependency=afterok:"$REPAIR_ID" "$MALCT")
    GATED+=("$id")
    echo "malcT  $(basename "$shard")   $id   after repair $REPAIR_ID"
done

# Untouched shards: no reason to wait.
for shard in "$UWYK_V3A_SHARD" "$UWYK_NOANC_SHARD"; do
    id=$(DUMPS_ROOT="$shard" OUT_ROOT="$MALCT_OUT/$(basename "$shard")" \
         sbatch --parsable "$MALCT")
    echo "malcT  $(basename "$shard")   $id   (no dependency)"
done

cat <<EOF

Submitted. Watch with:  squeue -u \$USER

If the repair fails, these two stay blocked and need cancelling:
  scancel ${GATED[*]}

When everything finishes, summarize each shard:
  for S in $(basename "$DOPFN_SHARD") $(basename "$CPFN_SHARD") $(basename "$UWYK_V3A_SHARD") $(basename "$UWYK_NOANC_SHARD"); do
    python benchmarks/eval_graph2d/summarize_density_tauC.py $MALCT_OUT/\$S
  done
EOF
