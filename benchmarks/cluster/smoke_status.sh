#!/bin/bash
# Status of today's smoke jobs: slurm state, the real error line, and whether
# any output was actually produced.
#
# Job state is not the test. Both J10 cells hit the same fatal traceback and
# slurm reported one FAILED and the other COMPLETED with exit 0, so this pairs
# every job with the first real error in its log and with an npz count.
#
#   bash R-PFN/benchmarks/cluster/smoke_status.sh

set -uo pipefail
# Paths are DERIVED, not hardcoded, so this runs on any cluster: the script lives
# at <kit>/R-PFN/benchmarks/cluster/, so the kit root is three levels up. SCRATCH
# must be set by the environment -- guessing a per-cluster scratch path is how a
# run silently writes to the wrong filesystem.
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_KIT_DEFAULT="$(cd "$_SELF_DIR/../../.." && pwd)"
KIT="${KIT:-$_KIT_DEFAULT}"
REPO="${REPO:-$KIT/R-PFN}"
SMOKE="${SMOKE:-${SCRATCH:?SCRATCH must be set}/smoke_new}"
cd "$KIT" || exit 1

echo "##### slurm state (today, array/batch rows collapsed)"
sacct -S today --format=JobID%16,JobName%20,State%12,ExitCode%8,Elapsed%10 -n \
  2>/dev/null | grep -vE '\.(batch|extern|[0-9]+) ' | sed 's/^/  /'

echo
echo "##### first error in each smoke log"
for d in logs_rc_dens_uni logs_cs_dvar; do
    [ -d "$d" ] || continue
    for f in $(ls -t "$d"/*.err 2>/dev/null | head -14); do
        msg=$(grep -m1 -E "Error|error:|Traceback|FATAL|invalid choice|SystemExit|no such" "$f" 2>/dev/null)
        real=$(grep -m1 -E "^(RuntimeError|ValueError|KeyError|SystemExit|OSError|AssertionError|TypeError|FileNotFoundError)" "$f" 2>/dev/null)
        [ -n "$real" ] && msg="$real"
        if [ -n "$msg" ]; then printf '  %-34s %s\n' "$(basename "$f")" "${msg:0:120}"
        else                   printf '  %-34s clean\n' "$(basename "$f")"; fi
    done
done

echo
echo "##### output actually written, per smoke root"
for d in "$SMOKE"/*; do
    [ -d "$d" ] || continue
    for b in rc cs; do
        [ -d "$d/$b" ] || continue
        n=$(find "$d/$b" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
        printf '  %-24s %-3s %5s npz\n' "$(basename "$d")" "$b" "$n"
    done
done

echo
echo "Now the real gate (scores each root, requires a row with n > 0):"
echo "  bash $REPO/benchmarks/cluster/check_smoke.sh"
