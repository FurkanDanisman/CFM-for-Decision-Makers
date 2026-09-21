#!/bin/bash
# Score every model on the semi-real datasets. ONE job per model.
#
#   bash R-PFN/benchmarks/cluster/submit_semireal_scores.sh            # dry run
#   bash R-PFN/benchmarks/cluster/submit_semireal_scores.sh --submit
set -uo pipefail
SUBMIT=0; [ "${1:-}" = "--submit" ] && SUBMIT=1
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
KIT="${KIT:-$(cd "$_SELF_DIR/../../.." && pwd)}"
REPO="${REPO:-$KIT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"
DUMPS="${SEMIREAL_DUMPS:-$SC/semireal_dumps}"
PERREAL="${PERREAL:-$SC/perreal}"
SB="$REPO/benchmarks/cluster/submit_semireal_score_one.sbatch"
ACCT="${ACCOUNT:-}"; PART="${PARTITION:-}"
TIME="${SCORE_TIME:-3:00:00}"; CPUS="${SCORE_CPUS:-32}"; MEM="${SCORE_MEM:-64G}"
STAGES="${STAGES:-point raw malc}"; ONLY="${ONLY:-}"

[ -d "$DUMPS" ] || { echo "FATAL: no dumps at $DUMPS" >&2; exit 1; }
cd "$KIT" || exit 1
N=0
for d in "$DUMPS"/*; do
    [ -d "$d" ] || continue
    name="$(basename "$d")"
    [ -n "$ONLY" ] && { case " $ONLY " in *" $name "*) ;; *) continue ;; esac; }
    n_npz=$(find "$d" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n_npz" = 0 ]; then printf '%-22s SKIP: no dumps\n' "$name"; continue; fi
    N=$((N+1))
    if [ "$SUBMIT" = 1 ]; then
        printf '%-22s (%s npz) -> ' "$name" "$n_npz"
        MODEL_NAME="$name" ROOT="$d" PERREAL="$PERREAL" STAGES="$STAGES" \
            sbatch --time="$TIME" --cpus-per-task="$CPUS" --mem="$MEM" \
                   ${ACCT:+--account=$ACCT} ${PART:+--partition=$PART} \
                   --job-name="srs-$name" "$SB"
    else
        printf '%-22s %s npz  stages="%s"\n' "$name" "$n_npz" "$STAGES"
    fi
done
echo
echo "jobs: $N   (one per model; both datasets, 5 splits each)"
[ "$SUBMIT" = 1 ] || echo "dry run -- add --submit"
