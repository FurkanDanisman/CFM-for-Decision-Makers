#!/bin/bash
# Score every ComplexMech model, one job each.
#
#   bash R-PFN/benchmarks/cluster/submit_cmech_scores.sh            # dry run
#   STAGES="point raw" bash ... --submit                            # cheap first
#   STAGES="malc" SCORE_TIME=12:00:00 bash ... --submit
set -uo pipefail
SUBMIT=0; [ "${1:-}" = "--submit" ] && SUBMIT=1
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
KIT="${KIT:-$(cd "$_SELF_DIR/../../.." && pwd)}"
REPO="${REPO:-$KIT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"
DUMPS="${CMECH_DUMPS:-$SC/cmech_dumps}"
PERREAL="${PERREAL:-$SC/perreal}"
SB="$REPO/benchmarks/cluster/submit_cmech_score_one.sbatch"
ACCT="${ACCOUNT:-}"; PART="${PARTITION:-}"
TIME="${SCORE_TIME:-6:00:00}"; CPUS="${SCORE_CPUS:-64}"; MEM="${SCORE_MEM:-128G}"
STAGES="${STAGES:-point raw malc}"; ONLY="${ONLY:-}"

[ -d "$DUMPS" ] || { echo "FATAL: no dumps at $DUMPS" >&2; exit 1; }
cd "$KIT" || exit 1
N=0
for d in "$DUMPS"/*; do
    [ -d "$d" ] || continue
    name="$(basename "$d")"
    [ -n "$ONLY" ] && { case " $ONLY " in *" $name "*) ;; *) continue ;; esac; }
    N=$((N+1))
    if [ "$SUBMIT" = 1 ]; then
        printf '%-22s -> ' "$name"
        MODEL_NAME="$name" ROOT="$d" PERREAL="$PERREAL" STAGES="$STAGES" \
            sbatch --time="$TIME" --cpus-per-task="$CPUS" --mem="$MEM" \
                   ${ACCT:+--account=$ACCT} ${PART:+--partition=$PART} \
                   --job-name="cms-$name" "$SB"
    else
        printf '%-22s stages="%s"\n' "$name" "$STAGES"
    fi
done
echo
echo "jobs: $N   (each walks 6 node counts, subset=total)"
[ "$SUBMIT" = 1 ] || echo "dry run -- add --submit"
