#!/bin/bash
# Scoring progress, split by what is actually being computed.
#
# The three columns cost wildly different amounts -- vx and raw are seconds, MALC is
# minutes to tens of minutes depending on B -- so one combined number hides which one is
# the holdup. MALC parts carry their B in the filename, so a table can never silently mix
# bootstrap sizes.
#
#   bash R-PFN/benchmarks/cluster/fq_score_progress.sh
set -uo pipefail
OUT_DIR="${OUT_DIR:-$HOME/projects/def-rgrosse/$USER/fq4_scores}"
NCS="${NCS:-45}"; NCM="${NCM:-15}"; NM="${NM:-13}"
TOT=$(( (NCS + NCM) * NM ))

cnt() { ls "$OUT_DIR"/$1 2>/dev/null | wc -l; }
csm() { ls "$OUT_DIR"/$1 2>/dev/null | grep -vc CMECH || true; }
cmm() { ls "$OUT_DIR"/$1 2>/dev/null | grep -c CMECH || true; }

printf '%-22s %8s %8s %8s   %s\n' part case cmech total "of $(( NCS*NM )) / $(( NCM*NM ))"
printf '%-22s %8s %8s %8s\n' ---------------------- -------- -------- --------
for spec in "vx:.*_vx_*.md" "raw (bayesian):.*_none_*.md.r" \
            "MALC B=100:.*_malc100_*.md.r" "MALC B=1000:.*_malc1000_*.md.r" \
            "MALC (no B, legacy):.*_malc_*.md.r"; do
    name="${spec%%:*}"; pat="${spec#*:}"
    printf '%-22s %8s %8s %8s\n' "$name" "$(csm "$pat")" "$(cmm "$pat")" "$(cnt "$pat")"
done
echo
printf 'SCORED cells (json):   case %s/%s   cmech %s/%s\n' \
    "$(csm '*_four.json')" "$NCS" "$(cmm '*_four.json')" "$NCM"
echo
echo "A part is one (cell, model). Full set is $(( NCS*NM )) case + $(( NCM*NM )) cmech = $TOT each."
echo "vx and raw are seconds per part; MALC is the slow one, and its B is in the name so"
echo "a run at one B never reuses a part computed at another."
