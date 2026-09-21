#!/bin/bash
# Move the ComplexMech dumps and the re-dumped PSID_bal results from killarney to
# nibi.
#
# RUN THIS ON YOUR LAPTOP, not on either cluster. killarney cannot ssh to nibi (no
# key), nibi cannot ssh to killarney, and nibi cannot even resolve killarney's
# hostname. The laptop is the only host with credentials for both.
#
# Two separate steps rather than a pipe: both ssh commands prompt for Duo, and in a
# pipeline they compete for stdin -- the passcode reaches one while the other times
# out. Each step authenticates alone.
#
# tar, not rsync: these are tens of thousands of small files, where rsync's
# per-file round trip dominates (466k files measured at ~5 days vs ~1 hour).
#
#   bash xfer_killarney_to_nibi.sh --pull     # killarney -> laptop
#   bash xfer_killarney_to_nibi.sh --push     # laptop -> nibi
#   bash xfer_killarney_to_nibi.sh --verify   # compare file counts
#
# NOTE: this deliberately does NOT touch nibi's `perreal`. That holds the
# case-study MALC scoring, which has been running for hours and does not exist on
# killarney -- overwriting it would destroy work.

set -uo pipefail
KUSER="${KUSER:-furkanbd}"
KHOST="${KHOST:-killarney.alliancecan.ca}"
NHOST="${NHOST:-nibi.alliancecan.ca}"
KSCRATCH="${KSCRATCH:-/scratch/furkanbd}"
NSCRATCH="${NSCRATCH:-/scratch/furkanbd}"
STAGE="${STAGE:-$HOME/xfer_k2n.tar}"

# What moves. cmech_dumps is the new work; the PSID_bal directories are the
# re-dumps that fixed the balance-subsample bug.
PATHS=(
  "cmech_dumps"
  "rc_dens_uni/dopfn_native/PSID_bal"
  "dumps_all/dopfn_repro_1d_J10/rc/dopfn_native/PSID_bal"
  "dumps_all/dopfn_repro_1d_J100/rc/dopfn_native/PSID_bal"
  "dumps_all/dopfn_repro_joint2d/rc/dopfn_native/PSID_bal"
)

case "${1:---help}" in
--pull)
    echo "checking which paths exist on $KHOST ..."
    EXISTS=$(ssh "$KUSER@$KHOST" "cd $KSCRATCH && for p in ${PATHS[*]}; do \
        [ -e \"\$p\" ] && echo \"\$p\"; done")
    [ -n "$EXISTS" ] || { echo "FATAL: none of the paths exist on $KHOST" >&2; exit 1; }
    echo "$EXISTS" | sed 's/^/  /'
    echo
    echo "streaming to $STAGE ..."
    # shellcheck disable=SC2086
    ssh "$KUSER@$KHOST" "tar cf - -C $KSCRATCH $(echo $EXISTS | tr '\n' ' ')" > "$STAGE"
    ls -lh "$STAGE" | awk '{print "  staged: "$5" -> "$9}'
    echo "now: bash $0 --push"
    ;;
--push)
    [ -s "$STAGE" ] || { echo "FATAL: no staged archive at $STAGE (run --pull)" >&2; exit 1; }
    echo "pushing $(ls -lh "$STAGE" | awk '{print $5}') to $NHOST ..."
    ssh "$KUSER@$NHOST" "mkdir -p $NSCRATCH && tar xf - -C $NSCRATCH" < "$STAGE"
    echo "done. verify, then: rm $STAGE"
    ;;
--verify)
    for h in "$KHOST" "$NHOST"; do
        s="$KSCRATCH"; [ "$h" = "$NHOST" ] && s="$NSCRATCH"
        echo "=== $h"
        # shellcheck disable=SC2086
        ssh "$KUSER@$h" "cd $s 2>/dev/null && for p in ${PATHS[*]}; do \
            if [ -e \"\$p\" ]; then \
              printf '  %-58s %6s npz\n' \"\$p\" \
                \"\$(find \"\$p\" -name '*.npz' | wc -l | tr -d ' ')\"; \
            else printf '  %-58s %6s\n' \"\$p\" absent; fi; done"
    done
    echo
    echo "counts must match. A short count on nibi means the tar was truncated."
    ;;
*)
    sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
