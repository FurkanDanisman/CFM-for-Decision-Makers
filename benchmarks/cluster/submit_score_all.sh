#!/bin/bash
# Re-score EVERY model under the per-realization aggregation. Scoring only.
#
# MALC is fitted from the dumped densities at score time, so nothing needs
# re-dumping -- but every existing MALC number was computed with the POOLED mean
# and is therefore stale. B=1000 is the whole cost, so this fans out one job per
# (root, slice) instead of one job for everything.
#
# Roots, not models: the six original models share rc_dens_uni / cs_dvar_dens, and
# scoring that root covers all of them in one pass because METHODS walks its
# subdirs. Each new model has its own root.
#
#   bash R-PFN/benchmarks/cluster/submit_score_all.sh            # dry run
#   bash R-PFN/benchmarks/cluster/submit_score_all.sh --submit
#   STAGES="point raw" bash ...   # skip MALC for a fast first table

set -uo pipefail
# Paths are DERIVED, not hardcoded, so this runs on any cluster: the script lives
# at <kit>/R-PFN/benchmarks/cluster/, so the kit root is three levels up. SCRATCH
# must be set by the environment -- guessing a per-cluster scratch path is how a
# run silently writes to the wrong filesystem.
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_KIT_DEFAULT="$(cd "$_SELF_DIR/../../.." && pwd)"
SUBMIT=0; [ "${1:-}" = "--submit" ] && SUBMIT=1
KIT="${KIT:-$_KIT_DEFAULT}"
REPO="${REPO:-$KIT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"
SB="$REPO/benchmarks/cluster/submit_full_table.sbatch"
STAGES="${STAGES:-point raw malc indep}"
RC_DS="${RC_DS:-IHDP ACIC CPS PSID PSID_bal}"
SHIFTS="${SHIFTS:-0 +2 -2}"
# CS_DS empty = one job per shift (fine for single-method roots). Set it to the d
# list to split per (shift, d), which MALC needs on the multi-method root.
CS_DS="${CS_DS:-}"
# Default 3h so scoring lands in the 3h tier (168 l40s nodes) rather than the 12h
# tier (126). Raw/point/indep finish well inside that. MALC at B=1000 is the one
# stage that may not, so run it as its own wave with SCORE_TIME=12:00:00 --
# which is also what keeps partial results flowing: the cheap columns land first.
T="${SCORE_TIME:-3:00:00}"
# SCORE_PARTITION: there is no CPU partition on this cluster, so this normally
# stays unset and the walltime tier does the work instead. Kept for portability.
PART=""; [ -n "${SCORE_PARTITION:-}" ] && PART="--partition=$SCORE_PARTITION"

# label | rc root | cs root
ROOTS=(
  "orig|$SC/rc_dens_uni|$SC/cs_dvar_dens"
  "eta0|$SC/rc_dens_eta0|$SC/cs_dvar_eta0"
  "J10|$SC/dumps_all/dopfn_repro_1d_J10/rc|$SC/dumps_all/dopfn_repro_1d_J10/cs"
  "J100|$SC/dumps_all/dopfn_repro_1d_J100/rc|$SC/dumps_all/dopfn_repro_1d_J100/cs"
  "joint2d|$SC/dumps_all/dopfn_repro_joint2d/rc|$SC/dumps_all/dopfn_repro_joint2d/cs"
  "j32|$SC/dumps_all/cpfn1d_j32/rc|$SC/dumps_all/cpfn1d_j32/cs"
  "botharms|$SC/dumps_all/cpfn1d_botharms/rc|$SC/dumps_all/cpfn1d_botharms/cs"
  "cpfn_v0|$SC/dumps_all/cpfn_v0/rc|$SC/dumps_all/cpfn_v0/cs"
  "uwyk_bin|$SC/dumps_all/uwyk_bin/rc|$SC/dumps_all/uwyk_bin/cs"
)

cd "$KIT" || exit 1
N=0
for r in "${ROOTS[@]}"; do
    IFS='|' read -r lbl rc cs <<<"$r"
    # ONLY accepts a LIST, so a wave can name several roots in one submission.
    if [ -n "${ONLY:-}" ]; then
        case " $ONLY " in *" $lbl "*) ;; *) continue ;; esac
    fi
    case " ${SKIP:-} " in *" $lbl "*) printf 'score %-10s SKIPPED\n' "$lbl"; continue ;; esac
    # RealCause: one job per dataset.
    if [ -d "$rc" ]; then
        for ds in $RC_DS; do
            N=$((N+1))
            if [ "$SUBMIT" = 1 ]; then
                printf 'score %-10s rc/%-9s -> ' "$lbl" "$ds"
                ONE_RC_ROOT="$rc" SKIP_CS=1 RC_DATASETS="$ds" STAGES="$STAGES" LABEL="$lbl" \
                    sbatch --time="$T" $PART --job-name="sc-$lbl-$ds" "$SB"
            else printf 'score %-10s rc/%s\n' "$lbl" "$ds"; fi
        done
    else printf 'score %-10s rc  SKIP (no root)\n' "$lbl"; fi
    # Case studies: one job per shift.
    if [ -d "$cs" ]; then
        for sh in $SHIFTS; do
            for dv in ${CS_DS:-ALL}; do
                N=$((N+1))
                _d=""; _t="cs$sh"
                [ "$dv" != ALL ] && { _d="$dv"; _t="cs$sh-d$dv"; }
                if [ "$SUBMIT" = 1 ]; then
                    printf 'score %-10s %-14s -> ' "$lbl" "$_t"
                    ONE_CS_ROOT="$cs" SKIP_RC=1 CS_SHIFT="$sh" CS_D="$_d" \
                    STAGES="$STAGES" LABEL="$lbl" \
                        sbatch --time="$T" $PART --job-name="sc-$lbl-$_t" "$SB"
                else printf 'score %-10s %s\n' "$lbl" "$_t"; fi
            done
        done
    else printf 'score %-10s cs  SKIP (no root)\n' "$lbl"; fi
done
echo
echo "jobs: $N"
[ "$SUBMIT" = 1 ] || echo "dry run -- add --submit"
