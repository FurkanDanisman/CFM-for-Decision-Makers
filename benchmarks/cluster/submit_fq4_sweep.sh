#!/bin/bash
# The fq4 case-study sweep: generation + dumps for every (case, shift), as arrays.
#
# COST, because this is large. cases x shifts x d x realizations x models =
# 6 x 3 x 6 x 10 x 13 = 14,040 model runs. Each is DRAWS resampled datasets at
# QUERIES queries; the smoke run took ~2 min for 2 datasets, most of it model load,
# so 100 datasets is roughly 15-30 min -> 3,500-7,000 CPU-hours total. Submit a
# subset first (CASES / SHIFTS / DS) and widen once the numbers are worth it.
#
#   bash R-PFN/benchmarks/cluster/submit_fq4_sweep.sh              # dry run
#   bash R-PFN/benchmarks/cluster/submit_fq4_sweep.sh --submit
#   CASES=Observed_Confounder DS=5 bash ... --submit                # one slice
#
# Env: CASES, SHIFTS, DS, REALS, QUERIES, DRAWS, ACCOUNT, THROTTLE, CTX, FQ4, OUT4
set -uo pipefail
SUBMIT=0; [ "${1:-}" = "--submit" ] && SUBMIT=1
KIT="${KIT:-$PWD}"; REPO="${REPO:-$KIT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"
CASES="${CASES:-Observed_Confounder Observed_Mediator Observed_Mediator_and_Confounder Unobserved_Confounder Frontdoor_Criterion Backdoor_Criterion}"
SHIFTS="${SHIFTS:-0 +2 -2}"
DS="${DS:-5 10 20 30 40 50}"
REALS="${REALS:-10}"
QUERIES="${QUERIES:-10}"
DRAWS="${DRAWS:-100}"
CTX="${CTX:-1000}"
ACCT="${ACCOUNT:-def-rgrosse}"
# %N caps how many array tasks run at once. Without it one (case, shift) can put 780
# tasks in flight and starve everything else the account is running.
THROTTLE="${THROTTLE:-40}"
# PHASE picks which half to submit. Needed because a 780-task dump array counts
# against the account's MaxSubmitJobs, so a full sweep gets partially REJECTED and
# leaves cells with generation done and no dumps queued. Re-running the pair would
# burn a slot on a generation that only skips; PHASE=dump submits the missing half
# alone, with no dependency since the cells already exist.
PHASE="${PHASE:-both}"
case "$PHASE" in both|gen|dump) ;; *) echo "PHASE must be both|gen|dump" >&2; exit 1 ;; esac
mkdir -p "$KIT/logs_fq"

ND=$(echo "$DS" | wc -w); NC=$(echo "$CASES" | wc -w); NS=$(echo "$SHIFTS" | wc -w)
NM=13
GEN_N=$(( ND * REALS ))
DUMP_N=$(( ND * REALS * NM ))
echo "cases=$NC shifts=$NS d=$ND reals=$REALS models=$NM draws=$DRAWS queries=$QUERIES"
echo "per (case,shift): gen array 0-$((GEN_N-1)), dump array 0-$((DUMP_N-1))"
echo "TOTAL: $(( NC * NS * GEN_N )) generations, $(( NC * NS * DUMP_N )) model runs"
echo "throttle: $THROTTLE concurrent tasks per array   phase: $PHASE"
echo "CONCURRENT TASKS if all of these land: $(( NC * NS * THROTTLE ))"
echo

for case in $CASES; do
  for sh in $SHIFTS; do
    if [ "$SUBMIT" = 1 ]; then
        gid=""
        if [ "$PHASE" != dump ]; then
            gid=$(CASE="$case" SHIFT="$sh" DS="$DS" REALS="$REALS" QUERIES="$QUERIES" \
                  DRAWS="$DRAWS" CTX="$CTX" \
                  sbatch --parsable --account="$ACCT" \
                         --array="0-$((GEN_N-1))%$THROTTLE" \
                         --job-name="fq4g-$case-$sh" \
                         "$REPO/benchmarks/cluster/submit_fq4_gen.sbatch") || gid=""
        fi
        did=""
        if [ "$PHASE" != gen ]; then
            # afterok on the whole gen array when we just submitted it: a dump that
            # starts against a half-written cell reads a truncated npz and reports a
            # NUMBER rather than failing. With PHASE=dump the cells are already there.
            did=$(CASE="$case" SHIFT="$sh" DS="$DS" REALS="$REALS" QUERIES="$QUERIES" \
                  CTX="$CTX" \
                  sbatch --parsable --account="$ACCT" \
                         --array="0-$((DUMP_N-1))%$THROTTLE" \
                         ${gid:+--dependency=afterok:$gid} \
                         --job-name="fq4d-$case-$sh" \
                         "$REPO/benchmarks/cluster/submit_fq4_dump.sbatch") || did=""
        fi
        printf '%-36s shift%-3s gen=%-10s dump=%-10s%s\n' "$case" "$sh" \
               "${gid:--}" "${did:--}" \
               "$([ "$PHASE" != gen ] && [ -z "$did" ] && echo '  <- REJECTED')"
    else
        printf '%-36s shift%-3s gen 0-%d, dump 0-%d\n' "$case" "$sh" \
               "$((GEN_N-1))" "$((DUMP_N-1))"
    fi
  done
done
echo
[ "$SUBMIT" = 1 ] || echo "dry run -- add --submit"
