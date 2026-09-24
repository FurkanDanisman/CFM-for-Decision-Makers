#!/bin/bash
# The fq4 case-study sweep: 1 generation job per (case, shift), 1 dump job per model.
#
# For one case that is 3 + 13 = 16 jobs. Each loops its own cells internally, so the
# scheduler sees 16 entries instead of the 15,120 array tasks the first version
# submitted -- which is also why most of that version was REJECTED on the account's
# submit limit. Cost is unchanged; only the packaging is.
#
# Dump jobs wait on afterok of every generation job, and each is resumable: a cell
# already holding DRAWS npz is skipped, so a job killed at the wall continues on
# resubmission rather than starting over.
#
#   bash R-PFN/benchmarks/cluster/submit_fq4_sweep.sh              # dry run
#   bash R-PFN/benchmarks/cluster/submit_fq4_sweep.sh --submit
#   PHASE=dump ONLY="uwyk1d graph2d" bash ... --submit
#
# Env: CASES, SHIFTS, DS, REALS, QUERIES, DRAWS, ACCOUNT, PHASE, ONLY, CTX, FQ4, OUT4
set -uo pipefail
SUBMIT=0; [ "${1:-}" = "--submit" ] && SUBMIT=1
KIT="${KIT:-$PWD}"; REPO="${REPO:-$KIT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"
CK="${CK:-$REPO/Required_checkpoints}"
UWYKD="${UWYK_CKPT_DIR:-$KIT/external/uwyk_reproduce/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
# shellcheck disable=SC1091
source "$REPO/benchmarks/cluster/fq_models.sh"

CASES="${CASES:-Observed_Confounder}"
SHIFTS="${SHIFTS:-0 +2 -2}"
DS="${DS:-5 10 20 30 40 50}"
REALS="${REALS:-5}"
QUERIES="${QUERIES:-10}"
DRAWS="${DRAWS:-30}"
CTX="${CTX:-1000}"
ACCT="${ACCOUNT:-def-rgrosse}"
PHASE="${PHASE:-both}"
ONLY="${ONLY:-}"
GEN_TIME="${GEN_TIME:-06:00:00}"
DUMP_TIME="${DUMP_TIME:-24:00:00}"
# GRES / CPU_ONLY for the dump jobs. nibi rejects a bare gpu:N, so a TYPE is required.
# Default stays CPU: the ten fast models gain nothing from a GPU and would only wait
# longer in its queue.
GRES="${GRES:-none}"
CPU_ONLY="${CPU_ONLY:-1}"
case "$PHASE" in both|gen|dump) ;; *) echo "PHASE must be both|gen|dump" >&2; exit 1 ;; esac
mkdir -p "$KIT/logs_fq"

NC=$(echo "$CASES" | wc -w); NS=$(echo "$SHIFTS" | wc -w); ND=$(echo "$DS" | wc -w)
NGEN=$(( NC * NS ))
CELLS=$(( NC * NS * ND * REALS ))
MODELS=()
for r in "${ROWS[@]}"; do
    IFS='|' read -r name _h _e <<<"$r"
    [ -n "$ONLY" ] && { case " $ONLY " in *" $name "*) ;; *) continue ;; esac; }
    MODELS+=("$name")
done
NDUMP=${#MODELS[@]}

echo "cases=$NC shifts=$NS d=$ND reals=$REALS draws=$DRAWS queries=$QUERIES"
echo "cells: $CELLS   models: $NDUMP"
NG=$NGEN; NDU=$NDUMP
[ "$PHASE" = dump ] && NG=0
[ "$PHASE" = gen ] && NDU=0
echo "JOBS TO SUBMIT: $NG gen + $NDU dump = $(( NG + NDU ))"
echo "CONCURRENT: at most that many, one task each (no arrays)"
echo "each gen job walks $(( ND * REALS )) cells; each dump job walks $CELLS cells"
echo "dump gres=$GRES cpu_only=$CPU_ONLY"
echo

GIDS=()
if [ "$PHASE" != dump ]; then
  for case in $CASES; do
    for sh in $SHIFTS; do
      if [ "$SUBMIT" = 1 ]; then
        gid=$(CASE="$case" SHIFT="$sh" DS="$DS" REALS="$REALS" QUERIES="$QUERIES" \
              DRAWS="$DRAWS" CTX="$CTX" \
              sbatch --parsable --account="$ACCT" --time="$GEN_TIME" \
                     --job-name="fq4g-$case-$sh" \
                     "$REPO/benchmarks/cluster/submit_fq4_gen.sbatch") || gid=""
        [ -n "$gid" ] && GIDS+=("$gid")
        printf '  gen  %-36s shift%-3s %s\n' "$case" "$sh" "${gid:-REJECTED}"
      else
        printf '  gen  %-36s shift%-3s (dry)\n' "$case" "$sh"
      fi
    done
  done
fi

if [ "$PHASE" != gen ]; then
  DEP=""
  # afterok on every generation job: a dump reading a half-written cell gets a
  # truncated npz and reports a NUMBER rather than failing.
  [ "${#GIDS[@]}" -gt 0 ] && DEP="afterok:$(IFS=:; echo "${GIDS[*]}")"
  for m in "${MODELS[@]}"; do
    if [ "$SUBMIT" = 1 ]; then
      did=$(MODEL="$m" CASES="$CASES" SHIFTS="$SHIFTS" DS="$DS" REALS="$REALS" \
            QUERIES="$QUERIES" DRAWS="$DRAWS" CTX="$CTX" CPU_ONLY="$CPU_ONLY" \
            sbatch --parsable --account="$ACCT" --time="$DUMP_TIME" \
                   --gres="$GRES" \
                   ${DEP:+--dependency=$DEP} --job-name="fq4d-$m" \
                   "$REPO/benchmarks/cluster/submit_fq4_dump.sbatch") || did=""
      printf '  dump %-36s %s\n' "$m" "${did:-REJECTED}"
    else
      printf '  dump %-36s (dry)\n' "$m"
    fi
  done
fi
echo
[ "$SUBMIT" = 1 ] || echo "dry run -- add --submit"
