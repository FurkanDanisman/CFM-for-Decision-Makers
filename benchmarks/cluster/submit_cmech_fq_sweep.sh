#!/bin/bash
# ComplexMech fixed-query sweep: 1 generation job per node count + 1 dump job per model.
#
# 6 + 13 = 19 jobs. Each loops its own cells, matching the case-study sweep's shape --
# not an array, because 60 cells x 13 models would be 780 scheduler entries and the
# account's submit limit refuses those.
#
#   bash R-PFN/benchmarks/cluster/submit_cmech_fq_sweep.sh              # dry run
#   bash R-PFN/benchmarks/cluster/submit_cmech_fq_sweep.sh --submit
#   NODES_LIST=5 PHASE=gen bash ... --submit
#
# Env: NODES_LIST, REALS, DRAWS, NTEST, QUERIES, RHO_MIN, REGIME, HIDE, SUBSET,
#      ACCOUNT, PHASE, ONLY, CMFQ, CMFQ_DUMPS
set -uo pipefail
SUBMIT=0; [ "${1:-}" = "--submit" ] && SUBMIT=1
KIT="${KIT:-$PWD}"; REPO="${REPO:-$KIT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"
CK="${CK:-$REPO/Required_checkpoints}"
# shellcheck disable=SC1091
source "$REPO/benchmarks/cluster/cmech_models.sh"

NODES_LIST="${NODES_LIST:-5 10 20 30 40 50}"
REALS="${REALS:-10}"
DRAWS="${DRAWS:-100}"
NTEST="${NTEST:-100}"
QUERIES="${QUERIES:-10}"
RHO_MIN="${RHO_MIN:-0.99}"
REGIME="${REGIME:-path_TY}"
HIDE="${HIDE:-0.0}"
SUBSET="${SUBSET:-nonzero}"
ACCT="${ACCOUNT:-def-rgrosse}"
PHASE="${PHASE:-both}"
ONLY="${ONLY:-}"
GEN_TIME="${GEN_TIME:-12:00:00}"
DUMP_TIME="${DUMP_TIME:-24:00:00}"
case "$PHASE" in both|gen|dump) ;; *) echo "PHASE must be both|gen|dump" >&2; exit 1 ;; esac
mkdir -p "$KIT/logs_fq"

NN=$(echo "$NODES_LIST" | wc -w)
MODELS=()
for r in "${ROWS[@]}"; do
    IFS='|' read -r name _i _e _c _x <<<"$r"
    [ -n "$ONLY" ] && { case " $ONLY " in *" $name "*) ;; *) continue ;; esac; }
    MODELS+=("$name")
done
NG=$NN; ND=${#MODELS[@]}
[ "$PHASE" = dump ] && NG=0
[ "$PHASE" = gen ] && ND=0
echo "nodes=$NN reals=$REALS draws=$DRAWS queries=$QUERIES rho>=$RHO_MIN subset=$SUBSET"
echo "cells: $(( NN * REALS ))   models: ${#MODELS[@]}"
echo "JOBS TO SUBMIT: $NG gen + $ND dump = $(( NG + ND ))"
echo "each dump job walks $(( NN * REALS )) cells"
echo

GIDS=()
if [ "$PHASE" != dump ]; then
  for n in $NODES_LIST; do
    if [ "$SUBMIT" = 1 ]; then
      gid=$(NODES="$n" REGIME="$REGIME" HIDE="$HIDE" REALS="$REALS" DRAWS="$DRAWS" \
            NTEST="$NTEST" QUERIES="$QUERIES" RHO_MIN="$RHO_MIN" \
            sbatch --parsable --account="$ACCT" --time="$GEN_TIME" \
                   --job-name="cmfqg-n$n" \
                   "$REPO/benchmarks/cluster/submit_cmech_fq_gen.sbatch") || gid=""
      [ -n "$gid" ] && GIDS+=("$gid")
      printf '  gen  nodes=%-4s %s\n' "$n" "${gid:-REJECTED}"
    else
      printf '  gen  nodes=%-4s (dry)\n' "$n"
    fi
  done
fi

if [ "$PHASE" != gen ]; then
  DEP=""
  # afterok on every generation job: a dump against a half-written cell reads a
  # truncated npz and reports a NUMBER rather than failing.
  [ "${#GIDS[@]}" -gt 0 ] && DEP="afterok:$(IFS=:; echo "${GIDS[*]}")"
  for m in "${MODELS[@]}"; do
    if [ "$SUBMIT" = 1 ]; then
      did=$(MODEL="$m" NODES_LIST="$NODES_LIST" REALS="$REALS" REGIME="$REGIME" \
            HIDE="$HIDE" DRAWS="$DRAWS" SUBSET="$SUBSET" \
            sbatch --parsable --account="$ACCT" --time="$DUMP_TIME" \
                   ${DEP:+--dependency=$DEP} --job-name="cmfqd-$m" \
                   "$REPO/benchmarks/cluster/submit_cmech_fq_dump.sbatch") || did=""
      printf '  dump %-24s %s\n' "$m" "${did:-REJECTED}"
    else
      printf '  dump %-24s (dry)\n' "$m"
    fi
  done
fi
echo
[ "$SUBMIT" = 1 ] || echo "dry run -- add --submit"
