#!/bin/bash
#SBATCH --job-name=cmfq-dump
#SBATCH --output=logs_fq/cmfq_dump_%j.out
#SBATCH --error=logs_fq/cmfq_dump_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# Dump ONE model across every ComplexMech fixed-query cell: loops node count x
# realization. 6 node counts x 10 realizations = 60 cells, each of DRAWS replicates.
#
# One job per model, so 13 jobs -- not 60 x 13. The checkpoint loads once and is
# reused for every cell, which is most of the per-cell cost, and 780 scheduler entries
# would be refused by the account's submit limit anyway.
#
# Resumable: a cell whose output already holds DRAWS npz is skipped, so a job killed
# at the wall continues on resubmission.
#
# Required: MODEL (a name from cmech_models.sh)
# Env: NODES_LIST, REALS, REGIME, HIDE, DRAWS, SUBSET, CMFQ, CMFQ_DUMPS, CTX
set -uo pipefail
mkdir -p logs_fq
: "${MODEL:?MODEL required, e.g. MODEL=graph2d}"

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"
CK="${CK:-$REPO/Required_checkpoints}"
# shellcheck disable=SC1091
source "$REPO/benchmarks/cluster/cmech_models.sh"

read -r -a NL <<<"${NODES_LIST:-5 10 20 30 40 50}"
REALS="${REALS:-10}"
REGIME="${REGIME:-path_TY}"
HIDE="${HIDE:-0.0}"
DRAWS="${DRAWS:-100}"
SUBSET="${SUBSET:-nonzero}"
CTX="${CTX:-1000}"
CMFQ="${CMFQ:-$SC/cmech_fq}"
CMFQ_DUMPS="${CMFQ_DUMPS:-$SC/cmech_fq_dumps}"
INNER="$REPO/benchmarks/cluster/submit_cmech_dump_one.sbatch"

row=""
for r in "${ROWS[@]}"; do
    IFS='|' read -r name _i _e _c _x <<<"$r"
    [ "$name" = "$MODEL" ] && { row="$r"; break; }
done
[ -n "$row" ] || { echo "FATAL: no model '$MODEL' in cmech_models.sh" >&2; exit 1; }
IFS='|' read -r name idx envv ck extra <<<"$row"

if [ "$envv" != NONE ]; then
    sz=$(stat -Lc%s "$ck" 2>/dev/null || echo "")
    [ -n "$sz" ] || { echo "FATAL: missing $ck" >&2; exit 1; }
    # A git-lfs pointer is a few hundred bytes and dies as 'invalid load key' --
    # checked once, not 60 cells into a 24h allocation.
    [ "$sz" -ge 1000000 ] || { echo "FATAL: $ck is $sz bytes -- lfs pointer" >&2; exit 1; }
fi

echo "[$(date '+%F %T')] cmech-fq dump $MODEL (harness_idx=$idx)"
echo "  nodes=${NL[*]} reals=$REALS subset=$SUBSET draws=$DRAWS"
cd "$DEPLOY_ROOT"
ok=0; skip=0; fail=0
for n in "${NL[@]}"; do
 for (( ri=0; ri<REALS; ri++ )); do
  ROOT="$CMFQ/n$n/r$ri"
  CELL="$ROOT/complexmech/${n}node/$REGIME/hide_$HIDE"
  TAG="CMECH_n${n}_${SUBSET}_r${ri}"
  OUTR="$CMFQ_DUMPS/$TAG/$MODEL"
  if [ ! -d "$CELL" ]; then
      echo "  $TAG: cell absent -- SKIP"; fail=$((fail+1)); continue
  fi
  HAVE=$(find "$OUTR" -name '*.npz' ! -name 'summary.npz' 2>/dev/null | wc -l)
  if [ "$HAVE" -ge "$DRAWS" ]; then skip=$((skip+1)); continue; fi
  # CMECH_NODES pinned to this root's single node count: the inner index is
  # model_idx*(n_nodes*n_contexts)+..., so with one node and one context it reduces
  # to model_idx and the dispatch picks the right harness.
  if env MODEL_NAME="$name" MODEL_IDX="$idx" CKPT_ENV="$envv" CKPT_PATH="$ck" \
         EXTRA_ENV="$extra" OUT_ROOT="$OUTR" UWYK_FIG34_DATA="$ROOT" \
         CMECH_NODES="$n" SUBSETS="$SUBSET" CMECH_CONTEXTS="$CTX" CPU_ONLY=1 \
         bash "$INNER" >>"logs_fq/cmfq_dump_${MODEL}_cells.log" 2>&1; then
      ok=$((ok+1)); echo "  $TAG ok ($(date '+%H:%M:%S'))"
  else
      fail=$((fail+1)); echo "  $TAG FAILED -- see logs_fq/cmfq_dump_${MODEL}_cells.log"
  fi
 done
done
echo "[$(date '+%F %T')] cmech-fq dump $MODEL done: $ok new, $skip already there, $fail failed"
