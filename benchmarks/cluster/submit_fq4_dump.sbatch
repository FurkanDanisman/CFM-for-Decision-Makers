#!/bin/bash
#SBATCH --job-name=fq4-dump
#SBATCH --output=logs_fq/fq4dump_%j.out
#SBATCH --error=logs_fq/fq4dump_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# Dump ONE model across every fq4 cell: loops case x shift x d x realization.
#
# One job per model, not an array over cells. The model is loaded once and then
# reused for every cell, which is most of the per-cell cost at these sizes -- the
# smoke run spent ~2 min on 2 datasets, nearly all of it startup. An array paid that
# startup 780 times and burned 780 entries against the account's submit limit.
#
# Resumable, and it must be: 60 cells x ~15 min is around 15h, close enough to the
# wall that being killed mid-sweep is expected. A cell whose output already has
# DRAWS npz is skipped, so resubmitting continues where it stopped.
#
# Required: MODEL (a name from fq_models.sh)
# Env: CASES, SHIFTS, DS, REALS, QUERIES, DRAWS, CTX, FQ4, OUT4
set -uo pipefail
mkdir -p logs_fq
: "${MODEL:?MODEL required, e.g. MODEL=dopfn_native}"

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
UWYKD="${UWYK_CKPT_DIR:-$DEPLOY_ROOT/external/uwyk_reproduce/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"
# shellcheck disable=SC1091
source "$REPO/benchmarks/cluster/fq_models.sh"

CASES="${CASES:-Observed_Confounder}"
SHIFTS="${SHIFTS:-0 +2 -2}"
CTX="${CTX:-1000}"
read -r -a DSA <<<"${DS:-5 10 20 30 40 50}"
REALS="${REALS:-10}"
QUERIES="${QUERIES:-10}"
DRAWS="${DRAWS:-100}"
FQ4="${FQ4:-$SC/fq4}"
OUT4="${OUT4:-$SC/fq4_dumps}"

row=""
for r in "${ROWS[@]}"; do
    IFS='|' read -r name _h _e <<<"$r"
    [ "$name" = "$MODEL" ] && { row="$r"; break; }
done
[ -n "$row" ] || { echo "FATAL: no model '$MODEL' in fq_models.sh" >&2; exit 1; }
IFS='|' read -r name harness extra <<<"$row"

# A git-lfs POINTER is a few hundred bytes and dies as 'invalid load key' -- checked
# once, before spending a 24h allocation on it.
for kv in $extra; do
    case "$kv" in *=*.pt)
        f="${kv#*=}"
        sz=$(stat -Lc%s "$f" 2>/dev/null || echo "")
        [ -n "$sz" ] || { echo "FATAL: missing $f" >&2; exit 1; }
        [ "$sz" -ge 1000000 ] || { echo "FATAL: $f is $sz bytes -- lfs pointer" >&2; exit 1; } ;;
    esac
done

echo "[$(date '+%F %T')] dump $MODEL (harness=$harness)"
echo "  cases=$CASES shifts=$SHIFTS d=${DSA[*]} reals=$REALS"
cd "$DEPLOY_ROOT"
ok=0; skip=0; fail=0
for case in $CASES; do
 for sh in $SHIFTS; do
  for D in "${DSA[@]}"; do
   for (( REAL=0; REAL<REALS; REAL++ )); do
    TAG="${case}_shift${sh}_d${D}_r${REAL}"
    ROOT="$FQ4/${TAG}_root"
    OUT="$OUT4/$TAG/$MODEL"
    if [ ! -d "$ROOT/shift0/d0/$case/N$CTX" ]; then
        echo "  $TAG: cell not generated -- SKIP"; fail=$((fail+1)); continue
    fi
    HAVE=$(find "$OUT" -name '*.npz' ! -name 'summary.npz' 2>/dev/null | wc -l)
    if [ "$HAVE" -ge "$DRAWS" ]; then skip=$((skip+1)); continue; fi
    # shellcheck disable=SC2086
    if env $extra CUDA_VISIBLE_DEVICES= MODEL_OVERRIDE="$harness" D=0 SHIFT=0 \
           CTX="$CTX" NQ="$QUERIES" DATA="$ROOT" CASES_OVERRIDE="$case" \
           OUT_ROOT="$OUT" SLURM_ARRAY_TASK_ID=0 \
           bash "$REPO/benchmarks/cluster/submit_cs_dvar_density.sbatch" \
           >>"logs_fq/fq4dump_${MODEL}_cells.log" 2>&1; then
        ok=$((ok+1)); echo "  $TAG ok ($(date '+%H:%M:%S'))"
    else
        fail=$((fail+1)); echo "  $TAG FAILED -- see logs_fq/fq4dump_${MODEL}_cells.log"
    fi
   done
  done
 done
done
echo "[$(date '+%F %T')] dump $MODEL done: $ok new, $skip already there, $fail failed"
