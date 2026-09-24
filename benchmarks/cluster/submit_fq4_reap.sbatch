#!/bin/bash
#SBATCH --job-name=fq4-reap
#SBATCH --output=logs_fq/fq4reap_%j.out
#SBATCH --error=logs_fq/fq4reap_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#
# Score complete fq4 cells INCLUDING MALC, then delete their dumps.
#
# WHY A SEPARATE JOB. A cell cannot be scored until all 13 models have dumped it, and
# each model is its own job, so no dump job can know when a cell is finished. This one
# polls instead: each pass it finds cells where every model has its R replicates,
# scores them, and removes the raw npz.
#
# WHY DELETE AT ALL. One cell is R data files + 13 x R dumps. At R=30 that is ~420
# files; 90 cells is 37,800, and the six-case scope would be 1.5M against a 1M inode
# quota. After scoring, the only thing the final table needs is the cell's _four.json,
# a few KB. So the raw files are dead weight the moment MALC has run.
#
# WHY MALC MUST RUN FIRST. MALC recomputes from the dumped densities. Delete those and
# its two columns can never be produced for that cell -- which is exactly what
# happened to the first four cells, scored with SKIP_MALC=1.
#
# Scores land OUTSIDE scratch by default: writing them is what failed once scratch hit
# its inode cap, and a scoring pass that cannot save its own output wastes the compute.
#
# Env: CASES, SHIFTS, DS, REALS, DRAWS, QUERIES, OUT_DIR, FQ4, OUT4, MALC_B,
#      KEEP=1 (score but do not delete), PASSES, SLEEP
set -uo pipefail
mkdir -p logs_fq

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
# shellcheck disable=SC1091
source "$DEPLOY_ROOT/venv/bin/activate"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"

CASES="${CASES:-Observed_Confounder}"
SHIFTS="${SHIFTS:-0 +2 -2}"
read -r -a DSA <<<"${DS:-5 10 20 30 40 50}"
REALS="${REALS:-5}"
DRAWS="${DRAWS:-30}"
QUERIES="${QUERIES:-10}"
CTX="${CTX:-1000}"
FQ4="${FQ4:-$SC/fq4}"
OUT4="${OUT4:-$SC/fq4_dumps}"
OUT_DIR="${OUT_DIR:-$HOME/projects/def-rgrosse/$USER/fq4_scores}"
MALC_B="${MALC_B:-1000}"
NMODELS="${NMODELS:-13}"
PASSES="${PASSES:-48}"
SLEEP="${SLEEP:-900}"
mkdir -p "$OUT_DIR"

echo "[$(date '+%F %T')] reaper: R=$DRAWS, $NMODELS models, scores -> $OUT_DIR"
echo "  ${KEEP:+KEEP=1 -- scoring only, nothing deleted}"

# MODE=cmech watches the ComplexMech cells instead. Their tag and their data-cell
# path differ from the case study's, and the reaper has to know BOTH: the scorer needs
# --data-cell to read truths, and the delete has to remove the right directories.
MODE="${MODE:-cs}"
TAGS=(); CELLS=(); DATAS=()
if [ "$MODE" = cmech ]; then
  CMFQ="${CMFQ:-$SC/cmech_fq}"
  OUT4="${OUT4:-$SC/cmech_fq_dumps}"
  REGIME="${REGIME:-path_TY}"; HIDE="${HIDE:-0.0}"; SUBSET="${SUBSET:-nonzero}"
  for n in "${DSA[@]}"; do
    for (( r=0; r<REALS; r++ )); do
      TAGS+=("CMECH_n${n}_${SUBSET}_r${r}")
      CELLS+=("$CMFQ/n$n/r$r/complexmech/${n}node/$REGIME/hide_$HIDE")
      DATAS+=("$CMFQ/n$n/r$r")
    done
  done
else
  for c in $CASES; do for s in $SHIFTS; do for d in "${DSA[@]}"; do
    for (( r=0; r<REALS; r++ )); do
      TAGS+=("${c}_shift${s}_d${d}_r${r}")
      CELLS+=("$FQ4/${c}_shift${s}_d${d}_r${r}/$c/N$CTX")
      DATAS+=("$FQ4/${c}_shift${s}_d${d}_r${r}")
    done
  done; done; done
fi
echo "  mode=$MODE, watching ${#TAGS[@]} cell(s)"

for (( pass=1; pass<=PASSES; pass++ )); do
  did=0; pend=0; done_n=0
  for i in "${!TAGS[@]}"; do
    TAG="${TAGS[$i]}"; CELL="${CELLS[$i]}"; DATA="${DATAS[$i]}"
    [ -f "$OUT_DIR/${TAG}_four.json" ] && { done_n=$((done_n+1)); continue; }
    DUMPS="$OUT4/$TAG"
    [ -d "$DUMPS" ] || { pend=$((pend+1)); continue; }
    # complete = every model directory holds at least R replicates
    ready=1; nm=0
    for md in "$DUMPS"/*/; do
      [ -d "$md" ] || continue
      nm=$((nm+1))
      n=$(find "$md" -name '*.npz' ! -name 'summary.npz' 2>/dev/null | wc -l)
      [ "$n" -ge "$DRAWS" ] || { ready=0; break; }
    done
    [ "$nm" -ge "$NMODELS" ] || ready=0
    [ "$ready" = 1 ] || { pend=$((pend+1)); continue; }

    echo "[$(date '+%F %T')] scoring $TAG ($nm models)"
    if DUMPS="$DUMPS" QUERIES="$QUERIES" DRAWS="$DRAWS" CTX="$CTX" \
       OUT_DIR="$OUT_DIR" CELL_OVERRIDE="$CELL" MALC_B="$MALC_B" \
       CASE_OVERRIDE="${CASE_FOR_SCORER:-}" \
       SKIP_MALC="${SKIP_MALC:-0}" \
       WORKERS="${SLURM_CPUS_PER_TASK:-8}" \
       bash "$REPO/benchmarks/cluster/score_fq4.sh" >>"logs_fq/fq4reap_cells.log" 2>&1
    then
        # Only delete once the JSON exists on disk. A scorer that ran but failed to
        # write -- the inode failure -- must not take the dumps with it.
        if [ -f "$OUT_DIR/${TAG}_four.json" ]; then
            if [ "${KEEP:-0}" = 1 ]; then
                echo "  $TAG scored (kept)"
            else
                rm -rf "$DUMPS" "$DATA" "${DATA}_root"
                echo "  $TAG scored and reaped"
            fi
            did=$((did+1)); done_n=$((done_n+1))
        else
            echo "  $TAG scored but NO json -- kept, see logs_fq/fq4reap_cells.log"
        fi
    else
        echo "  $TAG scoring FAILED -- kept, see logs_fq/fq4reap_cells.log"
    fi
  done
  echo "[$(date '+%F %T')] pass $pass: $did scored, $done_n/${#TAGS[@]} complete, $pend pending"
  [ "$done_n" -ge "${#TAGS[@]}" ] && { echo "all cells done"; break; }
  [ "$pass" -lt "$PASSES" ] && sleep "$SLEEP"
done

python "$REPO/benchmarks/fq4_aggregate.py" --scores "$OUT_DIR" \
    --group shift,d --out "$OUT_DIR/fq4_final.md" || true
echo "[$(date '+%F %T')] reaper done"
