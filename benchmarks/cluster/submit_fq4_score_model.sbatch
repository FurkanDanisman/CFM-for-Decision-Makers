#!/bin/bash
#SBATCH --job-name=fq4-score1
#SBATCH --output=logs_fq/fq4score_%j.out
#SBATCH --error=logs_fq/fq4score_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#
# Score ONE model across every cell, writing the cached per-(cell, model) parts.
#
# The reaper scored a whole cell in one job, so all 13 models queued behind each other
# and MALC at B=1000 came to ~8.4 h for 45 cells. But the parts are per model and
# independent -- scoring graph2d has nothing to do with scoring dopfn_bb -- so 13 jobs
# do the same work in the time of one model, ~45 min. Same parallelism the dumps use.
#
# This writes parts only. The reaper still merges them into the per-cell table, and
# still owns deletion; it finds the parts already present and merges in seconds.
#
# Required: MODEL
# Env: MODE (cs|cmech), CASES, SHIFTS, DS, NODES_LIST, REALS, DRAWS, QUERIES,
#      MALC_B, MALC_K, SKIP_MALC, OUT_DIR, FQ4, OUT4, CMFQ, CMFQ_DUMPS, CTX, SUBSET
set -uo pipefail
mkdir -p logs_fq
: "${MODEL:?MODEL required}"

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

MODE="${MODE:-cs}"
CASE="${CASE:-Observed_Confounder}"
SHIFTS="${SHIFTS:-0 +2 -2}"
read -r -a DSA <<<"${DS:-5 20 50}"
read -r -a NLA <<<"${NODES_LIST:-5 20 50}"
REALS="${REALS:-5}"; DRAWS="${DRAWS:-30}"; QUERIES="${QUERIES:-10}"
CTX="${CTX:-1000}"; SUBSET="${SUBSET:-nonzero}"
MALC_B="${MALC_B:-1000}"; MALC_K="${MALC_K:-1}"
FQ4="${FQ4:-$SC/fq4}"
OUT4="${OUT4:-$SC/fq4_dumps}"
CMFQ="${CMFQ:-$SC/cmech_fq}"
CMFQ_DUMPS="${CMFQ_DUMPS:-$SC/cmech_fq_dumps}"
OUT_DIR="${OUT_DIR:-$HOME/projects/def-rgrosse/$USER/fq4_scores}"
W="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$OUT_DIR"
QLIST=$(seq 0 $((QUERIES-1)) | tr '\n' ' ')

# tag | model-root | data-cell | dataset | nodes
UNITS=()
if [ "$MODE" = cmech ]; then
  for n in "${NLA[@]}"; do for (( r=0; r<REALS; r++ )); do
    UNITS+=("CMECH_n${n}_${SUBSET}_r${r}|$CMFQ_DUMPS/CMECH_n${n}_${SUBSET}_r${r}/$MODEL/N$CTX|-|CMECH_n${n}_${SUBSET}|$n")
  done; done
else
  for sh in $SHIFTS; do for d in "${DSA[@]}"; do for (( r=0; r<REALS; r++ )); do
    T="${CASE}_shift${sh}_d${d}_r${r}"
    UNITS+=("$T|$OUT4/$T/$MODEL/shift0/d0/ctx$CTX|$FQ4/$T/$CASE/N$CTX|$CASE|-")
  done; done; done
fi

echo "[$(date '+%F %T')] score $MODEL  mode=$MODE  ${#UNITS[@]} cell(s)  B=$MALC_B  workers=$W"
ok=0; skip=0; miss=0; fail=0
for u in "${UNITS[@]}"; do
  IFS='|' read -r TAG ROOT CELL DSET NODES <<<"$u"
  [ -d "$ROOT" ] || { miss=$((miss+1)); continue; }
  n=$(find "$ROOT" -name '*.npz' ! -name 'summary.npz' 2>/dev/null | wc -l)
  [ "$n" -ge "$DRAWS" ] || { miss=$((miss+1)); continue; }

  did=0
  # --- vx part ---
  VP="$OUT_DIR/.${TAG}_vx_${MODEL}.md"
  if [ ! -s "$VP" ]; then
    PFT=""; [ "$MODE" = cmech ] && PFT="--per-file-truth"
    if python "$REPO/benchmarks/fixedq_ci_coverage.py" \
        --root "$ROOT" --dataset "$DSET" --query $QLIST \
        ${PFT:-} $([ -z "$PFT" ] && echo "--data-cell $CELL") \
        --workers "$W" --label "$TAG" --max-replicates "$DRAWS" \
        --json-out "$OUT_DIR/.${TAG}_vx_${MODEL}.json" \
        --out "$VP" >/dev/null 2>&1
    then did=1; else echo "  $TAG vx FAILED"; fail=$((fail+1)); continue; fi
  fi

  # --- density parts, raw then malc ---
  for sm in none malc; do
    [ "$sm" = malc ] && [ "${SKIP_MALC:-0}" = 1 ] && continue
    DP="$OUT_DIR/.${TAG}_${sm}_${MODEL}.md"
    [ -s "$DP.r" ] && continue
    sel=()
    if [ "$MODE" = cmech ]; then
      sel=(--root "$(dirname "$ROOT")" --nodes "$NODES" --subset "$SUBSET" --context "$CTX")
    else
      sel=(--root "$ROOT" --dataset "$DSET" --context "$CTX")
    fi
    extra=()
    [ "$sm" = malc ] && extra=(--malc-B "$MALC_B" --malc-K "$MALC_K" --malc-workers "$W")
    if python "$REPO/UWYK_Fig3_4/cate_density_metrics.py" \
        "${sel[@]}" --max-real "$DRAWS" --tau-smoother "$sm" "${extra[@]}" \
        --out "$DP" >/dev/null 2>&1
    then
      # Rename the harness row to the model key, as score_fq4 does -- six harness names
      # cover thirteen checkpoints, and unrenamed rows would merge onto each other.
      MODEL="$MODEL" python - "$DP" <<'PYREN' > "$DP.r"
import os, sys
m = os.environ["MODEL"]
for ln in open(sys.argv[1]):
    if ln.lstrip().startswith("|"):
        c = ln.strip().strip("|").split("|")
        name = c[0].strip()
        if name.lower() != "method" and set(name) - set("-: "):
            suf = next((x for x in ("-noanc", "-v3ab", "-v3a", "-v3b")
                        if name.endswith(x)), "")
            c[0] = f" {m}{suf} "
            ln = "|" + "|".join(c) + "|\n"
    sys.stdout.write(ln)
PYREN
      did=1
    else
      echo "  $TAG $sm FAILED"
    fi
  done
  if [ "$did" = 1 ]; then ok=$((ok+1)); echo "  $TAG ok ($(date '+%H:%M:%S'))"
  else skip=$((skip+1)); fi
done
echo "[$(date '+%F %T')] score $MODEL done: $ok scored, $skip cached, $miss not dumped, $fail failed"
