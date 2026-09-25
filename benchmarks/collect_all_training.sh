#!/bin/bash
# EVERYTHING needed to compute training cost, in one pass. Run once per cluster.
#
#   CLUSTER=fir REPO=/path/to/clone bash .../collect_all_training.sh
#
# Emits under $REPO/training_cost_evidence/$CLUSTER/:
#   ledger.tsv      one row per training job: out_dir, resume/last step, wall window
#   ckpts.tsv       every step checkpoint with its STEP and MTIME
#   sacct.psv       month-by-month accounting
#   logs/...        extracted head/tail of every train_*.out, unfiltered
#
# ckpts.tsv is the decisive artifact. A model's cost is the time to reach its
# FINAL step, and several of these trainings span multiple jobs with requeues
# that Slurm no longer accounts for. Checkpoint mtimes date each step directly,
# so the elapsed between the first training job's start and the mtime of the
# final checkpoint bounds it, and gaps between consecutive checkpoint mtimes
# expose queue waits that must NOT be counted as GPU time.
set -uo pipefail
CLUSTER="${CLUSTER:?CLUSTER required}"
REPO="${REPO:?REPO required (path to the git clone)}"
OUT="$REPO/training_cost_evidence/$CLUSTER"
mkdir -p "$OUT/logs"

echo "[1/4] ledger"
{
printf 'cluster\tlogdir\tjob\tout_dir\ttag\tresume_step\tlast_step\tstart\tmtime\thours\n'
for D in logs_*; do
  [ -d "$D" ] || continue
  for F in "$D"/train_*.out; do
    [ -f "$F" ] || continue
    jid=$(basename "$F" | grep -oE '[0-9]+' | head -1)
    out=$(grep -ohE "(OUT_DIR|CHECKPOINT_DIR|OUT)=[^ ]+" "$F" 2>/dev/null | head -1 | cut -d= -f2-)
    tag=$(grep -ohE "TAG=[A-Za-z0-9_]+|variant=[A-Za-z0-9_]+|NUM_BARS=[^ ]*|BINARIZE=[0-9.]+|BOTH_ARMS=[0-9]|J=[0-9]+|n_out=[0-9]+" "$F" 2>/dev/null | head -5 | tr '\n' ',' | sed 's/,$//')
    res=$(grep -ohE "resuming from actual_step=[0-9]+" "$F" 2>/dev/null | head -1 | grep -oE '[0-9]+')
    [ -z "$res" ] && grep -q "no latest.pt" "$F" 2>/dev/null && res=0
    [ -z "$res" ] && grep -q "\[resume\] found" "$F" 2>/dev/null && res=RESUMED
    last=$(tr '\r' '\n' < "$F" 2>/dev/null | grep -oE "done at step [0-9]+|reached actual_step=[0-9]+|^ *[0-9]{4,}" | grep -oE '[0-9]+' | sort -n | tail -1)
    st=$(grep -ohE '^\[2026-[0-9-]+ [0-9:]+\]' "$F" 2>/dev/null | head -1 | tr -d '[]')
    mt=$(stat -c '%y' "$F" 2>/dev/null | cut -d. -f1)
    h=""
    if [ -n "$st" ] && [ -n "$mt" ]; then
      a=$(date -d "$st" +%s 2>/dev/null); b=$(date -d "$mt" +%s 2>/dev/null)
      [ -n "$a" ] && [ -n "$b" ] && [ "$b" -gt "$a" ] && h=$(awk -v x=$((b-a)) 'BEGIN{printf "%.3f", x/3600}')
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$CLUSTER" "$D" "$jid" "${out:-?}" "${tag:-?}" "${res:-?}" "${last:-?}" "${st:-?}" "${mt:-?}" "${h:-?}"
  done
done
} > "$OUT/ledger.tsv"
echo "  $(( $(wc -l < "$OUT/ledger.tsv") - 1 )) job(s)"

echo "[2/4] checkpoint mtimes"
{
printf 'dir\tfile\tstep\tmtime\tbytes\n'
find . -maxdepth 6 -name "*.pt" \( -path "*checkpoint*" -o -path "*_output*" \) 2>/dev/null | while read -r P; do
  b=$(basename "$P")
  s=$(echo "$b" | grep -oE '(step|epoch)[_-]?0*([0-9]+)' | grep -oE '[0-9]+$')
  [ -n "$s" ] || continue
  printf '%s\t%s\t%s\t%s\t%s\n' "$(dirname "$P")" "$b" "$s" \
     "$(stat -c '%y' "$P" 2>/dev/null | cut -d. -f1)" "$(stat -c '%s' "$P" 2>/dev/null)"
done
} > "$OUT/ckpts.tsv"
echo "  $(( $(wc -l < "$OUT/ckpts.tsv") - 1 )) checkpoint(s)"

echo "[3/4] sacct"
{ for M in 04 05 06 07 08 09 10; do
    sacct -S 2026-$M-01 -E 2026-$M-15 -X --units=G -P -o JobID,JobName%60,State,Elapsed,AllocTRES%120,Start,End 2>/dev/null
    sacct -S 2026-$M-15 -E 2026-$M-31 -X --units=G -P -o JobID,JobName%60,State,Elapsed,AllocTRES%120,Start,End 2>/dev/null
  done; } | awk 'NR==1 || $0 !~ /^JobID\|/' > "$OUT/sacct.psv"
echo "  $(wc -l < "$OUT/sacct.psv") row(s)"

echo "[4/4] log extracts (ALL train_*.out, no filter)"
n=0
for D in logs_*; do
  [ -d "$D" ] || continue
  for F in "$D"/train_*.out; do
    [ -f "$F" ] || continue
    mkdir -p "$OUT/logs/$D"
    { echo "### HEAD"; head -50 "$F" | tr '\r' '\n' | grep -vE '^\s*$' | head -35
      echo "### STEPS"; grep -E "^ *[0-9]{3,} +-?[0-9]+\.[0-9]{4} " "$F" 2>/dev/null | awk 'NR%500==1' | head -20
      echo "### TAIL"; tr '\r' '\n' < "$F" | grep -vE '^\s*$' | tail -20
    } > "$OUT/logs/$D/$(basename "$F").txt" 2>/dev/null
    n=$((n+1))
  done
done
echo "  $n log(s)"
du -sh "$OUT"
