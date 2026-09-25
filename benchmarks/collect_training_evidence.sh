#!/bin/bash
# Compact, committable evidence for tab:train_cost. Run on EVERY cluster that
# trained something (fir, killarney, nibi, tamia), then git add/commit/push.
#
#   CLUSTER=fir bash R-PFN/benchmarks/collect_training_evidence.sh
#   cd R-PFN && git add training_cost_evidence && git commit -m "evidence: fir" && git push
#
# Training logs are mostly tqdm carriage-return spam and can be hundreds of MB,
# so each log is reduced to: the config header, a subsample of the s/step table,
# and the tail that says which step it finished at. That is everything needed to
# tell a fresh 150k run from a resume, which raw GPU-hours cannot.
set -uo pipefail
CLUSTER="${CLUSTER:?CLUSTER required (fir|killarney|nibi|tamia)}"
REPO="${REPO:-$PWD/R-PFN}"
OUT="$REPO/training_cost_evidence/$CLUSTER"
mkdir -p "$OUT"

echo "[1/3] log extracts -> $OUT"
n=0
for D in logs_*; do
    [ -d "$D" ] || continue
    for F in "$D"/*.out "$D"/*.err; do
        [ -f "$F" ] || continue
        # Only logs that look like TRAINING: a config header or a step table.
        grep -qiE "variant=|params=|step-limit|OUT_DIR=|s/step|step-ckpt" "$F" || continue
        mkdir -p "$OUT/$D"
        T="$OUT/$D/$(basename "$F").txt"
        {
            echo "##### HEAD (config, device, fresh-vs-resume) #####"
            head -60 "$F" | tr '\r' '\n' | grep -vE "^\s*$" | head -45
            echo
            echo "##### s/step rows (every 200th) #####"
            # training_dopfn_repro/train.py: step loss lr s/step skipped
            grep -E "^ *[0-9]{3,} +-?[0-9]+\.[0-9]{4} " "$F" 2>/dev/null \
              | awk 'NR % 200 == 1' | head -40
            echo
            echo "##### tqdm rate samples #####"
            tr '\r' '\n' < "$F" | grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, +[0-9.]+ *(it/s|s/it)" \
              | awk 'NR % 500 == 1' | head -20
            echo
            echo "##### TAIL (final step / DONE) #####"
            tr '\r' '\n' < "$F" | grep -vE "^\s*$" | tail -25
        } > "$T" 2>/dev/null
        n=$((n+1))
    done
done
echo "  $n log(s) extracted"

echo "[2/3] sacct -> $OUT/sacct.psv"
# Some clusters refuse a wide range, so go month by month and keep one header.
{
  for Y in 2026; do for M in 04 05 06 07 08 09 10; do
    sacct -S $Y-$M-01 -E $Y-$M-15 -X --units=G -P \
      -o JobID,JobName%60,State,Elapsed,AllocTRES%120,ExitCode 2>/dev/null
    sacct -S $Y-$M-15 -E $Y-$M-31 -X --units=G -P \
      -o JobID,JobName%60,State,Elapsed,AllocTRES%120,ExitCode 2>/dev/null
  done; done
} | awk 'NR==1 || $0 !~ /^JobID\|/' > "$OUT/sacct.psv"
echo "  $(wc -l < "$OUT/sacct.psv") row(s)"

echo "[3/3] checkpoint step ceilings -> $OUT/ckpt_steps.txt"
python3 - "$OUT/ckpt_steps.txt" <<'PY' 2>/dev/null || echo "  (python3 unavailable, skipped)"
import glob, os, re, sys
rows = []
for d in sorted(set(glob.glob('checkpoints_*') + glob.glob('*_output')
                    + glob.glob('*/step_checkpoints') + glob.glob('*/checkpoints'))):
    if not os.path.isdir(d):
        continue
    s = [int(m.group(1)) for f in glob.glob(os.path.join(d, '**', '*.pt'), recursive=True)
         for m in [re.search(r'step[_-](\d+)', os.path.basename(f))] if m]
    e = [int(m.group(1)) for f in glob.glob(os.path.join(d, '**', '*.pt'), recursive=True)
         for m in [re.search(r'epoch[_-](\d+)', os.path.basename(f))] if m]
    if s or e:
        rows.append(f"{d:60s} max_step={max(s) if s else '-'} "
                    f"max_epoch={max(e) if e else '-'} n={len(s)+len(e)}")
open(sys.argv[1], 'w').write("\n".join(rows) + "\n")
print(f"  {len(rows)} dir(s)")
PY

echo
du -sh "$OUT"
echo "now:  cd $REPO && git add training_cost_evidence && git commit -m 'evidence: $CLUSTER' && git push"
