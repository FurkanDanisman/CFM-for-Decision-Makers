#!/bin/bash
# Score one fq4 cell: the four coverage columns for every model that dumped it.
#
# Runs the two existing scorers rather than a new one -- fixedq_ci_coverage.py for
# the moment-based v(x) pair, cate_density_metrics.py twice (raw, then MALC B=1000
# K=1) for the credible-interval pair -- then merges. Any number computed a second
# way here would be a second definition of a figure already reported elsewhere.
#
# cate_density_metrics takes ONE root and names rows by harness, while these dumps
# are one root per checkpoint and six harness names cover thirteen models. So it is
# run once per model root and each row is renamed to <model dir><tag>, matching how
# fixedq_ci_coverage names the same model.
#
#   bash R-PFN/benchmarks/cluster/score_fq4.sh
# Env: DUMPS (cell dump parent), FQ4 (generated cells), CASE, CTX, QUERIES,
#      OUT_DIR, MALC_B, MALC_K, WORKERS, SKIP_MALC=1
set -uo pipefail
KIT="${KIT:-$PWD}"; REPO="${REPO:-$KIT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"
# Activate the venv if it is not already: this is a plain script, so it gets run
# from whatever shell is handy and dies on 'No module named numpy' otherwise.
if ! python -c "import numpy" >/dev/null 2>&1; then
    D="$KIT"
    for _ in 1 2 3; do
        [ -f "$D/venv/bin/activate" ] && break
        D="$(cd "$D/.." && pwd)"
    done
    if [ -f "$D/venv/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$D/venv/bin/activate"
        echo "(activated $D/venv)"
    else
        echo "FATAL: no numpy and no venv found from $KIT" >&2; exit 1
    fi
fi
DUMPS="${DUMPS:?DUMPS required (e.g. \$SCRATCH/fq4_dumps/Observed_Confounder_shift0_d5_r0)}"
CASE="${CASE:-Observed_Confounder}"
CTX="${CTX:-1000}"
QUERIES="${QUERIES:-10}"
# R: the number of resampled datasets to score per query. Capped here, not at dump
# time, because eval_dopfn_bb_raw ignores MAX_REAL -- so without this one model would
# be scored on 100 replicates while the rest used 30.
DRAWS="${DRAWS:-30}"
TAG="$(basename "$DUMPS")"
FQ4="${FQ4:-$SC/fq4}"          # where the generated cells live (truths come from here)
CELL="$FQ4/$TAG/$CASE/N$CTX"
OUT_DIR="${OUT_DIR:-$SC/fq4_scores}"
MALC_B="${MALC_B:-1000}"; MALC_K="${MALC_K:-1}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
mkdir -p "$OUT_DIR"

QLIST=$(seq 0 $((QUERIES-1)) | tr '\n' ' ')
ROOTS=()
for m in "$DUMPS"/*/shift0/d0/"ctx$CTX"; do [ -d "$m" ] && ROOTS+=("$m"); done
[ "${#ROOTS[@]}" -gt 0 ] || { echo "no ctx$CTX cells under $DUMPS"; exit 1; }
[ -d "$CELL" ] || { echo "no data cell at $CELL -- cannot read the true taus"; exit 1; }
echo "== $TAG: ${#ROOTS[@]} model root(s), queries $QLIST"

# ---- 1. the two moment-based columns (one pass, all models, all queries) -----
python "$REPO/benchmarks/fixedq_ci_coverage.py" \
    --root "${ROOTS[@]}" --dataset "$CASE" --query $QLIST \
    --data-cell "$CELL" \
    --workers "$WORKERS" --label "$TAG" --max-replicates "$DRAWS" \
    --json-out "$OUT_DIR/${TAG}_vx.json" \
    --out "$OUT_DIR/${TAG}_vx.md" >/dev/null || {
        echo "FATAL: fixedq_ci_coverage failed" >&2; exit 1; }
echo "  vx    -> $OUT_DIR/${TAG}_vx.md"

# ---- 2. the two credible-interval columns -----------------------------------
run_dens() {   # run_dens <smoother> <outfile>
    local sm="$1" out="$2" parts=()
    for r in "${ROOTS[@]}"; do
        local model part
        model=$(basename "$(dirname "$(dirname "$(dirname "$r")")")")
        part="$OUT_DIR/.${TAG}_${sm}_${model}.md"
        local extra=()
        [ "$sm" = malc ] && extra=(--malc-B "$MALC_B" --malc-K "$MALC_K"
                                   --malc-workers "$WORKERS")
        python "$REPO/UWYK_Fig3_4/cate_density_metrics.py" \
            --root "$r" --dataset "$CASE" --context "$CTX" \
            --max-real "$DRAWS" \
            --tau-smoother "$sm" "${extra[@]}" \
            --out "$part" >/dev/null 2>&1 || {
                echo "  [$sm] $model FAILED" >&2; continue; }
        MODEL="$model" python - "$part" <<'PYREN' > "$part.r"
import os, re, sys
# Rename each harness row to <model dir><tag>, the same key fixedq_ci_coverage
# uses. Without this, thirteen models collapse onto six harness names and the
# merge silently pairs the wrong rows.
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
        parts+=("$part.r")
    done
    [ "${#parts[@]}" -gt 0 ] || return 1
    # Concatenate in python, keyed on the HEADER row rather than on line numbers.
    # 'head -n 2 | grep ^|' assumed the table starts at line 1; cate_density_metrics
    # prints a title first, so the header was dropped and the merge then read a data
    # row as the header and found no coverage95 column at all.
    OUT="$out" python - "${parts[@]}" <<'PYCAT'
import os, sys
out, hdr, rows = os.environ["OUT"], None, []
for path in sys.argv[1:]:
    seen_hdr = False
    for ln in open(path):
        if not ln.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if not seen_hdr:
            if "method" in [c.lower() for c in cells]:
                seen_hdr = True
                if hdr is None:
                    hdr = ln.rstrip("\n")
                continue
            # a part with no header at all: treat every pipe line as data
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(ln.rstrip("\n"))
if hdr is None:
    print(f"no header row in any of {len(sys.argv)-1} part(s)", file=sys.stderr)
    raise SystemExit(1)
n = hdr.count("|") - 1
with open(out, "w") as fh:
    fh.write(hdr + "\n" + "|" + "---|" * n + "\n" + "\n".join(rows) + "\n")
print(len(rows))
PYCAT
    echo "  $sm -> $out ($(grep -c '^| ' "$out") rows incl. header)"
}

run_dens none "$OUT_DIR/${TAG}_raw.md" || echo "  raw: no rows"
if [ "${SKIP_MALC:-0}" = 1 ]; then
    echo "  malc: SKIPPED (SKIP_MALC=1)"
    MALC_ARG=()
else
    run_dens malc "$OUT_DIR/${TAG}_malc.md" || echo "  malc: no rows"
    MALC_ARG=(--malc "$OUT_DIR/${TAG}_malc.md")
fi

# ---- 3. merge ---------------------------------------------------------------
python "$REPO/benchmarks/fq4_table.py" \
    --vx "$OUT_DIR/${TAG}_vx.md" \
    --bayes "$OUT_DIR/${TAG}_raw.md" "${MALC_ARG[@]}" \
    --label "$TAG" --out "$OUT_DIR/${TAG}_four.md" \
    --json-out "$OUT_DIR/${TAG}_four.json"
