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
# CELL_OVERRIDE / CASE_OVERRIDE: ComplexMech cells sit at a different path and carry a
# different dataset name, and deriving either from the tag would need a second parser
# here. The caller already knows both, so it passes them.
CELL="${CELL_OVERRIDE:-$FQ4/$TAG/$CASE/N$CTX}"
[ -n "${CASE_OVERRIDE:-}" ] && CASE="$CASE_OVERRIDE"
# ComplexMech truths are in per-replicate units, so each replicate is scored against
# its OWN stored tau; the case study has one fixed truth per query.
PFT=""
case "$TAG" in CMECH_*) PFT="--per-file-truth" ;; esac
OUT_DIR="${OUT_DIR:-$SC/fq4_scores}"
MALC_B="${MALC_B:-1000}"; MALC_K="${MALC_K:-1}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
mkdir -p "$OUT_DIR"

QLIST=$(seq 0 $((QUERIES-1)) | tr '\n' ' ')
# The two benchmarks nest their dumps differently:
#   case study  <dumps>/<model>/shift0/d0/ctx<N>/<harness>/<case>
#   ComplexMech <dumps>/<model>/N<N>/<harness>/CMECH_n<d>_<subset>
# fixedq_ci_coverage resolves <harness>/<dataset> under whatever root it is given, so
# only the prefix differs -- but it has to be the right prefix, or every cell reports
# "no cells" and the reaper sees nothing to do.
ROOTS=(); NODES=""
if [ -n "$PFT" ]; then
    NODES=$(echo "$TAG" | sed -E 's/^CMECH_n([0-9]+)_.*/\1/')
    for m in "$DUMPS"/*/"N$CTX"; do [ -d "$m" ] && ROOTS+=("$m"); done
else
    for m in "$DUMPS"/*/shift0/d0/"ctx$CTX"; do [ -d "$m" ] && ROOTS+=("$m"); done
fi
if [ "${#ROOTS[@]}" = 0 ]; then
    if [ -n "$PFT" ]; then echo "no N$CTX cells under $DUMPS"
    else echo "no shift0/d0/ctx$CTX cells under $DUMPS"; fi
    exit 1
fi
# The dataset directory the scorer resolves under each root. Set explicitly rather
# than with ${PFT:-...}, which expands to PFT itself when PFT is non-empty and yielded
# the dataset name "CMECH_n20_nonzero--per-file-truth".
if [ -n "$PFT" ]; then DSET="CMECH_n${NODES}_${SUBSET:-nonzero}"; else DSET="$CASE"; fi
# Only the case study needs a data cell: under --per-file-truth the truth is read from
# each dump, so ComplexMech does not consult one at all.
if [ -z "$PFT" ] && [ ! -d "$CELL" ]; then
    echo "no data cell at $CELL -- cannot read the true taus"; exit 1
fi
echo "== $TAG: ${#ROOTS[@]} model root(s), queries $QLIST"

# ---- 1. the two moment-based columns, CACHED PER MODEL ----------------------
# One invocation per model root, not one over all of them. Models are independent --
# scoring graph2d has nothing to do with scoring dopfn_bb -- so when the three slow
# harnesses finish a cell hours after the other ten, re-scoring should cost three
# models, not thirteen. Each part is kept and reused; only missing ones are computed.
#
# Delete a part file to force that model to be re-scored.
model_of() {   # model_of <root>
    if [ -n "$PFT" ]; then basename "$(dirname "$1")"
    else basename "$(dirname "$(dirname "$(dirname "$1")")")"; fi
}

vx_parts=(); n_new=0; n_cached=0
for r in "${ROOTS[@]}"; do
    mdl=$(model_of "$r")
    part="$OUT_DIR/.${TAG}_vx_${mdl}.md"
    if [ -s "$part" ]; then
        vx_parts+=("$part"); n_cached=$((n_cached+1)); continue
    fi
    if python "$REPO/benchmarks/fixedq_ci_coverage.py" \
        --root "$r" --dataset "$DSET" --query $QLIST \
        ${PFT:-} $([ -z "$PFT" ] && echo "--data-cell $CELL") \
        --workers "$WORKERS" --label "$TAG" --max-replicates "$DRAWS" \
        --json-out "$OUT_DIR/.${TAG}_vx_${mdl}.json" \
        --out "$part" >/dev/null 2>&1
    then
        vx_parts+=("$part"); n_new=$((n_new+1))
    else
        echo "  [vx] $mdl FAILED" >&2
    fi
done
[ "${#vx_parts[@]}" -gt 0 ] || { echo "FATAL: no vx parts for $TAG" >&2; exit 1; }
OUT="$OUT_DIR/${TAG}_vx.md" python - "${vx_parts[@]}" <<'PYVX'
import os, sys
# One header, then every data row. Each part is a single-model table from the same
# writer, so the columns line up by construction.
out, hdr, rows = os.environ["OUT"], None, []
for path in sys.argv[1:]:
    seen = False
    for ln in open(path):
        if not ln.lstrip().startswith("|"):
            continue
        cells = [c.strip().lower() for c in ln.strip().strip("|").split("|")]
        if not seen:
            if "model" in cells:
                seen = True
                if hdr is None:
                    hdr = ln.rstrip("\n")
                continue
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(ln.rstrip("\n"))
if hdr is None:
    raise SystemExit("no header in any vx part")
n = hdr.count("|") - 1
open(out, "w").write(hdr + "\n" + "|" + "---|" * n + "\n" + "\n".join(rows) + "\n")
PYVX
echo "  vx    -> $OUT_DIR/${TAG}_vx.md ($n_new scored, $n_cached cached)"

# ---- 2. the two credible-interval columns -----------------------------------
run_dens() {   # run_dens <smoother> <outfile>
    local sm="$1" out="$2" parts=()
    for r in "${ROOTS[@]}"; do
        local model part sel=()
        if [ -n "$PFT" ]; then
            # cmech: root is <dumps>/<model>/N<ctx>, and the scorer wants the model dir
            # plus the node count and subset rather than a dataset name.
            model=$(basename "$(dirname "$r")")
            sel=(--root "$(dirname "$r")" --nodes "$NODES"
                 --subset "${SUBSET:-nonzero}" --context "$CTX")
        else
            model=$(basename "$(dirname "$(dirname "$(dirname "$r")")")")
            sel=(--root "$r" --dataset "$CASE" --context "$CTX")
        fi
        local tag_sm="$sm"
        [ "$sm" = malc ] && tag_sm="malc${MALC_B}"
        part="$OUT_DIR/.${TAG}_${tag_sm}_${model}.md"
        # Cached like the vx parts: a model already scored under this smoother is not
        # recomputed when a later pass adds other models.
        if [ -s "$part.r" ]; then parts+=("$part.r"); continue; fi
        local extra=()
        [ "$sm" = malc ] && extra=(--malc-B "$MALC_B" --malc-K "$MALC_K"
                                   --malc-workers "$WORKERS")
        python "$REPO/UWYK_Fig3_4/cate_density_metrics.py" \
            "${sel[@]}" --max-real "$DRAWS" \
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

# ---- 2b. sd(Y) for this cell ------------------------------------------------
# Recorded now because it is computable ONLY from the generated data, which the
# reaper deletes once this cell is scored. Interval lengths are in the outcome's
# units, so without it they cannot be compared across d values or benchmarks.
python "$REPO/benchmarks/fq4_cell_sdy.py" --cell "$CELL" \
    --max-files "$DRAWS" --out "$OUT_DIR/${TAG}_sdy.json" >/dev/null 2>&1 \
    && echo "  sd(Y) -> $OUT_DIR/${TAG}_sdy.json" \
    || echo "  sd(Y): FAILED (lengths will not be normalisable for this cell)"

# ---- 3. merge ---------------------------------------------------------------
python "$REPO/benchmarks/fq4_table.py" \
    --vx "$OUT_DIR/${TAG}_vx.md" \
    --bayes "$OUT_DIR/${TAG}_raw.md" "${MALC_ARG[@]}" \
    --label "$TAG" --out "$OUT_DIR/${TAG}_four.md" \
    --sdy "$OUT_DIR/${TAG}_sdy.json" \
    ${N_MODELS:+--n-models "$N_MODELS"} \
    ${SKIP_MALC:+ } $([ "${SKIP_MALC:-0}" = 1 ] || echo "--malc-b $MALC_B") \
    --json-out "$OUT_DIR/${TAG}_four.json"
