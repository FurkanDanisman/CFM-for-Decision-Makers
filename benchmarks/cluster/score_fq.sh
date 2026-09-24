#!/bin/bash
# Score the fixed-query CI coverage for both settings, as far as each has got.
#
# Safe to run while the dumps are still going: fixedq_ci_coverage.py reports the
# number of replicates each model actually has in its `datasets` column, so a
# partial run reads as partial instead of as a finished result. Read that column
# before reading the coverage -- a model at 18/1000 has a coverage figure with a
# +-0.1 standard error on it, which is not evidence of anything.
#
# The true tau comes from each set's OWN data cell (cate[query] of a replicate),
# never a hardcoded constant: set A and set B must share the estimand, and reading
# it from the data is what makes a mismatch visible rather than silent.
#
#   bash R-PFN/benchmarks/cluster/score_fq.sh
# Env: ROOT_A/ROOT_B (data roots), A/B (dump parents), CASE, CTX, QUERY, OUT_DIR
set -uo pipefail
KIT="${KIT:-$PWD}"; REPO="${REPO:-$KIT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"
ROOT_A="${ROOT_A:-$SC/cs_fq1000_root}";       A="${A:-$SC/cs_fq1000_dumps}"
ROOT_B="${ROOT_B:-$SC/cs_fq1000_f100_root}";  B="${B:-$SC/cs_fq1000_f100_dumps}"
CASE="${CASE:-Observed_Confounder}"
CTX="${CTX:-1000}"; QUERY="${QUERY:-0}"
OUT_DIR="${OUT_DIR:-$SC}"

score() {   # score <tag> <data root> <dump parent> <description>
    local tag="$1" droot="$2" dumps="$3" desc="$4"
    local cell="$droot/shift0/d0/$CASE/N$CTX"
    echo
    echo "############ set $tag -- $desc"
    if [ ! -d "$dumps" ]; then echo "  no dumps at $dumps -- not submitted yet"; return; fi
    if [ ! -d "$cell" ]; then echo "  no data cell at $cell -- cannot read the true tau"; return; fi
    # One --root per model: each model has its own OUT_ROOT (the dump path is keyed
    # by HARNESS, and six harnesses cover the thirteen checkpoints, so they would
    # otherwise overwrite one another).
    local roots=()
    for m in "$dumps"/*/shift0/d0/"ctx$CTX"; do [ -d "$m" ] && roots+=("$m"); done
    if [ "${#roots[@]}" = 0 ]; then echo "  $dumps exists but holds no ctx$CTX cells yet"; return; fi
    echo "  ${#roots[@]} model root(s)"
    python "$REPO/benchmarks/fixedq_ci_coverage.py" \
        --root "${roots[@]}" --dataset "$CASE" --query "$QUERY" \
        --data-cell "$cell" --label "$desc" \
        --out "$OUT_DIR/fq_${tag}_coverage.md"
}

score A "$ROOT_A" "$A" "row 0 frozen only"
score B "$ROOT_B" "$B" "rows 0-99 frozen"
echo
echo "wrote $OUT_DIR/fq_A_coverage.md and $OUT_DIR/fq_B_coverage.md (whichever ran)"
