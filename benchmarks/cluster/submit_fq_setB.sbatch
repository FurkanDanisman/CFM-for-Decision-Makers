#!/bin/bash
#SBATCH --job-name=fq-setB-gen
#SBATCH --output=logs_fq/setB_gen_%j.out
#SBATCH --error=logs_fq/setB_gen_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#
# Set B of the fixed-query study: 1000 replicate datasets with rows 0..99 frozen,
# then the 13 model dumps that read them.
#
# Set A froze row 0 alone, so every replicate was a nearly independent draw of the
# observational sample. Set B additionally freezes 99 of its neighbours, so the
# context the model conditions on barely moves between replicates. Both are dumped
# with NQ=1, i.e. ONLY row 0 is ever a query, so the two runs estimate the same
# estimand and differ purely in how much of the surrounding sample is held fixed.
# That isolates how much of a head's sampling spread comes from the context.
#
# Generates, VERIFIES, then submits -- one job instead of generating, waiting at a
# terminal, and firing the dumps by hand. The verification is the point: it refuses
# to submit 13 jobs unless row 0's true tau reproduces set A's to 1e-9. Nothing
# else guarantees the two sets share an estimand, and if they do not, every number
# the comparison produces is meaningless.
#
#   bash R-PFN/benchmarks/cluster/fq_submit.sh setB
# NOT plain sbatch: logs_fq must exist BEFORE Slurm opens --output, or the
# job dies at launch with no log and no squeue entry. fq_submit.sh mkdirs it.
#
# Env: ROOT_A / ROOT_B (data roots), OUT_B (dump parent), DRAWS, NFROZEN,
#      CASE, CTX, ACCOUNT, TIME_DUMP, GRES, SEED, SEED_CANDIDATES,
#      FROM_NPZ + SRC_D (if set A was built from a d_variation source)

set -uo pipefail
mkdir -p logs_fq

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }
# shellcheck disable=SC1091
source "$DEPLOY_ROOT/venv/bin/activate"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"

ROOT_A="${ROOT_A:-$SC/cs_fq1000_root}"
ROOT_B="${ROOT_B:-$SC/cs_fq1000_f100_root}"
OUT_B="${OUT_B:-$SC/cs_fq1000_f100_dumps}"
CASE="${CASE:-Observed_Confounder}"
CTX="${CTX:-1000}"
DRAWS="${DRAWS:-1000}"
NFROZEN="${NFROZEN:-100}"
TIME_DUMP="${TIME_DUMP:-8:00:00}"
GRES="${GRES:-none}"
ACCT="${ACCOUNT:-}"
SEED="${SEED:-}"

CELL_A="$ROOT_A/shift0/d0/$CASE/N$CTX"
CELL_B="$ROOT_B/shift0/d0/$CASE/N$CTX"
echo "[$(date '+%F %T')] set B: $DRAWS draws, rows 0..$((NFROZEN-1)) frozen"
echo "  A root : $ROOT_A"
echo "  B root : $ROOT_B"
echo "  B dumps: $OUT_B"

# ---- 1. the estimand set A actually used -------------------------------------
# From A's DATA CELL, not a manifest: a manifest is written by one version of the
# generator and may simply be absent (set A's is), whereas cate[query] inside any
# replicate IS the estimand and is always there. It is also the exact source
# score_fq.sh reads, so the two cannot disagree.
tau_of() {   # tau_of <cell dir> -> cate[query 0] of its first replicate
    python - "$1" <<'PYTAU'
import glob, os, sys
import numpy as np
fs = sorted(f for f in glob.glob(os.path.join(sys.argv[1], "*.npz"))
            if os.path.basename(f) != "summary.npz")
if not fs:
    raise SystemExit(1)
with np.load(fs[0], allow_pickle=True) as z:
    print(repr(float(np.asarray(z["cate"]).ravel()[0])))
PYTAU
}

A_TAU=$(tau_of "$CELL_A") || {
    echo "FATAL: no replicate npz under $CELL_A -- ROOT_A is wrong." >&2
    echo "Candidate roots on this filesystem:" >&2
    ls -d "$SC"/*/shift0/d0/"$CASE"/N"$CTX" 2>/dev/null >&2
    exit 1; }
echo "  A cell : $CELL_A"
echo "  A true tau[0] = $A_TAU"

# The generator needs the SEED that produced that tau, and set A's npz files do not
# record it. Probe for it with 1-draw generations (seconds each) rather than asking:
# row 0's exogenous draws depend only on (case, seed, n_context), not on how many
# rows are frozen, so a 1-draw probe settles it exactly.
# Searched over d as well as seed: plain seeds 0..9 do NOT reproduce set A, which
# means A came from a d_variation DAG -- build_dag_d(case, d), a different graph
# from build_dag(case). Neither d nor the seed is recorded in the replicate files,
# so both are probed. "" in D_CANDIDATES means the plain DAG.
if [ -z "$SEED" ] && [ -z "${FROM_NPZ:-}" ]; then
    PROBE="${PROBE_DIR:-$SC/.fq_seed_probe}"
    for dcand in ${D_CANDIDATES:-none 5 10 20 30 40 50}; do
        [ -n "$SEED" ] && break
        if [ "$dcand" = none ]; then DARG=(); else DARG=(--d "$dcand"); fi
        for cand in ${SEED_CANDIDATES:-0 1 2 3 4 5 6 7 8 9}; do
            rm -rf "$PROBE"
            python "$REPO/case_study/gen_fixed_query.py" \
                --case "$CASE" --seed "$cand" --n-context "$CTX" \
                --query 0 --n-queries 1 --n-draws 1 "${DARG[@]}" \
                --out "$PROBE/shift0/d0" >/dev/null 2>&1 || continue
            got=$(tau_of "$PROBE/shift0/d0/$CASE/N$CTX" 2>/dev/null) || continue
            same=$(python -c "import sys;print(int(abs(float(sys.argv[1])-float(sys.argv[2]))<1e-9))" "$got" "$A_TAU")
            if [ "$same" = 1 ]; then
                echo "    d=$dcand seed=$cand -> tau $got   <-- MATCH"
                SEED="$cand"; MATCH_D="$dcand"; break
            fi
            echo "    d=$dcand seed=$cand -> tau $got"
        done
    done
    rm -rf "$PROBE"
fi
if [ -z "$SEED" ] && [ -z "${FROM_NPZ:-}" ]; then
    echo "FATAL: no (d, seed) in the probe grid reproduces A's tau ($A_TAU)." >&2
    echo "       Widen it with D_CANDIDATES / SEED_CANDIDATES, or pass the source" >&2
    echo "       directly: FROM_NPZ=<d_variation npz> SRC_D=<d>." >&2
    exit 1
fi
MATCH_D="${MATCH_D:-none}"
[ "$MATCH_D" = none ] || SRC_D="${SRC_D:-$MATCH_D}"
echo "  seed: ${SEED:-<from FROM_NPZ>}   d: $MATCH_D"

# ---- 2. generate, unless the root is already complete ------------------------
HAVE=$(find "$CELL_B" -name "${CASE}_*.npz" 2>/dev/null | wc -l)
if [ "$HAVE" -ge "$DRAWS" ]; then
    echo "  generation: SKIP ($HAVE >= $DRAWS npz already in $CELL_B)"
else
    echo "  generation: $HAVE/$DRAWS present -- running gen_fixed_query"
    EXTRA=()
    [ -n "${FROM_NPZ:-}" ] && EXTRA+=(--from-npz "$FROM_NPZ")
    [ -n "${SRC_D:-}" ]    && EXTRA+=(--d "$SRC_D")   # set from the probe match
    [ -n "$SEED" ]         && EXTRA+=(--seed "$SEED")
    python "$REPO/case_study/gen_fixed_query.py" \
        --case "$CASE" --n-context "$CTX" \
        --query 0 --n-queries "$NFROZEN" --n-draws "$DRAWS" \
        --out "$ROOT_B/shift0/d0" "${EXTRA[@]}" || {
            echo "FATAL: generation failed (or the fixed-estimand assert tripped)" >&2
            exit 1; }
fi

# ---- 3. refuse to submit unless B's row 0 IS A's row 0 -----------------------
# From the data cells on both sides, and across EVERY B replicate rather than just
# the first: if row 0's tau moves between replicates the estimand is not fixed and
# "coverage for one estimand" means nothing.
python - "$CELL_A" "$CELL_B" <<'PYVER' || exit 1
import glob, os, sys
import numpy as np


def taus(cell):
    fs = sorted(f for f in glob.glob(os.path.join(cell, "*.npz"))
                if os.path.basename(f) != "summary.npz")
    out = []
    for f in fs:
        with np.load(f, allow_pickle=True) as z:
            out.append(float(np.asarray(z["cate"]).ravel()[0]))
    return np.asarray(out)


ta, tb = taus(sys.argv[1]), taus(sys.argv[2])
if ta.size == 0 or tb.size == 0:
    print("FATAL: empty data cell", file=sys.stderr)
    raise SystemExit(1)
dev = abs(float(ta[0]) - float(tb[0]))
spread = float(np.abs(tb - tb[0]).max())
print(f"  A true tau[0] = {ta[0]:.10f}  ({ta.size} replicates)")
print(f"  B true tau[0] = {tb[0]:.10f}  ({tb.size} replicates)")
print(f"  |A - B|                    = {dev:.3e}")
print(f"  B spread across replicates = {spread:.3e}")
if dev >= 1e-9:
    print("FATAL: B's row 0 is NOT A's row 0 -- a different estimand, so the two\n"
          "       sets cannot be compared. Pass FROM_NPZ/SRC_D or SEED to match A.",
          file=sys.stderr)
    raise SystemExit(1)
if spread >= 1e-6:
    print("FATAL: tau moves across B's replicates -- the estimand is not fixed.",
          file=sys.stderr)
    raise SystemExit(1)
print("  ESTIMAND MATCHES SET A -- submitting dumps")
PYVER

# ---- 4. the 13 dumps --------------------------------------------------------
cd "$(dirname "$REPO")"
DATA="$ROOT_B" CASE="$CASE" CTX="$CTX" NQ=1 OUT_PARENT="$OUT_B" \
    TIME="$TIME_DUMP" GRES="$GRES" ${ACCT:+ACCOUNT="$ACCT"} \
    bash "$REPO/benchmarks/cluster/submit_cs_fixedq_all.sh" --submit
echo "[$(date '+%F %T')] set B submitted"
