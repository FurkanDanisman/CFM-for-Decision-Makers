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
# This job generates, VERIFIES, then submits -- one sbatch instead of generating,
# waiting at a terminal, and firing the dumps by hand. The verification is the
# point: it refuses to submit 13 jobs unless row 0's true tau reproduces set A's
# to 1e-9. Nothing else guarantees the two sets share an estimand, and if they do
# not, every number the comparison produces is meaningless.
#
#   sbatch --account=def-zhijing R-PFN/benchmarks/cluster/submit_fq_setB.sbatch
#
# Env: ROOT_A / ROOT_B (data roots), OUT_B (dump parent), DRAWS, NFROZEN,
#      CASE, CTX, ACCOUNT, TIME_DUMP, GRES, FROM_NPZ + SRC_D (if set A used them)

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

CELL_B="$ROOT_B/shift0/d0/$CASE/N$CTX"
echo "[$(date '+%F %T')] set B: $DRAWS draws, rows 0..$((NFROZEN-1)) frozen"
echo "  A root : $ROOT_A"
echo "  B root : $ROOT_B"
echo "  B dumps: $OUT_B"

# ---- 1. the estimand set A actually used -------------------------------------
# Read it from A's own manifest rather than retyping the number: a hardcoded tau
# that silently drifts from the data would make every coverage figure wrong while
# still looking perfectly reasonable.
MAN_A=$(ls "$ROOT_A"/shift0/d0/manifest_*.json 2>/dev/null | head -1)
[ -n "$MAN_A" ] || { echo "FATAL: no manifest under $ROOT_A/shift0/d0 -- was set A generated there?" >&2; exit 1; }
read -r A_CASE A_SEED A_CTX A_TAU < <(python - "$MAN_A" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(m["case"], m["seed"], m["n_context"], repr(float(m["true_tau"][0])))
PY
)
echo "  A manifest: $MAN_A"
echo "    case=$A_CASE seed=$A_SEED n_context=$A_CTX  true_tau[0]=$A_TAU"
[ "$A_CASE" = "$CASE" ] || { echo "FATAL: A is case $A_CASE, asked for $CASE" >&2; exit 1; }
[ "$A_CTX" = "$CTX" ]   || { echo "FATAL: A is n_context $A_CTX, asked for $CTX" >&2; exit 1; }
SEED="${SEED:-$A_SEED}"

# ---- 2. generate, unless the root is already complete ------------------------
HAVE=$(find "$CELL_B" -name "${CASE}_*.npz" 2>/dev/null | wc -l)
if [ "$HAVE" -ge "$DRAWS" ]; then
    echo "  generation: SKIP ($HAVE >= $DRAWS npz already in $CELL_B)"
else
    echo "  generation: $HAVE/$DRAWS present -- running gen_fixed_query"
    # FROM_NPZ/SRC_D reproduce a d_variation source realization's SCM. If set A was
    # built that way, set B must be too: build_dag_d(case, d) is a different DAG
    # from build_dag(case), so omitting it changes tau. The check below catches it
    # either way, but passing them through means it does not have to.
    EXTRA=()
    [ -n "${FROM_NPZ:-}" ] && EXTRA+=(--from-npz "$FROM_NPZ")
    [ -n "${SRC_D:-}" ]    && EXTRA+=(--d "$SRC_D")
    python "$REPO/case_study/gen_fixed_query.py" \
        --case "$CASE" --seed "$SEED" --n-context "$CTX" \
        --query 0 --n-queries "$NFROZEN" --n-draws "$DRAWS" \
        --out "$ROOT_B/shift0/d0" "${EXTRA[@]}" || {
            echo "FATAL: generation failed (or the fixed-estimand assert tripped)" >&2; exit 1; }
fi

# ---- 3. refuse to submit unless B's row 0 IS A's row 0 -----------------------
MAN_B=$(ls "$ROOT_B"/shift0/d0/manifest_*.json 2>/dev/null | head -1)
[ -n "$MAN_B" ] || { echo "FATAL: generation left no manifest under $ROOT_B/shift0/d0" >&2; exit 1; }
python - "$MAN_A" "$MAN_B" <<'PY' || exit 1
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
ta, tb = float(a["true_tau"][0]), float(b["true_tau"][0])
dev = abs(ta - tb)
print(f"  A true_tau[0] = {ta:.10f}")
print(f"  B true_tau[0] = {tb:.10f}")
print(f"  |A - B|       = {dev:.3e}")
print(f"  B frozen rows = {b['n_queries']}, draws = {b['n_draws']}, "
      f"tau spread across replicates = {b['tau_spread']:.3e}")
if dev >= 1e-9:
    print("FATAL: B's row 0 is NOT A's row 0 -- different estimand, so the two\n"
          "       sets cannot be compared. Pass FROM_NPZ/SRC_D to match how set A\n"
          "       was generated, or point ROOT_A at the root A really used.",
          file=sys.stderr)
    raise SystemExit(1)
if b["tau_spread"] >= 1e-6:
    print("FATAL: tau moves across B's replicates -- the estimand is not fixed.",
          file=sys.stderr)
    raise SystemExit(1)
print("  ESTIMAND MATCHES SET A -- submitting dumps")
PY

# ---- 4. the 13 dumps --------------------------------------------------------
cd "$(dirname "$REPO")"
DATA="$ROOT_B" CASE="$CASE" CTX="$CTX" NQ=1 OUT_PARENT="$OUT_B" \
    TIME="$TIME_DUMP" GRES="$GRES" ${ACCT:+ACCOUNT="$ACCT"} \
    bash "$REPO/benchmarks/cluster/submit_cs_fixedq_all.sh" --submit
echo "[$(date '+%F %T')] set B submitted"
