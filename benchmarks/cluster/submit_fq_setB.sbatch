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
# with NQ=1, i.e. ONLY row 0 is ever a query, so the two estimate the same estimand
# and differ purely in how much of the surrounding sample is held fixed. That
# isolates how much of a head's sampling spread comes from the context.
#
# THE ESTIMAND. Set A was generated as
#   gen_fixed_query.py --from-npz case_study_data/d_variation/shift0/d5/
#       Observed_Confounder/N1000/Observed_Confounder_0.npz --d 5 --query 0
#       --n-draws 1000 --out $SCRATCH/cs_fq1000
# i.e. from a d_variation source realization -- DAG build_dag_d(case, 5), seed
# 2251341562 read out of that npz, true tau at query 0 = -0.2445281297. Set B is
# the same command plus --n-queries 100. The seed is NOT guessable and is recorded
# nowhere in the replicate files, so the source npz is the input that matters here;
# it is a parameter below rather than something rediscovered at runtime.
#
# LAYOUT. The generator writes <GEN_OUT>/<case>/N<ctx>, but the dump harness wants
# <root>/shift<S>/d<K>/<case>/N<ctx>. Set A bridged that with a symlink, and so
# does this.
#
#   bash R-PFN/benchmarks/cluster/fq_submit.sh setB
# NOT plain sbatch: logs_fq must exist BEFORE Slurm opens --output, or the job dies
# at launch with no log and no squeue entry. fq_submit.sh mkdirs it.
#
# Env: FROM_NPZ, SRC_D, ROOT_A, GEN_OUT, ROOT_B, OUT_B, DRAWS, NFROZEN, CASE, CTX,
#      ACCOUNT, TIME_DUMP, GRES

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

CASE="${CASE:-Observed_Confounder}"
CTX="${CTX:-1000}"
DRAWS="${DRAWS:-1000}"
NFROZEN="${NFROZEN:-100}"
SRC_D="${SRC_D:-5}"
FROM_NPZ="${FROM_NPZ:-$DEPLOY_ROOT/case_study_data/d_variation/shift0/d$SRC_D/$CASE/N$CTX/${CASE}_0.npz}"
ROOT_A="${ROOT_A:-$SC/cs_fq1000_root}"
GEN_OUT="${GEN_OUT:-$SC/cs_fq1000_f100}"
ROOT_B="${ROOT_B:-$SC/cs_fq1000_f100_root}"
OUT_B="${OUT_B:-$SC/cs_fq1000_f100_dumps}"
TIME_DUMP="${TIME_DUMP:-8:00:00}"
GRES="${GRES:-none}"
ACCT="${ACCOUNT:-}"

CELL_A="$ROOT_A/shift0/d0/$CASE/N$CTX"
CELL_B="$GEN_OUT/$CASE/N$CTX"
echo "[$(date '+%F %T')] set B: $DRAWS draws, rows 0..$((NFROZEN-1)) frozen"
echo "  source : $FROM_NPZ  (d=$SRC_D)"
echo "  gen out: $GEN_OUT"
echo "  B root : $ROOT_B   (symlink wrapper)"
echo "  B dumps: $OUT_B"
[ -f "$FROM_NPZ" ] || { echo "FATAL: source npz not found: $FROM_NPZ" >&2; exit 1; }

# ---- 1. generate, unless already complete ------------------------------------
HAVE=$(find "$CELL_B" -name "${CASE}_*.npz" 2>/dev/null | wc -l)
if [ "$HAVE" -ge "$DRAWS" ]; then
    echo "  generation: SKIP ($HAVE >= $DRAWS npz already in $CELL_B)"
else
    echo "  generation: $HAVE/$DRAWS present -- running gen_fixed_query"
    python "$REPO/case_study/gen_fixed_query.py" \
        --from-npz "$FROM_NPZ" --d "$SRC_D" \
        --query 0 --n-queries "$NFROZEN" --n-draws "$DRAWS" \
        --out "$GEN_OUT" || {
            echo "FATAL: generation failed (or the fixed-estimand assert tripped)" >&2
            exit 1; }
fi

# ---- 2. the shift0/d0 view the dump harness expects --------------------------
mkdir -p "$ROOT_B/shift0/d0"
ln -sfn "$GEN_OUT/$CASE" "$ROOT_B/shift0/d0/$CASE"
[ -d "$ROOT_B/shift0/d0/$CASE/N$CTX" ] || {
    echo "FATAL: symlink did not resolve to $GEN_OUT/$CASE/N$CTX" >&2; exit 1; }

# ---- 3. refuse to submit unless B's row 0 IS A's row 0 -----------------------
# Read from the data cells on both sides -- cate[0] inside a replicate IS the
# estimand, and it is the same value score_fq.sh reads, so the two cannot disagree.
# Checked across EVERY B replicate, not just the first: if row 0's tau moves
# between replicates the estimand is not fixed and the whole design is void.
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
    print(f"FATAL: empty data cell (A={ta.size}, B={tb.size})", file=sys.stderr)
    raise SystemExit(1)
dev = abs(float(ta[0]) - float(tb[0]))
spread = float(np.abs(tb - tb[0]).max())
print(f"  A true tau[0] = {ta[0]:.10f}  ({ta.size} replicates)")
print(f"  B true tau[0] = {tb[0]:.10f}  ({tb.size} replicates)")
print(f"  |A - B|                    = {dev:.3e}")
print(f"  B spread across replicates = {spread:.3e}")
if dev >= 1e-9:
    print("FATAL: B's row 0 is NOT A's row 0 -- a different estimand, so the two\n"
          "       sets cannot be compared. Check FROM_NPZ / SRC_D against set A.",
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
