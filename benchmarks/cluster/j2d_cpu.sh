#!/bin/bash
# Smoke the joint_2d native path on CPU, on the login node. No queue, no GPU.
#
# Runs the RealCause/IHDP cell through the dopfn_native harness (array task 0)
# with DOPFN_CKPT pointing at the ORIGINAL repro joint_2d checkpoint. The .sbatch
# body is plain bash, so it runs directly with SLURM_ARRAY_TASK_ID set by hand.
#
# What to look for in the output, in order:
#   [dopfn_native] joint_2d: J=10 edges=[-2.4865, +3.7314]   <- checkpoint read
#   [dopfn_native] checkpoint J differs ... rebuilt N tensors <- head 100 -> 113
#   [dopfn_native][2d] data_std=... edges_raw=[...]           <- THE check: this
#       is the training grid mapped into raw Y units. If it does not span a
#       plausible IHDP outcome range, the affine recovery is wrong and nothing
#       downstream means anything.
#
#   bash R-PFN/benchmarks/cluster/j2d_cpu.sh

set -uo pipefail
# Paths are DERIVED, not hardcoded, so this runs on any cluster: the script lives
# at <kit>/R-PFN/benchmarks/cluster/, so the kit root is three levels up. SCRATCH
# must be set by the environment -- guessing a per-cluster scratch path is how a
# run silently writes to the wrong filesystem.
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_KIT_DEFAULT="$(cd "$_SELF_DIR/../../.." && pwd)"
KIT="${KIT:-$_KIT_DEFAULT}"
REPO="${REPO:-$KIT/R-PFN}"
CK="${CK:-$REPO/Required_checkpoints}"
OUT="${OUT:-${SCRATCH:?SCRATCH must be set}/j2d_cpu}"
NT="${NTHREAD:-8}"
CKPT="${CKPT:-$CK/dopfn_repro_joint2d_step150000.pt}"

[ -f "$CKPT" ] || { echo "FATAL: no checkpoint at $CKPT" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="$NT" MKL_NUM_THREADS="$NT" OPENBLAS_NUM_THREADS="$NT"
export SLURM_CPUS_PER_TASK="$NT" COMPILE=0
cd "$KIT" || exit 1
rm -rf "$OUT"; mkdir -p "$OUT"

echo "joint_2d CPU smoke: $CKPT"
echo "OUT=$OUT  threads=$NT"
echo
SLURM_ARRAY_TASK_ID=0 \
DOPFN_CKPT="$CKPT" OUT_ROOT="$OUT" DENSITY_DUMP=1 \
    bash "$REPO/benchmarks/cluster/submit_realcause_density_unified.sbatch" 2>&1 \
    | tee "$OUT/run.log" | grep -E "dopfn_native|2d\]|density|WARN" | head -30
# A truncated traceback hides which call failed, so print the tail in full.
if grep -q "Traceback" "$OUT/run.log"; then
    echo; echo "--- traceback (full)"; sed -n '/Traceback/,$p' "$OUT/run.log" | head -40
fi
echo
n=$(find "$OUT" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "npz written: $n   (full log: $OUT/run.log)"
[ "$n" = 0 ] && { echo "NO DUMP -- see the log"; exit 1; }

source "$KIT/venv/bin/activate"
echo
echo "--- point estimate"
python -u "$REPO/realcause_eval/point_raw_em.py" --root "$OUT" --dataset IHDP --modes raw --ate-metric rel 2>&1 \
  | grep -E '^\|' | grep -viE '^\| *method|^\|[- :|]*$'
echo
echo "--- calibration (raw)"
python -u "$REPO/UWYK_Fig3_4/cate_density_metrics.py" --root "$OUT" --dataset IHDP \
    --target cate --tau-smoother none 2>&1 | grep -E '^\|' | head -8
echo
echo "compare against the bb-harness numbers: PEHE 6.8324 / IS 41.9639 (min_max)"
echo "                                        PEHE 6.2052 / IS 31.4368 (std 1.0)"
