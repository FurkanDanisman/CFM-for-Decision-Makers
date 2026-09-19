#!/bin/bash
# Verify every model produces output -- on CPU, without slurm.
#
# Slurm is refusing submissions, but the question the smoke test answers ("does
# each checkpoint load and emit a scoreable density?") does not need a GPU or a
# scheduler. Three facts make this work:
#   * the .sbatch files are plain bash; #SBATCH lines are comments, so they run
#     directly once SLURM_ARRAY_TASK_ID is set by hand
#   * eval_native_dopfn.py picks its device with torch.cuda.is_available(), so
#     CUDA_VISIBLE_DEVICES="" sends it to CPU rather than failing
#   * the case-study cell is sized by NQ and CASES_OVERRIDE, so 4 queries on one
#     case is a few forward passes, not a benchmark run
#
# Deliberately the cheapest cell that still exercises the whole path: load
# weights -> forward -> dump density -> score it. It does NOT produce reportable
# numbers (4 queries, 1 realization) -- it proves the pipeline runs.
#
# Thread count is capped to stay a polite login-node citizen.
#
#   bash R-PFN/benchmarks/cluster/cpu_smoke.sh
#   ONLY=cpfn1d_j32 bash R-PFN/benchmarks/cluster/cpu_smoke.sh
#   NQ=8 bash R-PFN/benchmarks/cluster/cpu_smoke.sh

set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
REPO="${REPO:-$KIT/R-PFN}"
CK="${CK:-$REPO/Required_checkpoints}"
SC="${SCRATCH:-/scratch/furkanbd}"
OUT="${OUT:-$SC/cpu_smoke}"
NQ="${NQ:-4}"
CASE="${CASE:-Observed_Confounder}"
NTHREAD="${NTHREAD:-4}"
ONLY="${ONLY:-}"
CS_SB="$REPO/benchmarks/cluster/submit_cs_dvar_density.sbatch"
CD="$REPO/UWYK_Fig3_4/cate_density_metrics.py"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="$NTHREAD" MKL_NUM_THREADS="$NTHREAD"
export OPENBLAS_NUM_THREADS="$NTHREAD"
export SLURM_CPUS_PER_TASK="$NTHREAD"
export COMPILE=0
mkdir -p "$OUT"

# name | env var | checkpoint | CS array task (8*model_idx + 2 for d=5)
ROWS=(
  "dopfn_repro_1d_J10|DOPFN_CKPT|$CK/dopfn_repro_1d_J10_step150000.pt|2"
  "dopfn_repro_1d_J100|DOPFN_CKPT|$CK/dopfn_repro_1d_J100_step150000.pt|2"
  "dopfn_repro_joint2d|CKPT_DOPFN_BB|$CK/dopfn_repro_joint2d_bb.pt|10"
  "cpfn1d_j32|CKPT_CPFN1D|$CK/cpfn1d_j32_step50000.pt|34"
  "cpfn1d_botharms|CKPT_CPFN1D|$CK/cpfn1d_botharms_step50000.pt|34"
)

cd "$KIT" || exit 1
echo "CPU smoke: case=$CASE  NQ=$NQ  threads=$NTHREAD  out=$OUT"
echo

for r in "${ROWS[@]}"; do
    IFS='|' read -r name env ck task <<<"$r"
    [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue
    [ -f "$ck" ] || { printf '%-22s SKIP (no checkpoint)\n' "$name"; continue; }
    root="$OUT/$name"; log="$OUT/$name.log"
    rm -rf "$root"; mkdir -p "$root"
    printf '%-22s running ... ' "$name"
    SLURM_ARRAY_TASK_ID="$task" \
    env "$env=$ck" OUT_ROOT="$root" SHIFT=0 DENSITY_DUMP=1 \
        NQ="$NQ" CASES_OVERRIDE="$CASE" \
        bash "$CS_SB" > "$log" 2>&1
    st=$?
    n_npz=$(find "$root" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
    if [ "$st" != 0 ]; then printf 'HARNESS FAILED (exit %s, see %s)\n' "$st" "$log"; continue; fi
    if [ "$n_npz" = 0 ]; then printf 'NO DUMP written (exit 0! see %s)\n' "$log"; continue; fi

    cell=$(find "$root" -type d -name 'ctx*' | head -1)
    sc=$(python -u "$CD" --root "$cell" --dataset "$CASE" --target cate \
             --tau-smoother none --max-real 1 2>&1)
    echo "$sc" >> "$log"
    rows=$(printf '%s\n' "$sc" | awk -F'|' 'NF>4 { gsub(/ /,"",$NF); if ($NF ~ /^[0-9]+$/ && $NF+0>0) c++ } END {print c+0}')
    if [ "$rows" -gt 0 ]; then printf 'PASS  (%s npz, %s scored row)\n' "$n_npz" "$rows"
    else printf 'dumped %s npz but 0 scored rows -- see %s\n' "$n_npz" "$log"; fi
done

echo
echo "logs in $OUT/*.log"
