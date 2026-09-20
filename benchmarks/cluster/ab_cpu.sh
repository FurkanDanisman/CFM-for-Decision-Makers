#!/bin/bash
# Run the joint2d y-scaling A/B on CPU, on the login node. No queue, no GPU.
#
# The equivalent GPU job took 18 SECONDS (job 5563060_5, sm-rc-joint2d on IHDP),
# so this is minutes of CPU work -- not worth waiting behind a busy queue for.
# The .sbatch body is plain bash, so it runs directly with SLURM_ARRAY_TASK_ID
# set by hand; array task 5 is dopfn_bb x IHDP.
#
#   bash R-PFN/benchmarks/cluster/ab_cpu.sh
# then
#   bash R-PFN/benchmarks/cluster/ab_joint2d_yscaling.sh --compare

set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
REPO="${REPO:-$KIT/R-PFN}"
CK="${CK:-$REPO/Required_checkpoints}"
SC="${SCRATCH:-/scratch/furkanbd}"
AB="${AB:-$SC/ab_joint2d}"
NT="${NTHREAD:-8}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="$NT" MKL_NUM_THREADS="$NT" OPENBLAS_NUM_THREADS="$NT"
export SLURM_CPUS_PER_TASK="$NT" COMPILE=0
cd "$KIT" || exit 1

for a in "minmax|min_max|0.3" "zscore|std|1.0"; do
    IFS='|' read -r arm ys st <<<"$a"
    echo "=== arm=$arm  Y_SCALING=$ys  STD_TARGET=$st  (CPU, $NT threads)"
    rm -rf "$AB/$arm"; mkdir -p "$AB/$arm"
    SLURM_ARRAY_TASK_ID=5 \
    CKPT_DOPFN_BB="$CK/dopfn_repro_joint2d_bb.pt" OUT_ROOT="$AB/$arm" \
    DENSITY_DUMP=1 Y_SCALING="$ys" STD_TARGET="$st" \
        bash "$REPO/benchmarks/cluster/submit_realcause_density_unified.sbatch" \
        > "$AB/$arm.log" 2>&1
    st_=$?
    n=$(find "$AB/$arm" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
    echo "    exit=$st_  npz=$n  log=$AB/$arm.log"
done
echo
echo "compare:  bash $REPO/benchmarks/cluster/ab_joint2d_yscaling.sh --compare"
