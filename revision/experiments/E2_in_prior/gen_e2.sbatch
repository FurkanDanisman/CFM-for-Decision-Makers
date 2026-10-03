#!/bin/bash
#SBATCH --job-name=e2-gen
#SBATCH --output=logs_e2/gen_%a.out
#SBATCH --error=logs_e2/gen_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-49
# E2 data: 1000 worlds from each training prior, CPU only. One task = one prior x 100 worlds.
#   task id = prior * 10 + shard;  prior 0 dopfn  1 uwyk_2d  2 uwyk_1d  3 cpfn_1d  4 cpfn_2d
#   e.g. only uwyk_2d, uwyk_1d, cpfn_2d:  --array=10-29,40-49
# Every world is seeded by its own index, so any shard can be (re)run on its own.
# Submit from the deploy root:  E2_DATA=$SCRATCH/e2/data sbatch R-PFN/revision/experiments/E2_in_prior/gen_e2.sbatch
set -euo pipefail
mkdir -p logs_e2
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="$DEPLOY_ROOT/R-PFN"
E2="$REPO/revision/experiments/E2_in_prior"
OUT="${E2_DATA:?E2_DATA required}"
P=$(( SLURM_ARRAY_TASK_ID / 10 )); SH=$(( SLURM_ARRAY_TASK_ID % 10 ))
RANGE="$(( SH * 100 )):$(( SH * 100 + 100 ))"
export UWYK="$DEPLOY_ROOT/external/uwyk_reproduce" CAUSALPFN="$DEPLOY_ROOT/external/causalpfn"
source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
cd "$E2"
case "$P" in
  0) DOPFN_SRC="$DEPLOY_ROOT/external/dopfn" python gen_dopfn.py "$RANGE" "$OUT/dopfn" ;;
  1|2) M=$([ "$P" = 1 ] && echo 2d || echo 1d)
     PYTHONPATH="$REPO:$UWYK/src" python gen_uwyk.py "$M" "$RANGE" "$OUT/uwyk_$M" ;;
  3|4) M=$([ "$P" = 3 ] && echo 1d || echo 2d)
     PYTHONPATH="$CAUSALPFN/src:$REPO/benchmarks/uwyk_table1/shims" python gen_cpfn.py "$M" "$RANGE" "$OUT/cpfn_$M" ;;
esac
