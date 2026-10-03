#!/bin/bash
#SBATCH --job-name=e2-gen
#SBATCH --output=logs_e2/gen_%a.out
#SBATCH --error=logs_e2/gen_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4
# E2 data: 1000 worlds from each training prior. CPU only.
#   0 dopfn   1 uwyk_2d   2 uwyk_1d   3 cpfn_1d   4 cpfn_2d
# Submit from the deploy root:  E2_DATA=$SCRATCH/e2/data sbatch R-PFN/revision/experiments/E2_in_prior/gen_e2.sbatch
set -euo pipefail
mkdir -p logs_e2
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="$DEPLOY_ROOT/R-PFN"
E2="$REPO/revision/experiments/E2_in_prior"
OUT="${E2_DATA:?E2_DATA required}"
R="${E2_R:-1000}"
source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
cd "$E2"
case "$SLURM_ARRAY_TASK_ID" in
  0) DOPFN_SRC="$DEPLOY_ROOT/external/dopfn" python gen_dopfn.py "$R" "$OUT/dopfn" ;;
  1) PYTHONPATH="$REPO:$DEPLOY_ROOT/external/uwyk_reproduce/src" python gen_uwyk.py 2d "$R" "$OUT/uwyk_2d" ;;
  2) PYTHONPATH="$REPO:$DEPLOY_ROOT/external/uwyk_reproduce/src" python gen_uwyk.py 1d "$R" "$OUT/uwyk_1d" ;;
  3|4) M=$([ "$SLURM_ARRAY_TASK_ID" = 3 ] && echo 1d || echo 2d)
     CAUSALPFN="$DEPLOY_ROOT/external/causalpfn" \
     PYTHONPATH="$DEPLOY_ROOT/external/causalpfn/src:$REPO/benchmarks/uwyk_table1/shims" \
       python gen_cpfn.py "$M" "$R" "$OUT/cpfn_$M" ;;
esac
