#!/bin/bash
#SBATCH --job-name=e2-gen
#SBATCH --output=logs_e2/gen_%x_%a.out
#SBATCH --error=logs_e2/gen_%x_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
# E2 data, CPU only. Priors: 0 dopfn  1 uwyk_2d  2 uwyk_1d (binarized T)  3 cpfn_1d  4 cpfn_2d
#                            5 uwyk_1dc (UWYK 1D's own prior, continuous T)
#   E2_RUN=new : 10,000 worlds per prior, 500 queries each. Task = prior * 10 + chunk of 1000 worlds.
#                sbatch --array=0-59%6
#   E2_RUN=same: ONE world per prior, 1000 observational sets, 500 fresh queries each. Task = prior.
#                sbatch --array=0-5
# Finished datasets are skipped, so a task that times out is simply resubmitted.
# Submit from the deploy root, e.g.
#   E2_RUN=new  E2_DATA=$SCRATCH/e2/new/data  sbatch --array=0-59%6 R-PFN/revision/experiments/E2_in_prior/gen_e2.sbatch
#   E2_RUN=same E2_DATA=$SCRATCH/e2/same/data sbatch --array=0-5    R-PFN/revision/experiments/E2_in_prior/gen_e2.sbatch
set -euo pipefail
mkdir -p logs_e2
DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
REPO="$DEPLOY_ROOT/R-PFN"
E2="$REPO/revision/experiments/E2_in_prior"
OUT="${E2_DATA:?E2_DATA required}"
RUN="${E2_RUN:?E2_RUN = new or same}"
if [ "$RUN" = new ]; then
    P=$(( SLURM_ARRAY_TASK_ID / 10 )); C=$(( SLURM_ARRAY_TASK_ID % 10 ))
    RANGE="$(( C * 1000 )):$(( C * 1000 + 1000 ))"
else
    P=$SLURM_ARRAY_TASK_ID; RANGE="0:1000"
fi
export UWYK="$DEPLOY_ROOT/external/uwyk_reproduce" CAUSALPFN="$DEPLOY_ROOT/external/causalpfn"
source "$DEPLOY_ROOT/venv/bin/activate"
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
cd "$E2"
echo "[$(date '+%F %T')] run=$RUN prior=$P range=$RANGE"
case "$P" in
  0) DOPFN_SRC="$DEPLOY_ROOT/external/dopfn" python gen_dopfn.py "$RUN" "$RANGE" "$OUT/dopfn" ;;
  1|2) M=$([ "$P" = 1 ] && echo 2d || echo 1d)
     PYTHONPATH="$REPO:$UWYK/src" python gen_uwyk.py "$M" "$RUN" "$RANGE" "$OUT/uwyk_$M" ;;
  3|4) M=$([ "$P" = 3 ] && echo 1d || echo 2d)
     PYTHONPATH="$CAUSALPFN/src:$REPO/benchmarks/uwyk_table1/shims" python gen_cpfn.py "$M" "$RUN" "$RANGE" "$OUT/cpfn_$M" ;;
  5) PYTHONPATH="$REPO:$UWYK/src" python gen_uwyk_cont.py "$RUN" "$RANGE" "$OUT/uwyk_1dc" ;;
esac
echo "[$(date '+%F %T')] done"
