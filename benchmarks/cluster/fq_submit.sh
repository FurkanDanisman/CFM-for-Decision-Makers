#!/bin/bash
# Submit the fixed-query jobs, creating the log dir FIRST.
#
# Slurm opens --output before the script's first line runs, so an `mkdir -p
# logs_fq` inside the sbatch is too late: with logs_fq absent the job dies at
# launch, writes no log, and simply is not in squeue -- which looks exactly like
# "it crashed silently". Hence this wrapper.
#
#   bash R-PFN/benchmarks/cluster/fq_submit.sh setB    # generate set B + 13 dumps
#   bash R-PFN/benchmarks/cluster/fq_submit.sh score   # score A and B
#   bash R-PFN/benchmarks/cluster/fq_submit.sh both
#   bash R-PFN/benchmarks/cluster/fq_submit.sh fq4cs   # four-column, one cell
# Env: ACCOUNT (default def-rgrosse), plus whatever the target job reads.
set -uo pipefail
WHAT="${1:-both}"
KIT="${KIT:-$PWD}"; REPO="${REPO:-$KIT/R-PFN}"
ACCT="${ACCOUNT:-def-rgrosse}"
mkdir -p "$KIT/logs_fq"
cd "$KIT"
case "$WHAT" in
  setB|both)  echo "-- set B (generate, verify, submit 13 dumps)"
              ACCOUNT="$ACCT" sbatch --account="$ACCT" \
                  "$REPO/benchmarks/cluster/submit_fq_setB.sbatch" ;;
esac
case "$WHAT" in
  fq4cs)      echo "-- fq4 case-study cell (generate, verify, submit 13 dumps)"
              ACCOUNT="$ACCT" sbatch --account="$ACCT" \
                  "$REPO/benchmarks/cluster/submit_fq4_cs.sbatch" ;;
esac
case "$WHAT" in
  score|both) echo "-- score A and B"
              sbatch --account="$ACCT" \
                  "$REPO/benchmarks/cluster/submit_score_fq.sbatch" ;;
esac
echo "logs: $KIT/logs_fq/"
