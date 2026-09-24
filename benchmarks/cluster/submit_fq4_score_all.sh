#!/bin/bash
# One scoring job per model -- the same parallelism the dumps use.
#
# The reaper scores a cell at a time, so all 13 models queue behind each other and MALC
# at B=1000 costs ~8.4 h for 45 cells. The per-(cell, model) parts are independent, so
# 13 jobs finish in the time of one model, ~45 min. The reaper then merges the parts and
# owns deletion as before.
#
#   bash R-PFN/benchmarks/cluster/submit_fq4_score_all.sh            # dry run
#   MODE=cmech bash ... --submit
set -uo pipefail
SUBMIT=0; [ "${1:-}" = "--submit" ] && SUBMIT=1
KIT="${KIT:-$PWD}"; REPO="${REPO:-$KIT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"
CK="${CK:-$REPO/Required_checkpoints}"; UWYKD="${UWYK_CKPT_DIR:-/nonexistent}"
# shellcheck disable=SC1091
source "$REPO/benchmarks/cluster/fq_models.sh"
ACCT="${ACCOUNT:-def-rgrosse}"; ONLY="${ONLY:-}"
TIME="${SCORE_TIME:-6:00:00}"
# 13 jobs x 16 CPUs is 208 cores per benchmark, which queues. Fewer CPUs per job means
# slower MALC within a job (it is what --malc-workers gets) but the jobs start sooner,
# and 13 jobs starting at 8 cores beats 13 jobs waiting at 16.
CPUS="${SCORE_CPUS:-16}"
MEM="${SCORE_MEM:-48G}"
mkdir -p "$KIT/logs_fq"
N=0
for r in "${ROWS[@]}"; do
    IFS='|' read -r m _rest <<<"$r"
    [ -n "$ONLY" ] && { case " $ONLY " in *" $m "*) ;; *) continue ;; esac; }
    N=$((N+1))
    if [ "$SUBMIT" = 1 ]; then
        jid=$(MODEL="$m" sbatch --parsable --account="$ACCT" --time="$TIME" \
              --cpus-per-task="$CPUS" --mem="$MEM" \
              --job-name="fq4s-$m" \
              "$REPO/benchmarks/cluster/submit_fq4_score_model.sbatch") || jid=REJECTED
        printf '  score %-24s %s\n' "$m" "$jid"
    else
        printf '  score %-24s (dry)\n' "$m"
    fi
done
echo
echo "JOBS: $N   mode=${MODE:-cs}  B=${MALC_B:-1000}  cpus=$CPUS mem=$MEM"
[ "$SUBMIT" = 1 ] || echo "dry run -- add --submit"
