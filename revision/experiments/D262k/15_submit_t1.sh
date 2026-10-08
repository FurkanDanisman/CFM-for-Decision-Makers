#!/bin/bash
# Stage 15: every Do-PFN 2D (262k) evaluation again with Do-PFN's inference temperature OFF, as the ADDED
# model dopfn_repro_joint2d_262k_t1, plus the 2D dumps the IHDP/ACIC NLL tables need. Nothing existing
# is overwritten (stage 1/4 refuse to write into existing folders; E2 writes to results_t1 roots).
#   cd $KIT && git -C R-PFN pull --rebase && bash R-PFN/revision/experiments/D262k/15_submit_t1.sh
set -euo pipefail
export D262_NAME=dopfn_repro_joint2d_262k_t1 DOPFN_SOFTMAX_TEMP=0
D="$(cd "$(dirname "$0")" && pwd)"
source "$D/common.sh"; cd "$KIT"
# 1) RealCause, case studies (3 shifts), ComplexMech all rho: 5 jobs. The RealCause dumps also carry
#    logits_2d, which the NLL tables use for Do-PFN 2D.
bash "$D/1_submit_dumps.sh"
# 2) ComplexMech rho > 0.99: 1 job
bash "$D/4_rho99_dumps.sh"
# 3) E2 in-prior: new world (10,000 worlds; shards 0-9 already done by stage 14) then same world, 1 job
mkdir -p logs_e2
sbatch --account=$ACCOUNT --gres=$GRES --time=3:00:00 --cpus-per-task=4 --mem=32G --job-name=e2-t1 \
  -o logs_e2/t1_%j.out --wrap "cd $KIT && \
  SLURM_ARRAY_TASK_ID=8 E2_DATA=$SCRATCH/e2/new/data  E2_RESULTS=$SCRATCH/e2/new/results_t1  E2_SHARDS=100 bash $REPO/revision/experiments/E2_in_prior/eval_e2.sbatch && \
  SLURM_ARRAY_TASK_ID=8 E2_DATA=$SCRATCH/e2/same/data E2_RESULTS=$SCRATCH/e2/same/results_t1 E2_SHARDS=10  bash $REPO/revision/experiments/E2_in_prior/eval_e2.sbatch"
# 4) NLL tables: UWYK 2D and CausalPFN-C 2D on IHDP and ACIC with the full head output saved, same harness
#    and settings as rc_dens_uni / rc_dens_eta0, into a new root. Do-PFN 2D comes from step 1.
NLL="$SCRATCH/rc_nll2d"
[ -e "$NLL" ] && { echo "FATAL: $NLL already exists"; exit 1; }
env -u DOPFN_SOFTMAX_TEMP OUT_ROOT="$NLL" CKPT_CPFN2D="$REPO/Required_checkpoints/cpfn2d_j32_eta0_y01_step50000.pt" \
  sbatch --account=$ACCOUNT --gres=$GRES --time=3:00:00 --array=15,16,25,26 --job-name=nll2d \
  "$REPO/benchmarks/cluster/submit_realcause_density_unified.sbatch"
echo "submitted: 5 dumps + 1 rho99 + 1 E2 + 1 NLL array (4 tasks)"
