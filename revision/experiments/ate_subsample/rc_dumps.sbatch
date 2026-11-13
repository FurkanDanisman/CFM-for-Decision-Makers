#!/bin/bash
#SBATCH --job-name=ate-rc
#SBATCH --output=logs_ate/%x_%j.out
#SBATCH --error=logs_ate/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# RealCause density dumps for the OT ATE coverage check, one job per model, all 5 datasets, with the
# paper's dump script (benchmarks/cluster/submit_realcause_density_unified.sbatch, run under bash per
# dataset as submit_dump_one_model.sbatch does). Model settings as the paper: Do-PFN 2D = 262k with the
# inference temperature off (D262k); CausalPFN-C 1D = cpfn1d_j1024; CausalPFN-C 2D = cpfn2d eta0.
# A dataset with a .done marker is skipped, so a timed-out model is simply resubmitted.
# Dumps: $SCRATCH/ate_rc_dumps/<model>/<inner model>/<dataset>/. Score afterwards with rc_score.sbatch.
# From the kit root (no dependency jobs):
#   for M in dopfn_native dopfn_joint2d_262k cpfn1d cpfn2d; do ATE_MODEL=$M sbatch --account=aip-rgrosse \
#     --gres=gpu:l40s:1 R-PFN/revision/experiments/ate_subsample/rc_dumps.sbatch; done
set -uo pipefail
mkdir -p logs_ate
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; CK="$REPO/Required_checkpoints"; FC="$KIT/final_checkpoints_paper"
INNER="$REPO/benchmarks/cluster/submit_realcause_density_unified.sbatch"
MODEL="${ATE_MODEL:?ATE_MODEL required}"
T0="${ATE_T0:-$(date +%s)}"   # job start, shared by the steps of one job
ROOT="${ATE_RC_DUMPS:-$SCRATCH/ate_rc_dumps}/$MODEL"; mkdir -p "$ROOT"
ENVS=(OUT_ROOT="$ROOT" DENSITY_DUMP=1 DEPLOY_ROOT="$KIT")
case "$MODEL" in
  dopfn_native)       IDX=0 ;;
  dopfn_joint2d_262k) IDX=0; ENVS+=(DOPFN_CKPT="$CK/dopfn_repro_joint2d_step262144.pt" DOPFN_SOFTMAX_TEMP=0) ;;
  cpfn1d)             IDX=4; ENVS+=(CKPT_CPFN1D="$CK/cpfn1d_j1024_headrand_step_50000.pt") ;;
  cpfn2d)             IDX=5; ENVS+=(CKPT_CPFN2D="$CK/cpfn2d_j32_eta0_y01_step50000.pt") ;;
  uwyk1d)             IDX=2 ;;                  # published UWYK, its dump script's own settings
  graph2d)            IDX=3 ;;                  # UWYK 2D, graph2d_step_50000 (the script's default)
  # matched-resolution / matched-label ablations: the checkpoints and settings of E2_in_prior/eval_e2.sbatch
  dopfn1d_mr)         IDX=0; ENVS+=(DOPFN_CKPT="$FC/dopfn1d_J10_step262144.pt" DOPFN_SOFTMAX_TEMP=0) ;;
  dopfn1d_ml)         IDX=0; ENVS+=(DOPFN_CKPT="$FC/dopfn1d_botharms_step262144.pt" DOPFN_SOFTMAX_TEMP=0) ;;
  cpfn1d_mr)          IDX=4; ENVS+=(CKPT_CPFN1D="$FC/cpfnC_mr_1d_j32_step50000.pt") ;;
  cpfn1d_ml)          IDX=4; ENVS+=(CKPT_CPFN1D="$FC/cpfnC_ml_1d_botharms_step50000.pt") ;;
  uwyk1d_mr)          IDX=2; ENVS+=(CKPT="$FC/uwyk_mr_D_j32_step50000.pt" CONFIG="$FC/uwyk_mr_D_j32_config.yaml" UWYK_T_ENCODING=target) ;;
  uwyk1d_ml)          IDX=2; ENVS+=(CKPT="$FC/uwyk_ml_C_botharms_step50000.pt" CONFIG="$FC/uwyk_ml_C_botharms_config.yaml" UWYK_T_ENCODING=binary) ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
SETS=(IHDP ACIC CPS PSID PSID_bal)
for c in 0 1 2 3 4; do
  D="${SETS[$c]}"; [ -f "$ROOT/$D.done" ] && { echo "skip $D (done)"; continue; }
  [ $(( $(date +%s) - T0 )) -gt 9600 ] && { echo "[$(date '+%F %T')] stopping before the time limit; resubmit to continue"; exit 0; }
  echo "===== [$(date '+%F %T')] $MODEL / $D"
  SLURM_ARRAY_TASK_ID=$(( IDX * 5 + c )) env "${ENVS[@]}" bash "$INNER" && touch "$ROOT/$D.done" \
    || echo "  DATASET FAILED $D"
done
echo "[$(date '+%F %T')] $MODEL: $(ls "$ROOT"/*.done 2>/dev/null | wc -l)/5 datasets done"
