#!/bin/bash
#SBATCH --job-name=ate-cscm
#SBATCH --output=logs_ate/%x_%j.out
#SBATCH --error=logs_ate/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Case-study and ComplexMech density dumps for the OT ATE coverage check, with the paper's dump scripts
# run under bash per cell (as submit_dump_one_model.sbatch / submit_cmech_dump_one.sbatch do).
#   ATE_BENCH=cs ATE_SHIFT=0|+2|-2 : the 6 cases at d = 5..50 (inner cells 2..7), one job per model x shift
#   ATE_BENCH=cm                   : ComplexMech N = 1000, 6 node counts x 2 query subsets, one job per model
# Model settings as the paper (Do-PFN 2D = 262k, inference temperature off; CausalPFN-C 1D = cpfn1d_j1024;
# CausalPFN-C 2D = cpfn2d eta0). A cell with a .done marker is skipped: resubmit a timed-out job.
# From the kit root (no dependency jobs):
#   for M in dopfn_native dopfn_joint2d_262k cpfn1d cpfn2d; do
#     for S in 0 +2 -2; do ATE_MODEL=$M ATE_BENCH=cs ATE_SHIFT=$S sbatch --account=aip-rgrosse --gres=gpu:l40s:1 \
#       R-PFN/revision/experiments/ate_subsample/cs_cm_dumps.sbatch; done
#     ATE_MODEL=$M ATE_BENCH=cm sbatch --account=aip-rgrosse --gres=gpu:l40s:1 \
#       R-PFN/revision/experiments/ate_subsample/cs_cm_dumps.sbatch
#   done
set -uo pipefail
mkdir -p logs_ate
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; CK="$REPO/Required_checkpoints"
MODEL="${ATE_MODEL:?ATE_MODEL = dopfn_native | dopfn_joint2d_262k | cpfn1d | cpfn2d}"
BENCH="${ATE_BENCH:?ATE_BENCH = cs | cm}"
ENVS=(DENSITY_DUMP=1 DEPLOY_ROOT="$KIT")
case "$MODEL" in
  dopfn_native)       IDX=0 ;;
  dopfn_joint2d_262k) IDX=0; ENVS+=(DOPFN_CKPT="$CK/dopfn_repro_joint2d_step262144.pt" DOPFN_SOFTMAX_TEMP=0) ;;
  cpfn1d)             IDX=4; ENVS+=(CKPT_CPFN1D="$CK/cpfn1d_j1024_headrand_step_50000.pt") ;;
  cpfn2d)             IDX=5; ENVS+=(CKPT_CPFN2D="$CK/cpfn2d_j32_eta0_y01_step50000.pt") ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
run_cell () {   # $1 = marker name, rest = env assignments; runs $INNER once
  local mk="$ROOT/$1.done"; shift
  [ -f "$mk" ] && { echo "skip $(basename "$mk" .done) (done)"; return; }
  [ "$SECONDS" -gt 9600 ] && { echo "[$(date '+%F %T')] stopping before the time limit; resubmit to continue"; exit 0; }
  echo "===== [$(date '+%F %T')] $MODEL $(basename "$mk" .done)"
  env "${ENVS[@]}" "$@" bash "$INNER" && touch "$mk" || echo "  CELL FAILED $(basename "$mk" .done)"
}
if [ "$BENCH" = cs ]; then
  SHIFT="${ATE_SHIFT:?ATE_SHIFT = 0 | +2 | -2}"
  INNER="$REPO/benchmarks/cluster/submit_cs_dvar_density.sbatch"
  ROOT="${ATE_CS_DUMPS:-$SCRATCH/ate_cs_dumps}/$MODEL"; mkdir -p "$ROOT"
  LABELS=(d2 d3 d5 d10 d20 d30 d40 d50)
  for c in 2 3 4 5 6 7; do
    run_cell "shift${SHIFT}_${LABELS[$c]}" OUT_ROOT="$ROOT" SHIFT="$SHIFT" SLURM_ARRAY_TASK_ID=$(( IDX * 8 + c ))
  done
else
  INNER="$REPO/benchmarks/cluster/submit_cmech_pehe_1d_vs_2d.sbatch"
  ROOT="${ATE_CM_DUMPS:-$SCRATCH/ate_cm_dumps}/$MODEL"; mkdir -p "$ROOT"
  NODES=(5 10 20 30 40 50)
  for sub in nonzero zero; do
    for i in 0 1 2 3 4 5; do
      run_cell "n${NODES[$i]}_$sub" OUT_ROOT="$ROOT" UWYK_FIG34_DATA="$SCRATCH/cmech_data_v2" SUBSET="$sub" \
        CMECH_CONTEXTS=1000 CMECH_NODES="${NODES[*]}" SLURM_ARRAY_TASK_ID=$(( IDX * 6 + i ))
    done
  done
fi
echo "[$(date '+%F %T')] $MODEL $BENCH: $(ls "$ROOT"/*.done 2>/dev/null | wc -l) cells done"
