#!/bin/bash
# A/B test the pooled-vs-per-arm Y-scaling hypothesis for the 2D heads, on ONE
# context size and a small set of cases, via the same dispatcher sbatch.
#
# Hypothesis: pooled Y-scaling is inflated by the +ATE between-arm gap, crushing
# the within-arm CID below the J=10 grid's bin width — the failure mode on the
# shifted cases. Per-arm scaling removes the gap.
#
# Cases (dispatcher array indices): 1=Observed_Mediator (should IMPROVE with
# per-arm), 5=Backdoor_Criterion (τ≈0 control — should barely move).
#
# Usage:
#   DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit \
#     bash case_study/cluster/ab_scaling.sh <DATA_ROOT> <AB_OUT_ROOT>
#
# Env: N (default 1000), ARRAY (default "1,5").
set -euo pipefail

DATA_ROOT="${1:?DATA_ROOT required (npz root, e.g. .../case_study_data/shift+2)}"
AB="${2:?AB_OUT_ROOT required}"

DEPLOY_ROOT="${DEPLOY_ROOT:?DEPLOY_ROOT required}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SBATCH="$REPO/benchmarks/cluster/submit_eval_scm_case_studies.sbatch"
N="${N:-1000}"
ARRAY="${ARRAY:-1,5}"

export DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
CPFN2D_CKPT="${CPFN2D_CKPT:-$DEPLOY_ROOT/cpfn2d_j32_random_A1_h100/step_checkpoints/run/step_0050000.pt}"
DOPFNBB_CKPT="${DOPFNBB_CKPT:-$REPO/Required_checkpoints/dopfn_bb_j10_step_150000.pt}"

common="CASE_STUDY_DATA_ROOT=$DATA_ROOT CASE_STUDY_N=$N SCM_TRUE_ATE_SHIFT=0 \
SCM_N_QUERY=100 DOPFN_ROOT=$DOPFN_ROOT CAUSALPFN=$CAUSALPFN DEPLOY_ROOT=$DEPLOY_ROOT REPO=$REPO"

echo "[ab] data_root=$DATA_ROOT  N=$N  cases(array)=$ARRAY  out=$AB"

# ── cpfn2d: pooled (A) vs per_arm_std_target (B) ──
env $common MODEL=cpfn2d CKPT="$CPFN2D_CKPT" STD_MODE=pooled \
    OUT_ROOT="$AB/cpfn2d_pooled"  sbatch --array="$ARRAY" "$SBATCH"
env $common MODEL=cpfn2d CKPT="$CPFN2D_CKPT" STD_MODE=per_arm_std_target STD_TARGET=3.0 \
    OUT_ROOT="$AB/cpfn2d_perarm" sbatch --array="$ARRAY" "$SBATCH"

# ── dopfn_bb: min_max (A, = current cluster default) vs std_per_arm (B) ──
env $common MODEL=dopfn_bb CKPT="$DOPFNBB_CKPT" BB_Y_SCALING=min_max \
    OUT_ROOT="$AB/dopfn_bb_minmax" sbatch --array="$ARRAY" "$SBATCH"
env $common MODEL=dopfn_bb CKPT="$DOPFNBB_CKPT" BB_Y_SCALING=std_per_arm BB_STD_TARGET=0.3 \
    OUT_ROOT="$AB/dopfn_bb_perarm" sbatch --array="$ARRAY" "$SBATCH"

echo "[ab] submitted 4 variants × cases[$ARRAY] @ N=$N"
echo "[ab] when done:  python $REPO/case_study/cluster/ab_report.py --ab $AB"
