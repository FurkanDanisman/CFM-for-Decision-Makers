#!/bin/bash
# Standardization sweep for the two fixed-grid 2D heads (cpfn2d, dopfn_bb) on
# the shift+2 case studies at N=500. Grid and shift are held fixed (can't change
# them); the ONLY lever is how Y is standardized onto the model's [-1,1] (bb) /
# [-10,10] (cpfn2d) grid. Also runs the 1D references (native, cpfn1d) once as
# the target line. per_arm variants are included but flagged INVALID (they
# reintroduce confounded arm-mean gaps) — reference upper-bound only.
#
# Usage:
#   DEPLOY_ROOT=$SCRATCH/rpfn_bench_kit \
#     bash case_study/cluster/std_sweep.sh <DATA_ROOT> <STD_ROOT>
#   DATA_ROOT : npz root with N=500 (e.g. .../case_study_data/shift+2)
set -euo pipefail

DATA_ROOT="${1:?DATA_ROOT required (npz root incl. N=500)}"
ROOT="${2:?STD_ROOT required}"
DEPLOY_ROOT="${DEPLOY_ROOT:?DEPLOY_ROOT required}"
REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
SB="$REPO/benchmarks/cluster/submit_eval_scm_case_studies.sbatch"
N=500

export DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
CPFN2D_CKPT="${CPFN2D_CKPT:-$DEPLOY_ROOT/cpfn2d_j32_random_A1_h100/step_checkpoints/run/step_0050000.pt}"
CPFN1D_CKPT="${CPFN1D_CKPT:-$DEPLOY_ROOT/causalpfn_j1024_headrand_A1_h100/step_checkpoints/run/step_0050000.pt}"
DOPFNBB_CKPT="${DOPFNBB_CKPT:-$REPO/Required_checkpoints/dopfn_bb_j10_step_150000.pt}"

common="CASE_STUDY_DATA_ROOT=$DATA_ROOT CASE_STUDY_N=$N SCM_TRUE_ATE_SHIFT=0 \
SCM_N_QUERY=100 DOPFN_ROOT=$DOPFN_ROOT CAUSALPFN=$CAUSALPFN DEPLOY_ROOT=$DEPLOY_ROOT REPO=$REPO"

sub() { env $common "$@" sbatch --array=0-5 "$SB" >/dev/null; }   # 6 cases

echo "[std_sweep] data=$DATA_ROOT  N=$N  root=$ROOT"

# ── dopfn_bb (edges ±1, J=10): --y-scaling / --std-target via BB_* env ──
for spec in "minmax:min_max:0.3" "std0.1:std:0.1" "std0.2:std:0.2" \
            "std0.3:std:0.3" "std0.5:std:0.5" "perarm0.3:std_per_arm:0.3"; do
  name="${spec%%:*}"; rest="${spec#*:}"; ys="${rest%%:*}"; tgt="${rest##*:}"
  sub MODEL=dopfn_bb CKPT="$DOPFNBB_CKPT" BB_Y_SCALING="$ys" BB_STD_TARGET="$tgt" \
      OUT_ROOT="$ROOT/dopfn_bb/$name"
done

# ── cpfn2d (edges ±10, J=32): STD_MODE / STD_TARGET env ──
for spec in "pooled:pooled:1" "std1:std_target:1" "std3:std_target:3" \
            "std5:std_target:5" "perarm3:per_arm_std_target:3"; do
  name="${spec%%:*}"; rest="${spec#*:}"; mode="${rest%%:*}"; tgt="${rest##*:}"
  sub MODEL=cpfn2d CKPT="$CPFN2D_CKPT" STD_MODE="$mode" STD_TARGET="$tgt" \
      OUT_ROOT="$ROOT/cpfn2d/$name"
done

# ── 1D references (no std knob) ──
sub MODEL=dopfn_native CKPT=none      OUT_ROOT="$ROOT/ref/dopfn_native"
sub MODEL=cpfn1d       CKPT="$CPFN1D_CKPT" STD_MODE=per_arm OUT_ROOT="$ROOT/ref/cpfn1d"

echo "[std_sweep] submitted. Report when done:"
echo "  python $REPO/case_study/cluster/std_report.py --root $ROOT"
