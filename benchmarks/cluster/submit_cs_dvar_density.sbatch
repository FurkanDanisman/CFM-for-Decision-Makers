#!/bin/bash
#SBATCH --job-name=csdvar-dens
#SBATCH --output=logs_cs_dvar/dens_%A_%a.out
#SBATCH --error=logs_cs_dvar/dens_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-47
#
# Case-study d-variation density dumps using EXACTLY the ComplexMech path:
# every harness writes the cpfn schema itself, at its OWN native resolution,
# with DENSITY_DUMP=1. No tauC, no format conversion, no shared bin grid.
# UWYK_Fig3_4/cate_density_metrics.py then scores every model identically.
#
# This replaces the eval_density_tauC route for these cells. That route saved
# raw logits + bar geometry instead of a density, which (a) forced an n_y0
# quadrature per tau per query at scoring time and (b) needed a converter,
# whose 1D branch coarsened UWYK 1004->32 bins and DoPFN ~100->10 against the
# joint's grid -- i.e. compared resolutions rather than models. The harnesses
# below dump p_y0_scaled / p_joint_scaled on each model's own edges, which is
# what ComplexMech and RealCause have always done.
#
#   48 tasks = 6 models x 8 d, for ONE shift. Submit once per shift:
#       for S in 0 +2 -2; do SHIFT=$S sbatch ... ; done
#   task/8 -> model, task%8 -> d. Each task loops all 6 case studies.
#
# Required env:
#   OUT_ROOT   dumps land at OUT_ROOT/shift<S>/d<K>/ctx<N>/<model>/<CASE>/
# Optional:
#   SHIFT (default 0)   DATA (default $DEPLOY_ROOT/case_study_data/d_variation)
#   CTX (default 1000 = SCM_N_TRAIN)   NQ (default 100 = SCM_N_QUERY)
#   CKPT_* overrides

set -euo pipefail
mkdir -p logs_cs_dvar

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
export CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
export DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
export UWYK="${UWYK:-$DEPLOY_ROOT/external/uwyk_reproduce}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
SHIFT="${SHIFT:-0}"
DATA="${DATA:-$DEPLOY_ROOT/case_study_data/d_variation}"
CTX="${CTX:-1000}"
NQ="${NQ:-100}"

CKPT_CPFN1D="${CKPT_CPFN1D:-/scratch/furkanbd/rpfn_bench_kit/causalpfn_j1024_headrand_A1_h100/step_checkpoints/run/step_0050000.pt}"
CKPT_CPFN2D="${CKPT_CPFN2D:-/scratch/furkanbd/rpfn_bench_kit/cpfn2d_j32_random_A1_h100/step_checkpoints/run/step_0050000.pt}"
CKPT_GRAPH2D="${CKPT_GRAPH2D:-$REPO/Required_checkpoints/graph2d_step_50000.pt}"
CKPT_DOPFN_BB="${CKPT_DOPFN_BB:-$REPO/Required_checkpoints/dopfn_bb_j10_step_150000.pt}"
UWYK_CKPT_DIR="${UWYK_CKPT_DIR:-$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0}"

# ANC_MODE=case_family, NOT v3ab_only. v3a/v3b from build_anc_v3* hardcode a
# confounder layout (X->T, X->Y) for EVERY dataset; case_family builds the
# per-case DAG via build_case_adj, so mediators (T->X->Y), the spurious X of
# Unobserved_Confounder and the front-door structure are encoded correctly.
# Feeding a confounder graph to the mediator cases would make "anc" carry
# WRONG information and lose to noanc for reasons unrelated to the method.
# It emits the same noanc / v3a / v3b tags the scorer's METHODS expects.
MODELS=(dopfn_native dopfn_bb uwyk1d graph2d cpfn1d cpfn2d_pooled)
DS=(2 3 5 10 20 30 40 50)
# CASES_OVERRIDE (space-separated) narrows the case list, so a smoke run can do
# one case instead of all six. Unset leaves the full list, so existing
# invocations are unchanged.
if [ -n "${CASES_OVERRIDE:-}" ]; then
    read -r -a CASES <<<"$CASES_OVERRIDE"
else
    CASES=(Observed_Confounder Backdoor_Criterion Observed_Mediator
           Observed_Mediator_and_Confounder Unobserved_Confounder
           Frontdoor_Criterion)
fi

MODEL="${MODELS[$(( SLURM_ARRAY_TASK_ID / 8 ))]}"
D="${DS[$(( SLURM_ARRAY_TASK_ID % 8 ))]}"

# Point the case-study loader at this (shift, d) cell.
#
# The d_variation tree is NPZ, laid out as <shift>/<d>/<Case>/N<ctx>/..., not
# DoPFN's pkls at <root>/<Case>/*.pkl. benchmarks/scm_case_study_dataset.py
# swaps in the npz backend when CASE_STUDY_DATA_ROOT is set (see its
# "Opt-in npz backend" block), and that backend REQUIRES CASE_STUDY_N to pick
# the N<ctx> subdirectory. Setting DOPFN_DATA_ROOT/SCM_N_TRAIN instead left
# the pkl loader in force: it globbed *.pkl, found none, and every case
# reported "summary (n=0)" with exit 0 and no error.
export CASE_STUDY_DATA_ROOT="$DATA/shift$SHIFT/d$D"
export CASE_STUDY_N="$CTX"
export SCM_N_QUERY="$NQ"
[ -d "$CASE_STUDY_DATA_ROOT/Observed_Confounder/N$CTX" ] || {
    echo "FATAL: no N$CTX data under $CASE_STUDY_DATA_ROOT/Observed_Confounder" >&2
    exit 1; }

# DENSITY_DUMP=1 (the default) writes per-query densities, which the scorer
# needs and which is the expensive, large part of a run. DENSITY_DUMP=0 gives
# point estimates only -- PEHE / ATE error -- in a fraction of the time, for
# when only the point table is wanted. Default stays 1 so every existing
# invocation is unchanged.
export DENSITY_DUMP="${DENSITY_DUMP:-1}"
export SKIP_EM="${SKIP_EM:-1}"
export COMPILE="${COMPILE:-0}"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%F %T')] task=$SLURM_ARRAY_TASK_ID model=$MODEL shift$SHIFT d$D"
echo "  CASE_STUDY_DATA_ROOT=$CASE_STUDY_DATA_ROOT  N=$CTX  N_QUERY=$NQ"

for CASE in "${CASES[@]}"; do
    OUT="$OUT_ROOT/shift$SHIFT/d$D/ctx$CTX/$MODEL/$CASE"
    mkdir -p "$OUT"
    export DATASET="$CASE" OUT
    echo "=== $MODEL  $CASE ==="
    case "$MODEL" in
      dopfn_native)
        python -u "$REPO/benchmarks/eval_scm_case_studies/eval_native_dopfn.py" \
            --dataset "$CASE" || echo "WARN: $MODEL $CASE failed"
        ;;
      dopfn_bb)
        python -u "$REPO/benchmarks/l2_ihdp/eval_dopfn_bb_raw.py" \
            --dataset "$CASE" --out "$OUT/summary.npz" \
            --repo "$REPO" --dopfn "$DOPFN_ROOT" --causalpfn "$CAUSALPFN" \
            --checkpoint "$CKPT_DOPFN_BB" \
            --y-scaling "${Y_SCALING:-min_max}" \
            --std-target "${STD_TARGET:-0.3}" || echo "WARN: $MODEL $CASE failed"
        ;;
      uwyk1d)
        # Overridable so a UWYK VARIANT (e.g. the binarized head) can be run
        # without inventing a fake checkpoint directory around it.
        export CKPT="${CKPT:-$UWYK_CKPT_DIR/best_model.pt}"
        export CONFIG="${CONFIG:-$UWYK_CKPT_DIR/best_model_config.yaml}"
        export ANC_MODE=case_family
        # --t-encoding: uwyk1d.py defaults to 'target'. A binarized model needs
        # 'binary' (T fed as {0,1}); without this passthrough the flag could
        # never be set from a job, and a binary head would run under target
        # encoding and quietly score a model fed the wrong treatment variable.
        # See the RealCause sbatch: exporting CKPT is not enough, uwyk1d.py
        # overwrites it from args.ckpt.
        python -u "$REPO/realcause_eval/Table1/uwyk1d.py" \
            --dataset "$CASE" --outdir "$OUT" \
            --ckpt "$CKPT" --config "$CONFIG" \
            --uwyk-repro "$UWYK" --repo "$REPO" --causalpfn "$CAUSALPFN" \
            --anc-mode case_family \
            --t-encoding "${UWYK_T_ENCODING:-target}" || echo "WARN: $MODEL $CASE failed"
        ;;
      graph2d)
        python -u "$REPO/realcause_eval/Table1/graph2d.py" \
            --dataset "$CASE" --outdir "$OUT" --density-dump "$DENSITY_DUMP" \
            --ckpt "$CKPT_GRAPH2D" --repo "$REPO" \
            --causalpfn "$CAUSALPFN" --uwyk "$UWYK" \
            --anc-mode case_family --density-primary-mode noanc \
          || echo "WARN: $MODEL $CASE failed"
        ;;
      cpfn1d)
        # pooled, not per_arm: the dump asserts both arms share y_shift/y_scale
        # so the shift cancels in tau. Same constraint as the ComplexMech run.
        python -u "$REPO/realcause_eval/Table1/cpfn1d.py" \
            --dataset "$CASE" --outdir "$OUT" --density-dump "$DENSITY_DUMP" \
            --ckpt "$CKPT_CPFN1D" --repo "$REPO" --causalpfn "$CAUSALPFN" \
            --std-mode pooled || echo "WARN: $MODEL $CASE failed"
        ;;
      cpfn2d_pooled)
        python -u "$REPO/realcause_eval/Table1/cpfn2d.py" \
            --dataset "$CASE" --outdir "$OUT" --density-dump "$DENSITY_DUMP" \
            --ckpt "$CKPT_CPFN2D" --repo "$REPO" --causalpfn "$CAUSALPFN" \
            --std-mode pooled || echo "WARN: $MODEL $CASE failed"
        ;;
    esac
done
echo "[$(date '+%F %T')] done  $MODEL shift$SHIFT d$D"
