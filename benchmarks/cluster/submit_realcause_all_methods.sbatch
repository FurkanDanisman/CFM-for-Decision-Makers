#!/bin/bash
#SBATCH --job-name=rc-all
#SBATCH --output=logs_realcause_all/eval_%A_%a.out
#SBATCH --error=logs_realcause_all/eval_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-34
#
# 35-task array — 7 methods × 5 RealCause datasets.
#
#   method_idx = SLURM_ARRAY_TASK_ID / 5    0=cpfn1d 1=cpfn2d 2=dopfn 3=dopfnbb
#                                           4=fn_50  5=graph2d 6=uwyk1d
#   dataset_idx = SLURM_ARRAY_TASK_ID % 5   0=IHDP 1=ACIC 2=CPS 3=PSID 4=PSID_bal
#
# graph2d and uwyk1d run with ANC_MODE=v3b_only which emits BOTH v3b and noanc
# keys per realization NPZ — the aggregator reads both into separate rows.
#
# Env overrides:
#   OUT_ROOT           required — per-method outputs land at OUT_ROOT/<method>/<DATASET>/
#   CKPT_CPFN2D        default cpfn2d_j32_random_A1_h100/step_0050000.pt
#   CKPT_CPFN1D        default causalpfn_j1024_headrand_A1_h100/step_0050000.pt
#   CKPT_DOPFNBB       default Required_checkpoints/dopfn_bb_j10_step_150000.pt
#   CKPT_FN50          default checkpoints/step_50000_final.pt
#   CKPT_GRAPH2D       default Required_checkpoints/graph2d_step_50000.pt
#
# Aggregate the whole run with:
#   python R-PFN/realcause_eval/aggregate_all_methods.py --out-root $OUT_ROOT

set -euo pipefail
mkdir -p logs_realcause_all

DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD}"
if [ ! -f "$DEPLOY_ROOT/venv/bin/activate" ]; then
    for _ in 1 2 3; do
        [ -f "$DEPLOY_ROOT/venv/bin/activate" ] && break
        DEPLOY_ROOT="$(cd "$DEPLOY_ROOT/.." && pwd)"
    done
fi
[ -f "$DEPLOY_ROOT/venv/bin/activate" ] || { echo "FATAL: venv not found under $DEPLOY_ROOT" >&2; exit 1; }

REPO="${REPO:-$DEPLOY_ROOT/R-PFN}"
CAUSALPFN="${CAUSALPFN:-$DEPLOY_ROOT/external/causalpfn}"
DOPFN_ROOT="${DOPFN_ROOT:-$DEPLOY_ROOT/external/dopfn}"
UWYK_REPRO="${UWYK_REPRO:-$DEPLOY_ROOT/external/uwyk_reproduce}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required (per-method dirs land under it)}"

CKPT_CPFN2D="${CKPT_CPFN2D:-/scratch/furkanbd/rpfn_bench_kit/cpfn2d_j32_random_A1_h100/step_checkpoints/run/step_0050000.pt}"
CKPT_CPFN1D="${CKPT_CPFN1D:-/scratch/furkanbd/rpfn_bench_kit/causalpfn_j1024_headrand_A1_h100/step_checkpoints/run/step_0050000.pt}"
CKPT_DOPFNBB="${CKPT_DOPFNBB:-$DEPLOY_ROOT/Required_checkpoints/dopfn_bb_j10_step_150000.pt}"
CKPT_FN50="${CKPT_FN50:-$DEPLOY_ROOT/checkpoints/step_50000_final.pt}"
CKPT_GRAPH2D="${CKPT_GRAPH2D:-$DEPLOY_ROOT/Required_checkpoints/graph2d_step_50000.pt}"

METHOD_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
DATASET_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))

case "$METHOD_IDX" in
    0) METHOD=cpfn1d  ;;
    1) METHOD=cpfn2d  ;;
    2) METHOD=dopfn   ;;
    3) METHOD=dopfnbb ;;
    4) METHOD=fn_50   ;;
    5) METHOD=graph2d ;;
    6) METHOD=uwyk1d  ;;
    *) echo "invalid METHOD_IDX=$METHOD_IDX" >&2; exit 2 ;;
esac

case "$DATASET_IDX" in
    0) DATASET=IHDP     ;;
    1) DATASET=ACIC     ;;
    2) DATASET=CPS      ;;
    3) DATASET=PSID     ;;
    4) DATASET=PSID_bal ;;
    *) echo "invalid DATASET_IDX=$DATASET_IDX" >&2; exit 2 ;;
esac

OUT="$OUT_ROOT/$METHOD/$DATASET"
mkdir -p "$OUT"

source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] METHOD=$METHOD  DATASET=$DATASET  OUT=$OUT"

case "$METHOD" in
    cpfn1d)
        [ -f "$CKPT_CPFN1D" ] || { echo "FATAL: CKPT_CPFN1D not found: $CKPT_CPFN1D" >&2; exit 1; }
        python -u "$REPO/realcause_eval/cpfn1d.py" \
            --dataset "$DATASET" --outdir "$OUT" \
            --ckpt "$CKPT_CPFN1D" --repo "$REPO" --causalpfn "$CAUSALPFN"
        ;;
    cpfn2d)
        [ -f "$CKPT_CPFN2D" ] || { echo "FATAL: CKPT_CPFN2D not found: $CKPT_CPFN2D" >&2; exit 1; }
        python -u "$REPO/realcause_eval/cpfn2d.py" \
            --dataset "$DATASET" --outdir "$OUT" \
            --ckpt "$CKPT_CPFN2D" --repo "$REPO" --causalpfn "$CAUSALPFN"
        ;;
    dopfn)
        [ -d "$DOPFN_ROOT" ] || { echo "FATAL: DOPFN_ROOT not found: $DOPFN_ROOT" >&2; exit 1; }
        # dopfn.py takes --realization; loop 0..(n-1) inside the wrapper via env
        # or do it here. Simplest: --realization -1 means "all realizations".
        # Look at do_pfn.py's actual CLI: it appears to accept --dataset and
        # iterates internally when --realization omitted.
        python -u "$REPO/realcause_eval/do_pfn.py" \
            --dataset "$DATASET" --outdir "$OUT" \
            --dopfn "$DOPFN_ROOT" --causalpfn "$CAUSALPFN"
        ;;
    dopfnbb)
        [ -f "$CKPT_DOPFNBB" ] || { echo "FATAL: CKPT_DOPFNBB not found: $CKPT_DOPFNBB" >&2; exit 1; }
        # do_pfn_bb.py uses 'PSIDbal' (no underscore) as dataset choice.
        DS_BB="$DATASET"
        [ "$DATASET" = "PSID_bal" ] && DS_BB="PSIDbal"
        python -u "$REPO/realcause_eval/do_pfn_bb.py" \
            --dataset "$DS_BB" --outdir "$OUT" \
            --ckpt "$CKPT_DOPFNBB" --repo "$REPO" \
            --dopfn "$DOPFN_ROOT" --causalpfn "$CAUSALPFN"
        ;;
    fn_50)
        [ -f "$CKPT_FN50" ] || { echo "FATAL: CKPT_FN50 not found: $CKPT_FN50" >&2; exit 1; }
        # fn_50.py uses $TABLE1_OUT_ROOT/$exp_name/ — put each dataset under
        # its own exp_name so results are cleanly separated.
        python -u "$REPO/realcause_eval/fn_50.py" \
            --dataset "$DATASET" --exp-name "rc_all_$DATASET" \
            --out "$OUT" --ckpt "$CKPT_FN50" \
            --causalpfn "$CAUSALPFN" --repo "$REPO"
        ;;
    graph2d)
        [ -f "$CKPT_GRAPH2D" ] || { echo "FATAL: CKPT_GRAPH2D not found: $CKPT_GRAPH2D" >&2; exit 1; }
        python -u "$REPO/realcause_eval/graph2d.py" \
            --dataset "$DATASET" --outdir "$OUT" \
            --ckpt "$CKPT_GRAPH2D" --repo "$REPO" \
            --causalpfn "$CAUSALPFN" --uwyk "$UWYK_REPRO" \
            --anc-mode v3b_only --density-primary-mode v3b
        ;;
    uwyk1d)
        [ -d "$UWYK_REPRO" ] || { echo "FATAL: UWYK_REPRO not found: $UWYK_REPRO" >&2; exit 1; }
        python -u "$REPO/realcause_eval/uwyk1d.py" \
            --dataset "$DATASET" --outdir "$OUT" \
            --uwyk-repro "$UWYK_REPRO" --repo "$REPO" \
            --causalpfn "$CAUSALPFN" --anc-mode v3b_only
        ;;
esac

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done  METHOD=$METHOD  DATASET=$DATASET"
