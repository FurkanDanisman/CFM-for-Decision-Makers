#!/bin/bash
#SBATCH --job-name=uwyk1d-rc
#SBATCH --output=logs_uwyk1d_realcause/eval_%A_%a.out
#SBATCH --error=logs_uwyk1d_realcause/eval_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-9

# CONTROL run: UWYK's own 1D checkpoint through OUR graph2d RealCause harness.
# Companion to submit_eval_graph2d_realcause.sbatch — identical datasets,
# preprocessing, adjacency matrices (ANC_MODE) and npz schema; the ONLY
# difference is that CATE comes from UWYK's 1D bar-distribution model run
# twice (do(T=1), do(T=0)) instead of from our 2D joint head run once.
#
# 10-task array — 5 datasets × 2 treatment encodings:
#   0-4: T_ENCODING=binary   T ∈ {0,1}, query T_intv ∈ {0,1} — what OUR
#                            harness feeds the joint model. Apples-to-apples.
#   5-9: T_ENCODING=target   T ← mean(Y|T) — what UWYK's own
#                            dofm_full_conditioning.py does. UWYK at its best.
# Within each block: 0 IHDP, 1 ACIC, 2 CPS, 3 PSID, 4 PSID_bal.
#
# Run both: the binary-vs-target gap tells you how much of UWYK's reported
# advantage is carried by the treatment encoding rather than by the graph.
#
# Submit:  sbatch benchmarks/cluster/submit_eval_uwyk1d_realcause.sbatch
# Override the adjacency variant:
#          ANC_MODE=focus4 sbatch benchmarks/cluster/submit_eval_uwyk1d_realcause.sbatch

set -euo pipefail
mkdir -p logs_uwyk1d_realcause

REPO="/home/lukez/projects/aip-rgrosse/lukez/CFM-for-Decision-Makers"

export UWYK="$REPO/g4cfm"
export CAUSALPFN="/home/lukez/projects/aip-rgrosse/lukez/CausalPFN"

# UWYK's full-conditioning checkpoint (the anc/noanc model from their Table 1).
# Required_checkpoints/uwyk_reproduce_best_model.pt is a copy of the same file;
# the config lives only in the g4cfm checkpoint dir, so pair them from there.
UWYK_CKPT_DIR="$UWYK/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"
export CKPT="${CKPT:-$UWYK_CKPT_DIR/best_model.pt}"
export CONFIG="${CONFIG:-$UWYK_CKPT_DIR/best_model_config.yaml}"

# Must match whatever submit_eval_graph2d_realcause.sbatch used for the numbers
# you are comparing against, or the two columns are not measuring the same
# thing. v6a_only → keys pehe_raw_v6a / pehe_raw_noanc.
# (No FIX_DIAG here: that env var no longer exists in the harness — v6a sets
# the real-block diagonal to -1 itself. The FIX_DIAG=1 line still in the
# graph2d sbatch is a no-op.)

# UWYK's config sets max_number_train_samples_per_dataset=1000 and their own
# wrapper truncates to it; feeding CPS/PSID's full 14.5k rows is both far
# outside what their model saw in training and slow. 1000 matches their
# pipeline. Set EVAL_MAX_CONTEXT='' to feed full context instead (parity with
# a joint run that also left it unset).
export EVAL_MAX_CONTEXT="${EVAL_MAX_CONTEXT-1000}"

export OUT_ROOT="${OUT_ROOT:-$REPO/results_uwyk1d_realcause/$SLURM_ARRAY_JOB_ID}"

[ -f "$CKPT" ]   || { echo "FATAL: checkpoint not found at $CKPT" >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "FATAL: config not found at $CONFIG" >&2; exit 1; }

case "$(( SLURM_ARRAY_TASK_ID / 5 ))" in
    0) T_ENCODING=binary ;;
    1) T_ENCODING=target ;;
    *) echo "invalid task id: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
case "$(( SLURM_ARRAY_TASK_ID % 5 ))" in
    0) DATASET=IHDP     ;;
    1) DATASET=ACIC     ;;
    2) DATASET=CPS      ;;
    3) DATASET=PSID     ;;
    4) DATASET=PSID_bal ;;
    *) echo "invalid task id: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
export T_ENCODING DATASET
export OUT="$OUT_ROOT/$T_ENCODING/$DATASET"

source "$REPO/.venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
# shims/ = sitecustomize denylist that keeps causalpfn's __init__ from
# dragging in deps we don't have; same one the graph2d eval uses.
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] uwyk1d RealCause  dataset=$DATASET  " \
     "T_ENCODING=$T_ENCODING  ANC_MODE=$ANC_MODE  OUT=$OUT"
python -u "$REPO/benchmarks/eval_graph2d/eval_uwyk1d_realcause.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
