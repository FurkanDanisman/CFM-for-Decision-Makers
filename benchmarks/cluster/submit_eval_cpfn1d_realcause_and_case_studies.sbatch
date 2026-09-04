#!/bin/bash
#SBATCH --job-name=cpfn1d-rc+cs
#SBATCH --output=logs_cpfn1d_rc_cs/eval_%A_%a.out
#SBATCH --error=logs_cpfn1d_rc_cs/eval_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-10

# 11-task array — one per dataset.
#   0..4 : RealCause  (IHDP, ACIC, CPS, PSID, PSID_bal)
#   5..10: Case study (Observed_Confounder, Observed_Mediator,
#          Observed_Mediator_and_Confounder, Unobserved_Confounder,
#          Frontdoor_Criterion, Backdoor_Criterion) — uses the ORIGINAL
#          DoPFN pkls at $DOPFN_DATA_ROOT/prior_sampling.
#
# Same Python entry point for both (eval_causalpfn_v0_realcause.py); the
# --dataset name routes the loader.  Point-estimate CATE only (no density).
#
# Required env vars:
#   OUT_ROOT   per-dataset outputs land at OUT_ROOT/{realcause|case_studies}/<DATASET>
#
# Optional (defaults to the final 50k cpfn-orig-headrand ckpt):
#   CKPT       path to the 1D CausalPFN ckpt
#              default: /scratch/furkanbd/rpfn_bench_kit/causalpfn_j1024_headrand_A1_h100/
#                       step_checkpoints/run/step_0050000.pt
#
# Optional env vars:
#   STD_MODE           per_arm | pooled | log | log_per_arm | log_winsor  (default per_arm)
#   DOPFN_DATA_ROOT    parent of prior_sampling/  (default = deploy_root/external/dopfn/data)
#   DOPFN_ROOT         DoPFN repo root — needed to unpickle case-study .pkls
#                      (default = derived from DOPFN_DATA_ROOT, stripping trailing /data)
#   SKIP_EM            (case studies) 1 = raw CATE only, no EM (default 1)

set -euo pipefail
mkdir -p logs_cpfn1d_rc_cs

# ── deploy-root discovery (venv, repo) ───────────────────────────────────────
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

# ── required knobs ───────────────────────────────────────────────────────────
export CKPT="${CKPT:-/scratch/furkanbd/rpfn_bench_kit/causalpfn_j1024_headrand_A1_h100/step_checkpoints/run/step_0050000.pt}"
export OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
export STD_MODE="${STD_MODE:-per_arm}"

# ── case-study env: point at the ORIGINAL DoPFN pkls, not any regen ──────────
# DOPFN_DATA_ROOT must be the DIRECT parent of the case-study subdirs
# (Observed_Confounder/, Observed_Mediator/, ...). That is prior_sampling/.
export DOPFN_DATA_ROOT="${DOPFN_DATA_ROOT:-$DEPLOY_ROOT/external/dopfn/data/prior_sampling}"
# DOPFN_ROOT points at the dopfn repo root so pickle can import
# `datasets.InterventionalDataset`.
export DOPFN_ROOT="${DOPFN_ROOT:-${DOPFN_DATA_ROOT%/data/prior_sampling}}"
export SKIP_EM="${SKIP_EM:-1}"

[ -f "$CKPT" ] || { echo "FATAL: ckpt not found: $CKPT" >&2; exit 1; }

# ── task dispatch ────────────────────────────────────────────────────────────
case "$SLURM_ARRAY_TASK_ID" in
    # RealCause
    0) DATASET=IHDP;                                TYPE=realcause  ;;
    1) DATASET=ACIC;                                TYPE=realcause  ;;
    2) DATASET=CPS;                                 TYPE=realcause  ;;
    3) DATASET=PSID;                                TYPE=realcause  ;;
    4) DATASET=PSID_bal;                            TYPE=realcause  ;;
    # Case studies (use ORIGINAL DoPFN pkls under $DOPFN_DATA_ROOT/prior_sampling)
    5) DATASET=Observed_Confounder;                 TYPE=case_studies ;;
    6) DATASET=Observed_Mediator;                   TYPE=case_studies ;;
    7) DATASET=Observed_Mediator_and_Confounder;    TYPE=case_studies ;;
    8) DATASET=Unobserved_Confounder;               TYPE=case_studies ;;
    9) DATASET=Frontdoor_Criterion;                 TYPE=case_studies ;;
   10) DATASET=Backdoor_Criterion;                  TYPE=case_studies ;;
    *) echo "invalid task id: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
export DATASET
export OUT="$OUT_ROOT/$TYPE/$DATASET"
mkdir -p "$OUT"

# ── env / activation ─────────────────────────────────────────────────────────
source "$DEPLOY_ROOT/venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] cpfn1d  TYPE=$TYPE  DATASET=$DATASET"
echo "  CKPT=$CKPT"
echo "  OUT=$OUT   STD_MODE=$STD_MODE"
if [ "$TYPE" = "case_studies" ]; then
    echo "  DOPFN_DATA_ROOT=$DOPFN_DATA_ROOT   DOPFN_ROOT=$DOPFN_ROOT   SKIP_EM=$SKIP_EM"
fi

python -u "$REPO/benchmarks/eval_causalpfn2d/eval_causalpfn_v0_realcause.py" --dataset "$DATASET"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
