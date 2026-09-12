#!/bin/bash
#SBATCH --output=logs_density/%x_%j.out
#SBATCH --error=logs_density/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#
# One (model, context) job: loops the six cases. Driven entirely by env set by
# 04_submit_density.sh. Add --partition/--account on the sbatch command line, or
# via SBATCH_PARTITION / SBATCH_ACCOUNT, so this file stays cluster-agnostic.
set -euo pipefail
mkdir -p logs_density

: "${REPO:?}" "${HERE:?}" "${OUT:?}" "${KIND:?}" "${CASE_STUDY_DATA_ROOT:?}" "${CASE_STUDY_N:?}"
source "${VENV:-$DEPLOY_ROOT/venv}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# sitecustomize denylist: causalpfn/__init__.py imports causal_estimator, which
# pulls faiss / transformers / wandb at module scope. The evals only need
# causalpfn.benchmarks dataset loaders, so the shim stubs those top-level
# modules. Without it every causalpfn-backed eval dies at import with
# ModuleNotFoundError: No module named 'faiss'. Same line the working
# dispatcher (benchmarks/cluster/submit_eval_scm_case_studies.sbatch) uses.
export PYTHONPATH="$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
[ -f "$REPO/benchmarks/uwyk_table1/shims/sitecustomize.py" ] || {
    echo "FATAL: shim missing at $REPO/benchmarks/uwyk_table1/shims" >&2; exit 1; }
IFS=',' read -ra CASE_LIST <<< "${CASES:?}"

echo "[density] kind=$KIND N=$CASE_STUDY_N out=$OUT cases=${#CASE_LIST[@]}"
mkdir -p "$OUT"

for CASE in "${CASE_LIST[@]}"; do
  CELL="$OUT/$CASE"; mkdir -p "$CELL"
  echo "--- $CASE ---"
  case "$KIND" in
    cpfn1d)
      # LOCAL copy: saves per-arm (shift, scale) so STD_MODE=per_arm dumps.
      DATASET="$CASE" OUT="$CELL" DENSITY_DUMP=1 \
        python -u "$HERE/eval_cpfn1d_perarm.py" --dataset "$CASE"
      python -u "$HERE/run_density_scm.py" --dump-root "$CELL" \
        --dataset "$CASE" --data-root "$CASE_STUDY_DATA_ROOT" \
        --n "$CASE_STUDY_N" --out "$CELL/metrics.json"
      ;;
    cpfn2d)
      DATASET="$CASE" OUT="$CELL" DENSITY_DUMP=1 \
        python -u "$REPO/benchmarks/eval_causalpfn2d/eval_cpfn2d_realcause.py" \
        --dataset "$CASE"
      python -u "$HERE/run_density_scm.py" --dump-root "$CELL" \
        --dataset "$CASE" --data-root "$CASE_STUDY_DATA_ROOT" \
        --n "$CASE_STUDY_N" --out "$CELL/metrics.json"
      ;;
    tauC)
      # The copied runner puts $HERE first on sys.path, so the SCM-aware
      # density_truth.py in this folder shadows the IHDP/ACIC-only original.
      DATASET="$CASE" OUT="$CELL" REAL_START="${REAL_START:-0}" \
        REAL_END="${REAL_END:-}" \
        python -u "$HERE/eval_density_tauC.py"
      ;;
    *) echo "FATAL: unknown KIND=$KIND" >&2; exit 2 ;;
  esac
done
echo "[density] done -> $OUT"
