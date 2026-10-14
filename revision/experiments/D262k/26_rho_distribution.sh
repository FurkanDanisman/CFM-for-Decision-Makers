#!/bin/bash
# Stage 26: per-dataset cross-world correlation rho, ONE estimator (Table 9's ols-on-X) for every source
#   ComplexMech ($SCRATCH/cmech_data_v2) + 1,000 worlds of each 2D training prior (Do-PFN joint, UWYK
#   graph2d, CausalPFN eta1:=eta0) -> revision/results/d262t1/rho_{distribution.csv,distribution.md,per_dataset.csv}
# ONE CPU job (benchmarks/rho/rho_distribution.py); the three prior samplers run side by side, 2 threads each.
#   cd $KIT && git -C R-PFN pull --rebase && bash R-PFN/revision/experiments/D262k/26_rho_distribution.sh
# The same file is the job body: sbatch --wrap runs "bash <this file> run" (never `source` under sh).
# Samplers resume from $WORK/<source>.csv, so a timed-out job is simply resubmitted.
set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
SELF="$KIT/R-PFN/revision/experiments/D262k/26_rho_distribution.sh"

if [ "${1:-}" != run ]; then
  cd "$KIT" && mkdir -p logs_rho
  sbatch --account=aip-rgrosse --time="${RHO_TIME:-3:00:00}" --cpus-per-task=8 --mem=32G \
    --job-name=rhodist -o logs_rho/rhodist_%j.out --wrap "KIT=$KIT bash $SELF run"
  exit $?
fi

# ---- job body ----
REPO="$KIT/R-PFN"; cd "$KIT"; source venv/bin/activate
export UWYK="$KIT/external/uwyk_reproduce" CAUSALPFN="$KIT/external/causalpfn" DOPFN_SRC="$KIT/external/dopfn"
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY="$REPO/benchmarks/rho/rho_distribution.py"
N="${RHO_N:-1000}"
WORK="$SCRATCH/rho_distribution_work"; mkdir -p "$WORK"
O="$REPO/revision/results/d262t1"; mkdir -p "$O"

# PYTHONPATHs exactly as revision/experiments/E2_in_prior/gen_e2.sbatch sets them per prior.
python -u "$PY" sample --source dopfn  --n "$N" --work "$WORK" > "$WORK/dopfn.log" 2>&1 &
P1=$!
PYTHONPATH="$REPO:$UWYK/src" python -u "$PY" sample --source uwyk2d --n "$N" --work "$WORK" \
  > "$WORK/uwyk2d.log" 2>&1 &
P2=$!
PYTHONPATH="$CAUSALPFN/src:$REPO/benchmarks/uwyk_table1/shims" python -u "$PY" sample --source cpfn2d \
  --n "$N" --work "$WORK" > "$WORK/cpfn2d.log" 2>&1 &
P3=$!
python -u "$PY" cmech --data "$SCRATCH/cmech_data_v2" --work "$WORK" || echo "cmech exited non-zero"
for p in "$P1:dopfn" "$P2:uwyk2d" "$P3:cpfn2d"; do
  wait "${p%%:*}" || echo "${p#*:} sampler exited non-zero"
  echo "==== ${p#*:} (tail of log)"; tail -n 5 "$WORK/${p#*:}.log"
done

python -u "$PY" summarize --work "$WORK" --out-dir "$O" --n "$N" || echo "summarize exited non-zero"
cd "$REPO" && git add revision/results/d262t1/rho_distribution.* revision/results/d262t1/rho_per_dataset.csv \
  && git commit -q -m "d262t1: per-dataset cross-world rho (ols-on-X) for ComplexMech and the three 2D priors" \
  && git pull -q --rebase && git push -q && echo pushed
