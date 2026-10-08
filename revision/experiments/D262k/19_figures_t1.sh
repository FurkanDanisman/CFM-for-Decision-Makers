#!/bin/bash
# Stage 19: the case-study by-d figures (PEHE, eps) and the IHDP density figure with the temperature-off
# Do-PFN 2D; written to revision/results/d262t1/figures and pushed. Only the new model's case-study points
# are computed; every other model's rows come from stage 9's CSV ($SCRATCH/cs_point_by_d_d262.csv).
#   cd $KIT && git -C R-PFN pull --rebase && sbatch --account=aip-rgrosse --time=3:00:00 --cpus-per-task=16 --mem=64G \
#        -o logs_hist/t1_fig_%j.out --wrap "bash R-PFN/revision/experiments/D262k/19_figures_t1.sh"
set -uo pipefail
export D262_NAME=dopfn_repro_joint2d_262k_t1 DOPFN_SOFTMAX_TEMP=0
KIT="${KIT:-$PWD}"; D="$KIT/R-PFN/revision/experiments/D262k"; source "$D/common.sh"; cd "$KIT"; source venv/bin/activate
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
F="$REPO/revision/results/$TAG/figures"; mkdir -p "$F"
OLD="$SCRATCH/cs_point_by_d_d262.csv"; NEWC="$SCRATCH/cs_point_by_d_${TAG}_only.csv"; CSV="$SCRATCH/cs_point_by_d_$TAG.csv"
[ -s "$OLD" ] || { echo "FATAL: no $OLD"; exit 1; }
python -u "$REPO/benchmarks/cs_point_total.py" --by-d --roots joint2d_262k_t1 --csv "$NEWC" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" --out "${NEWC%.csv}.md" || { echo "FAILED cs_point_total"; exit 1; }
python - "$OLD" "$NEWC" "$CSV" <<'PY'
import sys, pandas as pd
old, new = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
m = set(new["model"]); out = pd.concat([old[~old["model"].isin(m)], new], ignore_index=True)
out.to_csv(sys.argv[3], index=False); print("merged:", sorted(m), len(out), "rows")
PY
python -u "$REPO/case_study/cluster/plot_fig3_dotgrid.py" --csv "$CSV" --kind case --mode by-d \
  --metrics pehe eps --out "$F/fig3_cen3" || exit 1
cd "$REPO" && git add "revision/results/$TAG/figures" && git commit -q -m "$TAG: case-study by-d figures" \
  && git pull -q --rebase && git push -q && echo pushed
cd "$KIT" && bash "$D/10_density_fig.sh"
