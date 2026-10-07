#!/bin/bash
# Stage 9: the three paper figures with the 262k Do-PFN 2D.
#   fig3_cen3_pehe_by_d.png, fig3_cen3_eps_by_d.png  (case studies by d; cs_point_total.py -> plot_fig3_dotgrid.py)
#   ihdp_r0_ate_2x4.png                              (IHDP r0 densities; Do-PFN 2D slot -> the 262k dumps)
# Writes only revision/results/d262/figures/ and $SCRATCH/*_d262*; then commits and pushes the figures.
#   cd $KIT && sbatch --account=aip-rgrosse --time=3:00:00 --cpus-per-task=16 --mem=64G \
#        -o logs_hist/d262_fig_%j.out --wrap "bash R-PFN/revision/experiments/D262k/9_figures.sh"
set -uo pipefail
KIT="${KIT:-$PWD}"; source "$KIT/R-PFN/revision/experiments/D262k/common.sh"
source "$KIT/venv/bin/activate"; export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
F="$REPO/revision/results/d262/figures"; mkdir -p "$F"
CSV="$SCRATCH/cs_point_by_d_d262.csv"
python -u "$REPO/benchmarks/cs_point_total.py" --by-d --csv "$CSV" --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --out "$SCRATCH/cs_point_by_d_d262.md" || { echo "FAILED cs_point_total"; exit 1; }
grep -c "dopfn_repro_joint2d_262k" "$CSV" | sed 's/^/rows for the 262k model: /'
python -u "$REPO/case_study/cluster/plot_fig3_dotgrid.py" --csv "$CSV" --kind case --mode by-d \
  --metrics pehe eps --out "$F/fig3_cen3"
# IHDP density figure: a view of the 8-model root with the Do-PFN 2D slot pointing at the 262k dumps
SRC="${IHDP_DENS_ROOT:-$SCRATCH/ihdp_dens_8models}"
if [ -d "$SRC" ]; then
  V="$SCRATCH/ihdp_dens_d262_view"; rm -rf "$V"; mkdir -p "$V"
  for m in "$SRC"/*; do ln -s "$m" "$V/$(basename "$m")"; done
  rm -f "$V/dopfn_repro_joint2d" "$V/dopfn_joint2d" "$V/dopfn_bb"
  ln -s "$RC_OUT/dopfn_native" "$V/dopfn_repro_joint2d"
  python -u "$REPO/benchmarks/plots/plot_ate_density_2x4.py" --root "$V" --dataset IHDP --realization 0 \
    --repo "$REPO" --out "$F/ihdp_r0_ate_2x4.png"
else
  echo "SKIP density figure: no $SRC (set IHDP_DENS_ROOT to the root the paper figure was made from)"
fi
ls -la "$F"
cd "$REPO" && git add revision/results/d262/figures && git commit -q -m "d262: figures with the 262k Do-PFN 2D" \
  && git pull -q --rebase --autostash && git push -q && git log --oneline -1
