#!/bin/bash
# Implied (predicted) rho of the three 2D heads on the benchmarks, from the dumps behind the paper's tables
# (roots as in benchmarks/final_table.py ROOTS and the main hist scoring) and from the controlled-rho dumps,
# where the true rho of every dataset is known. ONE CPU job; pushes revision/results/d262t1/rho_pred_bench.{csv,md}.
#   cd $KIT && git -C R-PFN pull --rebase && bash R-PFN/revision/experiments/rho_pred/2_implied_rho_bench.sh
# The first log lines list every root with its file count, so a root that holds nothing is visible.
set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
SELF="$KIT/R-PFN/revision/experiments/rho_pred/2_implied_rho_bench.sh"
if [ "${1:-}" != run ]; then
  cd "$KIT" && mkdir -p logs_rho_pred
  sbatch --account=aip-rgrosse --time="${RP_TIME:-3:00:00}" --cpus-per-task=2 --mem=32G \
    --job-name=rho-implied -o logs_rho_pred/implied_%j.out --wrap "KIT=$KIT bash $SELF run"
  exit $?
fi
REPO="$KIT/R-PFN"; cd "$KIT"; source venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
S=$SCRATCH; C=$S/rho_ctrl/dumps
O="$REPO/revision/results/d262t1"; mkdir -p "$O"
python -u "$REPO/revision/experiments/rho_pred/implied_rho_bench.py" "$O/rho_pred_bench" \
  "Do-PFN 2D|RealCause|$S/dumps_all/dopfn_repro_joint2d_262k_t1/rc|" \
  "Do-PFN 2D|CaseStudy|$S/dumps_all/dopfn_repro_joint2d_262k_t1/cs|" \
  "Do-PFN 2D|ComplexMech|$S/cmech_dumps/dopfn_repro_joint2d_262k_t1|" \
  "UWYK 2D No-Anc|RealCause|$S/rc_dens_uni/graph2d|" \
  "UWYK 2D No-Anc|CaseStudy|$S/cs_dvar_dens|" \
  "UWYK 2D No-Anc|ComplexMech|$S/cmech_dumps/graph2d|" \
  "CausalPFN-C 2D|RealCause|$S/rc_dens_eta0|" \
  "CausalPFN-C 2D|CaseStudy|$S/cs_dvar_eta0|" \
  "CausalPFN-C 2D|ComplexMech|$S/cmech_dumps/cpfn2d_eta0|" \
  "Do-PFN 2D|rho_ctrl_cs|$C/dopfn2d/cs|" "Do-PFN 2D|rho_ctrl_cm|$C/dopfn2d/cm|" \
  "UWYK 2D No-Anc|rho_ctrl_cs|$C/graph2d/cs|" "UWYK 2D No-Anc|rho_ctrl_cm|$C/graph2d/cm|" \
  "UWYK 2D Anc|rho_ctrl_cs|$C/graph2d/cs|v3a" "UWYK 2D Anc|rho_ctrl_cm|$C/graph2d/cm|v3a" \
  "CausalPFN-C 2D|rho_ctrl_cs|$C/cpfn2d/cs|" "CausalPFN-C 2D|rho_ctrl_cm|$C/cpfn2d/cm|" \
  || { echo "implied_rho_bench.py failed; nothing pushed"; exit 1; }
cd "$REPO" && git add revision/results/d262t1/rho_pred_bench.csv revision/results/d262t1/rho_pred_bench.md \
  && git commit -q -m "d262t1: implied rho of the 2D heads on the benchmarks" && git pull -q --rebase && git push -q && echo pushed
