#!/bin/bash
# Fixed-SCM experiment, 300k steps: the four runs of submit.sh (same setups, same everything) trained 30x longer,
# fresh, with the cosine schedule over 300,000 steps. Do-PFN 1D (100 bins), four runs. The in-prior "same" SCM (seed 20261003)
# enters training as (training_dopfn_repro/fixed_scm.py):
#   natural  normal training (the SCM is never forced)
#   setup1   each step, 10% chance to use this SCM instead of the prior's SCM, 4 datasets
#   setup2   exactly one step (fixed random position) uses this SCM with 1000 datasets (250 x 4, one update)
#   setup3   each step, 10% chance to use this SCM, 1000 datasets
# Everything else as the 262k 1D run: same prior stream (stream seed 1,000,000), backbone init, optimizer,
# cosine schedule over 300k steps. One L40S job per run, 24 h (~0.22-0.28 s/step -> ~18-23 h), --requeue: on the
# time limit the job requeues itself and train.py resumes from latest.pt (written every 5,000 steps, with step_<N>.pt).
#   cd $KIT && git -C R-PFN pull --rebase && bash R-PFN/revision/experiments/fixed_scm/submit_300k.sh
set -euo pipefail
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; cd "$KIT"
ROOT="$KIT/checkpoints_fixed_scm_300k"
DATA="${E2_SAME:-$SCRATCH/e2/same/data}/dopfn/shard0/complexmech/5node/path_TY/hide_0.0/r000.npz"
INIT="$KIT/checkpoints_dopfn_repro_262k/backbone_init_s0.pt"
[ -f "$DATA" ] || { echo "FATAL: no in-prior dataset $DATA (set E2_SAME)"; exit 1; }
[ -f "$INIT" ] || { echo "FATAL: no backbone init $INIT"; exit 1; }
mkdir -p "$ROOT"
for spec in "natural|natural|1" "setup1|p10|1" "setup2|once|250" "setup3|p10|250"; do
  IFS='|' read -r RUN MODE CH <<< "$spec"
  if squeue -u "$USER" -h -o %j | grep -qx "fscm300k-$RUN"; then echo "$RUN: already queued, skipped"; continue; fi
  STEPS=300000 OUT="$ROOT/$RUN" \
  EXTRA="--num-buckets 100 --backbone-init $INIT --ckpt-every 5000 --fixed-scm $MODE --fixed-chunks $CH --fixed-scm-data $DATA" \
    sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --time=24:00:00 --job-name="fscm300k-$RUN" \
    "$REPO/training_dopfn_repro/cluster/submit_train_dopfn_1d.sbatch"
done
