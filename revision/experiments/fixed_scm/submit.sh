#!/bin/bash
# Fixed-SCM experiment: Do-PFN 1D (100 bins), 10,000 steps, four runs. The in-prior "same" SCM (seed 20261003)
# enters training as (training_dopfn_repro/fixed_scm.py):
#   natural  normal training (the SCM is never forced)
#   setup1   each step, 10% chance to use this SCM instead of the prior's SCM, 4 datasets
#   setup2   exactly one step (fixed random position) uses this SCM with 1000 datasets (250 x 4, one update)
#   setup3   each step, 10% chance to use this SCM, 1000 datasets
# Everything else as the 262k 1D run: same prior stream (stream seed 1,000,000), backbone init, optimizer,
# cosine schedule over 10k steps. One L40S job per run, 12 h, --requeue: on the time limit the job requeues
# itself and train.py resumes from latest.pt (written every 250 steps).
#   cd $KIT && git -C R-PFN pull --rebase && bash R-PFN/revision/experiments/fixed_scm/submit.sh
set -euo pipefail
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; cd "$KIT"
ROOT="$KIT/checkpoints_fixed_scm"
DATA="${E2_SAME:-$SCRATCH/e2/same/data}/dopfn/shard0/complexmech/5node/path_TY/hide_0.0/r000.npz"
INIT="$KIT/checkpoints_dopfn_repro_262k/backbone_init_s0.pt"
[ -f "$DATA" ] || { echo "FATAL: no in-prior dataset $DATA (set E2_SAME)"; exit 1; }
[ -f "$INIT" ] || { echo "FATAL: no backbone init $INIT"; exit 1; }
mkdir -p "$ROOT"
for spec in "natural|natural|1" "setup1|p10|1" "setup2|once|250" "setup3|p10|250"; do
  IFS='|' read -r RUN MODE CH <<< "$spec"
  if squeue -u "$USER" -h -o %j | grep -qx "fscm-$RUN"; then echo "$RUN: already queued, skipped"; continue; fi
  STEPS=10000 OUT="$ROOT/$RUN" \
  EXTRA="--num-buckets 100 --backbone-init $INIT --ckpt-every 250 --fixed-scm $MODE --fixed-chunks $CH --fixed-scm-data $DATA" \
    sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --time=12:00:00 --job-name="fscm-$RUN" \
    "$REPO/training_dopfn_repro/cluster/submit_train_dopfn_1d.sbatch"
done
