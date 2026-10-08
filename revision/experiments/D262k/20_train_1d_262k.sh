#!/bin/bash
# Stage 20: retrain the two Do-PFN 1D ablation models at the published 1D budget (262,144 steps), as ADDED
# models next to the 150k ones:  matched resolution (dopfn_1d, 10 bins) and matched label (dopfn_1d_botharms).
# Settings are read from each 150k checkpoint's own provenance; only --steps changes. One 24 h L40S job each,
# --requeue: train.py resumes from <OUT>/<variant>/latest.pt.
#   cd $KIT && git -C R-PFN pull --rebase && bash R-PFN/revision/experiments/D262k/20_train_1d_262k.sh
set -euo pipefail
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"; cd "$KIT"; source venv/bin/activate
OUT="$KIT/checkpoints_dopfn_repro_262k"            # joint_2d of the 262k 2D already lives here
J10="$REPO/Required_checkpoints/dopfn_repro_1d_J10_step150000.pt"
BOTH="$KIT/from_aurora/dopfn_1d_botharms_step150000.pt"
for spec in "dopfn_1d|$J10|submit_train_dopfn_1d.sbatch" "dopfn_1d_botharms|$BOTH|submit_train_dopfn_1d_botharms.sbatch"; do
  IFS='|' read -r VAR CK SB <<< "$spec"
  [ -f "$CK" ] || { echo "FATAL: no $CK"; exit 1; }
  [ -e "$OUT/$VAR" ] && { echo "FATAL: $OUT/$VAR exists -- refusing to write into it"; exit 1; }
  read -r PV NB STEPS SEED BS SL LR AMP <<< "$(python - "$CK" <<'PY'
import sys, torch
p = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["provenance"]
print(p["variant"], p["spec"]["num_buckets"], p["steps"], p["stream_seed"],
      p["prior_cfg"]["batch_size"], p["prior_cfg"]["seq_len"], p["optimizer"]["lr"], p["amp"])
PY
)"
  echo "$VAR: 150k checkpoint says variant=$PV num_buckets=$NB steps=$STEPS stream_seed=$SEED batch=$BS seq_len=$SL lr=$LR amp=$AMP"
  [ "$PV" = "$VAR" ] || { echo "FATAL: $CK is variant $PV, expected $VAR"; exit 1; }
  [ "$SEED" = 1000000 ] || { echo "FATAL: unexpected stream_seed $SEED"; exit 1; }
  STEPS=262144 OUT="$OUT" EXTRA="--num-buckets $NB --batch-size $BS --seq-len $SL --lr $LR --amp $AMP" \
    sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --time=24:00:00 --job-name="t262-$VAR" \
    "$REPO/training_dopfn_repro/cluster/$SB"
done
