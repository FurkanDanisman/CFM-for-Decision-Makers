#!/bin/bash
# Stage 23: every paper evaluation of a 262k Do-PFN ablation model, as an ADDED model with Do-PFN's inference temperature OFF. ONE job, sequential
# (body: 23_run_1d_262k.sh):
#   dumps   RealCause (5 datasets), case studies (shifts 0/+2/-2, d = 5..50), ComplexMech all rho (the ablation table is all rho)
#   scoring hist (CDM_INTERVAL=hist), exactly as submit_hist_extra.sbatch scored the 150k J10 model
# These feed tab:rc-ablation-cal, tab:cs-abl-cal and tab:cm-abl-r99-cal. Nothing existing is overwritten.
#   matched resolution (dopfn_repro_1d_J10_262k_t1): bash R-PFN/revision/experiments/D262k/23_eval_1d_262k.sh
#   matched label (dopfn_1d_botharms_262k_t1):     VARIANT=dopfn_1d_botharms bash R-PFN/revision/experiments/D262k/23_eval_1d_262k.sh
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
export KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"; cd "$KIT"
# VARIANT=dopfn_1d (matched resolution, default) | dopfn_1d_botharms (matched label)
VARIANT="${VARIANT:-dopfn_1d}"
case "$VARIANT" in
  dopfn_1d)          export NAME=dopfn_repro_1d_J10_262k_t1 NB=10 ;;
  dopfn_1d_botharms) export NAME=dopfn_1d_botharms_262k_t1 NB=100 ;;
  *) echo "FATAL: VARIANT must be dopfn_1d or dopfn_1d_botharms"; exit 1 ;;
esac
export VARIANT
export CKPT="$KIT/checkpoints_dopfn_repro_262k/$VARIANT/step_262144_final.pt"
export RC="$SCRATCH/dumps_all/$NAME/rc" CS="$SCRATCH/dumps_all/$NAME/cs" CMALL="$SCRATCH/cmech_dumps/$NAME"
[ -f "$CKPT" ] || { echo "FATAL: no $CKPT"; ls "$(dirname "$CKPT")"; exit 1; }
for d in "$RC" "$CS" "$CMALL" "$SCRATCH/hist/perreal/$NAME" "$SCRATCH/hist/perreal_e1/$NAME"; do
  [ -e "$d" ] && { echo "FATAL: $d already exists -- refusing to write into it"; exit 1; }
done
[ -d "$SCRATCH/hist/truth" ] || { echo "FATAL: no $SCRATCH/hist/truth"; exit 1; }
source venv/bin/activate
python - "$CKPT" <<'PY'
import sys, torch
b = torch.load(sys.argv[1], map_location="cpu", weights_only=False); p = b["provenance"]
print("checkpoint:", p["variant"], "num_buckets", p["spec"]["num_buckets"], "step", b.get("step"))
import os
assert (p["variant"] == os.environ["VARIANT"] and p["spec"]["num_buckets"] == int(os.environ["NB"])
        and b.get("step") == 262144), "wrong checkpoint"
PY
mkdir -p logs_hist
sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --time="${TIME:-3:00:00}" --cpus-per-task=16 --mem=96G \
  --job-name=t262-1d-eval -o logs_hist/t262_1d_eval_%j.out --export=ALL \
  --wrap "bash $D/23_run_1d_262k.sh"
echo "submitted $NAME (one job); log: $KIT/logs_hist/t262_1d_eval_<jobid>.out"
