# Do-PFN 2D trained to the published 1D budget (262,144 steps), evaluated as an ADDED model.
# Every output goes under its own name; nothing of the 150k model is touched.
NAME=dopfn_repro_joint2d_262k
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
REPO="$KIT/R-PFN"
CKPT="$REPO/Required_checkpoints/dopfn_repro_joint2d_step262144.pt"
RC_OUT="$SCRATCH/dumps_all/$NAME/rc"           # same layout as dumps_all/dopfn_repro_joint2d/rc
CS_OUT="$SCRATCH/dumps_all/$NAME/cs"
CM_OUT="$SCRATCH/cmech_dumps/$NAME"            # beside cmech_dumps/dopfn_repro_joint2d
HIST="$SCRATCH/hist"                           # scores land in $HIST/perreal{,_e1}/$NAME
ACCOUNT="${ACCOUNT:-aip-rgrosse}"
GRES="${GRES:-gpu:l40s:1}"
TIME="${TIME:-3:00:00}"
