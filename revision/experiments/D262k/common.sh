# Do-PFN 2D trained to the published 1D budget (262,144 steps), evaluated as an ADDED model.
# Every output goes under its own name; nothing of the 150k model is touched.
NAME="${D262_NAME:-dopfn_repro_joint2d_262k}"
# The _t1 model is the same checkpoint with Do-PFN's inference temperature OFF (softmax_temperature=0;
# the released config divides every output by 0.8). The name and the switch must go together.
#   export D262_NAME=dopfn_repro_joint2d_262k_t1 DOPFN_SOFTMAX_TEMP=0
case "$NAME" in
  *_t1) [ "${DOPFN_SOFTMAX_TEMP:-}" = 0 ] || { echo "FATAL: $NAME needs DOPFN_SOFTMAX_TEMP=0"; exit 1; } ;;
  *)    [ -z "${DOPFN_SOFTMAX_TEMP:-}" ] || { echo "FATAL: DOPFN_SOFTMAX_TEMP set for $NAME"; exit 1; } ;;
esac
export DOPFN_SOFTMAX_TEMP
TAG=d262; case "$NAME" in *_t1) TAG=d262t1 ;; esac   # suffix of derived files (csv, results folder)
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
REPO="$KIT/R-PFN"
CKPT="$REPO/Required_checkpoints/dopfn_repro_joint2d_step262144.pt"
RC_OUT="$SCRATCH/dumps_all/$NAME/rc"           # same layout as dumps_all/dopfn_repro_joint2d/rc
CS_OUT="$SCRATCH/dumps_all/$NAME/cs"
CM_OUT="$SCRATCH/cmech_dumps/$NAME"            # beside cmech_dumps/dopfn_repro_joint2d
HIST="$SCRATCH/hist"                           # scores land in $HIST/perreal{,_e1}/$NAME
ACCOUNT="${ACCOUNT:-aip-rgrosse}"
GRES="${GRES:-gpu:l40s:1}"
TIME="${TIME:-3:00:00}"            # ComplexMech
TIME_LONG="${TIME_LONG:-3:00:00}"  # RealCause (5 datasets) and each case-study shift (6 d x 6 cases) in one job
