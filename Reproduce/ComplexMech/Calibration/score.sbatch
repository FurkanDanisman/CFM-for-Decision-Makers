#!/bin/bash
# Thin wrapper -- a real file, NOT a symlink: writing through a symlink edits
# the target, and a relative `source ../../env.sh` resolves differently from
# the target's own directory.
#
#   OUT_ROOT=$SCRATCH/cmech_1d2d sbatch score.sbatch
#   TARGETS=ate OUT_ROOT=... sbatch score.sbatch    # skip the slower CATE pass
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/env.sh"
exec sbatch --export=ALL \
    "$REPO/benchmarks/cluster/submit_cmech_density_score.sbatch" "$@"
