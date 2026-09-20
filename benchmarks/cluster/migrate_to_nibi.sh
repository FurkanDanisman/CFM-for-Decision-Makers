#!/bin/bash
# Bulk-transfer the benchmark kit from one cluster to another.
#
# Run this ON THE SOURCE cluster (killarney), pushing to the destination.
#
#   bash R-PFN/benchmarks/cluster/migrate_to_nibi.sh --size      # what would move
#   bash R-PFN/benchmarks/cluster/migrate_to_nibi.sh --dry-run
#   bash R-PFN/benchmarks/cluster/migrate_to_nibi.sh --go
#   bash R-PFN/benchmarks/cluster/migrate_to_nibi.sh --verify    # after transfer
#
# DEST_HOST / DEST_PATH override the target.
#
# Two things make a naive `rsync -a` wrong here:
#
#  1. venv/ MUST NOT be copied. It is built against this cluster's modules,
#     compilers and CPU; a copied venv fails at import in ways that look like code
#     bugs. It is excluded and must be rebuilt on the destination.
#
#  2. Two checkpoints are SYMLINKS into other trees --
#       cpfn_v0_original.pt            -> warmstart/causalpfn_v0.pt
#       uwyk_USED_IN_RESULTS_best_model.pt -> external/uwyk_reproduce/.../best_model.pt
#     and they use ABSOLUTE targets. Plain rsync copies the link, not the bytes, so
#     they land dangling unless the destination happens to have the identical
#     absolute path. Checkpoints are therefore transferred with -L (follow links)
#     so real files arrive.

set -uo pipefail
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
KIT="${KIT:-$(cd "$_SELF_DIR/../../.." && pwd)}"
DEST_HOST="${DEST_HOST:-nibi.alliancecan.ca}"
DEST_PATH="${DEST_PATH:-/scratch/$USER/rpfn_bench_kit}"
MODE="${1:---size}"

# Excluded from the bulk pass. Everything here is either rebuildable on the
# destination or pure noise, and all of it is large.
EXCLUDES=(
  --exclude 'venv/'                 # rebuild: built against this cluster
  --exclude '**/__pycache__/'
  --exclude '**/*.pyc'
  --exclude 'logs_*/'               # job logs; regenerate
  --exclude 'slurm-*.out'
  --exclude '**/.ipynb_checkpoints/'
  --exclude 'wandb/'
)

human() { awk -v b="$1" 'BEGIN{u="B";s=b;if(s>1024){s/=1024;u="KB"}if(s>1024){s/=1024;u="MB"}if(s>1024){s/=1024;u="GB"}printf "%.1f %s", s, u}'; }

case "$MODE" in
--size)
    echo "source: $KIT"
    echo "dest:   $DEST_HOST:$DEST_PATH"
    echo
    printf '%-42s %10s\n' COMPONENT SIZE
    printf '%.0s-' {1..54}; echo
    for d in R-PFN external case_study_data warmstart \
             R-PFN/Required_checkpoints final_checkpoints; do
        [ -e "$KIT/$d" ] || continue
        printf '%-42s %10s\n' "$d" "$(du -sh "$KIT/$d" 2>/dev/null | cut -f1)"
    done
    echo
    echo "dump / result trees on scratch (these are the expensive artefacts):"
    for d in "$SCRATCH"/rc_dens_uni "$SCRATCH"/cs_dvar_dens "$SCRATCH"/rc_dens_eta0 \
             "$SCRATCH"/cs_dvar_eta0 "$SCRATCH"/dumps_all "$SCRATCH"/perreal \
             "$SCRATCH"/cmech_data_v2; do
        [ -e "$d" ] || continue
        printf '  %-40s %10s\n' "$(basename "$d")" "$(du -sh "$d" 2>/dev/null | cut -f1)"
    done
    echo
    echo "excluded from transfer: venv (rebuild), logs_*, __pycache__, wandb"
    echo
    echo "NOTE: cmech data is cheap to REGENERATE (~40 min, CPU only) -- transferring"
    echo "it is usually not worth the bytes. The dump trees are not: they need GPUs."
    ;;
--dry-run|--go)
    DRY=""; [ "$MODE" = "--dry-run" ] && DRY="--dry-run"
    echo "### 1/3  kit (code, externals, case-study data) -- links preserved"
    rsync -aHP $DRY "${EXCLUDES[@]}" \
        "$KIT/" "$DEST_HOST:$DEST_PATH/" || exit 1
    echo
    echo "### 2/3  checkpoints -- with -L so symlinked weights arrive as real files"
    for d in R-PFN/Required_checkpoints warmstart; do
        [ -e "$KIT/$d" ] || continue
        rsync -aLP $DRY "$KIT/$d/" "$DEST_HOST:$DEST_PATH/$d/" || exit 1
    done
    echo
    echo "### 3/3  dump + result trees on scratch"
    for d in rc_dens_uni cs_dvar_dens rc_dens_eta0 cs_dvar_eta0 dumps_all perreal; do
        [ -e "$SCRATCH/$d" ] || continue
        echo "  -> $d"
        rsync -aHP $DRY "$SCRATCH/$d/" "$DEST_HOST:/scratch/$USER/$d/" || exit 1
    done
    echo
    if [ "$MODE" = "--go" ]; then
        echo "Transfer done. On the destination, IN THIS ORDER:"
        echo "  1. module load python/3.11  (or the destination's equivalent)"
        echo "  2. python -m venv $DEST_PATH/venv && source $DEST_PATH/venv/bin/activate"
        echo "  3. pip install -r <the requirements you used here>"
        echo "  4. bash R-PFN/benchmarks/cluster/migrate_to_nibi.sh --verify"
    fi
    ;;
--verify)
    # Checksums, not sizes: a truncated transfer often keeps a plausible size.
    echo "checkpoint sha256 on THIS cluster -- run the same on the destination"
    echo "and diff the two outputs:"
    echo
    for f in "$KIT"/R-PFN/Required_checkpoints/*.pt; do
        [ -e "$f" ] || continue
        printf '%s  %s\n' "$(sha256sum "$f" 2>/dev/null | cut -c1-16)" "$(basename "$f")"
    done
    echo
    echo "dump counts:"
    for d in rc_dens_uni cs_dvar_dens rc_dens_eta0 cs_dvar_eta0 dumps_all perreal; do
        [ -e "$SCRATCH/$d" ] || continue
        printf '  %-16s %6s npz\n' "$d" "$(find "$SCRATCH/$d" -name '*.npz' | wc -l | tr -d ' ')"
    done
    ;;
*)
    echo "usage: $0 [--size|--dry-run|--go|--verify]" >&2; exit 1 ;;
esac
