#!/bin/bash
# Resolve cpfn_v0 -- and re-check every checkpoint's TRUE size through symlinks.
#
# cpfn_v0_original.pt reports 58 bytes but reads out a real torch zip header
# naming the member 'causalpfn_v0_tabdpt_late/data.pkl' plus pickle content, far
# more than 58 bytes. That is what a SYMLINK looks like to stat/find: they report
# the link's own size, which is the length of its target path. So the earlier
# "git-lfs pointer" call was a measurement error, and the weights may be fine.
#
# Stage 2 searches by zip member name rather than filename. A torch .pt is a zip
# archive whose member names appear as plain text near the start of the file, so
# grepping for the run name finds the real weights wherever they live and
# whatever they are called.
#
#   bash R-PFN/benchmarks/cluster/find_cpfn_v0.sh

set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
REPO="${REPO:-$KIT/R-PFN}"
CK="${CK:-$REPO/Required_checkpoints}"
SC="${SCRATCH:-/scratch/furkanbd}"

echo "##### 1. every checkpoint: link status and TRUE size (-L follows symlinks)"
printf '  %-46s %-12s %-12s %s\n' FILE LINK_SIZE TARGET_SIZE TARGET
for f in "$CK"/*.pt; do
    [ -e "$f" ] || continue
    lsz=$(stat -c%s  "$f" 2>/dev/null || echo ?)
    tsz=$(stat -Lc%s "$f" 2>/dev/null || echo ?)
    tgt=""
    [ -L "$f" ] && tgt="-> $(readlink -f "$f" 2>/dev/null || readlink "$f")"
    printf '  %-46s %-12s %-12s %s\n' "$(basename "$f")" "$lsz" "$tsz" "$tgt"
done

echo
echo "##### 2. is cpfn_v0 actually loadable?"
V0="$CK/cpfn_v0_original.pt"
if [ -e "$V0" ]; then
    file "$V0" 2>/dev/null | sed 's/^/  /'
    echo "  first zip member:"
    head -c 120 "$V0" 2>/dev/null | strings 2>/dev/null | head -2 | sed 's/^/      /'
    if [ -f "$KIT/venv/bin/activate" ]; then
        source "$KIT/venv/bin/activate"
        python -u "$REPO/benchmarks/inspect_ckpt.py" "$V0" 2>&1 | sed 's/^/  /'
    fi
else
    echo "  $V0 does not exist"
fi

echo
echo "##### 3. if that failed: find the real weights by zip member name"
echo "  (searching .pt files for the run name 'causalpfn_v0')"
for d in "$CK" "$SC" "$HOME"; do
    [ -d "$d" ] || continue
    find "$d" -name '*.pt' -size +1M -not -path '*/venv/*' \
         -not -path '*/site-packages/*' 2>/dev/null \
      | while read -r f; do
            if head -c 4096 "$f" 2>/dev/null | grep -qa "causalpfn_v0"; then
                printf '  MATCH %10s  %s\n' "$(stat -Lc%s "$f")" "$f"
            fi
        done
done
echo
echo "##### 4. directories named after the run"
find "$SC" "$HOME" -maxdepth 4 -type d \( -name '*causalpfn_v0*' -o -name '*tabdpt_late*' \) \
     2>/dev/null | sed 's/^/  /'
