#!/bin/bash
# Copy the three third-party dependencies into vendor/ so a clone of this
# repo is self-sufficient.
#
# Run ONCE, from a machine that has the upstream checkouts, then commit
# vendor/. Strips .git (we want a snapshot, not their history) and records
# the upstream URL and commit in vendor/VENDORED.md so provenance survives.
#
#   SRC_ROOT=$SCRATCH/rpfn_bench_kit/external bash Reproduce/vendor_externals.sh
#
# Env:
#   SRC_ROOT   where the upstream checkouts live (default $DEPLOY_ROOT/external)
#   VENDOR     destination (default <repo>/vendor)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
SRC_ROOT="${SRC_ROOT:-${DEPLOY_ROOT:-$(cd "$REPO/.." && pwd)}/external}"
VENDOR="${VENDOR:-$REPO/vendor}"

# upstream dir name -> vendored name
declare -A MAP=(
  [causalpfn]=causalpfn
  [dopfn]=dopfn
  [uwyk_reproduce]=uwyk
)

mkdir -p "$VENDOR"
MANIFEST="$VENDOR/VENDORED.md"
{
  echo "# Vendored dependencies"
  echo
  echo "Third-party code copied in so this repository is self-contained."
  echo "Each retains its upstream licence. Do not edit these trees; local"
  echo "modifications belong in \`rpfn_patches/\` and are applied at runtime."
  echo
  echo "| vendored as | upstream | commit | licence |"
  echo "|---|---|---|---|"
} > "$MANIFEST"

for src in "${!MAP[@]}"; do
    dst="${MAP[$src]}"
    s="$SRC_ROOT/$src"
    d="$VENDOR/$dst"
    if [ ! -d "$s" ]; then
        echo "SKIP $src -- not found at $s" >&2
        continue
    fi
    url="(unknown)"; sha="(unknown)"
    if [ -d "$s/.git" ]; then
        url="$(git -C "$s" remote get-url origin 2>/dev/null || echo '(no remote)')"
        sha="$(git -C "$s" rev-parse --short HEAD 2>/dev/null || echo '(detached)')"
    fi
    lic="$(ls "$s"/LICENSE* "$s"/COPYING* 2>/dev/null | head -1)"
    lic="${lic:+$(basename "$lic")}"; lic="${lic:-MISSING}"

    echo "vendoring $src -> vendor/$dst  ($sha)"
    rm -rf "$d"
    mkdir -p "$d"
    # Snapshot without history, caches, or build junk.
    tar -C "$s" \
        --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='.ipynb_checkpoints' --exclude='*.egg-info' \
        -cf - . | tar -C "$d" -xf -
    echo "| \`vendor/$dst\` | $url | $sha | $lic |" >> "$MANIFEST"
done

{
  echo
  echo "## Checkpoints are NOT vendored here"
  echo
  echo "Trained weights live in \`Required_checkpoints/\` (git LFS)."
  echo "DoPFN-native's three artifact files are the exception documented in"
  echo "\`Reproduce/Training/DoPFN_Native/README.md\`."
  echo
  echo "## Regenerating"
  echo
  echo "\`bash Reproduce/vendor_externals.sh\` from a machine with the"
  echo "upstream checkouts. It overwrites, so re-running is safe."
} >> "$MANIFEST"

echo
echo "wrote $MANIFEST"
du -sh "$VENDOR"/* 2>/dev/null
echo
echo "Check for large files before committing:"
echo "  find $VENDOR -type f -size +10M | head"
