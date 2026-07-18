#!/bin/bash
# Bundle everything needed into a payload directory, ready to rsync to killarney.
#
# Usage:
#   bash benchmarks/cluster/deploy_local.sh /tmp/rpfn_bench_kit

set -euo pipefail

TARGET="${1:-/tmp/rpfn_bench_kit}"
REPO="/Users/furkandanisman/R-PFN"
DOPFN="/tmp/dopfn"
UWYK="/tmp/g4cfm_uwyk"
CAUSALPFN="/tmp/causalpfn_full"

mkdir -p "$TARGET/external" "$TARGET/R-PFN" "$TARGET/logs" "$TARGET/results"

echo "[1/4] Copying R-PFN repo (excluding heavy / non-essential dirs)…"
rsync -a \
    --exclude '.venv/' --exclude '__pycache__/' \
    --exclude '.git/' \
    "$REPO/" "$TARGET/R-PFN/"

echo "[2/4] Copying patched Do-PFN…"
rsync -a --exclude '__pycache__/' --exclude '.git/' "$DOPFN/" "$TARGET/external/dopfn/"

echo "[3/4] Copying patched UWYK (incl. git-lfs checkpoint)…"
rsync -a --exclude '__pycache__/' --exclude '.git/' "$UWYK/" "$TARGET/external/uwyk/"

echo "[4/4] Copying CausalPFN benchmarks…"
rsync -a --exclude '__pycache__/' --exclude '.git/' "$CAUSALPFN/" "$TARGET/external/causalpfn/"

echo
echo "Payload size:"
du -sh "$TARGET"/* | sort -h

echo
echo "Ready. Now rsync to killarney:"
echo "  rsync -avz --progress $TARGET/ <user>@killarney:/path/to/rpfn_bench_kit/"
