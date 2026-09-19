#!/bin/bash
# Locate the checkpoints for the models that still need dumps.
#
# /scratch/furkanbd/final_checkpoints exists but is empty, so the six .pt files
# are somewhere else. One of them is a known-good anchor: cpfn2d_j32_eta0_y01
# already produced dumps on all three benchmarks, so its weights ARE on this
# filesystem -- finding it reveals the layout the others follow.
#
# Trained-on-cluster models live under <run_dir>/step_checkpoints/run/step_*.pt
# (that is where CKPT_CPFN1D and CKPT_CPFN2D already point), so run dirs are
# searched as well as flat checkpoint directories.
#
#   bash R-PFN/benchmarks/cluster/find_ckpts.sh

set -uo pipefail
SC="${SCRATCH:-/scratch/furkanbd}"
KIT="${KIT:-$SC/rpfn_bench_kit}"

echo "##### 1. flat checkpoint dirs (is final_checkpoints really empty?)"
for d in "$SC/final_checkpoints" "$KIT/final_checkpoints" "$KIT/R-PFN/final_checkpoints" \
         "$HOME/final_checkpoints" "$KIT/R-PFN/Required_checkpoints"; do
    [ -d "$d" ] || continue
    n=$(find "$d" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')
    echo "  $d  ->  $n .pt files"
    find "$d" -maxdepth 1 -name '*.pt' -printf '      %10s  %p\n' 2>/dev/null \
        || find "$d" -maxdepth 1 -name '*.pt' -exec ls -l {} \; 2>/dev/null | sed 's/^/      /'
done

echo
echo "##### 2. by distinctive name, anywhere under scratch and home"
for pat in '*eta0*' '*botharms*' '*repro*1d*' '*repro*joint*' '*j32*'; do
    echo "  --- $pat"
    find "$SC" "$HOME" -name "$pat" -name '*.pt' 2>/dev/null | head -12 | sed 's/^/      /'
done

echo
echo "##### 3. training run dirs (where CKPT_CPFN1D/CKPT_CPFN2D already point)"
for d in "$KIT"/*/step_checkpoints/run; do
    [ -d "$d" ] || continue
    run="${d%/step_checkpoints/run}"
    echo "  $(basename "$run")"
    ls -1 "$d"/*.pt 2>/dev/null | tail -3 | sed 's/^/      /'
done

echo
echo "##### 4. any .pt over 20MB modified in the last 60 days, outside venv"
find "$SC" "$HOME" -name '*.pt' -size +20M -mtime -60 \
     -not -path '*/venv/*' -not -path '*/site-packages/*' 2>/dev/null \
  | head -40 | sed 's/^/      /'
