#!/bin/bash
# What do we ACTUALLY have, per model, right now?
#
# Counts dumps on disk for all eleven models rather than trusting any prior
# claim that a model "has results". For RealCause it reports per-dataset counts,
# so a model that is complete on IHDP and empty on PSID_bal cannot read as done.
#
# Expected, when a model is fully dumped:
#   RealCause    5 datasets x ~100 realizations = ~500 npz
#   Case study   per (shift,d,ctx) cell: 6 cases x ~100 = ~600 npz
#
#   bash R-PFN/benchmarks/cluster/inventory_models.sh
#   FULL=1 bash ...    also score each root (slower, but the real test)

set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"
REPO="${REPO:-$KIT/R-PFN}"
SC="${SCRATCH:-/scratch/furkanbd}"
RC_DS="IHDP ACIC CPS PSID PSID_bal"

# label | rc root | cs root | subdir the harness writes under
# Model identity lives in the ROOT for the models that reuse a harness, and in
# the SUBDIR for the ones that have their own. Both are listed explicitly so
# nothing is inferred from a directory name.
ROWS=(
  "dopfn_native|$SC/rc_dens_uni|$SC/cs_dvar_dens|dopfn_native"
  "dopfn_bb|$SC/rc_dens_uni|$SC/cs_dvar_dens|dopfn_bb"
  "uwyk1d|$SC/rc_dens_uni|$SC/cs_dvar_dens|uwyk1d"
  "graph2d|$SC/rc_dens_uni|$SC/cs_dvar_dens|graph2d"
  "cpfn1d_j1024|$SC/rc_dens_uni|$SC/cs_dvar_dens|cpfn1d"
  "cpfn2d_baseline|$SC/rc_dens_uni|$SC/cs_dvar_dens|cpfn2d_pooled"
  "cpfn2d_eta0|$SC/rc_dens_eta0|$SC/cs_dvar_eta0|cpfn2d_pooled"
  "dopfn_repro_1d_J10|$SC/dumps_all/dopfn_repro_1d_J10/rc|$SC/dumps_all/dopfn_repro_1d_J10/cs|dopfn_native"
  "dopfn_repro_1d_J100|$SC/dumps_all/dopfn_repro_1d_J100/rc|$SC/dumps_all/dopfn_repro_1d_J100/cs|dopfn_native"
  "dopfn_repro_joint2d|$SC/dumps_all/dopfn_repro_joint2d/rc|$SC/dumps_all/dopfn_repro_joint2d/cs|dopfn_native"
  "cpfn1d_j32|$SC/dumps_all/cpfn1d_j32/rc|$SC/dumps_all/cpfn1d_j32/cs|cpfn1d"
  "cpfn1d_botharms|$SC/dumps_all/cpfn1d_botharms/rc|$SC/dumps_all/cpfn1d_botharms/cs|cpfn1d"
  "cpfn_v0|$SC/dumps_all/cpfn_v0/rc|$SC/dumps_all/cpfn_v0/cs|cpfn1d"
  "uwyk_bin|$SC/dumps_all/uwyk_bin/rc|$SC/dumps_all/uwyk_bin/cs|uwyk1d"
)
# Smoke roots hold ONE cell per benchmark, not the full grid, so they are
# reported separately rather than mistaken for a completed dump.
SMOKE="$SC/smoke_new"

printf '%-22s %-6s %-6s %-6s %-6s %-9s %-8s %s\n' MODEL IHDP ACIC CPS PSID PSID_bal CS_cells NOTE
printf '%.0s-' {1..92}; echo

for r in "${ROWS[@]}"; do
    IFS='|' read -r name rcroot csroot sub <<<"$r"
    if [ "$rcroot" = "-" ]; then
        printf '%-22s %-6s %-6s %-6s %-6s %-9s %-8s %s\n' "$name" - - - - - - "no checkpoint (lfs pointer)"
        continue
    fi
    counts=(); tot=0
    for ds in $RC_DS; do
        d="$rcroot/$sub/$ds"
        n=$(find "$d" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
        counts+=("$n"); tot=$((tot+n))
    done
    cs_cells=0; cs_npz=0
    if [ -d "$csroot" ]; then
        for cell in "$csroot"/shift*/d*/ctx*/"$sub"; do
            [ -d "$cell" ] || continue
            k=$(find "$cell" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
            [ "$k" -gt 0 ] && { cs_cells=$((cs_cells+1)); cs_npz=$((cs_npz+k)); }
        done
    fi
    note=""
    [ "$tot" = 0 ] && [ "$cs_npz" = 0 ] && note="NO DUMPS"
    if [ "$tot" != 0 ]; then
        for c in "${counts[@]}"; do [ "$c" = 0 ] && note="RC incomplete"; done
    fi
    # A smoke root means one cell was proven, not that the grid is dumped.
    if [ -d "$SMOKE/$name" ]; then
        sn=$(find "$SMOKE/$name" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
        [ "$sn" -gt 0 ] && note="${note:+$note; }smoke only ($sn npz)"
    fi
    printf '%-22s %-6s %-6s %-6s %-6s %-9s %-8s %s\n' \
        "$name" "${counts[0]}" "${counts[1]}" "${counts[2]}" "${counts[3]}" \
        "${counts[4]}" "$cs_cells" "$note"
done

echo
echo "RC columns are npz per dataset (~100 = one full dataset)."
echo "CS_cells counts (shift,d,ctx) cells holding at least one npz; a full"
echo "case-study sweep is 3 shifts x 8 d = 24 cells."
