#!/bin/bash
# Progress per model, in the reporting layout. Safe to run at any moment.
#
#   RealCause    PEHE/ATE (raw) | Cov (raw) | Cov (MALC) | Cov (indep, 2D only)
#   Case study   PEHE/ATE (raw) | Cov (raw) | Cov (MALC)
#
# Counts artefacts on disk, because slurm state is not trustworthy here: the
# case-study sbatch ends each harness call with `|| echo WARN`, so a cell that
# died on a traceback still reports COMPLETED with exit 0.
#
# Denominators:
#   RealCause  5 datasets
#   Case study 3 shifts x 8 d x 6 cases = 144 cells
#
# Point estimates are .md files written beside the dumps; coverage stages are the
# per-realization .npz under $PERREAL, which is what the table is built from.
#
#   bash R-PFN/benchmarks/cluster/progress.sh

set -uo pipefail
SC="${SCRATCH:?SCRATCH must be set}"
PERREAL="${PERREAL:-$SC/perreal}"
RC_N=5; CS_N=144

# label | rc dump root | cs dump root | 2D? | models covered
ROOTS=(
  "orig|$SC/rc_dens_uni|$SC/cs_dvar_dens|yes|dopfn_native,dopfn_bb,uwyk1d(x2),graph2d(x2),cpfn1d,cpfn2d"
  "eta0|$SC/rc_dens_eta0|$SC/cs_dvar_eta0|yes|cpfn2d_eta0"
  "J10|$SC/dumps_all/dopfn_repro_1d_J10/rc|$SC/dumps_all/dopfn_repro_1d_J10/cs|no|dopfn_repro_1d_J10"
  "J100|$SC/dumps_all/dopfn_repro_1d_J100/rc|$SC/dumps_all/dopfn_repro_1d_J100/cs|no|dopfn_repro_1d_J100"
  "joint2d|$SC/dumps_all/dopfn_repro_joint2d/rc|$SC/dumps_all/dopfn_repro_joint2d/cs|yes|dopfn_repro_joint2d"
  "j32|$SC/dumps_all/cpfn1d_j32/rc|$SC/dumps_all/cpfn1d_j32/cs|no|cpfn1d_j32"
  "botharms|$SC/dumps_all/cpfn1d_botharms/rc|$SC/dumps_all/cpfn1d_botharms/cs|no|cpfn1d_botharms"
  "cpfn_v0|$SC/dumps_all/cpfn_v0/rc|$SC/dumps_all/cpfn_v0/cs|no|cpfn_v0"
  "uwyk_bin|$SC/dumps_all/uwyk_bin/rc|$SC/dumps_all/uwyk_bin/cs|no|uwyk_bin"
)

pct() {  # count total -> "n/total (p%)"
    if [ "$2" -le 0 ]; then printf '%-13s' "-"; return; fi
    awk -v a="$1" -v b="$2" 'BEGIN{p=100*a/b; if(p>100)p=100; printf "%3d/%-3d %3.0f%%", a, b, p}'
}
# ComplexMech per-realization files share the "__-" slice suffix with RealCause
# (raw__cmech_n5__-.npz vs raw__IHDP__-.npz), so the RealCause glob matched them and
# reported 6/5. Every RealCause count excludes anything naming a cmech cell.
nf() {
    [ -d "$1" ] || { echo 0; return; }
    find "$1" -name "$2" 2>/dev/null | grep -v '__cmech_n' | wc -l | tr -d ' '
}
nf_cm() {
    [ -d "$1" ] || { echo 0; return; }
    find "$1" -name "$2" 2>/dev/null | wc -l | tr -d ' '
}
# Count the point tables that exist. The stamp is metadata, not a result: files
# written before it existed hold correct numbers, and requiring it reported 1/5 for
# a complete set.
nf_pt() {
    [ -d "$1" ] || { echo 0; return; }
    find "$1" -name 'point_raw_em_*.md' 2>/dev/null | wc -l | tr -d ' '
}

echo "=== RealCause  (denominator: $RC_N datasets)"
printf '%-10s %-14s %-14s %-14s %-14s %-14s %s\n' \
  ROOT PEHE/ATE Cov-raw Cov-MALC Cov-indep Cov-indepM DUMPS
printf '%.0s-' {1..112}; echo
for r in "${ROOTS[@]}"; do
    IFS='|' read -r lbl rc cs is2d models <<<"$r"
    p_pt=$(nf_pt "$rc")
    p_raw=$(nf "$PERREAL/$lbl" 'raw__*__-.npz')
    p_mal=$(nf "$PERREAL/$lbl" 'malc__*__-.npz')
    p_ind=$(nf "$PERREAL/$lbl" 'indep_raw__*__-.npz')
    # The indep stage runs BOTH a raw and a B=1000 MALC pass, and the MALC half was
    # being computed but never reported -- which is why jobs kept running long after
    # Cov-indep reached 100%.
    p_indm=$(nf "$PERREAL/$lbl" 'indep_malc__*__-.npz')
    nd=$(nf "$rc" '*.npz')
    ind_col="$(pct "$p_ind" "$RC_N")"
    indm_col="$(pct "$p_indm" "$RC_N")"
    [ "$is2d" = no ] && { ind_col="$(printf '%-13s' 'n/a (1D)')"
                          indm_col="$(printf '%-13s' 'n/a (1D)')"; }
    printf '%-10s %s %s %s %s %s %s\n' "$lbl" \
      "$(pct "$p_pt" "$RC_N")" "$(pct "$p_raw" "$RC_N")" \
      "$(pct "$p_mal" "$RC_N")" "$ind_col" "$indm_col" "$nd npz"
done

echo
echo "=== Case study  (denominator: $CS_N cells = 3 shifts x 8 d x 6 cases)"
printf '%-10s %-14s %-14s %-14s %s\n' ROOT PEHE/ATE Cov-raw Cov-MALC DUMPS
printf '%.0s-' {1..80}; echo
for r in "${ROOTS[@]}"; do
    IFS='|' read -r lbl rc cs is2d models <<<"$r"
    c_pt=$(nf_pt "$cs")
    # cs per-real files carry a shift slice; rc files carry "-".
    c_raw=$(find "$PERREAL/$lbl" -name 'raw__*__shift*.npz' 2>/dev/null | wc -l | tr -d ' ')
    c_mal=$(find "$PERREAL/$lbl" -name 'malc__*__shift*.npz' 2>/dev/null | wc -l | tr -d ' ')
    nd=$(nf "$cs" '*.npz')
    printf '%-10s %s %s %s %s npz\n' "$lbl" \
      "$(pct "$c_pt" "$CS_N")" "$(pct "$c_raw" "$CS_N")" "$(pct "$c_mal" "$CS_N")" "$nd"
done

# ── ComplexMech ─────────────────────────────────────────────────────────────
# Dumps only: the scoring side is not wired yet (submit_cmech_density_score still
# points at the old data root and has no per-model driver), so there is no coverage
# column to report here rather than a zero that would look like pending work.
CMECH="${CMECH_DUMPS:-$SC/cmech_dumps}"
if [ -d "$CMECH" ]; then
    CM_NODES="${CM_NODES:-5 10 20 30 40 50}"
    CM_CELLS=0
    for n in $CM_NODES; do CM_CELLS=$((CM_CELLS+2)); done   # nonzero + zero
    echo
    echo "=== ComplexMech  (denominator: $CM_CELLS cells = 6 node counts x 2 subsets, N=1000)"
    CM_N=$(echo $CM_NODES | wc -w)
    printf '%-22s %-12s %-10s %-14s %-14s %s\n' \
      MODEL CELLS NPZ Cov-raw Cov-MALC PEHE/ATE
    printf '%.0s-' {1..92}; echo
    for d in "$CMECH"/*; do
        [ -d "$d" ] || continue
        c=0; nz=0
        for n in $CM_NODES; do
            for sub in nonzero zero; do
                k=$(find "$d" -type d -name "CMECH_n${n}_${sub}" 2>/dev/null \
                    -exec find {} -name '*.npz' \; 2>/dev/null | wc -l | tr -d ' ')
                [ "$k" -gt 0 ] && { c=$((c+1)); nz=$((nz+k)); }
            done
        done
        m="$(basename "$d")"
        # scoring lands in perreal under the model name, one file per node count
        cr=$(nf_cm "$PERREAL/$m" 'raw__cmech_n*__-.npz')
        cm=$(nf_cm "$PERREAL/$m" 'malc__cmech_n*__-.npz')
        cp=$(find "$d" -name 'point_raw_em_CMECH_n*.md' 2>/dev/null | wc -l | tr -d ' ')
        printf '%-22s %-12s %-10s %s %s %s\n' "$m" "$c/$CM_CELLS" "$nz" \
          "$(pct "$cr" "$CM_N")" "$(pct "$cm" "$CM_N")" "$(pct "$cp" "$CM_N")"
    done
    echo "  Cov/PEHE denominators are the $CM_N node counts; subset=total"
fi

# ── Do-PFN semi-real (sales, law_race) ──────────────────────────────────────
SR="${SEMIREAL_DUMPS:-$SC/semireal_dumps}"
if [ -d "$SR" ]; then
    echo
    echo "=== Semi-real  (5 splits per dataset -- small n, state it when reporting)"
    printf '%-22s %-10s %-10s %s\n' MODEL SALES LAW_RACE NPZ
    printf '%.0s-' {1..56}; echo
    for d in "$SR"/*; do
        [ -d "$d" ] || continue
        a=$(find "$d" -path '*SEMIREAL_sales*' -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
        b=$(find "$d" -path '*SEMIREAL_law_race*' -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
        printf '%-22s %-10s %-10s %s\n' "$(basename "$d")" "$a/5" "$b/5" "$((a+b))"
    done
fi

echo
echo "models per root:"
for r in "${ROOTS[@]}"; do
    IFS='|' read -r lbl rc cs is2d models <<<"$r"
    printf '  %-10s %s\n' "$lbl" "$models"
done
echo
echo "Cov-* columns also carry Len, IS and CRPS -- same file, no extra work."
echo "Forced independence is RealCause-only and applies to 2D heads."
echo
echo "table from whatever exists now:"
echo "  python R-PFN/benchmarks/collect_table.py --perreal $PERREAL"
