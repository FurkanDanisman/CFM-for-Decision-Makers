#!/bin/bash
# Percentage complete, per model, per task. Safe to run at any moment.
#
# Counts files on disk against what a finished task produces, so it reflects real
# progress rather than slurm state -- which is unreliable here: the case-study
# sbatch ends each harness call with `|| echo WARN`, so a cell that died reports
# COMPLETED with exit 0.
#
# Expected counts per model:
#   dump rc   5 datasets x ~100 realizations                  = ~500 npz
#   dump cs   3 shifts x 8 d x 6 cases x ~100 realizations     = ~14400 npz
#   score     one per-realization .npz per (stage, group, slice):
#               rc  5 groups x 1 slice                         = 5
#               cs  48 groups (8 d x 6 cases) x 3 shifts       = 144
#
#   bash R-PFN/benchmarks/cluster/progress.sh

set -uo pipefail
SC="${SCRATCH:-/scratch/furkanbd}"
PERREAL="${PERREAL:-$SC/perreal}"
RC_EXP="${RC_EXP:-500}"; CS_EXP="${CS_EXP:-14400}"
RC_SCORE_EXP=5; CS_SCORE_EXP=144

# label | rc dump root | cs dump root | is2d
ROWS=(
  "orig|$SC/rc_dens_uni|$SC/cs_dvar_dens|yes"
  "eta0|$SC/rc_dens_eta0|$SC/cs_dvar_eta0|yes"
  "J10|$SC/dumps_all/dopfn_repro_1d_J10/rc|$SC/dumps_all/dopfn_repro_1d_J10/cs|no"
  "J100|$SC/dumps_all/dopfn_repro_1d_J100/rc|$SC/dumps_all/dopfn_repro_1d_J100/cs|no"
  "joint2d|$SC/dumps_all/dopfn_repro_joint2d/rc|$SC/dumps_all/dopfn_repro_joint2d/cs|yes"
  "j32|$SC/dumps_all/cpfn1d_j32/rc|$SC/dumps_all/cpfn1d_j32/cs|no"
  "botharms|$SC/dumps_all/cpfn1d_botharms/rc|$SC/dumps_all/cpfn1d_botharms/cs|no"
  "cpfn_v0|$SC/dumps_all/cpfn_v0/rc|$SC/dumps_all/cpfn_v0/cs|no"
  "uwyk_bin|$SC/dumps_all/uwyk_bin/rc|$SC/dumps_all/uwyk_bin/cs|no"
)

pct() { [ "$2" -le 0 ] && { echo "  -"; return; }
        awk -v a="$1" -v b="$2" 'BEGIN{p=100*a/b; if(p>100)p=100; printf "%3.0f%%", p}'; }
cnt() { [ -d "$1" ] && find "$1" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ' || echo 0; }
scnt() { [ -d "$PERREAL/$1" ] && \
         find "$PERREAL/$1" -name "$2__*.npz" 2>/dev/null | wc -l | tr -d ' ' || echo 0; }

printf '%-10s | %-14s %-14s | %-11s %-11s %-11s %-11s %s\n' \
  MODEL dump-rc dump-cs score-raw score-MALC score-indep cs-raw cs-MALC
printf '%.0s-' {1..108}; echo

for r in "${ROWS[@]}"; do
    IFS='|' read -r lbl rc cs is2d <<<"$r"
    nrc=$(cnt "$rc"); ncs=$(cnt "$cs")
    sraw=$(scnt "$lbl" raw); smal=$(scnt "$lbl" malc)
    sind=$(scnt "$lbl" indep_raw)
    # rc groups have slice "-", cs groups carry shift<S>
    rcraw=$(find "$PERREAL/$lbl" -name 'raw__*__-.npz'  2>/dev/null | wc -l | tr -d ' ')
    rcmal=$(find "$PERREAL/$lbl" -name 'malc__*__-.npz' 2>/dev/null | wc -l | tr -d ' ')
    csraw=$((sraw - rcraw)); csmal=$((smal - rcmal))
    ind_disp="$(pct "$sind" "$RC_SCORE_EXP")"
    [ "$is2d" = no ] && ind_disp="  n/a"
    printf '%-10s | %6s %-7s %6s %-7s | %5s %-5s %5s %-5s %5s %-5s %5s %-5s %5s\n' \
      "$lbl" "$nrc" "$(pct "$nrc" "$RC_EXP")" "$ncs" "$(pct "$ncs" "$CS_EXP")" \
      "$rcraw" "$(pct "$rcraw" "$RC_SCORE_EXP")" \
      "$rcmal" "$(pct "$rcmal" "$RC_SCORE_EXP")" \
      "$sind" "$ind_disp" \
      "$csraw" "$(pct "$csraw" "$CS_SCORE_EXP")" \
      "$csmal" "$(pct "$csmal" "$CS_SCORE_EXP")"
done

echo
echo "score-* columns count per-realization .npz files (the table's inputs)."
echo "score-indep is RealCause-only and applies to 2D heads, so n/a elsewhere."
echo
echo "Table from whatever exists right now:"
echo "  python R-PFN/benchmarks/collect_table.py --perreal $PERREAL"
