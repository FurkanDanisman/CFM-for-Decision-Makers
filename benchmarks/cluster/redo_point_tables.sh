#!/bin/bash
# Regenerate every point table with the correct ATE metric. Runs in place, fast.
#
# Why: progress.sh counts only tables carrying an "ate_metric=" stamp, and 4 of 5
# RealCause tables lack it. Those files were written before --ate-metric existed, so
# they hold plain |dATE| under an "eps_ATE" header -- the wrong quantity for
# RealCause, which reports RELATIVE error.
#
# RealCause gets --ate-metric rel; the case studies get l1.
#
# This is numpy over dumps that already exist -- seconds per cell, no GPU, no MALC
# -- so it runs here rather than as a job.
#
#   bash R-PFN/benchmarks/cluster/redo_point_tables.sh            # all roots
#   ONLY="orig eta0" bash R-PFN/benchmarks/cluster/redo_point_tables.sh

set -uo pipefail
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
KIT="${KIT:-$(cd "$_SELF_DIR/../../.." && pwd)}"
REPO="${REPO:-$KIT/R-PFN}"
SC="${SCRATCH:?SCRATCH must be set}"
PT="$REPO/realcause_eval/point_raw_em.py"
RC_DS="${RC_DS:-IHDP ACIC CPS PSID PSID_bal}"
CASES="${CASES:-Observed_Confounder Backdoor_Criterion Observed_Mediator Observed_Mediator_and_Confounder Unobserved_Confounder Frontdoor_Criterion}"
ONLY="${ONLY:-}"

ROOTS=(
  "orig|$SC/rc_dens_uni|$SC/cs_dvar_dens"
  "eta0|$SC/rc_dens_eta0|$SC/cs_dvar_eta0"
  "J10|$SC/dumps_all/dopfn_repro_1d_J10/rc|$SC/dumps_all/dopfn_repro_1d_J10/cs"
  "J100|$SC/dumps_all/dopfn_repro_1d_J100/rc|$SC/dumps_all/dopfn_repro_1d_J100/cs"
  "joint2d|$SC/dumps_all/dopfn_repro_joint2d/rc|$SC/dumps_all/dopfn_repro_joint2d/cs"
  "j32|$SC/dumps_all/cpfn1d_j32/rc|$SC/dumps_all/cpfn1d_j32/cs"
  "botharms|$SC/dumps_all/cpfn1d_botharms/rc|$SC/dumps_all/cpfn1d_botharms/cs"
  "cpfn_v0|$SC/dumps_all/cpfn_v0/rc|$SC/dumps_all/cpfn_v0/cs"
  "uwyk_bin|$SC/dumps_all/uwyk_bin/rc|$SC/dumps_all/uwyk_bin/cs"
)

source "$KIT/venv/bin/activate" 2>/dev/null
export PYTHONUNBUFFERED=1
OK=0; BAD=0

for r in "${ROOTS[@]}"; do
    IFS='|' read -r lbl rc cs <<<"$r"
    [ -n "$ONLY" ] && { case " $ONLY " in *" $lbl "*) ;; *) continue ;; esac; }

    if [ -d "$rc" ]; then
        for ds in $RC_DS; do
            printf '%-10s rc  %-9s ' "$lbl" "$ds"
            if python -u "$PT" --root "$rc" --dataset "$ds" --modes raw \
                   --ate-metric rel --out-md "$rc/point_raw_em_${ds}.md" \
                   > /tmp/.pt_$$.log 2>&1; then
                echo "ok"; OK=$((OK+1))
            else
                echo "FAILED"; sed 's/^/      /' /tmp/.pt_$$.log | tail -4; BAD=$((BAD+1))
            fi
        done
    fi

    if [ -d "$cs" ]; then
        n_ok=0; n_bad=0
        for cell in "$cs"/shift*/d*/ctx*; do
            [ -d "$cell" ] || continue
            for c in $CASES; do
                if python -u "$PT" --root "$cell" --dataset "$c" --modes raw \
                       --ate-metric l1 --out-md "$cell/point_raw_em_${c}.md" \
                       > /tmp/.pt_$$.log 2>&1; then
                    n_ok=$((n_ok+1))
                else
                    n_bad=$((n_bad+1))
                    [ "$n_bad" = 1 ] && { echo; echo "  first cs failure ($lbl):";
                                          sed 's/^/      /' /tmp/.pt_$$.log | tail -4; }
                fi
            done
        done
        printf '%-10s cs  %-9s ok=%s failed=%s\n' "$lbl" "(cells)" "$n_ok" "$n_bad"
        OK=$((OK+n_ok)); BAD=$((BAD+n_bad))
    fi
done
rm -f /tmp/.pt_$$.log
echo
echo "tables written=$OK  failed=$BAD"
echo "check with: bash $REPO/benchmarks/cluster/progress.sh"
