#!/bin/bash
#SBATCH --job-name=coupling-abl
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
# Revision v2, E1/E2 for the APPENDIX ablation models (matched resolution / matched label): their two
# marginals combined under other couplings. Same scorer and settings as score_coupling.sbatch and the paper's
# ablation tables (submit_hist_extra.sbatch; Do-PFN rows = the 262k temperature-off models of D262k stage 23).
# One job per benchmark part; inside it every (model, coupling) pair runs as its own process, 16 at a time.
# Resubmitting resumes (finished files are skipped).
#   cd $KIT && git -C R-PFN pull --rebase && for p in rc cs cm; do
#     sbatch --account=aip-rgrosse -o logs_coupling/%x_${p}_%j.out R-PFN/revision/experiments/coupling/score_coupling_abl.sbatch $p; done
set -uo pipefail
PART="${1:?part: rc | cs | cm}"
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"
source "$KIT/venv/bin/activate"
export CAUSALPFN="$KIT/external/causalpfn"
export CDM_INTERVAL=hist PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
export CBR="$REPO/benchmarks/coverage_by_realization.py" OUT="${OUT:-$SCRATCH/hist_coupling}" TRUTH="$SCRATCH/hist/truth" PART
mkdir -p "$KIT/logs_coupling"
[ -d "$TRUTH" ] || { echo "FATAL: no $TRUTH"; exit 1; }
S=$SCRATCH; A=$SCRATCH/aurora3_kill
# model | RealCause root | case-study root | ComplexMech all-rho root | methods   (as submit_hist_extra.sbatch)
export ROWS="
dopfn_repro_1d_J10_262k_t1|$S/dumps_all/dopfn_repro_1d_J10_262k_t1/rc|$S/dumps_all/dopfn_repro_1d_J10_262k_t1/cs|$S/cmech_dumps/dopfn_repro_1d_J10_262k_t1|dopfn_native
dopfn_1d_botharms_262k_t1|$S/dumps_all/dopfn_1d_botharms_262k_t1/rc|$S/dumps_all/dopfn_1d_botharms_262k_t1/cs|$S/cmech_dumps/dopfn_1d_botharms_262k_t1|dopfn_native
uwyk_D_j32_nobin|$A/rc/uwyk_D_j32_nobin|$A/cs/uwyk_D_j32_nobin|$S/aurora3_allrho/cm_rho99/uwyk_D_j32_nobin|uwyk1d-noanc uwyk1d-v3a
uwyk_C_botharms|$A/rc/uwyk_C_botharms|$A/cs/uwyk_C_botharms|$S/aurora3_allrho/cm_rho99/uwyk_C_botharms|uwyk1d-noanc uwyk1d-v3a
cpfn1d_j32|$S/dumps_all/cpfn1d_j32/rc|$S/dumps_all/cpfn1d_j32/cs|$S/cmech_dumps/cpfn1d_j32|cpfn1d
cpfn1d_botharms|$S/dumps_all/cpfn1d_botharms/rc|$S/dumps_all/cpfn1d_botharms/cs|$S/cmech_dumps/cpfn1d_botharms|cpfn1d"

# one (model, coupling) pair, every cell of PART
pair() {
    local model="$1" cpl="$2" row m rc cs cm meths pre F=0
    row="$(echo "$ROWS" | grep "^$model|")"; IFS='|' read -r m rc cs cm meths <<<"$row"
    pre="$(echo "$cpl" | tr ':' '_')"
    sc() {   # sc <out npz> <cbr args...>
        local o="$1"; shift
        [ -s "$o" ] && return 0
        mkdir -p "$(dirname "$o")"
        python -u "$CBR" "$@" --tau-smoother none --coupling "$cpl" --dump-per-real "$o" > "${o%.npz}.md" 2> "${o%.npz}.err" \
            || { echo "ERROR ${o%.npz}.err"; return 1; }
    }
    case "$PART" in
      rc) for ds in IHDP ACIC; do sc "$OUT/perreal_e1/$m/${pre}__${ds}__-.npz" --root "$rc" --dataset "$ds" --methods $meths --truth-dir "$TRUTH" || F=1; done
          for ds in CPS PSID PSID_bal; do sc "$OUT/perreal/$m/${pre}__${ds}__-.npz" --root "$rc" --dataset "$ds" --methods $meths || F=1; done ;;
      cs) for sh in shift-2 shift0 shift+2; do
            for cell in "$cs/$sh"/d*/ctx1000; do
              [ -d "$cell" ] || continue
              local d="${cell%/ctx1000}"; d="${d##*/}"
              for c in Observed_Confounder Observed_Mediator Observed_Mediator_and_Confounder Unobserved_Confounder Frontdoor_Criterion Backdoor_Criterion; do
                sc "$OUT/perreal/$m/${pre}__${d}_${c}__${sh}.npz" --root "$cell" --dataset "$c" --methods $meths || F=1
              done
            done
          done ;;
      cm) [ -d "$cm/N1000" ] || { echo "MISSING $cm/N1000"; return 1; }
          for n in 5 10 20 30 40 50; do
            compgen -G "$cm/N1000/*/CMECH_n${n}_*" >/dev/null || continue
            sc "$OUT/perreal/$m/${pre}__cmech_n${n}__-.npz" --root "$cm" --nodes "$n" --subset total --context 1000 --methods $meths || F=1
          done ;;
    esac
    echo "[$(date '+%F %T')] $model $cpl done (failed=$F)"
    return $F
}
export -f pair

echo "[$(date '+%F %T')] part=$PART"
for m in $(echo "$ROWS" | cut -d'|' -f1 | grep .); do
  for c in comonotonic gauss:-0.5 gauss:0.5 makarov; do echo "$m $c"; done
done | xargs -P 16 -L 1 bash -c 'pair "$0" "$1"'
echo "[$(date '+%F %T')] part $PART done"
