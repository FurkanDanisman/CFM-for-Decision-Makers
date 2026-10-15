#!/bin/bash
#SBATCH --job-name=coupling
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
# Revision v2, E1 + E2: rescore the EXISTING density dumps of the paper's models under other couplings.
# Same scorer, settings and truths as the paper's tables (submit_hist.sbatch: CDM_INTERVAL=hist,
# ATOM_GRID_MAX=4096, IHDP/ACIC against realized delta). No model is run; only reads dumps.
#   1D heads (--coupling):        comonotonic, gauss:-0.5, gauss:0.5, band, makarov
#   2D heads (--joint-coupling):  indep (LD=Off), band, makarov        [learned = the paper's tables]
# One job per benchmark part, all couplings run side by side inside it; resubmitting resumes (done files skipped).
#   cd $KIT && git -C R-PFN pull --rebase && for p in rc cs-2 cs0 cs+2 cm; do
#     sbatch --account=aip-rgrosse -o logs_coupling/%x_${p}_%j.out R-PFN/revision/experiments/coupling/score_coupling.sbatch $p; done
set -uo pipefail
PART="${1:?part: rc | cs-2 | cs0 | cs+2 | cm}"
KIT="${KIT:-$PWD}"; REPO="$KIT/R-PFN"
source "$KIT/venv/bin/activate"
export CAUSALPFN="$KIT/external/causalpfn"
export CDM_INTERVAL=hist PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ATOM_GRID_MAX="${ATOM_GRID_MAX:-4096}"
export PYTHONPATH="$REPO/benchmarks:$REPO/benchmarks/uwyk_table1/shims${PYTHONPATH:+:$PYTHONPATH}"
CBR="$REPO/benchmarks/coverage_by_realization.py"
OUT="${OUT:-$SCRATCH/hist_coupling}"
TRUTH="$SCRATCH/hist/truth"                       # built by submit_hist.sbatch task 0 (realized delta, IHDP/ACIC)
T1=dopfn_repro_joint2d_262k_t1                    # the paper's Do-PFN 2D (262k, temperature off)
mkdir -p "$KIT/logs_coupling"
[ -d "$TRUTH" ] || { echo "FATAL: no $TRUTH"; exit 1; }

score() {   # score <out npz> <cbr args...>   (skip finished files; log next to the output)
    local o="$1"; shift
    [ -s "$o" ] && { echo "skip (done) $o"; return 0; }
    mkdir -p "$(dirname "$o")"
    echo "[$(date '+%F %T')] $o"
    python -u "$CBR" "$@" --tau-smoother none --dump-per-real "$o" > "${o%.npz}.md" 2> "${o%.npz}.err" \
        || { echo "  ERROR: ${o%.npz}.err"; return 1; }
}

# label | RealCause root | case-study root | 1D methods | 2D methods     (as in submit_hist.sbatch)
ROOTS=(
  "orig|$SCRATCH/rc_dens_uni|$SCRATCH/cs_dvar_dens|dopfn_native uwyk1d-noanc uwyk1d-v3a cpfn1d|graph2d-noanc graph2d-v3a"
  "$T1|$SCRATCH/dumps_all/$T1/rc|$SCRATCH/dumps_all/$T1/cs||dopfn_native"
  "eta0|$SCRATCH/rc_dens_eta0|$SCRATCH/cs_dvar_eta0||cpfn2d"
)
C1="comonotonic gauss:-0.5 gauss:0.5 band makarov"
C2="indep band makarov"
CASES="Observed_Confounder Observed_Mediator Observed_Mediator_and_Confounder Unobserved_Confounder Frontdoor_Criterion Backdoor_Criterion"
CM1="dopfn_native uwyk1d cpfn1d_j1024"
CM2="graph2d cpfn2d_eta0 $T1"
fn() { echo "$1" | tr ':' '_'; }                  # gauss:-0.5 -> gauss_-0.5 in file names

# run_coupling <kind 1|2> <coupling>: every cell of this PART under one coupling (one process)
run_coupling() {
    local kind="$1" cpl="$2" F=0 flag
    [ "$kind" = 1 ] && flag="--coupling" || flag="--joint-coupling"
    local pre; pre="$(fn "$cpl")"; [ "$kind" = 2 ] && pre="joint_$pre"
    case "$PART" in
      rc) for row in "${ROOTS[@]}"; do
            IFS='|' read -r lbl rc cs m1 m2 <<<"$row"; local ms; [ "$kind" = 1 ] && ms="$m1" || ms="$m2"
            [ -n "$ms" ] || continue
            for ds in IHDP ACIC; do
              score "$OUT/perreal_e1/$lbl/${pre}__${ds}__-.npz" --root "$rc" --dataset "$ds" --methods $ms \
                    --truth-dir "$TRUTH" $flag "$cpl" || F=1
            done
            for ds in CPS PSID PSID_bal; do
              score "$OUT/perreal/$lbl/${pre}__${ds}__-.npz" --root "$rc" --dataset "$ds" --methods $ms $flag "$cpl" || F=1
            done
          done ;;
      cs-2|cs0|cs+2) local sh="shift${PART#cs}"
          for row in "${ROOTS[@]}"; do
            IFS='|' read -r lbl rc cs m1 m2 <<<"$row"; local ms; [ "$kind" = 1 ] && ms="$m1" || ms="$m2"
            [ -n "$ms" ] || continue
            for cell in "$cs/$sh"/d*/ctx1000; do
              [ -d "$cell" ] || continue
              local d="${cell%/ctx1000}"; d="${d##*/}"
              for c in $CASES; do
                score "$OUT/perreal/$lbl/${pre}__${d}_${c}__${sh}.npz" --root "$cell" --dataset "$c" --methods $ms $flag "$cpl" || F=1
              done
            done
          done ;;
      cm) local models; [ "$kind" = 1 ] && models="$CM1" || models="$CM2"
          for m in $models; do
            [ -d "$SCRATCH/cmech_dumps/$m/N1000" ] || { echo "MISSING $SCRATCH/cmech_dumps/$m/N1000"; F=1; continue; }
            for n in 5 10 20 30 40 50; do
              compgen -G "$SCRATCH/cmech_dumps/$m/N1000/*/CMECH_n${n}_*" >/dev/null || continue
              score "$OUT/perreal/$m/${pre}__cmech_n${n}__-.npz" --root "$SCRATCH/cmech_dumps/$m" \
                    --nodes "$n" --subset total --context 1000 $flag "$cpl" || F=1
            done
          done ;;
      *) echo "FATAL: unknown part $PART"; return 1 ;;
    esac
    echo "[$(date '+%F %T')] coupling $kind:$cpl done (failed=$F)"
    return $F
}

echo "[$(date '+%F %T')] part=$PART  out=$OUT"
PIDS=()
for c in $C1; do run_coupling 1 "$c" > "$KIT/logs_coupling/${PART}_1d_$(fn "$c").log" 2>&1 & PIDS+=($!); done
for c in $C2; do run_coupling 2 "$c" > "$KIT/logs_coupling/${PART}_2d_$(fn "$c").log" 2>&1 & PIDS+=($!); done
FAILED=0; for p in "${PIDS[@]}"; do wait "$p" || FAILED=1; done
grep -h "coupling .* done" "$KIT"/logs_coupling/${PART}_*.log
echo "[$(date '+%F %T')] part $PART done (failed=$FAILED)"
exit $FAILED
