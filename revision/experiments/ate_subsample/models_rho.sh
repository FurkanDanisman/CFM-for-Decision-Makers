# Model table of the rho run, sourced by score_rho / collect_rho / submit_rho.
# name | tags | couplings (empty = 2D, learned joint)
R1="indep,gauss:-0.5,gauss:0.5,gauss:1"
MODEL_SPECS=(
  "dopfn_native||$R1"        "dopfn_joint2d_262k||"     "dopfn1d_mr||$R1"        "dopfn1d_ml||$R1"
  "uwyk1d|noanc,v3a|$R1"     "graph2d|noanc,v3a|"       "uwyk1d_mr|noanc,v3a|$R1" "uwyk1d_ml|noanc,v3a|$R1"
  "cpfn1d||$R1"              "cpfn2d||"                 "cpfn1d_mr||$R1"         "cpfn1d_ml||$R1"
)
ABLATIONS=(dopfn1d_mr dopfn1d_ml uwyk1d_mr uwyk1d_ml cpfn1d_mr cpfn1d_ml)
# in-prior result directories of one model spec (as reduce_bary.py names them)
variants_of () {
  local name tags coup t r; IFS='|' read -r name tags coup <<<"$1"
  local TL=(); IFS=, read -r -a TL <<<"$tags"; [ ${#TL[@]} -eq 0 ] && TL=("")
  for t in "${TL[@]}"; do
    if [ -z "$coup" ]; then echo "$name${t:+-$t}"; else for r in 0 m0.5 0.5 1; do echo "$name${t:+-$t}-rho$r"; done; fi
  done
}
