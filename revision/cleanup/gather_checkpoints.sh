#!/bin/bash
# Step 1 of the cluster clean-up: copy every checkpoint the paper uses into ONE folder, checksum it, and
# report the size of the folders that are clean-up candidates. Copies only; nothing is deleted.
#   bash R-PFN/revision/cleanup/gather_checkpoints.sh          (from $KIT; login node is fine, it only copies)
# Output: $KIT/final_checkpoints_paper/ with MANIFEST.txt (sha256, size, source), and a report on stdout:
#   MISSING lines (a listed source does not exist), SAME/DIFFERENT for copies we only matched by name before,
#   and du -sh of every candidate folder.
set -uo pipefail
KIT="${KIT:-/scratch/furkanbd/rpfn_bench_kit}"; CK="$KIT/R-PFN/Required_checkpoints"; D262="$KIT/checkpoints_dopfn_repro_262k"
AUR="$KIT/from_aurora"; UW="$KIT/external/uwyk_reproduce/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0"
DEST="$KIT/final_checkpoints_paper"; mkdir -p "$DEST"; MAN="$DEST/MANIFEST.txt"; : > "$MAN"   # rebuilt on every run

grab() {   # grab <source> <name in DEST> <paper model>
  local src="$1" dst="$DEST/$2"
  if [ ! -e "$src" ]; then echo "MISSING  $2  <- $src  ($3)"; return; fi
  [ -L "$src" ] && echo "SYMLINK  $src -> $(readlink -f "$src")"
  [ -e "$dst" ] || cp -L --preserve=timestamps "$src" "$dst"
  printf '%s  %12s  %-45s  %s  | %s\n' "$(sha256sum "$dst" | cut -c1-64)" "$(stat -c %s "$dst")" "$2" "$3" "$(readlink -f "$src")" >> "$MAN"
  echo "ok       $2"
}
same() {   # same <copy> <original>: were these really the same file?
  [ -e "$1" ] && [ -e "$2" ] || { echo "CHECK?   cannot compare $1 vs $2 (one is missing)"; return; }
  if cmp -s "$1" "$2"; then echo "SAME     $(basename "$1") == $2"; else echo "DIFFERENT $(basename "$1") != $2"; fi
}

echo "== paper checkpoints -> $DEST"
grab "$CK/dopfn_repro_joint2d_step262144.pt"            dopfn2d_joint_step262144.pt           "Do-PFN 2D"
grab "$D262/dopfn_1d/step_262144_final.pt"              dopfn1d_J10_step262144.pt             "Do-PFN matched resolution"
grab "$D262/dopfn_1d_botharms/step_262144_final.pt"     dopfn1d_botharms_step262144.pt        "Do-PFN matched label"
grab "$CK/graph2d_step_50000.pt"                        uwyk2d_graph2d_step50000.pt           "UWYK 2D"
grab "$CK/uwyk_bin_step50000.pt"                        uwykB_bin_step50000.pt                "UWYK-B"
grab "$CK/uwyk_USED_IN_RESULTS_best_model_config.yaml"  uwykB_config.yaml                     "UWYK-B config"
grab "$AUR/uwyk_D_j32_nobin_step50000.pt"               uwyk_mr_D_j32_step50000.pt            "UWYK matched resolution"
grab "$AUR/uwyk_D_j32_config.yaml"                      uwyk_mr_D_j32_config.yaml             "UWYK matched resolution config"
grab "$AUR/uwyk_C_botharms_step50000.pt"                uwyk_ml_C_botharms_step50000.pt       "UWYK matched label"
grab "$AUR/best_model_config.yaml"                      uwyk_ml_C_botharms_config.yaml        "UWYK matched label config"
grab "$CK/cpfn1d_j1024_headrand_step_50000.pt"          cpfnC_1d_j1024_step50000.pt           "CausalPFN-C"
grab "$CK/cpfn2d_j32_eta0_y01_step50000.pt"             cpfnC_2d_j32_eta0_step50000.pt        "CausalPFN-C 2D"
grab "$CK/cpfn1d_j32_step50000.pt"                      cpfnC_mr_1d_j32_step50000.pt          "CausalPFN-C matched resolution"
grab "$CK/cpfn1d_botharms_step50000.pt"                 cpfnC_ml_1d_botharms_step50000.pt     "CausalPFN-C matched label"
grab "$CK/cpfn_v0_original.pt"                          cpfn_published_two_stage.pt           "published CausalPFN (symlink resolved)"
grab "$UW/best_model.pt"                                uwyk_published_best_model.pt          "UWYK (published)"
grab "$UW/best_model_config.yaml"                       uwyk_published_best_model_config.yaml "UWYK (published) config"
grab "$D262/backbone_init_s0.pt"                        dopfn_backbone_init_s0.pt             "shared backbone init (Do-PFN 262k + fixed-SCM)"
echo "== fixed-SCM Do-PFN 1D runs (natural, setup1-3): the final 10k checkpoint and the most recent 300k-run checkpoint"
for run in natural setup1 setup2 setup3; do
  grab "$KIT/checkpoints_fixed_scm/$run/dopfn_1d/step_10000_final.pt" "fscm10k_${run}_step10000.pt" "fixed-SCM $run, 10k run (final)"
  d="$KIT/checkpoints_fixed_scm_300k/$run/dopfn_1d"
  last=$(ls "$d"/step_*.pt 2>/dev/null | sort -V | tail -1)
  [ -z "$last" ] && { echo "MISSING  fixed-SCM $run 300k: no step_*.pt in $d"; continue; }
  st=$(basename "$last" .pt | sed 's/^step_//; s/_final$//')
  for old in "$DEST"/fscm300k_${run}_step*.pt; do      # a newer step replaces the older copy (setup3 is still training)
    [ -e "$old" ] && [ "$old" != "$DEST/fscm300k_${run}_step${st}.pt" ] && { rm -f "$old"; sed -i "/fscm300k_${run}_step/d" "$MAN"; }
  done
  grab "$last" "fscm300k_${run}_step${st}.pt" "fixed-SCM $run, 300k run (most recent: step $st)"
done
echo "== published Do-PFN weights (external/dopfn, whole folder kept in place; files listed here)"
find "$KIT/external/dopfn" -maxdepth 4 -type f \( -name '*.cpkt' -o -name '*.ckpt' -o -name '*.pt' -o -name '*.pth' \) -exec ls -la {} \; 2>/dev/null

echo "== copies we had matched only by name/size"
same "$CK/dopfn_repro_joint2d_step262144.pt" "$D262/joint_2d/step_262144_final.pt"
same "$CK/cpfn1d_j1024_headrand_step_50000.pt" "$KIT/causalpfn_j1024_headrand_A1_h100/step_checkpoints/run/step_0050000.pt"
same "$CK/cpfn2d_j32_eta0_y01_step50000.pt" "$KIT/cpfn2d_j32_eta0_y01_A1_h100/step_checkpoints/run/step_0050000.pt"
same "$CK/cpfn2d_j32_eta0_y01_step50000.pt" "$CK/cpfn2d_j32_random_step_50000.pt"
echo "== UWYK copy check: which published UWYK folder exists"
ls -d "$KIT/external/uwyk_reproduce" "$KIT/external/uwyk" 2>&1

echo "== size of the paper folder"; du -sh "$DEST"
echo "== fixed-SCM (kept for now)"; du -sh "$KIT"/checkpoints_fixed_scm* 2>/dev/null
echo "== clean-up candidates (sizes only, nothing deleted)"
cd "$KIT" && du -sh checkpoints_dopfn_repro checkpoints_dopfn_backbone* checkpoints_shared checkpoints_causalpfn2d* checkpoints_cpfn2d_* \
  cpfn2d_j32_random_A1_* cpfn2d_j1024_* cpfn2d_j32_armc* causalpfn_j1024_full_output causalpfn_j1024_headrand_output \
  causalpfn_original_output causalpfn_original_j100_output cpfn1d_j32_headrand_SMOKE checkpoints_graph2d_l40s checkpoints_graph2d_smoke \
  checkpoints_graph2d_ablation_baseline from_tamia R-PFN/checkpoints R-PFN/checkpoints_dopfn R-PFN/checkpoints_causalpfn2d* \
  R-PFN/Required_checkpoints/old R-PFN/Required_checkpoints/new R-PFN/Required_checkpoints/new_checkpoints 2>/dev/null | sort -h
echo "== total \$KIT"; du -sh "$KIT" 2>/dev/null
