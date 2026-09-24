# The 13 checkpoints, as "name|harness|extra env". Sourced by both
# submit_cs_fixedq_all.sh (one job per model, one cell) and submit_fq4_dump.sbatch
# (one array task per cell x model), so the two cannot drift apart -- a checkpoint
# path fixed in one copy and not the other would silently evaluate a different model.
#
# Expects CK (checkpoint dir) and UWYKD (uwyk checkpoint dir) already set.
# name | harness | extra env (space-separated VAR=VAL)
ROWS=(
  "dopfn_native|dopfn_native|"
  "dopfn_repro_1d_J10|dopfn_native|DOPFN_CKPT=$CK/dopfn_repro_1d_J10_step150000.pt"
  "dopfn_repro_1d_J100|dopfn_native|DOPFN_CKPT=$CK/dopfn_repro_1d_J100_step150000.pt"
  "dopfn_repro_joint2d|dopfn_native|DOPFN_CKPT=$CK/dopfn_repro_joint2d_step150000.pt"
  "dopfn_bb|dopfn_bb|CKPT_DOPFN_BB=$CK/dopfn_bb_j10_step_150000.pt"
  "uwyk1d|uwyk1d|CKPT=$UWYKD/best_model.pt CONFIG=$UWYKD/best_model_config.yaml"
  "uwyk_bin|uwyk1d|CKPT=$CK/uwyk_bin_step50000.pt CONFIG=$CK/uwyk_USED_IN_RESULTS_best_model_config.yaml UWYK_T_ENCODING=binary"
  "graph2d|graph2d|CKPT_GRAPH2D=$CK/graph2d_step_50000.pt"
  "cpfn1d_j1024|cpfn1d|CKPT_CPFN1D=$CK/cpfn1d_j1024_headrand_step_50000.pt"
  "cpfn1d_j32|cpfn1d|CKPT_CPFN1D=$CK/cpfn1d_j32_step50000.pt"
  "cpfn1d_botharms|cpfn1d|CKPT_CPFN1D=$CK/cpfn1d_botharms_step50000.pt"
  "cpfn_v0|cpfn1d|CKPT_CPFN1D=$CK/cpfn_v0_original.pt"
  "cpfn2d_eta0|cpfn2d_pooled|CKPT_CPFN2D=$CK/cpfn2d_j32_eta0_y01_step50000.pt"
)
