# The 13 checkpoints for ComplexMech, as "name|harness_idx|ckpt_env|ckpt|extra env".
# Sourced by submit_cmech_dumps.sh and by the fixed-query sweep, so a checkpoint fixed
# in one copy cannot be left stale in the other -- that would silently evaluate a
# different model. harness_idx selects the inner dispatch, NOT the model.
#
# Expects CK (checkpoint dir) already set.
ROWS=(
  "dopfn_native|0|NONE|-|"
  "dopfn_bb|1|CKPT_DOPFN_BB|$CK/dopfn_bb_j10_step_150000.pt|"
  "dopfn_repro_1d_J10|0|DOPFN_CKPT|$CK/dopfn_repro_1d_J10_step150000.pt|"
  "dopfn_repro_1d_J100|0|DOPFN_CKPT|$CK/dopfn_repro_1d_J100_step150000.pt|"
  "dopfn_repro_joint2d|0|DOPFN_CKPT|$CK/dopfn_repro_joint2d_step150000.pt|"
  "uwyk1d|2|NONE|-|"
  "uwyk_bin|2|CKPT|$CK/uwyk_bin_step50000.pt|CONFIG=$CK/uwyk_USED_IN_RESULTS_best_model_config.yaml UWYK_T_ENCODING=binary"
  "graph2d|3|CKPT_GRAPH2D|$CK/graph2d_step_50000.pt|"
  "cpfn1d_j1024|4|CKPT_CPFN1D|$CK/cpfn1d_j1024_headrand_step_50000.pt|"
  "cpfn1d_j32|4|CKPT_CPFN1D|$CK/cpfn1d_j32_step50000.pt|"
  "cpfn1d_botharms|4|CKPT_CPFN1D|$CK/cpfn1d_botharms_step50000.pt|"
  "cpfn_v0|4|CKPT_CPFN1D|$CK/cpfn_v0_original.pt|"
  "cpfn2d_eta0|5|CKPT_CPFN2D|$CK/cpfn2d_j32_eta0_y01_step50000.pt|"
)
