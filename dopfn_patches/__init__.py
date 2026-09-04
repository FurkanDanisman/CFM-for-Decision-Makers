"""Patches and shims for DoPFN's synthetic-SCM prior used by the case-study
regeneration pipeline.

Public entry points:
  make_structural_equations_bivariate.MakeStructuralEquationsBivariate
      Subclass of DoPFN's MakeStructuralEquations that samples a bivariate
      Gaussian (eps_obs, eps_int) for the Y-node with correlation rho, so
      tau = Y_int - Y_obs is a proper Gaussian rather than a delta.

  make_structural_equations_bivariate.install_bivariate_y_noise
      Post-hoc helper that patches an already-built StructuralCausalModel
      to add bivariate noise to the Y-node without touching non-Y nodes
      (so `exogenous_vars=exo_obs` reuse in scm.get_next_sample still
      leaves ancestor variables consistent between obs and int calls).

  case_study_graphs.build_case_study_graph
      Returns (nx.DiGraph, t_key, y_key, x_idcs) for each of the 6 named
      case studies from the DoPFN paper (App. D.1 Table 1).
"""
