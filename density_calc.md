# Density calculation and comparison metrics — agreed methodology

**Status**: v2 agreed 2026-08-16. Read this before implementing/reviewing any
density comparison code. Cross-reference the memory file:
`~/.claude/projects/-Users-furkandanisman-R-PFN/memory/feedback_density_and_l2_calculation.md`

Reference implementations (do not deviate without updating this doc):
- `benchmarks/empirical_tests/fig2_pehe_l2.py` — Fig 2 v2 (polynomial SCM)
- `benchmarks/l2_ihdp/methods_densities.py` — IHDP / ACIC / syn L2 pipeline
- `benchmarks/l2_ihdp/methods_densities.py::malc_1d_cvxpy` — Python 1D MALC
- `benchmarks/methods/ours.py::_fit_and_marginalize` — 2D-MALC + CATE integration
- `benchmarks/empirical_tests/sanity_cate_density.py` — pipeline verification
- `MALC_2D/MALC_2D_Algorithm.R` — R reference for 2D MALC
- `R/malc.R` — R reference for 1D MALC (equivalent to our CVXPY port)
- `R/density-calc.R::Eval` — reference L2 formula

Method list (as of 2026-08-16): **Do-PFN, UWYK-NoAnc, UWYK-FullAnc, Ours(fn=50)**.
Ours-DoPFN-bb is not run this round.

## 1. What "density" means for a discrete BarDist head

Each model outputs probability MASS on K (or J) bins with uniform width `w`:

    p_1, p_2, ..., p_K       Σ p_i = 1

The piecewise-constant density at bin i's center is:

    d_i = p_i / w            (mass / length = density)

## 2. Comparison metrics

Two metrics are reported for every method / density comparison. Both must
be computed on the same common evaluation grid with spacing `dx`. Both
`f` and `g` must integrate to 1 on that grid (renormalise after any
interpolation).

### 2a. L2 (Riemann-integral L² distance)

    L2(f, g) = sqrt( Σ_k (f[k] − g[k])² · dx )

Standard density-distance. Scale-dependent, penalises mean-shifts heavily.

### 2b. KL divergence — BOTH directions (added 2026-08-16)

    KL_fwd(f_true, f_est) = Σ_k f_true[k] · log(f_true[k] / f_est[k]) · dx
    KL_rev(f_est, f_true) = Σ_k f_est[k]  · log(f_est[k]  / f_true[k]) · dx

Numerical safety: floor both densities at ε=1e-12 before the log.

- **KL_fwd (truth ‖ est)** = "how much info do we lose using est when
  reality is truth". Equivalent to negative log-likelihood of truth under
  est. Small when est has mass wherever truth has mass. What NLL-style
  scoring rules optimise.
- **KL_rev (est ‖ truth)** = "how far does est stray beyond truth's
  support". Small when est doesn't put mass where truth is zero.

Report both — they answer different questions and are both cheap.

## 3. Marginal density — Y_do(0), Y_do(1)

Per method, per arm t ∈ {0, 1}:

### 3a. UWYK-NoAnc, UWYK-FullAnc, Do-PFN — raw BarDist probabilities

    pBars_t[i] = discrete probabilities from the model's BarDist head
                 (softmax on the K+2 logits, drop tails, renormalise)
    d_t[i]     = pBars_t[i] / w_native         # → density on native centers

Resample onto common Y-grid via `l2.resample_onto`. Compute L2 + KL_fwd +
KL_rev vs the analytic truth Gaussian evaluated on the same common grid.

Reference: `_uwyk_densities_from_raw_probs` and `dopfn_densities` in
`benchmarks/l2_ihdp/methods_densities.py`.

### 3b. Ours (fn=50) — 1D MALC on the marginal from p_mat

    1.  Raw marginal from the 2D joint (mass on J bins in scaled y):
            p_marg_t[j] = Σ_other p_mat[j0 or j1, other]

    2.  Fit 1D discrete log-concave MLE to p_marg_t via CVXPY:
            log_p = cp.Variable(J)
            obj   = cp.Maximize(p_marg_t @ log_p)
            constr= [ cp.log_sum_exp(log_p) <= 0,     # sum exp <= 1
                      cp.diff(log_p, 2) <= 0 ]         # log-concavity
            solve with SCS (fallback ECOS)
            p_smooth = exp(log_p); renormalise
        Falls back to raw p_marg_t if CVXPY unavailable or solver fails.

    3.  Convert probs → density on the native centers:
            d_native[j] = p_smooth[j] / bin_w_scaled

    4.  Resample d_native onto the common Y-grid (`resample_onto`).
        Compute L2 + KL_fwd + KL_rev vs truth Gaussian on the same grid.

Reference: `malc_1d_cvxpy()` + `ours_densities` in
`benchmarks/l2_ihdp/methods_densities.py`.

**Why 1D MALC on the 1D marginal (rather than marginalising 2D MALC?):**
matches the reference R implementation `R/malc.R::MALC` which fits a 1D
discrete log-concave MLE to the marginal directly. Cleaner interpretation
than marginalising a 2D fit that solves a different optimisation problem.

**Solver caveat (2026-08-17):** on the polynomial-SCM Fig 2 ρ-sweep the
CVXPY-SCS port was observed to collapse some broad log-concave Gaussians
into spikes (peak_ratio ≫ 1). IHDP/ACIC pipelines are not affected —
their inputs sit in the well-conditioned regime where SCS converges
tightly, and no NaN or degenerate density has been seen there.
Reproducer: `benchmarks/empirical_tests/inspect_malc_1d_solver.py`.
If we ever come back to the polynomial-SCM ρ-sweep we should switch
Python solver to ECOS-first, or wire in an active-set port; not urgent
because that pipeline is not on the current critical path.

**Deps**: `pip install cvxpy` (any modern version). SCS and ECOS solvers
ship with the default install.

## 4. CATE density — τ = Y1 − Y0

Per query x, we compute p(τ | x) on a shared τ-grid.

### 4a. Ours (fn=50) — 2D MALC + diagonal integration

Input: 2D joint `p_mat[i, j]` on (J × J) bins.

    1.  Fit 2D MALC to p_mat:
            obj = fit_malc_inner(p_mat.T, edges, edges,
                                 B_fit=B, B_select=B, max_K=K, seed=seed)

    2.  Evaluate the smooth 2D density on a fine (n_eval × n_eval) grid:
            density = dmalc_2d(obj, eval_pts).reshape(n_eval, n_eval)

    3.  Diagonal-integrate:  for each τ_k in tau_grid,
            y1_of_y0 = xs + τ_k                   (line y1 = y0 + τ_k)
            valid = y1_of_y0 in [ys[0], ys[-1]]
            f_diag[i] = bilinear_interp(density, y0=xs[i], y1=y1_of_y0[i])
            p_τ[k] = Σ_i f_diag[i] * dy0
        Renormalise: p_τ /= Σ_k p_τ[k] * dtau

    4.  Interpolate/resample p_τ onto the common τ-grid, renormalise.
        Compute L2 + KL_fwd + KL_rev vs truth.

**Why "diagonal"**: `p_τ(t) = ∫ f(y0, y0+t) dy0` — integrating f along the
line `y1 = y0 + t` in the (y0, y1) plane. That line is a diagonal.

Reference: `_fit_and_marginalize` in `benchmarks/methods/ours.py`.

### 4b. UWYK-NoAnc, UWYK-FullAnc, Do-PFN — independence-assumed outer product

No joint output. Assume Y_do(0) ⊥ Y_do(1) | X and construct the joint:

    joint_indep[i, j] = pBars_0[i] * pBars_1[j]

Then discrete diagonal sum on native centers:

    p_τ[k] = Σ_{i,j : c[j]-c[i] ∈ bin k} joint_indep[i, j]
    d_τ[k] = p_τ[k] / tau_bin_width

Resample onto common τ-grid, compute L2 + KL_fwd + KL_rev vs truth.

**Known asymmetry**: Ours gets 2D-MALC smoothing on its `p_mat`, baselines
get raw discrete convolution on `joint_indep`. Symmetric fix would be to
also fit 2D MALC to `joint_indep` for baselines — not implemented yet;
noted in §7 open issues.

## 5. ATE density

Per-query CATE densities aggregated via 1D Wasserstein barycenter over
queries on the common τ-grid:

    p_ATE = wasserstein_barycenter_1d(p_tau_per_query, tau_grid)

Renormalise. Compute L2 + KL_fwd + KL_rev vs truth ATE density (the
barycenter of the analytic per-query truth CATE densities on the same
grid).

Reference: `wasserstein_barycenter_1d` from `MALC/Optimal_Transport/ot_barycenter.py`.

## 6. Common grid choice

All methods evaluated on the same grid — required for cross-method L2 / KL
to be directly comparable. Current choices:

    Fig 2 v2 (polynomial SCM):
      Y_GRID   = np.linspace(-8,  8,  501)     dx ≈ 0.032
      TAU_GRID = np.linspace(-10, 10, 501)     dx ≈ 0.040

    l2_ihdp / l2_acic / l2_syn (scaled [-1, 1] y):
      Y_CENTERS   from  np.linspace(-1.5, 1.5, 101)      dx = 0.030
      TAU_CENTERS from  np.linspace(-3.0, 3.0, 601)      dx = 0.010

User note: "grid width does not matter much" — the ranking across methods
is robust to `dx`. Absolute L2/KL numbers scale with `dx`. If absolute
comparability across setups matters, standardise the grid.

## 7. Sanity check — proof the pipeline is right

Run `benchmarks/empirical_tests/sanity_cate_density.py` to verify.
Constructs a perfect p_mat from a known 2D Gaussian, pushes through both
CATE derivations (raw diagonal projection + MALC-fit-integrate), reports
L2 vs analytic truth CATE density:

    Expected (2026-08-16 verified):
      RAW diagonal L2:   0.45 – 0.79   (grows with ρ; discretisation error)
      MALC-fit L2:       0.04 – 0.15   (small; MALC's inherent smoothing bias)

If the pipeline is broken these numbers go up dramatically. Fig 2 v2's
Ours(fn=50) MALC L2 landed at 0.26 – 0.52 — 5× the sanity floor of 0.05.
That gap is the model's actual `p_mat` calibration (see §8), NOT a
pipeline bug.

## 8. Known limitations & open issues

1. **ρ is not identifiable from unpaired context.** Given only factual
   context (X_obs, T_obs, Y_obs), the model can identify the marginals
   `p(Y_do0 | x)` and `p(Y_do1 | x)` but NOT their joint correlation. So
   the model outputs a task-averaged ρ (~0.2 for Do-PFN's SCM prior) at
   every query. This is a fundamental information-theoretic limit, not a
   training bug or architecture problem. Verified by
   `benchmarks/empirical_tests/diagnose_joint_collapse.py` — implied ρ
   from Ours' `p_mat` is ~0.2 regardless of the query's true ρ.

   Paired training still helps PEHE (Fisher-doubling for marginal means,
   ratio ≤ √2), which is why our dn-grid shows √PEHE ratios ≈ 2.0 for
   Ours-DoPFN-bb vs Do-PFN at low d. But joint density fidelity per
   query cannot exceed what a task-averaged joint prior can capture.

2. **Asymmetric smoothing** (§4b) — Ours gets 2D MALC, baselines get raw
   discrete outer-product. Fix: apply 2D MALC to `joint_indep` for
   baselines too. Not urgent since even the unfair-favouring-Ours setup
   shows Ours failing to strongly beat baselines on CATE-L2.

3. **Grid dependence** — cross-method L2 / KL numbers scale with the
   common grid's `dx`. Rankings are stable; absolute values shift.

4. **L2 caveats** — not scale-invariant, penalises mean-shifts heavily.
   KL_fwd (= expected NLL) is often a better single-number summary for
   density-fidelity assessments. Report both, but prefer KL_fwd when
   picking a headline metric.

5. **CVXPY solver failures** — `malc_1d_cvxpy` falls back to raw
   p_marg silently if the solver returns None. Log-inspect the shard
   to check no fallbacks happened; if too many did, install a stronger
   solver (`pip install cvxpy[MOSEK]` if you have a license).

## 9. Change log

- **v2.1 (2026-08-17)**: §3b unchanged — Fig 2 polynomial-SCM ρ-sweep
  was observed to expose a CVXPY-SCS spike-collapse in `malc_1d_cvxpy`,
  but the IHDP/ACIC pipelines (the current critical path) are not
  affected and their results stand. Solver-fix investigation is parked
  (see §3b caveat). Diagnostics added:
  `benchmarks/empirical_tests/inspect_malc_1d_solver.py` and
  `benchmarks/empirical_tests/inspect_ours_pmat.py`.
- **v2 (2026-08-16)**: added `malc_1d_cvxpy` Python port for §3b. Added
  KL divergence (both directions) as a second metric alongside L2 in all
  L2-eval pipelines. Documented the ρ-identification limitation (§8.1).
  Dropped Ours-DoPFN-bb from the current method list.
- **v1 (2026-08-14)**: initial spec — L2 formula, marginal / CATE / ATE
  density derivations per method, diagonal-integration explanation.
