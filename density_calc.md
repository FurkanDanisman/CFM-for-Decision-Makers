# Density and L2 calculation — agreed methodology

**Status**: agreed 2026-08-16. Read this before implementing/reviewing any
density-L2 comparison. Cross-reference:
`~/.claude/projects/-Users-furkandanisman-R-PFN/memory/feedback_density_and_l2_calculation.md`

Reference implementations (do not deviate without updating this doc):
- `benchmarks/empirical_tests/fig2_pehe_l2.py` — Fig 2 v2
- `benchmarks/l2_ihdp/methods_densities.py` — IHDP / ACIC / syn L2
- `benchmarks/methods/ours.py::_fit_and_marginalize` — 2D-MALC CATE integration
- `MALC_2D/MALC_2D_Algorithm.R` — reference 2D MALC (log-concave 2D fit + `dmalc_2d` evaluator)
- `R/malc.R` — reference 1D MALC (log-concave 1D fit + `dmalc` evaluator)
- `R/density-calc.R::Eval` — reference L2 formula

## 1. What "density" means for a discrete BarDist head

Each model outputs probability MASS on K (or J) bins with uniform width `w`:

    p_1, p_2, ..., p_K       Σ p_i = 1

The piecewise-constant density at bin i's center is:

    d_i = p_i / w            (mass / length = density)

L2 between two densities on a common evaluation grid with spacing `dx`:

    L2(f, g) = sqrt( Σ_k (f[k] − g[k])² · dx )       # Riemann integral

Both f and g must integrate to 1 on the common grid. Renormalise after any
interpolation.

## 2. Marginal density (Y_do(0), Y_do(1))

Per method, per arm t ∈ {0, 1}:

- **UWYK, Do-PFN**: `pBars_t[i]` directly from BarDist head softmax; then
  `d_t[i] = pBars_t[i] / w_native`, resample onto common Y-grid, L2 vs
  truth Gaussian evaluated on the same grid.

- **Ours (fn=50, DoPFN-bb)**: marginal from the 2D joint,
  `p_marg_t[i] = Σ_other p_mat[i, j]`. Then **fit 1D MALC to `p_marg_t`**
  (log-concave fit on the length-J vector). Evaluate MALC's smooth 1D
  density on the common Y-grid. L2 vs truth.

  1D-MALC entry point: R `logcondens::MALC()` (in `/R/malc.R`). If Python
  port isn't ready, shell out to R. Alternatively, skip MALC on marginals
  (evaluate raw piecewise-constant density) — worse comparison but faster.

## 3. CATE density (τ = Y1 − Y0)

For all methods, per query x, we compute p(τ | x) on a **shared τ-grid**.

Two derivation paths depending on whether the method has a joint output:

### 3a. Ours (fn=50, DoPFN-bb) — MALC-then-integrate

Input: 2D joint `p_mat[i, j]` on (J × J) bins (bin centers `c` in scaled y).

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

    4.  Interpolate/resample p_τ onto the common τ-grid, renormalise,
        compare to truth via L2.

**Why "diagonal"**: p_τ(t) = ∫ f(y0, y0+t) dy0 — integrating f along the
line y1 = y0 + t in the (y0, y1) plane. That line is a diagonal of the
plane.

Reference: `_fit_and_marginalize` in `benchmarks/methods/ours.py`.

### 3b. UWYK, Do-PFN — independence-assumed outer product

No joint output. Assume Y_do(0) ⊥ Y_do(1) | X and construct the joint by
outer product:

    joint_indep[i, j] = pBars_0[i] * pBars_1[j]

Then optionally **fit 2D MALC to joint_indep** and diagonal-integrate as
in 3a (for symmetric smoothing vs Ours), OR do the discrete diagonal sum
directly (mass version):

    Discrete direct:
        p_τ[k] = Σ_{i,j : c[j]-c[i] ∈ bin k} joint_indep[i, j]
        d_τ[k] = p_τ[k] / tau_bin_width

**IMPORTANT** — the fairness question here: for a like-for-like comparison
with Ours' MALC-CATE, we should apply the SAME 2D-MALC-then-integrate to
UWYK/Do-PFN's `joint_indep`. Otherwise Ours gets a smoothing advantage
that isn't reflected in the baselines. Current Fig 2 v2 does NOT do this
symmetric smoothing (uses discrete outer-product for baselines vs
MALC-smoothed for Ours) — flagged as a known asymmetry.

## 4. ATE density

Per-query CATE densities aggregated via 1D Wasserstein barycenter over
queries on the common τ-grid:

    p_ATE = wasserstein_barycenter_1d(p_tau_per_query, tau_grid)

Renormalise, L2 vs truth ATE density (barycenter of the analytic per-query
truth CATE densities).

## 5. Common grid choice

All methods evaluated on the same grid — that's the requirement for L2
values to be directly comparable. Current choice in `fig2_pehe_l2.py`:

    Y_GRID   = np.linspace(-8,  8,  501)     dx ≈ 0.032
    TAU_GRID = np.linspace(-10, 10, 501)     dx ≈ 0.040

User note: "grid width does not matter much" — the ranking is robust, but
the absolute L2 numbers depend on `dx`. If precision matters, use a finer
common grid (at least as fine as UWYK's K=1000 native).

## 6. Sanity check — proof the pipeline is right

Run `benchmarks/empirical_tests/sanity_cate_density.py` (in-repo) to
verify. Constructs perfect p_mat from a known 2D Gaussian, pushes through
both derivations (raw diagonal + MALC-fit-integrate), and reports L2
vs analytic truth:

    Expected (2026-08-16 verified):
      RAW diagonal L2:    0.45 – 0.79   (grows with ρ; discretisation)
      MALC-fit L2:        0.04 – 0.15   (small; MALC bias only)

If the pipeline is broken these numbers go up dramatically. Fig 2 v2's
Ours(fn=50) MALC L2 is 0.26 – 0.52 — 5× the sanity floor of 0.05 — that's
the model's `p_mat` calibration gap, not a pipeline bug.

## 7. Known limitations & open issues

1. **Asymmetric smoothing** (§3b) — Ours gets 2D MALC, baselines get raw
   discrete convolution. Fix: apply 2D MALC to `joint_indep` for
   baselines too.

2. **Marginal-density MALC (§2) not yet implemented in Python** — either
   port `R/malc.R` or shell out to R.

3. **Grid dependence** — cross-method L2 numbers scale with the common
   grid's `dx`. Rankings are stable; absolute values shift.

4. **L2 is not scale-invariant** and heavily penalises mean-shifts. For
   density-fidelity comparisons where localization mismatches matter,
   L2 is defensible but Wasserstein/KL/Hellinger are alternatives.
