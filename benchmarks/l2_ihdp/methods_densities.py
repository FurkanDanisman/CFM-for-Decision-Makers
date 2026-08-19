"""Per-method density adapters for the IHDP L2 evaluation.

Every adapter returns, for a single realization, three arrays shaped
(n_queries, len(grid)):

    p_y0[i, :]  density of Y_do(0) | X_i  on true_ihdp.Y_CENTERS
    p_y1[i, :]  density of Y_do(1) | X_i  on true_ihdp.Y_CENTERS
    p_tau[i, :] density of tau     | X_i  on true_ihdp.TAU_CENTERS

Adapters implemented:
    ours_densities(cd, model, ckpt_meta, args)  — joint head + MALC + diag integration
    uwyk_noanc_densities(cd, uwyk_model, args)  — sample -> histogram + independence convolution
    dopfn_densities(cd, DoPFNRegressor, args)   — predict_full logits/borders + resample + independence conv

All adapters use the SAME rescaling (y_min, y_rng from training) as
plot_ihdp_n10*.py so their outputs share a coordinate frame with
true_ihdp.load_ihdp_truth().

Independence convolution for baselines: given per-arm marginals p0, p1 on the
common Y_CENTERS grid,
    p_tau(t) = sum_{y0} p0(y0) * p1(y0 + t) * dy   ~= diagonal sums of outer.
"""
from __future__ import annotations

import hashlib
import os
import sys
from typing import Any, Callable

import numpy as np
import torch

from l2 import resample_onto
from true_ihdp import TAU_BIN, TAU_CENTERS, Y_BIN, Y_CENTERS


# ── Independence-convolution helper (shared by UWYK-NoAnc and DoPFN) ─────
def naive_p_tau_from_marginals(p_y0: np.ndarray, p_y1: np.ndarray) -> np.ndarray:
    """CATE density under independence: p(tau) = ∫ p1(y0 + tau) p0(y0) dy0.

    Inputs on Y_CENTERS (uniform). Output on TAU_CENTERS.

    Uses np.correlate (equivalent to np.outer + diagonal sums, but with the
    correct sign convention: for outer[i, j] = p1[i] * p0[j] and
    trace(offset=off) = sum_k p1[k] * p0[k + off], each summand corresponds
    to tau = Y_CENTERS[k] - Y_CENTERS[k+off] = -off * Y_BIN. So the trace
    at offset off contributes to tau = -off * dY, not +off * dY. The
    sibling script plot_ihdp_n10_uwyk_noanc.py::_naive_p_tau uses the
    wrong sign convention (tau = +off * dY) and produces a mirrored p(tau).
    """
    p_y0 = np.asarray(p_y0, dtype=np.float64).reshape(-1)
    p_y1 = np.asarray(p_y1, dtype=np.float64).reshape(-1)
    n = p_y0.shape[0]
    # correlate(p1, p0)[k] = sum_i p1[i + (k - (n-1))] * p0[i]
    #   lag = k - (n-1)  <->  tau = lag * Y_BIN
    density = np.correlate(p_y1, p_y0, mode='full')            # length 2n-1
    tau_native = np.arange(-(n - 1), n) * Y_BIN
    out = np.interp(TAU_CENTERS, tau_native, density, left=0.0, right=0.0)
    s = out.sum() * TAU_BIN
    if s > 0:
        out = out / s
    return out


# Cache one CVXPY problem instance per J. Building a CVXPY Problem does
# a ~5-10s symbolic canonicalisation for J≈50; re-using the same Problem
# with a Parameter for p_marg lets SCS/ECOS reuse the compiled representation.
# Without this cache, 20 calls per SCM (2 arms × 10 queries) cost ~200s.
_MALC_1D_CACHE: dict = {}


def _get_malc_1d_problem(J: int):
    """Return (problem, p_param, log_p_var) for length-J MLE, cached per J."""
    entry = _MALC_1D_CACHE.get(J)
    if entry is not None:
        return entry
    import cvxpy as cp
    p_param = cp.Parameter(J, nonneg=True)
    log_p = cp.Variable(J)
    prob = cp.Problem(
        cp.Maximize(p_param @ log_p),
        [cp.log_sum_exp(log_p) <= 0, cp.diff(log_p, 2) <= 0],
    )
    entry = (prob, p_param, log_p)
    _MALC_1D_CACHE[J] = entry
    return entry


def evaluate_malc_1d_on_grid(p_malc, src_centers, dst_grid, dst_bin=None):
    """Evaluate a length-J log-concave discrete MLE at arbitrary points via
    log-linear interpolation — the canonical way to lift the discrete MLE to
    a continuous density (see R's logcondens::evaluateLogConDens).

    The discrete log-concave MLE fits log p at J bin centers. The
    corresponding CONTINUOUS log-concave density is uniquely determined:
    log p is piecewise-linear between the J fit points, and exp(log p) is a
    valid density on the support interval.

    Linear interpolation IN PROBABILITY (np.interp on p_malc) does NOT
    preserve log-concavity and gives a piecewise-linear staircase between
    the J bars — which is what we want to avoid, especially for small J.

    Args:
      p_malc     : (J,) length-J discrete probabilities from malc_1d_cvxpy
      src_centers: (J,) bin centers where p_malc lives
      dst_grid   : (M,) target evaluation grid (uniform assumed)
      dst_bin    : bin width of dst_grid (for normalisation); auto if None

    Returns:
      (M,) density on dst_grid, normalised to integrate to 1 on dst_grid,
      zero outside [src_centers[0], src_centers[-1]].
    """
    p_malc = np.asarray(p_malc, dtype=np.float64).reshape(-1)
    src_centers = np.asarray(src_centers, dtype=np.float64).reshape(-1)
    dst_grid = np.asarray(dst_grid, dtype=np.float64).reshape(-1)
    if dst_bin is None:
        dst_bin = float(dst_grid[1] - dst_grid[0])

    # Convert discrete probs to log-density values at src_centers so the
    # interpolation lives on the density scale (constant offset — doesn't
    # matter since we renormalise, but keeps semantics clean).
    src_bin = float(src_centers[1] - src_centers[0])
    d_src = p_malc / max(src_bin, 1e-12)
    # Log-linear interpolation. Guard zeros so log stays finite; anything
    # that maps to log(-inf) at a fit point stays -inf under linear interp
    # in log space, so mass-zero regions remain mass-zero.
    with np.errstate(divide='ignore'):
        log_d_src = np.log(d_src)
    finite = np.isfinite(log_d_src)
    if not finite.any():
        return np.zeros_like(dst_grid)
    # For log-linear interp, np.interp on log-density values, then exp.
    # Points where BOTH neighbours are -inf stay at -inf → 0 after exp.
    # Points where one neighbour is -inf will spike; safer to fall back to
    # linear-in-prob at those points. In practice log-concave MLE rarely
    # produces exact zeros in the interior, so this is edge-case cleanup.
    log_d_safe = np.where(finite, log_d_src, -1e18)
    log_d_dst = np.interp(dst_grid, src_centers, log_d_safe,
                          left=-1e18, right=-1e18)
    d_dst = np.exp(log_d_dst)
    # Zero out beyond src support (np.interp doesn't, since our sentinel is
    # -1e18 not -inf; exp(-1e18)=0 numerically but be explicit).
    outside = (dst_grid < src_centers[0]) | (dst_grid > src_centers[-1])
    d_dst[outside] = 0.0
    total = float(d_dst.sum() * dst_bin)
    if total > 0:
        d_dst = d_dst / total
    return d_dst


def malc_1d_cvxpy(p_marg, solver=None):
    """Fit a 1D discrete log-concave MLE to a length-J marginal p_marg.

    Solves the convex program (Rufibach-Duembgen MLE reformulated):

        max_{log_p}   p_marg @ log_p
        subject to    logsumexp(log_p) ≤ 0        (sum exp(log_p) ≤ 1)
                      diff(log_p, 2) ≤ 0          (log-concavity)

    Returns discrete probabilities of the same shape as p_marg, normalised.
    Falls back to the raw p_marg if CVXPY / solver is unavailable or fails.

    Reference: R implementation in /Users/furkandanisman/DensOLog_VS/R/malc.R
    (which uses logcondiscr::logConDiscrMLE — same optimisation problem).
    """
    p_marg = np.asarray(p_marg, dtype=np.float64).reshape(-1)
    s = p_marg.sum()
    if s <= 0:
        return p_marg.copy()
    p_marg = p_marg / s  # normalise input

    try:
        import cvxpy as cp  # noqa: F401 — probe availability
    except ImportError:
        # cvxpy not installed — return raw. Downstream will still normalise.
        return p_marg

    J = int(p_marg.shape[0])
    prob, p_param, log_p = _get_malc_1d_problem(J)
    p_param.value = p_marg
    solved = False
    for slv in ([solver] if solver else ['SCS', 'ECOS']):
        try:
            prob.solve(solver=slv, verbose=False, warm_start=True)
            if log_p.value is not None:
                solved = True; break
        except Exception:
            continue
    if not solved or log_p.value is None:
        return p_marg
    lp = np.asarray(log_p.value, dtype=np.float64)
    # Numerical safety — solver can return values > 0 that would blow up exp;
    # subtract max BEFORE exp (softmax trick), then normalise.
    if not np.all(np.isfinite(lp)):
        return p_marg
    lp = lp - float(lp.max())
    p_fit = np.exp(lp)
    p_fit = np.clip(p_fit, 0.0, None)
    total = float(p_fit.sum())
    if not np.isfinite(total) or total <= 0:
        return p_marg
    result = p_fit / total
    if not np.all(np.isfinite(result)):
        return p_marg
    return result


def _rescale_and_pad(X: np.ndarray, num_features: int) -> np.ndarray:
    """Pad X with NaN columns up to num_features (matches sibling scripts).

    If num_features is -1 (or 0) the backbone accepts any feature count
    (e.g., DoPFN's PerFeatureTransformer): return X unchanged.
    """
    if num_features is None or num_features <= 0:
        return X.astype(np.float32)
    d = X.shape[1]
    if d < num_features:
        pad = np.full((X.shape[0], num_features - d), np.nan, dtype=np.float32)
        return np.concatenate([X.astype(np.float32), pad], axis=1)
    return X.astype(np.float32)[:, :num_features]


# ─────────────────────────────────────────────────────────────────────────
# Ours (fn=50 or fn=10) — joint head + 2D MALC + diagonal integration
# ─────────────────────────────────────────────────────────────────────────
def ours_densities(cd,
                    model,
                    edges_np: np.ndarray,
                    J: int,
                    bin_width: float,
                    num_features: int,
                    y_min: float,
                    y_rng: float,
                    malc_B: int = 100,
                    malc_max_K: int = 1,
                    n_eval: int = 200,
                    n_context: int | None = None,
                    fit_malc_inner: Callable[..., Any] = None,
                    dmalc_2d: Callable[..., Any] = None,
                    y_scaling: str = 'min_max',
                    std_target: float = 0.3,
                    marginals_from_2d: bool = False,
                    ) -> dict[str, np.ndarray]:
    """Joint-head forward pass, per-query MALC fit, marginalise + diagonal
    integrate onto the common (Y_CENTERS, TAU_CENTERS) grids.

    fit_malc_inner / dmalc_2d are passed in to avoid a hard import at
    package-import time (the caller supplies them once).
    """
    X_train_full = _np(cd.X_train)
    t_train_full = _np(cd.t_train)
    y_train_full = _np(cd.y_train)
    X_test = _np(cd.X_test)

    N = X_train_full.shape[0] if n_context is None else min(n_context, X_train_full.shape[0])
    X_context = X_train_full[:N].astype(np.float32)
    T_context = t_train_full[:N].astype(np.float32).reshape(-1, 1)
    Y_context = y_train_full[:N].astype(np.float32).reshape(-1, 1)
    # Y rescaling: min_max (default, legacy) or std (matches eval_dopfn_bb_raw's
    # `--y-scaling std`). Both linear, so downstream un-scaling (y_rng/2) still
    # works if callers derive a matching y_rng from the same scheme.
    if y_scaling == 'std':
        _yt = y_train_full.astype(np.float64).reshape(-1)
        _mu = float(_yt.mean())
        _sig = float(_yt.std()) if _yt.size > 1 else 1.0
        _y_scale = max(_sig / max(std_target, 1e-6), 1e-8)
        Y_context = ((Y_context - _mu) / _y_scale).astype(np.float32)
        # override y_min/y_rng so the caller's downstream un-scaling matches.
        # y_min becomes the raw Y value that maps to scaled -1, y_rng is 2·y_scale.
        y_min = _mu - _y_scale
        y_rng = 2.0 * _y_scale
    else:
        Y_context = ((Y_context - y_min) / y_rng * 2.0 - 1.0).astype(np.float32)

    X_context = _rescale_and_pad(X_context, num_features)
    X_test_p = _rescale_and_pad(X_test.astype(np.float32), num_features)

    Xc = torch.from_numpy(X_context).unsqueeze(0)
    Tc = torch.from_numpy(T_context).unsqueeze(0)
    Yc = torch.from_numpy(Y_context).unsqueeze(0)
    Xq = torch.from_numpy(X_test_p).unsqueeze(0)

    from losses.BarDistribution2D import unpack_pred  # cluster-side import
    with torch.no_grad():
        pred = model(Xc, Tc, Yc, Xq)['predictions'][0]           # (n_test, D)

    n_test = X_test_p.shape[0]
    p_mats = np.zeros((n_test, J, J), dtype=np.float32)
    for q in range(n_test):
        p_mat, *_ = unpack_pred(pred[q], J, bin_width)
        p_mats[q] = p_mat.detach().cpu().numpy()

    # Evaluate MALC once per query on a fine (n_eval, n_eval) grid then
    # (a) marginalise -> p_y0, p_y1 on Y_CENTERS and
    # (b) diagonal-integrate -> p_tau on TAU_CENTERS.
    xs = np.linspace(edges_np[0], edges_np[-1], n_eval)
    ys = np.linspace(edges_np[0], edges_np[-1], n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    eval_pts = np.column_stack([XX.ravel(), YY.ravel()])
    dxs = float(xs[1] - xs[0]); dys = float(ys[1] - ys[0])

    p_y0 = np.zeros((n_test, len(Y_CENTERS)), dtype=np.float64)
    p_y1 = np.zeros((n_test, len(Y_CENTERS)), dtype=np.float64)
    p_tau = np.zeros((n_test, len(TAU_CENTERS)), dtype=np.float64)
    # Raw-marginal diagnostic: raw p_mat marginals converted to density and
    # resampled onto Y_CENTERS with no MALC smoothing at all. Used to
    # measure how much MALC helps (or hurts) the y0/y1 densities.
    p_y0_raw = np.zeros((n_test, len(Y_CENTERS)), dtype=np.float64)
    p_y1_raw = np.zeros((n_test, len(Y_CENTERS)), dtype=np.float64)
    # OLD-MALC diagnostic: 1D-MALC-smoothed marginal but with linear-in-prob
    # interp (resample_onto) — reproduces the pre-log-linear behaviour, useful
    # for A/B comparing whether log-linear evaluation actually helped.
    p_y0_malc_old = np.zeros((n_test, len(Y_CENTERS)), dtype=np.float64)
    p_y1_malc_old = np.zeros((n_test, len(Y_CENTERS)), dtype=np.float64)
    # Per-query "raw" CATE = E[Y1] - E[Y0] from raw p_mat marginals (no MALC),
    # in scaled Y units. Reported alongside the MALC-mean CATE so PEHE can be
    # computed both ways (matches Table 3's `ours_mean` variant).
    cate_raw_scaled     = np.zeros(n_test, dtype=np.float64)
    # EM-adjusted MALC-mean CATE — two variants:
    #   K=1    : force MALC to a single log-concave (no mixture over-split)
    #   mix    : use whatever K MALC selected via BIC, weighted by pi_k
    # By convention we fit MALC to p_mat.T so axis-0 (input) = Y1, axis-1 = Y0.
    # _fit_component_2d then computes mu_x = E[Y0], mu_y = E[Y1] via EM →
    # tau = mu_hat[1] - mu_hat[0] per component.
    cate_em_mix_scaled  = np.full(n_test, np.nan, dtype=np.float64)
    cate_em_k1_scaled   = np.full(n_test, np.nan, dtype=np.float64)
    _bin_centers_raw = 0.5 * (edges_np[:-1] + edges_np[1:])                # (J,)

    def _em_tau_from_fit(fit):
        if fit is None:
            return float('nan')
        mu_list = [c.mu_hat for c in fit.fits if c is not None]
        if not mu_list:
            return float('nan')
        mus = np.stack(mu_list)
        weights = np.asarray(fit.pi, dtype=np.float64)[:len(mus)]
        if weights.sum() <= 0:
            return float('nan')
        weights = weights / weights.sum()
        mu_y0 = float((weights * mus[:, 0]).sum())      # E[Y0]
        mu_y1 = float((weights * mus[:, 1]).sum())      # E[Y1]
        return mu_y1 - mu_y0                             # τ = E[Y1] - E[Y0]

    for q in range(n_test):
        # Raw p_mat marginals (no MALC): p_mat[j0, j1] convention has j0=y0, j1=y1
        _pnorm = p_mats[q] / max(p_mats[q].sum(), 1e-12)
        _E_y0 = float((_bin_centers_raw * _pnorm.sum(axis=1)).sum())
        _E_y1 = float((_bin_centers_raw * _pnorm.sum(axis=0)).sum())
        cate_raw_scaled[q] = _E_y1 - _E_y0

        seed = int(hashlib.md5(f'q{q}'.encode()).hexdigest()[:8], 16) % (10 ** 8)
        # MALC's log-concave fitter can raise RuntimeError on certain p_mat
        # shapes (e.g., mass too concentrated to fit a log-concave). Try
        # requested max_K, then a small ladder of retries, then fall back to
        # the raw piecewise-constant p_mat so evaluation of a single query
        # never kills the whole realization.
        density_2d = None
        _tried = []
        fit_mix = None                                              # holds the successful fit
        for _K in {malc_max_K, max(malc_max_K + 1, 2), 3, 1}:
            try:
                fit_mix = fit_malc_inner(p_mats[q].T, edges_np, edges_np,
                                          B_fit=malc_B, B_select=malc_B,
                                          max_K=_K, seed=seed, parallel=False)
                density_2d = dmalc_2d(fit_mix, eval_pts).reshape(n_eval, n_eval)
                break
            except Exception as e:
                _tried.append(f'K={_K}:{type(e).__name__}')
        if fit_mix is not None:
            cate_em_mix_scaled[q] = _em_tau_from_fit(fit_mix)

        # EM K=1 variant — separate fit forced to K=1 for the mean-adjusted
        # single-log-concave estimate. Cheaper than the mixture fit and never
        # over-splits. If K=1 fails on this p_mat we leave it NaN.
        try:
            fit_k1 = fit_malc_inner(p_mats[q].T, edges_np, edges_np,
                                    B_fit=malc_B, B_select=malc_B,
                                    max_K=1, seed=seed, parallel=False)
            cate_em_k1_scaled[q] = _em_tau_from_fit(fit_k1)
        except Exception:
            pass
        if density_2d is None:
            # Fallback: piecewise-constant density from the raw p_mat.
            # Nearest-neighbour lookup onto the fine (xs, ys) grid.
            print(f'[warn] q={q} MALC failed for all K ({" ".join(_tried)}); '
                  f'using raw p_mat fallback', flush=True)
            p_norm = p_mats[q] / max(p_mats[q].sum(), 1e-12)
            j0 = np.clip(np.searchsorted(edges_np, xs) - 1, 0, J - 1)
            j1 = np.clip(np.searchsorted(edges_np, ys) - 1, 0, J - 1)
            # density value = probability_mass / bin_area
            density_2d = (p_norm[np.ix_(j0, j1)] / (bin_width * bin_width)).T

        # Marginals — two variants:
        #   default (marginals_from_2d=False): fit 1D log-concave MLE via CVXPY
        #     on the raw marginals of p_mat. Two independent 1D MALCs.
        #   marginals_from_2d=True: marginalise the 2D-MALC-fitted joint
        #     (density_2d) — same joint that CATE integrates. Only used when
        #     the 2D fit succeeded (fit_mix is not None); falls back to 1D
        #     MALC otherwise so the raw-fallback path never regresses.
        # Convention: density_2d[row, col] is the density at (xs[col], ys[row]),
        # where xs is the y0 grid and ys is the y1 grid — matches the diagonal-
        # integration convention below (col = y0 index, row = y1 index).
        centers_raw_scaled = 0.5 * (edges_np[:-1] + edges_np[1:])          # (J,)
        bin_w_scaled = float(edges_np[1] - edges_np[0])
        # Raw-marginal diagnostic — no MALC, no log-linear interp, just
        # divide by bin width and resample. Always computed alongside
        # whichever main marginal path (1D-MALC or 2D-marginalise) is in
        # use, so we can compare per-realization L2s.
        _p_marg_y0_raw = p_mats[q].sum(axis=1)                             # (J,)
        _p_marg_y1_raw = p_mats[q].sum(axis=0)                             # (J,)
        _d_y0_raw_native = _p_marg_y0_raw / max(bin_w_scaled, 1e-12)
        _d_y1_raw_native = _p_marg_y1_raw / max(bin_w_scaled, 1e-12)
        p_y0_raw[q] = resample_onto(centers_raw_scaled, _d_y0_raw_native, Y_CENTERS)
        p_y1_raw[q] = resample_onto(centers_raw_scaled, _d_y1_raw_native, Y_CENTERS)
        if marginals_from_2d and fit_mix is not None:
            d_y0_native = density_2d.sum(axis=0) * dys                     # (n_eval,) on xs
            d_y1_native = density_2d.sum(axis=1) * dxs                     # (n_eval,) on ys
            p_y0[q] = resample_onto(xs, d_y0_native, Y_CENTERS)
            p_y1[q] = resample_onto(ys, d_y1_native, Y_CENTERS)
        else:
            # p_mat convention: p_mat[j0, j1] (j0 = y0 axis, j1 = y1 axis)
            p_marg_y0_raw = p_mats[q].sum(axis=1)                          # (J,)
            p_marg_y1_raw = p_mats[q].sum(axis=0)                          # (J,)
            p_marg_y0_malc = malc_1d_cvxpy(p_marg_y0_raw)
            p_marg_y1_malc = malc_1d_cvxpy(p_marg_y1_raw)
            # LOGLIN (default main output): evaluate the continuous log-
            # concave MLE directly on Y_CENTERS via log-linear interpolation.
            p_y0[q] = evaluate_malc_1d_on_grid(p_marg_y0_malc,
                                                centers_raw_scaled, Y_CENTERS,
                                                dst_bin=Y_BIN)
            p_y1[q] = evaluate_malc_1d_on_grid(p_marg_y1_malc,
                                                centers_raw_scaled, Y_CENTERS,
                                                dst_bin=Y_BIN)
            # OLD-MALC (diagnostic): same MALC fit, but resample_onto
            # (linear-in-prob) instead of log-linear. Reproduces pre-fix
            # behaviour; no extra MALC compute cost.
            _d_y0_native = p_marg_y0_malc / max(bin_w_scaled, 1e-12)
            _d_y1_native = p_marg_y1_malc / max(bin_w_scaled, 1e-12)
            p_y0_malc_old[q] = resample_onto(centers_raw_scaled, _d_y0_native, Y_CENTERS)
            p_y1_malc_old[q] = resample_onto(centers_raw_scaled, _d_y1_native, Y_CENTERS)

        # CATE via diagonal integration.
        # For each tau, integrate p(y0, y0+tau) over y0. y0 runs over xs
        # (the fine grid); we look up the density_2d value at that y0
        # column, and bilinearly interpolate in y1 = y0 + tau across rows.
        #
        # NOTE: earlier revisions (and the sibling plot_ihdp_n10.py) used
        #     col = np.clip(np.searchsorted(xs, xs[valid]) - 1, 0, len(xs)-1)
        # which is off-by-one for exact matches: searchsorted(side='left')
        # returns k for xs[valid][i] == xs[k], so -1 gives k-1 (wrong).
        # For smooth densities this shifts sampling by dxs ~ 1.5%; for
        # spiky joint outputs (fn=10 on hard queries) it can miss the peak
        # column entirely. Corrected form: use searchsorted output directly.
        p_tau_native = np.zeros(len(TAU_CENTERS))
        for k, t in enumerate(TAU_CENTERS):
            y1_target = xs + t
            valid = (y1_target >= ys[0]) & (y1_target <= ys[-1])
            if not np.any(valid):
                continue
            col = np.clip(np.searchsorted(xs, xs[valid]), 0, len(xs) - 1)
            row_f = (y1_target[valid] - ys[0]) / dys
            row_lo = np.clip(np.floor(row_f).astype(int), 0, len(ys) - 2)
            row_hi = row_lo + 1
            w_hi = row_f - row_lo
            w_lo = 1.0 - w_hi
            f_diag = w_lo * density_2d[row_lo, col] + w_hi * density_2d[row_hi, col]
            p_tau_native[k] = f_diag.sum() * dxs
        s = p_tau_native.sum() * TAU_BIN
        if s > 0:
            p_tau_native = p_tau_native / s
        p_tau[q] = p_tau_native

    return dict(p_y0=p_y0, p_y1=p_y1, p_tau=p_tau,
                p_y0_raw=p_y0_raw, p_y1_raw=p_y1_raw,
                p_y0_malc_old=p_y0_malc_old, p_y1_malc_old=p_y1_malc_old,
                cate_raw_scaled=cate_raw_scaled,
                cate_em_mix_scaled=cate_em_mix_scaled,
                cate_em_k1_scaled=cate_em_k1_scaled)


def _uwyk_capture_raw_preds(uwyk_model, X_ctx, T_ctx, Y_ctx, X_qry, T_intv, adj):
    """Capture raw_preds from the BarDistribution head by monkey-patching
    uwyk_model.bar_distribution.mode. Fires a single wrapper.predict call
    with prediction_type='mode'; the mode value is discarded — we take the
    raw K+4 tensor before postprocessing."""
    captured = {'raw': None}
    orig_mode = uwyk_model.bar_distribution.mode
    def _mode_capture(raw_preds):
        captured['raw'] = raw_preds.detach().cpu()
        return orig_mode(raw_preds)
    uwyk_model.bar_distribution.mode = _mode_capture
    try:
        _ = uwyk_model.predict(
            X_obs=X_ctx, T_obs=T_ctx, Y_obs=Y_ctx,
            X_intv=X_qry, T_intv=T_intv,
            adjacency_matrix=adj,
            prediction_type='mode', inverse_transform=False,
        )
    finally:
        uwyk_model.bar_distribution.mode = orig_mode
    return captured['raw']


def _uwyk_raw_bar_probs(uwyk_model, X_ctx, T_ctx, Y_ctx, X_qry, T_intv, adj, n_test):
    """Returns (n_test, K) discrete bar probabilities on UWYK's INTERIOR K
    centers (scaled y space [-1, 1]). Softmax over K+2 logits, then trim
    tails, then renormalise to unit mass."""
    raw = _uwyk_capture_raw_preds(uwyk_model, X_ctx, T_ctx, Y_ctx, X_qry, T_intv, adj)
    # raw shape: (1, n_test_padded, K+4). First K+2 are logits [pL, bars..., pR].
    w_logits = raw[..., :-2]
    probs = torch.softmax(w_logits, dim=-1)
    pBars = probs[..., 1:-1].squeeze(0).numpy()             # (n_test_padded, K)
    pBars = pBars[:n_test]                                    # trim padding
    s = pBars.sum(axis=1, keepdims=True)
    return pBars / np.maximum(s, 1e-12)


def _uwyk_densities_from_raw_probs(cd,
                                     uwyk_model,
                                     num_features: int,
                                     y_min: float,        # unused (scaled space)
                                     y_rng: float,        # unused
                                     n_context: int | None,
                                     adjacency_kind: str,
                                     ) -> dict[str, np.ndarray]:
    """Shared UWYK density computation. adjacency_kind ∈ {'noanc', 'anc'}."""
    X_train_full = _np(cd.X_train)
    t_train_full = _np(cd.t_train)
    y_train_full = _np(cd.y_train)
    X_test = _np(cd.X_test).astype(np.float32)

    N = X_train_full.shape[0] if n_context is None else min(n_context, X_train_full.shape[0])
    X_context = X_train_full[:N].astype(np.float32)
    t_train_orig = t_train_full[:N].astype(np.float32).reshape(-1, 1)
    y_train_ctx = y_train_full[:N].astype(np.float32).reshape(-1, 1)

    X_train_p = _rescale_and_pad(X_context, num_features)
    X_test_p  = _rescale_and_pad(X_test, num_features)

    mean_y_t0 = float(y_train_ctx[t_train_orig == 0].mean())
    mean_y_t1 = float(y_train_ctx[t_train_orig == 1].mean())
    t_train_enc = np.where(t_train_orig == 0, mean_y_t0, mean_y_t1).astype(np.float32)
    uwyk_model.fit(X_train_p, t_train_enc, y_train_ctx)

    # Adjacency: noanc = features connected, padded masked. anc adds
    # T→Y, X→T, X→Y edges for real features. Clamp n_real to num_features
    # because _rescale_and_pad truncates X to num_features columns
    # (X_context can have more original features than UWYK's model cap).
    n_real = min(X_context.shape[1], num_features)
    adj = np.zeros((num_features + 2, num_features + 2), dtype=np.float32)
    if adjacency_kind == 'anc':
        T_idx, Y_idx, off = 0, 1, 2
        adj[T_idx, Y_idx] = 1.0
        for i in range(n_real):
            adj[off + i, T_idx] = 1.0
            adj[off + i, Y_idx] = 1.0
    for i in range(n_real, num_features):
        fi = 2 + i
        adj[fi, :] = -1.0; adj[:, fi] = -1.0; adj[fi, fi] = -1.0

    n_test = X_test_p.shape[0]
    T_intv_0 = np.full((n_test, 1), mean_y_t0, dtype=np.float32)
    T_intv_1 = np.full((n_test, 1), mean_y_t1, dtype=np.float32)

    # Raw bar probabilities on UWYK's own K interior centers (scaled space).
    pBars_0 = _uwyk_raw_bar_probs(uwyk_model, X_train_p, t_train_enc, y_train_ctx,
                                    X_test_p, T_intv_0, adj, n_test)
    pBars_1 = _uwyk_raw_bar_probs(uwyk_model, X_train_p, t_train_enc, y_train_ctx,
                                    X_test_p, T_intv_1, adj, n_test)
    centers_scaled = uwyk_model.bar_distribution.centers.detach().cpu().numpy().astype(np.float64)
    bin_w = float(centers_scaled[1] - centers_scaled[0])

    # Convert probs → density on UWYK's native centers, then resample to Y_CENTERS.
    density_0_native = pBars_0 / bin_w                        # (n_test, K)
    density_1_native = pBars_1 / bin_w
    p_y0 = np.zeros((n_test, len(Y_CENTERS)), dtype=np.float64)
    p_y1 = np.zeros((n_test, len(Y_CENTERS)), dtype=np.float64)
    for q in range(n_test):
        p_y0[q] = resample_onto(centers_scaled, density_0_native[q], Y_CENTERS)
        p_y1[q] = resample_onto(centers_scaled, density_1_native[q], Y_CENTERS)

    # CATE under independence assumption via naive convolution on Y_CENTERS.
    p_tau = np.zeros((n_test, len(TAU_CENTERS)), dtype=np.float64)
    for q in range(n_test):
        p_tau[q] = naive_p_tau_from_marginals(p_y0[q], p_y1[q])
    return dict(p_y0=p_y0, p_y1=p_y1, p_tau=p_tau)


# ─────────────────────────────────────────────────────────────────────────
# UWYK No-Ancestral — raw BarDist probs + independence convolution
# ─────────────────────────────────────────────────────────────────────────
def uwyk_noanc_densities(cd,
                          uwyk_model,
                          num_features: int,
                          y_min: float,
                          y_rng: float,
                          n_context: int | None = None,
                          n_samples: int | None = None,   # kept for backward-compat, unused
                          ) -> dict[str, np.ndarray]:
    """UWYK-NoAnc raw bar probabilities via BarDist monkey-patch, then
    resample to Y_CENTERS; CATE under independence."""
    return _uwyk_densities_from_raw_probs(cd, uwyk_model, num_features,
                                            y_min, y_rng, n_context,
                                            adjacency_kind='noanc')


# ─────────────────────────────────────────────────────────────────────────
# UWYK Full-Ancestral — raw BarDist probs with full-graph adjacency
# ─────────────────────────────────────────────────────────────────────────
def uwyk_anc_densities(cd,
                        uwyk_model,
                        num_features: int,
                        y_min: float,
                        y_rng: float,
                        n_context: int | None = None,
                        n_samples: int | None = None,   # kept for backward-compat, unused
                        ) -> dict[str, np.ndarray]:
    """UWYK Full-Ancestral: raw bar probabilities via BarDist monkey-patch,
    full-graph adjacency, resample to Y_CENTERS, independence convolution."""
    return _uwyk_densities_from_raw_probs(cd, uwyk_model, num_features,
                                            y_min, y_rng, n_context,
                                            adjacency_kind='anc')


# ─────────────────────────────────────────────────────────────────────────
# Do-PFN — predict_full logits/borders + independence convolution
# ─────────────────────────────────────────────────────────────────────────
def dopfn_densities(cd,
                     DoPFNRegressor,
                     y_min: float,
                     y_rng: float,
                     dopfn_root: str,
                     n_context: int | None = None,
                     ) -> dict[str, np.ndarray]:
    """DoPFN per-arm density via predict_full()['logits'] over criterion.borders.

    Do-PFN convention (from benchmarks/methods/dopfn.py + upstream base.py):
    the treatment goes in column 0 of the covariate matrix. We call
    predict_full twice, once with column 0 = 0 (control) and once with
    column 0 = 1 (treatment); softmax(logits)/bar_width gives a density on
    criterion.borders, which we then resample onto Y_CENTERS.

    predict_full returns a dict with 'logits' (n_test, num_bars),
    'criterion' (a FullSupportBarDistribution with .borders), plus
    mean/median/mode/quantiles.

    DoPFNRegressor.__init__ opens 'artifacts/dopfn_config.pkl' by relative
    path, so we chdir to dopfn_root for the duration of the instantiation
    (matches the existing benchmark pipeline; see submit.sbatch:54).
    """
    X_train = _np(cd.X_train).astype(np.float32)
    t_train = _np(cd.t_train).astype(np.float32).reshape(-1)
    y_train = _np(cd.y_train).astype(np.float32).reshape(-1)
    X_test = _np(cd.X_test).astype(np.float32)

    N = X_train.shape[0] if n_context is None else min(n_context, X_train.shape[0])
    X_train_ctx = X_train[:N]
    t_train_ctx = t_train[:N]
    y_train_ctx = y_train[:N]

    # DoPFN convention (mirrors methods/dopfn.py::dopfn_pipeline). Both
    # __init__ and fit() open files with relative paths (artifacts/*.pkl,
    # artifacts/*.cpkt), so CWD must stay at dopfn_root through both calls
    # and through predict_full below (safest to keep it for the whole block).
    x_tr = np.concatenate([t_train_ctx[:, None], X_train_ctx], axis=1)
    x_te = np.concatenate([np.zeros((X_test.shape[0], 1), dtype=np.float32), X_test], axis=1)
    n_test = X_test.shape[0]

    _cwd = os.getcwd()
    try:
        os.chdir(dopfn_root)
        reg = DoPFNRegressor()
        reg.fit(torch.tensor(x_tr), torch.tensor(y_train_ctx))

        def _arm_density(arm: int) -> np.ndarray:
            x = x_te.copy()
            x[:, 0] = float(arm)
            full = reg.predict_full(torch.tensor(x))
            logits = np.asarray(full['logits'])                     # (n_test, num_bars)
            borders = np.asarray(full['criterion'].borders.cpu())    # (num_bars + 1,) in raw Y units
            bar_widths = np.diff(borders)                            # (num_bars,)
            # Softmax + convert to density on the (raw-Y) bar centres
            z = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(z); probs /= probs.sum(axis=1, keepdims=True)
            density_raw = probs / bar_widths[None, :]                # (n_test, num_bars)
            bar_centres_raw = 0.5 * (borders[:-1] + borders[1:])
            # Rescale border centres to [-1, 1] Y frame
            bar_centres_scaled = (bar_centres_raw - y_min) / y_rng * 2.0 - 1.0
            # Density transforms as (dy_raw / dy_scaled) = y_rng / 2
            density_scaled = density_raw * (y_rng / 2.0)
            out = np.zeros((n_test, len(Y_CENTERS)), dtype=np.float64)
            for q in range(n_test):
                out[q] = resample_onto(bar_centres_scaled, density_scaled[q], Y_CENTERS)
            return out

        p_y0 = _arm_density(0)
        p_y1 = _arm_density(1)
    finally:
        os.chdir(_cwd)

    p_tau = np.zeros((n_test, len(TAU_CENTERS)), dtype=np.float64)
    for q in range(n_test):
        p_tau[q] = naive_p_tau_from_marginals(p_y0[q], p_y1[q])
    return dict(p_y0=p_y0, p_y1=p_y1, p_tau=p_tau)


def _np(a):
    if isinstance(a, torch.Tensor):
        return a.numpy()
    return np.asarray(a)
