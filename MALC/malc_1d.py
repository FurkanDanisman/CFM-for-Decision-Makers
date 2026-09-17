"""MALC_1D: mixture of log-concave densities for 1D binned data.

The 1D analogue of malc_2d.py, written for variant T of the calibration
pipeline: project/convolve a model's prediction to a tau density FIRST, then
smooth that 1D density. malc_2d.py cannot be reused for this -- it fits a
density on a plane and there is no plane here.

Same four-step recipe as the 2D component fit, dimension-for-dimension:

  1. EM mean correction on the binned mass          (_em_mean_1d)
  2. Beta jitter parameter calibrated from that EM mean
  3. Sample B bins from p, place a point inside each with Beta jitter
  4. Fit a log-concave MLE on those B synthetic points   (mlelcd_1d)

and the same mixture EM on top (modal init -> Voronoi -> responsibilities ->
refit), the same patience-5 convergence, the same two-stage BIC scan for K.

WHAT DIFFERS FROM THE 2D CODE, AND WHY.

  The MLE itself. mlelcd_2d_fast solves a conic program over a Delaunay
  triangulation, approximating the integral of exp(y) by Dunavant quadrature
  over triangles. In one dimension the log-concave MLE has a closed-form
  objective: the density is piecewise log-linear with knots at the data, so

      integral over [x_i, x_{i+1}] of exp(linear) = d_i * exp(s) * sinh(t)/t

  with s = (y_i + y_{i+1})/2, t = (y_{i+1} - y_i)/2, d_i = x_{i+1} - x_i.
  That is exact, not quadrature, so the 1D fit is *more* accurate than the 2D
  one rather than a degraded port. Concavity is the linear condition that the
  slopes (y_{i+1} - y_i)/d_i are non-increasing.

  Parameter count in the BIC. The 2D version charges 3K - 1; the 1D analogue
  charges 2K - 1 (one location and one scale per component, K - 1 free
  mixture weights).

UNIFORM-GRID ASSUMPTION, INHERITED DELIBERATELY. Like the 2D code, the Beta
jitter is calibrated with a single bin width taken as grid[1] - grid[0]. A
non-uniform grid (DoPFN's quantile-spaced borders run 0.03 to 158) must be
rebinned onto a uniform one BEFORE calling this, exactly as it is before the
2D path, so variants J and T stay comparable. _check_uniform_grid raises
rather than silently mis-jittering.

SOLVER: BOUND-CONSTRAINED, NOT A QP. Concavity says the secant slopes are
non-increasing. Written directly that is a linear-constraint problem, and
SLSQP solves it as a DENSE QP costing ~B^3.7 -- 0.06 s at B=100 but 74 s at
B=800, which puts B >= 400 out of reach. Substituting the slope gaps

    u_i = s_i - s_{i+1} >= 0

turns every constraint into a BOUND, leaving only y0 and the final slope free,
so L-BFGS-B applies at O(m) per iteration. Measured per MLE after that change:

    B =  100 -> 0.09 s      B = 1000 -> 1.6 s
    B =  400 -> 0.24 s      B = 2000 -> 7.9 s

The change of variables is exact, not a relaxation; solver="slsqp" keeps the
direct encoding as a cross-check and agrees to ~3e-4 in objective.

Restarts are not optional. At the optimum the MLE has few knots, so MOST u_i
sit at their bound -- and L-BFGS-B stalls on heavily-bound-active problems,
halting up to 3e-2 above the optimum on one pass. Re-entering from the
returned point clears it (worst gap 7.6e-4, relative 3e-4, below MALC's own
Monte-Carlo noise). The closed-form shift y -> y - log(int exp(y)) then
restores int exp(yhat) = 1 exactly, which holds at the true optimum and is the
cheapest convergence check there is.

DO NOT TRUST THE BIC K-SELECTION -- PASS K EXPLICITLY. Measured on binned
targets at B=100, the BIC falls monotonically in K on data that is genuinely
unimodal, so MALC_1D(K=None) returns K* = max_K regardless:

    unimodal N(0,1)   L2  K=1 0.046  K=2 0.118  K=3 0.154   <- K=1 is right
                      BIC K=1 255    K=2 118    K=3  87     <- BIC says K=3
    bimodal +-1.2     L2  K=1 0.239  K=2 0.064  K=3 0.118   <- K=2 is right
                      BIC K=1 138    K=2 140    K=3 109     <- BIC says K=3

The complexity term (2K-1)*log(n_eff) is negligible against -2*loglik*n_eff.
This is INHERITED from malc_2d.MALC_2D_bic, not introduced here: on a unimodal
2D target that BIC also prefers K=2 (4567) to the correct K=1 (5344). It has
gone unnoticed because every caller in this repo pins K=1.

Choose K by the metric you actually care about -- IS_0.05 or CRPS on held-out
realizations -- not by this BIC. The mixture fit itself is sound: K=2 recovers
a bimodal target at 3.7x better L2 than K=1.

VALIDATED against N(0,1) binned at J=32, 20 seeds. B controls Monte-Carlo
error, and raising it pays off as theory says it should:

    B =  100   mean +0.037 (spread 0.101, theory SE 0.100)  sd 1.004  L2 0.077
    B = 1000   mean +0.011 (spread 0.041, theory SE 0.032)  sd 1.009  L2 0.034

No detectable bias at either setting; the ~1% sd excess is the shrinkage
expected of a log-concave MLE. Concavity holds to 1e-11 and the fitted density
integrates to 1.000000.

Usage:

    from malc_1d import MALC_1D, dmalc_1d
    fit = MALC_1D(p, edges, K=1, B=100, seed=20180621)
    dens = dmalc_1d(fit, tau_grid)          # density at arbitrary points
"""

from __future__ import annotations

import concurrent.futures
import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.stats import norm

__all__ = [
    "LogConcaveDensity1D",
    "mlelcd_1d",
    "dlcd_1d",
    "ComponentFit1D",
    "MALC1DFit",
    "MALC_1D_fit",
    "MALC_1D_bic",
    "MALC_1D",
    "dmalc_1d",
    "eval_grid_1d",
]

_UNIFORM_RTOL = 1e-6


def _check_uniform_grid(grid: np.ndarray) -> float:
    """Return the common bin width, or raise if the grid is not uniform."""
    d = np.diff(np.asarray(grid, dtype=float))
    if d.size == 0:
        raise ValueError("grid must have at least two edges")
    if np.any(d <= 0):
        raise ValueError("grid edges must be strictly increasing")
    if (d.max() - d.min()) > _UNIFORM_RTOL * d.mean():
        raise ValueError(
            f"MALC_1D requires a uniform grid; widths span "
            f"[{d.min():.6g}, {d.max():.6g}]. Rebin first (see "
            f"cate_density_metrics._rebin_nonuniform / _rebin_atoms_shared)."
        )
    return float(d.mean())


# ── EM mean correction ───────────────────────────────────────────────────────
# Identical to malc_2d._em_mean_2d, which is already a 1D routine applied to a
# marginal. Repeated here so this module stands alone.


def _em_mean_1d(
    props: np.ndarray,
    grid: np.ndarray,
    sigma: float,
    start: float,
    max_step: int = 1000,
    eps2: float = 1e-10,
    eps1: float = 1e-5,
) -> float:
    pn = props / props.sum()
    mu = start
    for _ in range(max_step):
        a = (grid - mu) / sigma
        G1 = norm.cdf(a)
        G2 = norm.pdf(a)
        temp = (np.diff(G2) + eps2) / (np.diff(G1) + eps2)
        mu_new = mu - sigma * float(np.sum(pn * temp))
        if abs(mu_new - mu) < eps1:
            return mu_new
        mu = mu_new
    return mu


# ── 1D log-concave MLE ───────────────────────────────────────────────────────


@dataclass
class LogConcaveDensity1D:
    x: np.ndarray        # (m,) sorted knots
    y: np.ndarray        # (m,) log-density at the knots
    w: np.ndarray        # (m,) weights used in the fit
    loglik: float
    integral: float      # integral of the fitted density; ~1 at the optimum


def _sinh_over_t(t: np.ndarray) -> np.ndarray:
    """sinh(t)/t, with the removable singularity at t = 0 handled by series."""
    out = np.empty_like(t)
    small = np.abs(t) < 1e-6
    ts = t[small]
    out[small] = 1.0 + ts**2 / 6.0 + ts**4 / 120.0
    tl = t[~small]
    out[~small] = np.sinh(tl) / tl
    return out


def _dg_dt(t: np.ndarray) -> np.ndarray:
    """d/dt [sinh(t)/t] = (t cosh t - sinh t)/t^2, series-stable near 0."""
    out = np.empty_like(t)
    small = np.abs(t) < 1e-4
    ts = t[small]
    out[small] = ts / 3.0 + ts**3 / 30.0
    tl = t[~small]
    out[~small] = (tl * np.cosh(tl) - np.sinh(tl)) / tl**2
    return out


def _objective(y: np.ndarray, x: np.ndarray, w: np.ndarray):
    """-[sum_i w_i y_i - integral exp(yhat)] and its gradient.

    The integral is exact for the piecewise-log-linear interpolant:
        I_i = d_i * exp(s_i) * sinh(t_i)/t_i,
        s_i = (y_i + y_{i+1})/2,  t_i = (y_{i+1} - y_i)/2,  d_i = x_{i+1} - x_i.
    """
    d = np.diff(x)
    s = 0.5 * (y[:-1] + y[1:])
    t = 0.5 * (y[1:] - y[:-1])

    # Overflow guard. L-BFGS-B's line search probes far outside the region of
    # interest, and exp(s) overflows once s > ~709. The overflow itself is
    # harmless -- the objective is genuinely enormous there and the search
    # should back off -- but inf propagates into the gradient as inf*0 = nan,
    # and a nan gradient makes L-BFGS-B halt silently at whatever point it is
    # standing on. That is a wrong answer that reports success.
    #
    # Returning +inf with a finite gradient pointing back downhill makes the
    # line search reject the step cleanly instead.
    # The product, not the factors: I_i ~ d * exp(s) * sinh(|t|)/|t|, which
    # grows like exp(s + |t|). Bounding s and |t| separately still overflows
    # when both are moderately large -- that is what leaked through the first
    # version of this guard and left 4 warnings in the cluster log.
    if not np.all(s + np.abs(t) < 700.0):
        return np.inf, np.sign(y) * 1e6
    g = _sinh_over_t(t)
    es = np.exp(s)
    I = d * es * g
    total = float(I.sum())

    # dI/ds = I ; dI/dt = d * exp(s) * g'(t)
    dI_ds = I
    dI_dt = d * es * _dg_dt(t)

    # s = (y_i + y_{i+1})/2 -> ds/dy_i = ds/dy_{i+1} = 1/2
    # t = (y_{i+1} - y_i)/2 -> dt/dy_i = -1/2, dt/dy_{i+1} = +1/2
    grad = np.zeros_like(y)
    np.add.at(grad, np.arange(len(y) - 1), 0.5 * dI_ds - 0.5 * dI_dt)
    np.add.at(grad, np.arange(1, len(y)), 0.5 * dI_ds + 0.5 * dI_dt)
    grad -= w

    return total - float(np.dot(w, y)), grad


def _objective_reparam(theta, x, w, d):
    """Objective + gradient in the (y0, s_last, u) parametrisation.

    Concavity is "the secant slopes are non-increasing", which as linear
    constraints on y needs a QP. Substituting

        u_i = s_i - s_{i+1} >= 0

    turns every one of them into a BOUND, leaving y0 and the final slope free.
    The problem becomes bound-constrained, so L-BFGS-B solves it at O(m) per
    iteration instead of SLSQP's dense O(m^3). Same optimum -- this is a change
    of variables on an equality-free convex problem, not a relaxation.

    Reconstruction (0-based, m knots, m-1 slopes, m-2 gaps):
        s[i] = s_last + sum_{k>=i} u[k]
        y[j] = y0 + sum_{i<j} s[i] * d[i]
    """
    m = x.size
    y0, s_last, u = theta[0], theta[1], theta[2:]
    # s[i] = s_last + sum_{k=i..m-3} u[k]  -> reverse cumulative sum
    s = np.empty(m - 1)
    s[m - 2] = s_last
    if m > 2:
        s[:m - 2] = s_last + np.cumsum(u[::-1])[::-1]
    y = np.empty(m)
    y[0] = y0
    y[1:] = y0 + np.cumsum(s * d)

    val, G = _objective(y, x, w)          # dF/dy, exact integral

    # chain rule, all O(m) via cumulative sums
    Gsuf = np.cumsum(G[::-1])[::-1]       # Gsuf[k] = sum_{j>=k} G[j]
    dF_ds = d * Gsuf[1:]                  # dF/ds[i] = d[i] * sum_{j>i} G[j]
    grad = np.empty_like(theta)
    grad[0] = G.sum()                     # dF/dy0
    grad[1] = dF_ds.sum()                 # dF/ds_last
    if m > 2:
        grad[2:] = np.cumsum(dF_ds)[:m - 2]   # dF/du[k] = sum_{i<=k} dF/ds[i]
    return val, grad


def _theta_from_y(y, d):
    s = np.diff(y) / d
    return np.concatenate([[y[0]], [s[-1]], np.maximum(-np.diff(s), 0.0)])


def _y_from_theta(theta, d, m):
    y0, s_last, u = theta[0], theta[1], theta[2:]
    s = np.empty(m - 1)
    s[m - 2] = s_last
    if m > 2:
        s[:m - 2] = s_last + np.cumsum(u[::-1])[::-1]
    y = np.empty(m)
    y[0] = y0
    y[1:] = y0 + np.cumsum(s * d)
    return y


def mlelcd_1d(
    x: np.ndarray,
    w: np.ndarray | None = None,
    max_iter: int = 2000,
    tol: float = 1e-10,
    solver: str = "lbfgsb",
    n_restart: int = 4,
) -> LogConcaveDensity1D:
    """Log-concave MLE on 1D points, exact objective.

    solver='lbfgsb' (default) uses the slope-gap reparametrisation above and
    scales to B in the thousands. solver='slsqp' keeps the direct
    linear-constraint formulation; it is kept because it is the obvious
    encoding of the problem and a useful cross-check, but it is O(m^3) and
    unusable past B ~= 300.

    Duplicate points are merged and their weights summed -- the MLE only sees
    the empirical measure, and duplicates would make d_i = 0.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    if n < 2:
        raise ValueError("need at least 2 points for the 1D MLE")
    w = np.full(n, 1.0 / n) if w is None else np.asarray(w, dtype=float) / np.sum(w)

    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    xu, inv = np.unique(x, return_inverse=True)
    wu = np.zeros(xu.size)
    np.add.at(wu, inv, w)
    if xu.size < 2:
        raise ValueError("all points identical; MLE undefined")
    x, w, m = xu, wu, xu.size
    d = np.diff(x)

    # Gaussian log-density through the weighted moments: concave, hence
    # feasible, and close for the unimodal cases that dominate here.
    mu = float(np.dot(w, x))
    sd = float(np.sqrt(max(np.dot(w, (x - mu) ** 2), 1e-12)))
    y0 = -0.5 * ((x - mu) / sd) ** 2 - np.log(sd * np.sqrt(2.0 * np.pi))

    if solver == "lbfgsb":
        theta0 = _theta_from_y(y0, d)
        bounds = [(None, None), (None, None)] + [(0.0, None)] * max(m - 2, 0)
        # Restart loop. The log-concave MLE has few knots, so at the optimum
        # MOST u_i sit at their bound of 0 -- and L-BFGS-B stalls on
        # heavily-bound-active problems, halting up to 3e-2 above the optimum
        # on a single pass. Re-entering from the returned point rebuilds the
        # limited-memory curvature model and clears the stall; measured worst
        # gap vs the SLSQP reference falls from 3.4e-2 to 7.6e-4 (relative
        # 3e-4, below MALC's own Monte-Carlo noise from sampling B points).
        #
        # ftol=0 disables the relative-decrease stop so only the gradient
        # criterion applies; maxcor=50 (vs the default 10) keeps a richer
        # curvature model, which matters because y0, s_last and the u gaps
        # differ in scale.
        theta, prev = theta0, np.inf
        for _ in range(n_restart):
            res = minimize(_objective_reparam, theta, args=(x, w, d), jac=True,
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": max_iter,
                                    "maxfun": 20 * max_iter,
                                    "ftol": 0.0, "gtol": 1e-14, "maxcor": 50})
            if not np.isfinite(res.fun):
                break
            theta = res.x
            if prev - res.fun < 1e-13:
                break
            prev = res.fun
        y = _y_from_theta(theta, d, m) if np.isfinite(res.fun) else y0
        if not np.isfinite(y).all():
            y = y0

        # Exact polish along the constant direction. F(y) = int exp(y) - <w,y>,
        # so for y - c the optimum is c = log(int exp(y)): dF/dc = 1 - I e^-c.
        # This is a closed-form improvement, and it restores int exp(yhat) = 1,
        # which holds at the true optimum and is the cheapest convergence check
        # available.
        I = float(np.sum(np.diff(x) * np.exp(0.5 * (y[:-1] + y[1:]))
                         * _sinh_over_t(0.5 * (y[1:] - y[:-1]))))
        if np.isfinite(I) and I > 0:
            y = y - np.log(I)
    else:
        if m > 2:
            A = np.zeros((m - 2, m))
            for i in range(m - 2):
                A[i, i] += -d[i + 1]
                A[i, i + 1] += d[i] + d[i + 1]
                A[i, i + 2] += -d[i]
            cons = [LinearConstraint(A, 0.0, np.inf)]
        else:
            cons = []
        res = minimize(
            _objective, y0, args=(x, w), jac=True, method="SLSQP",
            constraints=[{"type": "ineq",
                          "fun": (lambda yy, A=c.A: A @ yy),
                          "jac": (lambda yy, A=c.A: A)} for c in cons],
            options={"maxiter": max_iter, "ftol": tol},
        )
        y = res.x if res.success else y0

    integral = float(np.sum(np.diff(x) * np.exp(0.5 * (y[:-1] + y[1:]))
                            * _sinh_over_t(0.5 * (y[1:] - y[:-1]))))
    return LogConcaveDensity1D(x=x, y=y, w=w,
                               loglik=float(np.dot(w, y) - integral),
                               integral=integral)


def dlcd_1d(query_pts: np.ndarray, density: LogConcaveDensity1D) -> np.ndarray:
    """Evaluate the fitted density. Zero outside the convex support [x_0, x_m]."""
    q = np.asarray(query_pts, dtype=float).reshape(-1)
    out = np.zeros_like(q)
    inside = (q >= density.x[0]) & (q <= density.x[-1])
    if np.any(inside):
        out[inside] = np.exp(np.interp(q[inside], density.x, density.y))
    return out


# ── Component fit: EM mean -> Beta jitter -> sample B -> MLE ─────────────────


@dataclass
class ComponentFit1D:
    fhatn: LogConcaveDensity1D
    mu_hat: float
    alpha: float
    beta: float
    grid: np.ndarray
    p: np.ndarray
    xstar: np.ndarray


def _fit_component_1d(p, grid, B, alpha, rng, delta=None):
    p = np.asarray(p, dtype=float)
    tot = p.sum()
    if not np.isfinite(tot) or tot <= 0:
        return None
    p = p / tot

    delta = _check_uniform_grid(grid) if delta is None else delta
    grid_left = grid[:-1]
    centers = 0.5 * (grid_left + grid[1:])

    mu_low = float(np.sum(p * grid_left))
    mu_mid = 0.5 * (mu_low + float(np.sum(p * grid[1:])))

    sigma = float(np.sqrt(np.sum(p * (centers - mu_mid) ** 2) + delta**2 / 12.0))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = delta

    mu_n = _em_mean_1d(p, grid, sigma=sigma, start=mu_mid)
    beta = 2.0 * alpha * ((mu_n - mu_low) / delta - 0.5)
    if not np.isfinite(beta) or min(alpha + beta, alpha - beta) <= 0:
        return None

    bin_idx = rng.choice(p.size, size=B, p=p, replace=True)
    zstar = delta * rng.beta(alpha + beta, alpha - beta, size=B)
    xstar = grid_left[bin_idx] + zstar

    try:
        fhatn = mlelcd_1d(xstar)
    except Exception:
        return None

    return ComponentFit1D(fhatn=fhatn, mu_hat=mu_n, alpha=alpha, beta=beta,
                          grid=grid, p=p, xstar=xstar)


def _bin_probs_1d(fit_k: ComponentFit1D, grid: np.ndarray, n_eval: int = 512) -> np.ndarray:
    """Integrate a component over the bins of `grid` (the EM E-step)."""
    xs = np.linspace(grid.min(), grid.max(), n_eval)
    dx = xs[1] - xs[0]
    dens = np.nan_to_num(dlcd_1d(xs, fit_k.fhatn), nan=0.0)
    dens = np.maximum(dens, 0.0)

    nb = grid.size - 1
    b = np.clip(np.searchsorted(grid, xs, side="right") - 1, 0, nb - 1)
    probs = np.zeros(nb)
    np.add.at(probs, b, dens * dx)
    s = probs.sum()
    return probs / s if s > 0 else np.full(nb, 1.0 / nb)


# ── Modal init ───────────────────────────────────────────────────────────────


def _init_assignments_1d(p: np.ndarray, K: int) -> np.ndarray:
    n = p.size
    pmax = float(p.max())
    floor = 1e-10 * pmax
    rtol = 1e-12

    is_peak = np.zeros(n, dtype=bool)
    for i in range(n):
        nb = p[max(0, i - 1):min(n, i + 2)]
        nb_max = float(nb.max())
        if p[i] + rtol * nb_max >= nb_max and p[i] > floor:
            is_peak[i] = True

    cand = np.flatnonzero(is_peak)
    if cand.size == 0:
        cand = np.array([int(np.argmax(p))])

    # Non-maximum suppression over adjacent plateau ties (2D uses radius 1).
    order = np.argsort(-p[cand], kind="stable")
    kept: list[int] = []
    for idx in order:
        if all(abs(int(cand[idx]) - int(cand[k])) > 1 for k in kept):
            kept.append(int(idx))
    peaks = cand[kept]

    if peaks.size >= K:
        peaks = peaks[np.argsort(-p[peaks])[:K]]
    else:
        seeds = list(peaks)
        coords = np.arange(n, dtype=float)
        while len(seeds) < K:
            min_d2 = np.full(n, np.inf)
            for s in seeds:
                min_d2 = np.minimum(min_d2, (coords - s) ** 2)
            seeds.append(int(np.argmax(p * min_d2)))
        peaks = np.array(seeds)

    return np.argmin(np.abs(np.arange(n)[:, None] - peaks[None, :]), axis=1)


# ── Fit object, EM, BIC ──────────────────────────────────────────────────────


@dataclass
class MALC1DFit:
    fits: list
    pi: np.ndarray
    K: int
    loglik: float
    grid: np.ndarray
    p: np.ndarray
    bic_table: np.ndarray | None = None


def MALC_1D_fit(
    p: np.ndarray,
    grid: np.ndarray,
    K: int,
    B: int = 100,
    alpha: float = 2.0,
    max_iter: int = 30,
    tol: float = 1e-4,
    seed: int = 20180621,
    verbose: bool = False,
    bin_n_eval: int = 512,
) -> MALC1DFit:
    rng = np.random.default_rng(seed)
    p = np.maximum(np.asarray(p, dtype=float), 0.0)
    p = p / p.sum()
    grid = np.asarray(grid, dtype=float)
    if grid.size != p.size + 1:
        raise ValueError(f"grid must have len(p)+1 edges; got {grid.size} vs {p.size}+1")
    delta = _check_uniform_grid(grid)
    n = p.size

    if K == 1:
        fit = _fit_component_1d(p, grid, B=B, alpha=alpha, rng=rng, delta=delta)
        if fit is None:
            raise RuntimeError("MALC_1D_fit: K=1 component fit failed")
        pb = _bin_probs_1d(fit, grid, n_eval=bin_n_eval)
        loglik = float(np.sum(p * np.log(np.maximum(pb, 1e-300))))
        return MALC1DFit(fits=[fit], pi=np.array([1.0]), K=1, loglik=loglik,
                         grid=grid, p=p)

    asgn = _init_assignments_1d(p, K)
    pi_k = np.array([np.sum(asgn == k) for k in range(K)], dtype=float) / n
    pi_k = pi_k / pi_k.sum()

    fits = [_fit_component_1d(p * (asgn == k), grid, B=B, alpha=alpha, rng=rng,
                              delta=delta) for k in range(K)]

    best_loglik, best_fits, best_pi, no_improve = -np.inf, list(fits), pi_k.copy(), 0
    for it in range(max_iter):
        p_arr = np.zeros((n, K))
        for k in range(K):
            if fits[k] is not None:
                p_arr[:, k] = _bin_probs_1d(fits[k], grid, n_eval=bin_n_eval)

        mix_p = p_arr @ pi_k
        mix_safe = np.maximum(mix_p, 1e-300)
        gam = (pi_k[None, :] * p_arr) / mix_safe[:, None]
        p_k_list = [p * gam[:, k] for k in range(K)]

        loglik = float(np.sum(p * np.log(mix_safe)))
        if verbose:
            print(f"  iter {it + 1}  loglik = {loglik:.4f}")

        if loglik > best_loglik + tol:
            best_loglik, best_fits, best_pi, no_improve = loglik, list(fits), pi_k.copy(), 0
        else:
            no_improve += 1
            if no_improve >= 5:
                break

        pi_k = np.maximum(np.array([pk.sum() for pk in p_k_list]), 1e-6)
        pi_k = pi_k / pi_k.sum()
        for k in range(K):
            nf = _fit_component_1d(p_k_list[k], grid, B=B, alpha=alpha, rng=rng, delta=delta)
            if nf is not None:
                fits[k] = nf

    return MALC1DFit(fits=best_fits, pi=best_pi, K=K, loglik=best_loglik,
                     grid=grid, p=p)


def MALC_1D_bic(obj: MALC1DFit) -> float:
    """1D analogue of MALC_2D_bic: 2K - 1 parameters instead of 3K - 1."""
    n_eff = 1.0 / float(np.sum(obj.p ** 2))
    num_params = 2 * obj.K - 1
    return -2.0 * obj.loglik * n_eff + num_params * np.log(n_eff)


def _bic_worker(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    p, grid, K, B, alpha, max_iter, tol, seed, bin_n_eval = args
    try:
        fit = MALC_1D_fit(p, grid, K=K, B=B, alpha=alpha, max_iter=max_iter,
                          tol=tol, seed=seed, bin_n_eval=bin_n_eval)
        return K, float(MALC_1D_bic(fit))
    except Exception:
        return K, float("inf")


def MALC_1D(
    p: np.ndarray,
    grid: np.ndarray,
    K: int | None = None,
    max_K: int = 3,
    B_select: int = 100,
    B_fit: int = 300,
    B: int | None = None,
    alpha: float = 2.0,
    max_iter: int = 30,
    tol: float = 1e-4,
    seed: int = 20180621,
    verbose: bool = False,
    bin_n_eval: int = 512,
    parallel: bool = False,
) -> MALC1DFit:
    """Fit MALC-1D. K given -> single fit; K None -> two-stage BIC scan.

    `B` overrides both B_select and B_fit, for the K-given case where the
    caller wants one explicit sample size (the calibration pipeline uses
    K=1, B=100).
    """
    if K is not None:
        return MALC_1D_fit(p, grid, K=K, B=(B if B is not None else B_fit),
                           alpha=alpha, max_iter=max_iter, tol=tol, seed=seed,
                           verbose=verbose, bin_n_eval=bin_n_eval)

    args = [(p, grid, k, B_select, alpha, max_iter, tol, seed, bin_n_eval)
            for k in range(1, max_K + 1)]
    if parallel and max_K > 1:
        with concurrent.futures.ProcessPoolExecutor() as ex:
            scored = list(ex.map(_bic_worker, args))
    else:
        scored = [_bic_worker(a) for a in args]

    bic_table = np.array(sorted(scored), dtype=float)
    K_star = int(bic_table[np.argmin(bic_table[:, 1]), 0])
    if verbose:
        print(f"  BIC scan -> K* = {K_star}")

    obj = MALC_1D_fit(p, grid, K=K_star, B=(B if B is not None else B_fit),
                      alpha=alpha, max_iter=max_iter, tol=tol, seed=seed,
                      verbose=verbose, bin_n_eval=bin_n_eval)
    obj.bic_table = bic_table
    return obj


# ── Evaluation ───────────────────────────────────────────────────────────────


def dmalc_1d(obj: MALC1DFit, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1)
    total = np.zeros(pts.size)
    for k in range(obj.K):
        if obj.fits[k] is not None:
            total += obj.pi[k] * np.maximum(dlcd_1d(pts, obj.fits[k].fhatn), 0.0)
    return total


def eval_grid_1d(obj: MALC1DFit, n_eval: int = 4001):
    xs = np.linspace(obj.grid.min(), obj.grid.max(), n_eval)
    return xs, dmalc_1d(obj, xs)
