"""Shared density machinery for the Tier-C (CATE / tau) density eval.

Tier C scores p(tau | x), tau = Y_do1 - Y_do0, for two models that emit very
different objects:

  * UWYK-1D   two forward passes -> two 1D BarDistribution heads
              f0(y), f1(y);  joint assumed to factorise, so
              f(y0, y1) = f0(y0) * f1(y1)
  * Joint-2D  one forward pass -> BarDistribution2D head
              f(y0, y1) natively (J^2 inner bins + 9 region weights + 4 tails)

Both are reduced to p(tau) by the SAME operator:

    p(tau) = \\int f(y0, y0 + tau) dy0

implemented once in `tau_density`, which takes a callable f(y0, y1). The only
thing that differs between the two models is which callable it gets. That is
deliberate: any quadrature error is then common to both columns and cancels in
the comparison.

FULL DENSITIES, NOT INNER-ONLY. Both models are evaluated with their tails:
UWYK's two half-Gaussians, the joint's 8 non-inner regions. Truncating to the
inner support and renormalising would inflate each model's interior by
1/(1-eps) with a DIFFERENT eps per model (the joint loses ~2x the marginal
tail mass, being a square), producing a differential NLL bias larger than the
effect we are trying to measure. See density_eval_pipeline.md.

Conventions
-----------
Everything lives on the harness's scaled y axis (`H._scale_y`:
y -> 2(y - y_min)/y_rng - 1, from the *post-subsample* training context).
Densities are w.r.t. that axis; a raw-units density would need a 2/y_rng
Jacobian and is never mixed in here.

Reference implementations mirrored (do not let these drift):
  * losses/BarDistribution2D.py::neg_log_prob_2d   -> `Joint2D.density`
  * g4cfm/src/Losses/BarDistribution.py::_logpdf_from_pred -> `UWYK1D.density`
`test_density_common.py` asserts agreement with both to 1e-5.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Grids (match benchmarks/l2_ihdp/true_ihdp.py so numbers cross-reference)
# ---------------------------------------------------------------------------
Y_EDGES = np.linspace(-1.5, 1.5, 101)
Y_CENTERS = 0.5 * (Y_EDGES[:-1] + Y_EDGES[1:])
Y_BIN = float(Y_CENTERS[1] - Y_CENTERS[0])

TAU_EDGES = np.linspace(-3.0, 3.0, 601)
TAU_CENTERS = 0.5 * (TAU_EDGES[:-1] + TAU_EDGES[1:])
TAU_BIN = float(TAU_CENTERS[1] - TAU_CENTERS[0])

_LOG_2PI = math.log(2.0 * math.pi)
_EPS = 1e-300          # density floor, only to keep logs finite in KL
_TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


def _softplus(x):
    # threshold=20 matches BarDistribution._safe_softplus
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def _log_softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))


# ---------------------------------------------------------------------------
# Joint-2D head: the full 9-region density
# ---------------------------------------------------------------------------
@dataclass
class Joint2D:
    """Unpacked BarDistribution2D prediction for ONE query.

    Unpacking once and evaluating at many points avoids re-running the J^2
    softmax per evaluation point, which is what makes the tau quadrature
    affordable. `test_density_common.py::test_joint_matches_reference`
    checks this against neg_log_prob_2d on random inputs.
    """
    p_mat: np.ndarray          # (J, J) inner bin probabilities, sums to 1
    w: np.ndarray              # (9,)   region mixture weights, sums to 1
    sL0: float
    sR0: float
    sL1: float
    sR1: float
    rho: float
    edges: np.ndarray          # (J+1,)

    @property
    def lo(self) -> float:
        return float(self.edges[0])

    @property
    def hi(self) -> float:
        return float(self.edges[-1])

    @property
    def bw(self) -> float:
        return float((self.edges[-1] - self.edges[0]) / (len(self.edges) - 1))

    @property
    def max_scale(self) -> float:
        return float(max(self.sL0, self.sR0, self.sL1, self.sR1))

    @classmethod
    def from_pred(cls, pred: np.ndarray, J: int, edges: np.ndarray) -> "Joint2D":
        """pred: (J^2 + 9 + 4,) raw head output for one query."""
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        JJ = J * J
        bw = float((edges[-1] - edges[0]) / J)

        p_mat = np.exp(_log_softmax(pred[:JJ])).reshape(J, J)
        w = np.exp(_log_softmax(pred[JJ:JJ + 9]))
        tail_raw = pred[JJ + 9:JJ + 13]
        # _safe_scale(raw, base) = base * (softplus(raw) + SOFTPLUS_FLOOR)
        s = bw * (_softplus(tail_raw) + 1e-3)

        centers = 0.5 * (edges[:-1] + edges[1:])
        m0 = p_mat.sum(axis=1)
        m1 = p_mat.sum(axis=0)
        E0 = float((centers * m0).sum())
        E1 = float((centers * m1).sum())
        E01 = float((p_mat * centers[:, None] * centers[None, :]).sum())
        v0 = max(float((centers ** 2 * m0).sum()) - E0 ** 2, 1e-8)
        v1 = max(float((centers ** 2 * m1).sum()) - E1 ** 2, 1e-8)
        rho = (E01 - E0 * E1) / math.sqrt(v0 * v1)
        rho = float(np.clip(rho, -1 + 1e-6, 1 - 1e-6))

        return cls(p_mat=p_mat, w=w, sL0=float(s[0]), sR0=float(s[1]),
                   sL1=float(s[2]), sR1=float(s[3]), rho=rho,
                   edges=np.asarray(edges, dtype=np.float64))

    # -- the density ------------------------------------------------------
    def density(self, y0, y1) -> np.ndarray:
        """f(y0, y1) on the full plane. Broadcasting over arbitrary shapes."""
        y0 = np.asarray(y0, dtype=np.float64)
        y1 = np.asarray(y1, dtype=np.float64)
        y0, y1 = np.broadcast_arrays(y0, y1)

        J = self.p_mat.shape[0]
        lo, hi, bw = self.lo, self.hi, self.bw
        interior = self.edges[1:-1]
        j0 = np.clip(np.searchsorted(interior, y0, side='right'), 0, J - 1)
        j1 = np.clip(np.searchsorted(interior, y1, side='right'), 0, J - 1)

        in0 = (y0 >= lo) & (y0 <= hi)
        in1 = (y1 >= lo) & (y1 <= hi)
        L0, R0 = y0 < lo, y0 > hi
        L1, R1 = y1 < lo, y1 > hi

        w = self.w
        out = np.zeros(y0.shape, dtype=np.float64)

        # region 0: inner x inner
        m = in0 & in1
        if m.any():
            out[m] = w[0] * self.p_mat[j0[m], j1[m]] / (bw * bw)

        # regions 1,2: y0 outside, y1 inside.
        # boundary conditional f(y1 | y0 = boundary) ~ p_mat[row, j1] / rowsum
        row_lo, row_hi = self.p_mat[0, :], self.p_mat[J - 1, :]
        s_lo, s_hi = max(row_lo.sum(), 1e-300), max(row_hi.sum(), 1e-300)
        m = L0 & in1
        if m.any():
            out[m] = (w[1] * _half_gauss(y0[m], lo, self.sL0)
                      * row_lo[j1[m]] / (s_lo * bw))
        m = R0 & in1
        if m.any():
            out[m] = (w[2] * _half_gauss(y0[m], hi, self.sR0)
                      * row_hi[j1[m]] / (s_hi * bw))

        # regions 3,4: y0 inside, y1 outside
        col_lo, col_hi = self.p_mat[:, 0], self.p_mat[:, J - 1]
        c_lo, c_hi = max(col_lo.sum(), 1e-300), max(col_hi.sum(), 1e-300)
        m = in0 & L1
        if m.any():
            out[m] = (w[3] * col_lo[j0[m]] / (c_lo * bw)
                      * _half_gauss(y1[m], lo, self.sL1))
        m = in0 & R1
        if m.any():
            out[m] = (w[4] * col_hi[j0[m]] / (c_hi * bw)
                      * _half_gauss(y1[m], hi, self.sR1))

        # regions 5-8: both outside. Bivariate Gaussian anchored at the corner,
        # normalised by the quadrant probability (Sheppard's theorem).
        asin = math.asin(self.rho)
        n_same = 0.25 + asin / (2 * math.pi)
        n_opp = max(0.25 - asin / (2 * math.pi), 1e-300)
        for mask, idx, a0, a1, ss0, ss1, nrm in (
            (L0 & L1, 5, lo, lo, self.sL0, self.sL1, n_same),
            (L0 & R1, 6, lo, hi, self.sL0, self.sR1, n_opp),
            (R0 & L1, 7, hi, lo, self.sR0, self.sL1, n_opp),
            (R0 & R1, 8, hi, hi, self.sR0, self.sR1, n_same),
        ):
            if mask.any():
                out[mask] = w[idx] * _biv_gauss(
                    y0[mask], y1[mask], a0, a1, ss0, ss1, self.rho) / nrm
        return out


def _half_gauss(y, boundary, scale):
    """Half-normal density on the half-line nearest `boundary` (integrates 1)."""
    z = (y - boundary) / scale
    return 2.0 * np.exp(-0.5 * z * z) / (scale * math.sqrt(2.0 * math.pi))


def _biv_gauss(y0, y1, mu0, mu1, s0, s1, rho):
    z0 = (y0 - mu0) / s0
    z1 = (y1 - mu1) / s1
    r2 = max(1.0 - rho * rho, 1e-8)
    q = -(0.5 / r2) * (z0 * z0 - 2 * rho * z0 * z1 + z1 * z1)
    return np.exp(q) / (s0 * s1 * math.sqrt(r2) * 2.0 * math.pi)


# ---------------------------------------------------------------------------
# UWYK 1D BarDistribution head
# ---------------------------------------------------------------------------
@dataclass
class UWYK1D:
    """Unpacked BarDistribution prediction for ONE query / one arm.

    Mirrors BarDistribution._logpdf_from_pred: K bars of exact width plus a
    half-Gaussian on each side. `bar_state` comes from the checkpoint via the
    sklearn wrapper's restored bar_distribution.
    """
    log_pL: float
    log_pBars: np.ndarray      # (K,)
    log_pR: float
    sL: float
    sR: float
    edges: np.ndarray          # (K+1,)
    widths: np.ndarray         # (K,)

    @property
    def max_scale(self) -> float:
        return float(max(self.sL, self.sR))

    @classmethod
    def from_pred(cls, pred, edges, widths, base_sL, base_sR,
                  scale_floor=1e-3) -> "UWYK1D":
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        K = len(widths)
        lp = _log_softmax(pred[:K + 2])
        return cls(
            log_pL=float(lp[0]), log_pBars=lp[1:-1], log_pR=float(lp[-1]),
            sL=float(base_sL * (_softplus(pred[K + 2]) + scale_floor)),
            sR=float(base_sR * (_softplus(pred[K + 3]) + scale_floor)),
            edges=np.asarray(edges, dtype=np.float64),
            widths=np.asarray(widths, dtype=np.float64),
        )

    def density(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        K = len(self.widths)
        out = np.zeros(y.shape, dtype=np.float64)

        left = y < self.edges[0]
        right = y >= self.edges[-1]
        mid = ~(left | right)

        if left.any():
            out[left] = math.exp(self.log_pL) * _half_gauss(
                y[left], self.edges[0], self.sL)
        if right.any():
            out[right] = math.exp(self.log_pR) * _half_gauss(
                y[right], self.edges[-1], self.sR)
        if mid.any():
            k = np.clip(np.searchsorted(self.edges[1:-1], y[mid], side='right'),
                        0, K - 1)
            out[mid] = np.exp(self.log_pBars[k]) / self.widths[k]
        return out

    def rebin(self, new_edges) -> "UWYK1D":
        """Re-express this density on a coarser uniform grid (resolution match).

        CDF-interpolates the bar probabilities onto `new_edges`, which need NOT
        divide the native grid evenly (1000 bars vs 32 bins does not). Tail
        weights and scales are carried over unchanged, so only the interior
        resolution changes.
        """
        new_edges = np.asarray(new_edges, dtype=np.float64)
        pbars = np.exp(self.log_pBars)
        cdf = np.concatenate([[0.0], np.cumsum(pbars)])
        F = np.interp(new_edges, self.edges, cdf,
                      left=0.0, right=float(cdf[-1]))
        p_new = np.diff(F)
        tot = p_new.sum()
        if tot > 0:
            p_new *= (pbars.sum() / tot)      # preserve interior mass exactly
        w_new = np.diff(new_edges)
        return UWYK1D(
            log_pL=self.log_pL,
            log_pBars=np.log(np.maximum(p_new, 1e-300)),
            log_pR=self.log_pR,
            sL=self.sL, sR=self.sR,
            edges=new_edges, widths=w_new,
        )


# ---------------------------------------------------------------------------
# tau = y1 - y0 : the SAME operator for both models
# ---------------------------------------------------------------------------
# Both heads are STAIRCASES on their interior: piecewise-constant on a uniform
# bin grid. Integrating a staircase along the diagonal with plain trapezoid is
# O(h) wrong at every one of the ~2J discontinuities the diagonal crosses, and
# the error scales with the head's bin count -- so it would be LARGER for the
# J=32 joint than for the K=1000 UWYK, i.e. a differential bias between the two
# columns. Exactly the thing this eval must not have.
#
# So the interior is done in closed form instead. For uniform bins of width bw,
# writing tau = (d + phi) * bw with integer d and 0 <= phi < 1, the diagonal
# spends fraction (1-phi) of each y0-bin at bin-offset d and phi at d+1:
#
#     p_int(tau) = (weight / bw) * [ (1-phi) * S(d) + phi * S(d+1) ]
#     S(k)       = sum_i p[i, i+k]          (k-th diagonal sum of the joint)
#
# Exact, O(J) per tau, no quadrature error. It also integrates to `weight` by
# construction, since sum_k S(k) = 1. Only the 8 non-interior regions -- which
# are smooth and carry little mass -- are left to quadrature.


def _diag_sums(p_mat):
    """S[k] = sum_i p_mat[i, i+k] for k = -(J-1) .. (J-1). Index k+J-1."""
    J = p_mat.shape[0]
    return np.array([np.trace(p_mat, offset=k) for k in range(-(J - 1), J)])


def _diag_sums_product(p0, p1):
    """Same S(k) for a factorised joint p_mat = outer(p0, p1), without ever
    forming the outer product: S(k) = sum_i p0[i] * p1[i+k]."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    return np.correlate(p1, p0, mode='full')


def _interior_tau(S, tau_points, bw, weight):
    """Linear-interpolate the diagonal sums onto the tau axis."""
    tau = np.asarray(tau_points, dtype=np.float64)
    J = (len(S) + 1) // 2
    x = tau / bw
    d = np.floor(x).astype(int)
    phi = x - d
    i0, i1 = d + J - 1, d + J
    ok0 = (i0 >= 0) & (i0 < len(S))
    ok1 = (i1 >= 0) & (i1 < len(S))
    out = np.zeros(tau.shape, dtype=np.float64)
    out[ok0] += (1.0 - phi[ok0]) * S[i0[ok0]]
    out[ok1] += phi[ok1] * S[i1[ok1]]
    return out * (weight / bw)


def tau_density_quadrature(f2d, tau_points, lo=-1.0, hi=1.0, pad=0.75,
                           n_y0=4096, max_points=4_000_000):
    """Reference implementation: p(tau) = \\int f(y0, y0+tau) dy0 by trapezoid.

    Correct for smooth integrands, O(h)-biased on staircases. Used for the
    non-interior regions (smooth) and as the slow cross-check that the exact
    interior formula above agrees with brute force.

    Chunked over tau: the evaluation array is (n_tau, n_y0) and `density`
    allocates ~10 temporaries of that shape, so an unchunked call at large
    n_y0 will OOM long before it is slow.
    """
    tau_points = np.atleast_1d(np.asarray(tau_points, dtype=np.float64))
    g = np.linspace(lo - pad, hi + pad, n_y0)
    step = max(1, int(max_points // max(n_y0, 1)))
    out = np.empty(tau_points.size, dtype=np.float64)
    for i in range(0, tau_points.size, step):
        t = tau_points[i:i + step]
        Y0 = np.broadcast_to(g, (t.size, n_y0))
        Y1 = Y0 + t[:, None]
        out[i:i + step] = _TRAPZ(f2d(Y0, Y1), g, axis=1)
    return out


# Back-compat alias for the generic operator.
tau_density = tau_density_quadrature


def independent_f2d(f0: UWYK1D, f1: UWYK1D):
    """The independence reconstruction: f(y0,y1) = f0(y0) * f1(y1).

    This is what UWYK gets, and it is an assumption WE impose -- UWYK emits no
    joint. Any row built from this must be labelled 'UWYK (x) indep'.
    """
    return lambda y0, y1: f0.density(y0) * f1.density(y1)


def _outside_only(f2d, lo, hi):
    """f2d with the interior-x-interior region zeroed, so it can be added to
    the exact interior term without double counting."""
    def g(y0, y1):
        v = f2d(y0, y1)
        inner = (y0 >= lo) & (y0 <= hi) & (y1 >= lo) & (y1 <= hi)
        return np.where(inner, 0.0, v)
    return g


def joint_tau_density(jt: Joint2D, tau_points, n_pad_sigma=8.0, n_y0=4096):
    """p(tau) for the 2D head: exact interior + quadrature over regions 1-8."""
    S = _diag_sums(jt.p_mat)
    p = _interior_tau(S, tau_points, jt.bw, jt.w[0])
    pad = n_pad_sigma * jt.max_scale
    p += tau_density_quadrature(_outside_only(jt.density, jt.lo, jt.hi),
                                tau_points, jt.lo, jt.hi, pad, n_y0)
    return p


def uwyk_tau_density(f0: UWYK1D, f1: UWYK1D, tau_points, n_pad_sigma=8.0,
                     n_y0=4096):
    """p(tau) for UWYK under independence: exact interior + quadrature tails.

    Same decomposition as the joint, so the two columns share their numerical
    treatment and any residual quadrature bias is common to both.
    """
    if not np.allclose(f0.edges, f1.edges):
        raise ValueError('UWYK arms must share a bar grid')
    # The bars ARE uniform by construction (BarDistribution.fit uses linspace),
    # but checkpoints store `edges` in float32 and K=1000 gives bw=0.002, which
    # is not a power of two -- so consecutive stored edges do not differ by
    # exactly bw. Real spread on the shipped checkpoint: 4.7e-5 RELATIVE, which
    # np.allclose's 1e-5 default rtol rejects. Use the mean width and a
    # float32-sized relative tolerance; anything looser than 1e-4 is a
    # genuinely non-uniform grid and the exact interior formula does not apply.
    # (The J=32 2D head never trips this: bw = 0.0625 = 2^-4 is exact.)
    bw = float(np.mean(f0.widths))
    rel_dev = float(np.abs(f0.widths - bw).max() / bw) if bw > 0 else np.inf
    if rel_dev > 1e-4:
        raise ValueError(
            f'exact interior term assumes uniform bars; max relative width '
            f'deviation is {rel_dev:.2e} (> 1e-4). Either the grid is really '
            f'non-uniform, or edges were stored at lower precision than fp32.')
    p0, p1 = np.exp(f0.log_pBars), np.exp(f1.log_pBars)
    S = _diag_sums_product(p0, p1)
    # interior weight is the product of the two interior masses
    p = _interior_tau(S, tau_points, bw, 1.0)
    pad = n_pad_sigma * max(f0.max_scale, f1.max_scale)
    p += tau_density_quadrature(
        _outside_only(independent_f2d(f0, f1), float(f0.edges[0]),
                      float(f0.edges[-1])),
        tau_points, float(f0.edges[0]), float(f0.edges[-1]), pad, n_y0)
    return p


# ---------------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------------
def truth_tau_density(mu0, mu1, sigma, tau_points):
    """tau | x ~ N(mu1 - mu0, 2 sigma^2) under the IHDP / ACIC Gaussian DGP.

    Independent per-arm noise is a property of both DGPs (verified: pooled
    corr(eps0, eps1) is null on IHDP, ACIC, CPS and PSID), so the 2 sigma^2 has
    no covariance term.
    """
    tau_points = np.atleast_1d(np.asarray(tau_points, dtype=np.float64))
    s = math.sqrt(2.0) * float(sigma)
    z = (tau_points - (float(mu1) - float(mu0))) / s
    return np.exp(-0.5 * z * z) / (s * math.sqrt(2.0 * math.pi))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def l2_distance(f, g, grid):
    return float(np.sqrt(_TRAPZ((np.asarray(f) - np.asarray(g)) ** 2, grid)))


def kl(p, q, grid):
    """\\int p log(p/q). Call as kl(truth, est) for KL_fwd, kl(est, truth) for
    KL_rev. Both densities floored before the log."""
    p = np.maximum(np.asarray(p, dtype=np.float64), _EPS)
    q = np.maximum(np.asarray(q, dtype=np.float64), _EPS)
    return float(_TRAPZ(p * np.log(p / q), grid))


def nll(density_at_point):
    """-log f(tau*). No epsilon floor is applied here on purpose: with full
    tails the density is strictly positive everywhere, so a -inf means a real
    bug (or a truncated evaluation), and should surface rather than be hidden.
    """
    d = np.asarray(density_at_point, dtype=np.float64)
    return -np.log(d)


def mass(f, grid):
    """\\int f over the grid. Diagnostic: how much of the density the tau grid
    actually contains. Report it; do not silently renormalise NLL by it."""
    return float(_TRAPZ(np.asarray(f), grid))
