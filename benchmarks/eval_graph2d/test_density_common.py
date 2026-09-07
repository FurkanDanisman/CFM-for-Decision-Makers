"""Validation gates for density_common.py. Runs WITHOUT any model or GPU.

Every number the Tier-C eval produces depends on these passing, so run this
first and after any edit to density_common.py:

    python benchmarks/eval_graph2d/test_density_common.py

Gates
  1  Joint2D.density      == exp(-neg_log_prob_2d)        (the 9-region head)
  2  UWYK1D.density       == exp(_logpdf_from_pred)       (the 1D bar head)
  3  densities normalise  \\int f = 1, \\int\\int f = 1
  4  tau operator         \\int p(tau) dtau = 1 for both models
  5  joint == product     a joint built as an outer product must give the same
                          p(tau) as the independence path -- this is the one
                          that catches a wrong diagonal integration
  6  discretisation floor the best achievable score at J=32 vs K=1000, i.e.
                          how much of any model gap is just head resolution
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
for _p in (_REPO, os.path.join(_REPO, 'g4cfm', 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from losses.BarDistribution2D import neg_log_prob_2d                # noqa: E402
from Losses.BarDistribution import BarDistribution                  # noqa: E402

sys.path.insert(0, _HERE)
from density_common import (                                        # noqa: E402
    Joint2D, UWYK1D, independent_f2d, truth_tau_density,
    joint_tau_density, uwyk_tau_density, tau_density_quadrature,
    l2_distance, kl, mass, TAU_CENTERS, TAU_BIN,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok), detail))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))


# ---------------------------------------------------------------------------
def make_bar_distribution(K, lo=-1.0, hi=1.0):
    """A BarDistribution with fitted state, as restored from a checkpoint."""
    bd = BarDistribution(num_bars=K, min_width=1e-6, scale_floor=1e-3,
                         device=torch.device('cpu'))
    edges = torch.linspace(lo, hi, K + 1, dtype=torch.float32)
    bd.edges = edges
    bd.centers = 0.5 * (edges[:-1] + edges[1:])
    bd.widths = torch.diff(edges)
    base = float(max(bd.widths[0].item(), 1e-6))
    bd.base_s_left = torch.tensor(base)
    bd.base_s_right = torch.tensor(base)
    return bd



def realistic_pred(J, seed=0, w0_logit=3.5, tail_raw=-0.5, spread=0.25):
    """A prediction shaped like a TRAINED head: most mass in region 0, modest
    tail scales, a concentrated p_mat. Random N(0,1.5) logits put ~90% of the
    mass in the tail regions, which is nothing like a fitted model and makes
    grid-truncation dominate every normalisation check."""
    rng = np.random.default_rng(seed)
    edges = np.linspace(-1.0, 1.0, J + 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    m0 = rng.uniform(-0.3, 0.3)
    m1 = rng.uniform(-0.3, 0.3)
    q0 = np.exp(-0.5 * ((c - m0) / spread) ** 2); q0 /= q0.sum()
    q1 = np.exp(-0.5 * ((c - m1) / spread) ** 2); q1 /= q1.sum()
    pred = np.empty(J * J + 13)
    pred[:J * J] = np.log(np.maximum(np.outer(q0, q1), 1e-300)).reshape(-1)
    pred[J * J:J * J + 9] = 0.0
    pred[J * J] = w0_logit
    pred[J * J + 9:] = tail_raw
    return pred, edges



def gate7_knot_alignment():
    print('\nGate 7 - tau grid is tied to the model knots')
    from density_common import knots_aligned, TAU_BIN, TAU_CENTERS
    check('tau step divides both model bin widths', knots_aligned(),
          f'dtau={TAU_BIN:.6f}; 0.002/dtau={0.002/TAU_BIN:.1f}, '
          f'0.0625/dtau={0.0625/TAU_BIN:.1f}')
    check('grid is anchored at 0', float(np.abs(TAU_CENTERS).min()) == 0.0,
          'knots sit at multiples of the bin width, so 0 must be a node')

    # the property that makes trapezoid exact: interior is piecewise-LINEAR
    from density_common import _diag_sums_product, _interior_tau
    K = 1000
    be = np.linspace(-1.0, 1.0, K + 1); bw = float(be[1] - be[0])
    bc = 0.5 * (be[:-1] + be[1:])
    q = np.exp(-0.5 * ((bc + 0.05) / 0.11) ** 2); q /= q.sum()
    S = _diag_sums_product(q, q)
    t = np.linspace(10 * bw, 11 * bw, 9)
    v = _interior_tau(S, t, bw, 1.0)
    sec = float(np.abs(np.diff(v, 2)).max()) / max(float(np.abs(v).max()), 1e-30)
    check('interior is piecewise-linear between knots', sec < 1e-12,
          f'max |2nd difference| / scale = {sec:.1e}')


def gate8_tail_interpolation(J=32):
    print('\nGate 8 - decoupled tail quadrature stays accurate')
    # The tails are evaluated on a coarse tau grid and interpolated up, which
    # is what makes a 12001-point grid affordable. Bound the error that buys.
    import density_common as dc
    pred, edges = realistic_pred(J, seed=5)
    jt = Joint2D.from_pred(pred, J, edges)
    fast = joint_tau_density(jt, TAU_CENTERS, n_y0=4096)
    saved = dc.TAIL_TAU_STEP
    dc.TAIL_TAU_STEP = 1e-9                      # forces direct evaluation
    try:
        exact = joint_tau_density(jt, TAU_CENTERS, n_y0=4096)
    finally:
        dc.TAIL_TAU_STEP = saved
    d = abs(kl(exact, fast, TAU_CENTERS))
    check('interpolated tail == direct, in KL', d < 1e-3,
          f'KL(direct || interpolated) = {d:.2e} nats')

    m = mass(fast, TAU_CENTERS)
    check('midpoint quadrature conserves mass', abs(m - 1) < 1e-3,
          f'int p(tau) = {m:.6f}  (trapezoid gave a 1e-3 deficit here)')


# ---------------------------------------------------------------------------
def gate1_joint_matches_reference(seed=0, J=8, n=4000):
    print('\nGate 1 - Joint2D.density vs neg_log_prob_2d')
    rng = np.random.default_rng(seed)
    edges = np.linspace(-1.0, 1.0, J + 1)
    pred = rng.normal(0, 1.5, size=J * J + 13)

    # points spread across all 9 regions
    y0 = rng.uniform(-2.0, 2.0, n)
    y1 = rng.uniform(-2.0, 2.0, n)

    mine = Joint2D.from_pred(pred, J, edges).density(y0, y1)

    pt = torch.tensor(pred, dtype=torch.float64).view(1, 1, -1).expand(1, n, -1)
    ref = torch.exp(-neg_log_prob_2d(
        pt.contiguous(), torch.tensor(y0).view(1, n), torch.tensor(y1).view(1, n),
        J, torch.tensor(edges, dtype=torch.float64), reduce='none')).numpy().reshape(-1)

    rel = np.abs(mine - ref) / np.maximum(np.abs(ref), 1e-12)
    check('9-region density matches reference', rel.max() < 1e-5,
          f'max rel err {rel.max():.2e} over {n} pts in all regions')

    # region coverage, so the test is not silently only exercising region 0
    inside = ((np.abs(y0) <= 1) & (np.abs(y1) <= 1)).mean()
    check('test points cover inner + tails', 0.1 < inside < 0.5,
          f'{inside:.0%} inner, {1-inside:.0%} tail regions')


def gate2_uwyk_matches_reference(seed=1, K=64, n=3000):
    print('\nGate 2 - UWYK1D.density vs BarDistribution._logpdf_from_pred')
    rng = np.random.default_rng(seed)
    bd = make_bar_distribution(K)
    pred = rng.normal(0, 1.5, size=K + 4)
    y = rng.uniform(-2.0, 2.0, n)

    mine = UWYK1D.from_pred(
        pred, bd.edges.numpy(), bd.widths.numpy(),
        float(bd.base_s_left), float(bd.base_s_right), scale_floor=1e-3
    ).density(y)

    ref = torch.exp(bd._logpdf_from_pred(
        torch.tensor(pred, dtype=torch.float32).view(1, 1, -1).expand(1, n, -1).contiguous(),
        torch.tensor(y, dtype=torch.float32).view(1, n))).numpy().reshape(-1)

    rel = np.abs(mine - ref) / np.maximum(np.abs(ref), 1e-12)
    check('1D bar density matches reference', rel.max() < 1e-4,
          f'max rel err {rel.max():.2e} (fp32 reference)')


def gate2b_float32_bar_grid(K=1000):
    print('\nGate 2b - fp32-stored bar grid (K=1000, bw not a power of two)')
    # Regression for: "exact interior term assumes uniform bars". Checkpoints
    # store edges as float32; with bw=0.002 the stored widths wobble by ~5e-5
    # relative, which np.allclose's default rtol rejects. K=64 (bw = 2^-5) is
    # exactly representable and hides the bug -- this gate must use K=1000.
    bd = make_bar_distribution(K)                  # float32 edges, like a ckpt
    w = bd.widths.numpy().astype(np.float64)
    rel = float(np.abs(w - w.mean()).max() / w.mean())
    check('fp32 grid is non-uniform enough to matter', rel > 1e-5,
          f'max rel width deviation {rel:.2e} (np.allclose default rtol=1e-5)')

    rng = np.random.default_rng(11)
    bc = 0.5 * (bd.edges.numpy()[:-1] + bd.edges.numpy()[1:])

    def trained_like(mu, sd=0.11, tail=1e-3):
        """What a fitted head emits: mass smooth over ~50 bars, tiny tails.
        Random logits instead give a tau density with structure at the BAR
        scale (0.002). The OLD tau grid (dtau=0.01, bin midpoints) aliased
        that; the current grid (dtau=0.0005, anchored at 0) contains every bar
        knot, so the diagnostic below should now sit at ~1.0."""
        q = np.exp(-0.5 * ((bc - mu) / sd) ** 2); q /= q.sum()
        return UWYK1D(log_pL=np.log(tail), log_pBars=np.log(q * (1 - 2 * tail)),
                      log_pR=np.log(tail), sL=0.002, sR=0.002,
                      edges=bd.edges.numpy().astype(np.float64),
                      widths=bd.widths.numpy().astype(np.float64))

    f0, f1 = trained_like(-0.05), trained_like(0.12)
    try:
        m = mass(uwyk_tau_density(f0, f1, TAU_CENTERS, n_y0=4096), TAU_CENTERS)
        ok, detail = abs(m - 1) < 1e-3, f'int p(tau) = {m:.6f}'
    except ValueError as e:
        ok, detail = False, f'raised: {e}'
    check('uwyk_tau_density accepts an fp32 bar grid', ok, detail)

    e32 = np.linspace(-1.0, 1.0, 33)
    m = mass(uwyk_tau_density(f0.rebin(e32), f1.rebin(e32), TAU_CENTERS,
                              n_y0=4096), TAU_CENTERS)
    check('rebin to J=32 still integrates to 1', abs(m - 1) < 1e-3,
          f'int p(tau) = {m:.6f}')

    # Aliasing limit, reported not asserted: with random logits the K=1000
    # tau density varies at the 0.002 bar scale, which the old grid could not
    # resolve it. The eval's `mass` column is the live guard for this --
    # summarize_density_tauC warns if it strays >1% from 1.
    g0 = UWYK1D.from_pred(rng.normal(0, 1.5, K + 4), bd.edges.numpy(),
                          bd.widths.numpy(), float(bd.base_s_left),
                          float(bd.base_s_right))
    g1 = UWYK1D.from_pred(rng.normal(0, 1.5, K + 4), bd.edges.numpy(),
                          bd.widths.numpy(), float(bd.base_s_left),
                          float(bd.base_s_right))
    m = mass(uwyk_tau_density(g0, g1, TAU_CENTERS, n_y0=4096), TAU_CENTERS)
    print(f'    (diagnostic) random logits, structure at bar scale 0.002 vs '
          f'dtau={TAU_BIN:.5f}: int p(tau) = {m:.6f}')


def gate3_normalisation(J=16):
    print('\nGate 3 - densities integrate to 1 (trapezoid converges from below)')
    pred, edges = realistic_pred(J, seed=2)
    jt = Joint2D.from_pred(pred, J, edges)
    print(f'    w0={jt.w[0]:.3f}  tail mass={1-jt.w[0]:.3f}  max tail sigma={jt.max_scale:.4f}')
    pad = 10.0 * jt.max_scale
    tots = []
    for n in (1500, 4500, 9000):
        g = np.linspace(-1 - pad, 1 + pad, n)
        tot = 0.0
        for i in range(0, n, 500):        # chunk: density() allocates ~10 temporaries
            gi = g[i:i + 500]
            tot += np.trapezoid(np.trapezoid(jt.density(gi[:, None], g[None, :]), g, axis=1), gi)
        tots.append(tot)
        print(f'    n={n:6d}  int int f = {tot:.6f}')
    check('joint integrates to 1 (finest grid)', abs(tots[-1] - 1) < 3e-3,
          f'{tots[-1]:.6f}; trapezoid on a staircase converges as O(h)')
    check('joint normalisation converging', abs(tots[-1] - 1) < abs(tots[0] - 1),
          f'|err| {abs(tots[0]-1):.2e} -> {abs(tots[-1]-1):.2e}')

    bd = make_bar_distribution(64)
    rng = np.random.default_rng(2)
    f0 = UWYK1D.from_pred(rng.normal(0, 1.5, size=68), bd.edges.numpy(),
                          bd.widths.numpy(), float(bd.base_s_left),
                          float(bd.base_s_right))
    g1 = np.linspace(-1 - 10 * f0.max_scale, 1 + 10 * f0.max_scale, 200001)
    tot1 = np.trapezoid(f0.density(g1), g1)
    check('UWYK 1D integrates to 1', abs(tot1 - 1) < 1e-4, f'int f = {tot1:.6f}')


def gate4_tau_normalisation(J=16):
    print('\nGate 4 - tau operator preserves mass (exact interior + tail quad)')
    pred, edges = realistic_pred(J, seed=3)
    jt = Joint2D.from_pred(pred, J, edges)
    m = mass(joint_tau_density(jt, TAU_CENTERS, n_y0=16384), TAU_CENTERS)
    check('joint p(tau) integrates to 1', abs(m - 1) < 1e-3, f'int p(tau) = {m:.6f}')

    bd = make_bar_distribution(64)
    rng = np.random.default_rng(3)
    f0 = UWYK1D.from_pred(rng.normal(0, 1.5, 68), bd.edges.numpy(), bd.widths.numpy(),
                          float(bd.base_s_left), float(bd.base_s_right))
    f1 = UWYK1D.from_pred(rng.normal(0, 1.5, 68), bd.edges.numpy(), bd.widths.numpy(),
                          float(bd.base_s_left), float(bd.base_s_right))
    m = mass(uwyk_tau_density(f0, f1, TAU_CENTERS, n_y0=16384), TAU_CENTERS)
    check('UWYK p(tau) integrates to 1', abs(m - 1) < 1e-3, f'int p(tau) = {m:.6f}')

    # Mass lost off the ends of the tau grid is a REPORTED diagnostic, not a
    # failure -- with fat tail scales it is real model mass outside [-3, 3].
    jt_fat = Joint2D.from_pred(
        np.random.default_rng(9).normal(0, 1.5, J * J + 13), J, edges)
    m_fat = mass(joint_tau_density(jt_fat, TAU_CENTERS, n_y0=16384), TAU_CENTERS)
    print(f'    (diagnostic) pathological pred w0={jt_fat.w[0]:.3f}: '
          f'int p(tau) = {m_fat:.6f} -- mass off the tau grid, not an error')


def gate4b_exact_vs_bruteforce(J=12):
    print('\nGate 4b - exact interior formula vs brute-force quadrature')
    # Brute-force trapezoid is O(h)-biased on a staircase, so the right
    # assertion is that it CONVERGES to the exact formula as h -> 0, not that
    # it agrees at any fixed n.
    pred, edges = realistic_pred(J, seed=7, w0_logit=60.0, tail_raw=-20.0)
    jt = Joint2D.from_pred(pred, J, edges)      # interior only, isolates the formula
    taus = np.linspace(-1.7, 1.7, 41)
    fast = joint_tau_density(jt, taus)
    errs = []
    for n in (12501, 50001, 200001):
        slow = tau_density_quadrature(jt.density, taus, jt.lo, jt.hi,
                                      pad=8.0 * jt.max_scale, n_y0=n)
        errs.append(np.abs(fast - slow).max() / max(slow.max(), 1e-12))
        print(f'    brute n={n:7d}  max rel diff vs exact = {errs[-1]:.2e}')
    check('brute force converges to the exact formula',
          errs[-1] < errs[0] / 3 and errs[-1] < 3e-4,
          f'{errs[0]:.1e} -> {errs[-1]:.1e} as h shrinks 16x')


def gate5_joint_equals_product(seed=4, J=32):
    print('\nGate 5 - outer-product joint == independence path  (DECISIVE)')
    # Build a joint whose p_mat factorises and whose mass is entirely in the
    # inner region, plus 1D bars carrying the same marginals with no tail mass.
    # The two code paths must then agree; if the diagonal integration is wrong,
    # this is where it shows.
    rng = np.random.default_rng(seed)
    edges = np.linspace(-1.0, 1.0, J + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bw = float(edges[1] - edges[0])

    p0 = np.exp(-0.5 * ((centers - 0.10) / 0.22) ** 2); p0 /= p0.sum()
    p1 = np.exp(-0.5 * ((centers + 0.15) / 0.30) ** 2); p1 /= p1.sum()

    pred = np.empty(J * J + 13)
    pred[:J * J] = np.log(np.maximum(np.outer(p0, p1), 1e-300)).reshape(-1)
    pred[J * J:J * J + 9] = -50.0
    pred[J * J] = 50.0                        # all weight on region 0
    pred[J * J + 9:] = -20.0                  # negligible tail scales
    jt = Joint2D.from_pred(pred, J, edges)

    bars = [UWYK1D(log_pL=-800.0, log_pBars=np.log(p), log_pR=-800.0,
                   sL=1e-3, sR=1e-3, edges=edges, widths=np.diff(edges))
            for p in (p0, p1)]

    a = joint_tau_density(jt, TAU_CENTERS)
    b = uwyk_tau_density(*bars, TAU_CENTERS)
    d = np.abs(a - b).max() / max(b.max(), 1e-12)
    check('diagonal integration == convolution', d < 1e-9,
          f'max rel diff {d:.2e}  (bw={bw:.4f})')


def gate6_discretisation_floor():
    print('\nGate 6 - discretisation floor: best possible score per head grid')
    # A model that is EXACTLY right still pays a penalty for representing a
    # smooth density on its bin grid. Bin the analytic truth onto each head's
    # grid, push it through the same tau operator, and score against the truth.
    # The gap between the two floors is the resolution handicap -- everything
    # in the model comparison must be read against it.
    sigma, mu0, mu1 = 0.1041, -0.05, 0.12        # IHDP scaled-axis magnitudes
    truth = truth_tau_density(mu0, mu1, sigma, TAU_CENTERS)

    print(f'    truth: tau ~ N({mu1-mu0:+.3f}, ({math.sqrt(2)*sigma:.4f})^2)'
          f'   sigma_arm/bin(J=32) = {sigma/0.0625:.2f}')
    rows = []
    for J in (32, 1000):
        edges = np.linspace(-1.0, 1.0, J + 1)
        # exact bin masses of the true product density (not centre sampling)
        from scipy.stats import norm as _norm
        c0 = _norm.cdf(edges, mu0, sigma); c1 = _norm.cdf(edges, mu1, sigma)
        q0, q1 = np.diff(c0), np.diff(c1)
        q0 /= q0.sum(); q1 /= q1.sum()
        pm = np.outer(q0, q1)

        pred = np.empty(J * J + 13)
        pred[:J * J] = np.log(np.maximum(pm, 1e-300)).reshape(-1)
        pred[J * J:J * J + 9] = -50.0
        pred[J * J] = 50.0
        pred[J * J + 9:] = -20.0
        est = joint_tau_density(Joint2D.from_pred(pred, J, edges), TAU_CENTERS)
        rows.append((J, l2_distance(truth, est, TAU_CENTERS),
                     kl(truth, est, TAU_CENTERS)))
        print(f'    J={J:<5d} L2 = {rows[-1][1]:.5f}   KL_fwd = {rows[-1][2]:.6f} nats')

    gap = rows[0][2] - rows[1][2]
    print(f'    -> resolution handicap (J=32 minus K=1000): {gap:+.6f} nats')
    print(f'       compare with the ~0.013 nat effect of a spurious rho=0.2')
    check('floors computed', True, f'J=32 KL floor {rows[0][2]:.6f} nats')
    return rows


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    gate1_joint_matches_reference()
    gate2_uwyk_matches_reference()
    gate2b_float32_bar_grid()
    gate3_normalisation()
    gate4_tau_normalisation()
    gate4b_exact_vs_bruteforce()
    gate5_joint_equals_product()
    gate6_discretisation_floor()
    gate7_knot_alignment()
    gate8_tail_interpolation()

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f'\n{"="*66}\n{len(RESULTS)-n_fail}/{len(RESULTS)} gates passed')
    for name, ok, detail in RESULTS:
        if not ok:
            print(f'  FAILED: {name}  {detail}')
    sys.exit(1 if n_fail else 0)
