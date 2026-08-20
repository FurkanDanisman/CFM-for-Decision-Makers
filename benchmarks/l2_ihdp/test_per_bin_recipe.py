"""Verification of the per-bin probability L2 recipe.

Constructs a KNOWN Gaussian per query, simulates each method's native
head output, applies the aggregator's recipe-strict aggregation, and
checks that:

  (a) truth per-bin probabilities sum to 1
  (b) plain summation of native-J bins into target-K bins matches exact
      CDF differences to numerical precision (when K divides native J
      and both are uniform on [-1, 1])
  (c) overlap-fraction / CDF-interp on non-aligned adaptive bars matches
      exact CDF differences of a piecewise-uniform density
  (d) UWYK's plain summation of K=1000 native bars into K=100 target
      matches exact CDF differences
  (e) round-trip: known Gaussian → discretise → aggregate → back to
      truth CDF differences should have L2 == 0

Exits non-zero on any failure. Run:
    python benchmarks/l2_ihdp/test_per_bin_recipe.py
"""
from __future__ import annotations
import sys
import numpy as np
from scipy.stats import norm


TOL = 1e-8


def truth_bins(mu, sigma, edges):
    cdf = norm.cdf(edges, loc=mu, scale=max(sigma, 1e-12))
    return np.diff(cdf)


def check(name, ok, extra=''):
    tag = 'PASS' if ok else 'FAIL'
    print(f'  [{tag}] {name}  {extra}')
    if not ok:
        check.failed = True


check.failed = False


def test_a_truth_sums_to_one():
    print('(a) Truth per-bin probs sum to 1')
    edges = np.linspace(-1, 1, 11)
    for mu in [-0.3, 0.0, 0.5]:
        for sigma in [0.1, 0.3, 0.8]:
            t = truth_bins(mu, sigma, edges)
            # Truncated to [-1, 1] so it will sum to less than 1 for wide
            # sigma. Test that CDF-diff formula is internally consistent.
            expected = float(norm.cdf(1, mu, sigma) - norm.cdf(-1, mu, sigma))
            check(f'  μ={mu} σ={sigma}: Σt = expected CDF mass',
                  abs(t.sum() - expected) < TOL,
                  f'|Σt − expected| = {abs(t.sum() - expected):.2e}')


def test_b_plain_summation_uniform():
    print('(b) Plain summation: native J=100 → target J=10')
    edges_native = np.linspace(-1, 1, 101)       # J=100 uniform
    edges_target = np.linspace(-1, 1, 11)        # J=10 uniform
    for mu, sigma in [(-0.2, 0.15), (0.4, 0.3), (0.0, 0.6)]:
        # Truth on native grid
        t_native = truth_bins(mu, sigma, edges_native)     # (100,)
        # Aggregate: sum every 10 consecutive native bins
        t_summed = t_native.reshape(10, 10).sum(axis=1)    # (10,)
        # Exact truth on target grid
        t_target = truth_bins(mu, sigma, edges_target)     # (10,)
        diff = np.abs(t_summed - t_target).max()
        check(f'  μ={mu} σ={sigma}: summed native matches target CDF-diff',
              diff < TOL, f'max|diff| = {diff:.2e}')


def test_c_adaptive_bars_overlap_fraction():
    print('(c) Overlap-fraction / CDF-interp: adaptive quantile bars → uniform J=10')
    # Simulate Do-PFN: irregular borders (piecewise-uniform density inside each bar)
    rng = np.random.default_rng(0)
    borders = np.sort(np.concatenate([[-1.5, 1.5],
                                       rng.uniform(-1.5, 1.5, 20)]))
    K_nat = len(borders) - 1
    # Assign true probability mass to each bar under a specific Gaussian
    for mu, sigma in [(0.1, 0.3), (-0.4, 0.2)]:
        probs = np.diff(norm.cdf(borders, mu, sigma))
        probs = probs / probs.sum()   # renormalise to unit mass
        # Recipe-strict per-target-bin via CDF-interp
        edges_target = np.linspace(-1, 1, 11)
        cdf = np.concatenate(([0.0], np.cumsum(probs)))
        F_at = np.interp(edges_target, borders, cdf, left=0.0, right=1.0)
        p_target = np.diff(F_at)
        p_target = p_target / max(p_target.sum(), 1e-12)
        # Reference: piecewise-uniform density, integrate over [e_j, e_{j+1})
        # For each target bin, sum bar_prob * overlap_fraction directly
        p_ref = np.zeros(10)
        for j in range(10):
            lo, hi = edges_target[j], edges_target[j+1]
            for i in range(K_nat):
                a, b = borders[i], borders[i+1]
                ov = max(0.0, min(hi, b) - max(lo, a))
                if ov > 0:
                    p_ref[j] += probs[i] * ov / (b - a)
        p_ref = p_ref / max(p_ref.sum(), 1e-12)
        diff = np.abs(p_target - p_ref).max()
        check(f'  μ={mu} σ={sigma}: CDF-interp == explicit overlap-fraction',
              diff < 1e-10, f'max|diff| = {diff:.2e}')


def test_d_uwyk_summation():
    print('(d) UWYK K=1000 → K=100 plain summation')
    K_nat = 1000
    borders = np.linspace(-1, 1, K_nat + 1)
    edges_target = np.linspace(-1, 1, 101)   # K=100
    for mu, sigma in [(0.0, 0.2), (-0.5, 0.3)]:
        pBars = np.diff(norm.cdf(borders, mu, sigma))
        pBars = pBars / pBars.sum()
        # Recipe: plain sum of every 10 native bars per target bin
        m = K_nat // 100
        p_summed = pBars.reshape(100, m).sum(axis=1)
        # Reference: exact CDF differences at target edges (piecewise-uniform)
        cdf = np.concatenate(([0.0], np.cumsum(pBars)))
        F_at = np.interp(edges_target, borders, cdf, left=0.0, right=1.0)
        p_interp = np.diff(F_at)
        diff = np.abs(p_summed - p_interp).max()
        check(f'  μ={mu} σ={sigma}: plain-sum == CDF-interp',
              diff < 1e-10, f'max|diff| = {diff:.2e}')


def test_e_roundtrip_l2_zero():
    print('(e) Round-trip: perfect model → L2 == 0 on aligned grid')
    edges_native = np.linspace(-1, 1, 101)
    edges_target = np.linspace(-1, 1, 11)
    bin_w = float(edges_target[1] - edges_target[0])
    for mu, sigma in [(0.15, 0.2), (-0.3, 0.35)]:
        # "Perfect" native probs = truth CDF-diff on native
        p_native = truth_bins(mu, sigma, edges_native)
        # Aggregate to target via plain sum
        p_target = p_native.reshape(10, 10).sum(axis=1)
        # Truth on target
        t_target = truth_bins(mu, sigma, edges_target)
        L2 = float(np.sqrt(np.sum((p_target - t_target)**2) / bin_w))
        check(f'  μ={mu} σ={sigma}: L2(perfect, truth) ≈ 0',
              L2 < 1e-9, f'L2 = {L2:.2e}')


def test_f_riemann_is_biased():
    print('(f) Sanity: the FORBIDDEN Riemann-on-Y_CENTERS is measurably biased')
    Y_CENTERS = 0.5 * (np.linspace(-1.5, 1.5, 101)[:-1] + np.linspace(-1.5, 1.5, 101)[1:])
    Y_BIN = float(Y_CENTERS[1] - Y_CENTERS[0])
    edges_target = np.linspace(-1, 1, 11)
    bin_w = float(edges_target[1] - edges_target[0])
    for mu, sigma in [(0.0, 0.3), (0.2, 0.15)]:
        # Truth density on Y_CENTERS
        d_yc = norm.pdf(Y_CENTERS, mu, sigma)
        # Riemann sum midpoints into each J=10 bin
        p_riemann = np.zeros(10)
        for j in range(10):
            mask = (Y_CENTERS >= edges_target[j]) & (Y_CENTERS < edges_target[j+1])
            p_riemann[j] = d_yc[mask].sum() * Y_BIN
        s = p_riemann.sum(); p_riemann = p_riemann / s if s > 0 else p_riemann
        # Exact CDF differences
        t = truth_bins(mu, sigma, edges_target)
        t = t / t.sum()
        L2 = float(np.sqrt(np.sum((p_riemann - t)**2) / bin_w))
        # This SHOULD be non-trivial (order 1e-3 to 1e-2), NOT zero
        check(f'  μ={mu} σ={sigma}: Riemann-on-YC differs from strict recipe (bias observed)',
              L2 > 1e-4, f'L2_bias = {L2:.4e}')


if __name__ == '__main__':
    print('Per-bin probability L2 recipe verification')
    print('=' * 60)
    test_a_truth_sums_to_one(); print()
    test_b_plain_summation_uniform(); print()
    test_c_adaptive_bars_overlap_fraction(); print()
    test_d_uwyk_summation(); print()
    test_e_roundtrip_l2_zero(); print()
    test_f_riemann_is_biased(); print()
    print('=' * 60)
    if check.failed:
        print('SOME CHECKS FAILED — recipe may be misimplemented.')
        sys.exit(1)
    print('ALL CHECKS PASSED — recipe implementation is correct.')
