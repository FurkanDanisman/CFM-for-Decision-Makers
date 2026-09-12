"""Closed-form validation for interval_metrics.py. Run: python -m unittest -v"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interval_metrics import (                                    # noqa: E402
    predictive_cdf, interval_equal_tailed, interval_hpd, crps, winkler,
    query_metrics, summarize,
)

_ncdf = lambda x: 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x) / np.sqrt(2)))
_npdf = lambda x: np.exp(-np.asarray(x) ** 2 / 2) / np.sqrt(2 * np.pi)
GRID = np.linspace(-3.0, 3.0, 12001)          # the real TAU_CENTERS geometry


def gauss(mu, sig, grid=GRID):
    return _npdf((grid - mu) / sig) / sig


def crps_gauss_closed(mu, sig, y):
    w = (y - mu) / sig
    return float(sig * (w * (2 * _ncdf(w) - 1) + 2 * _npdf(w) - 1 / np.sqrt(np.pi)))


class TestCDF(unittest.TestCase):
    def test_mass_is_one_for_contained_density(self):
        self.assertAlmostEqual(predictive_cdf(gauss(0, 0.2), GRID)[-1], 1.0, places=9)

    def test_negative_density_rejected(self):
        p = gauss(0, 0.2).copy(); p[10] = -1e-9
        with self.assertRaises(ValueError):
            predictive_cdf(p, GRID)


class TestCRPS(unittest.TestCase):
    def test_matches_gaussian_closed_form(self):
        for sig in (0.05, 0.1, 0.25, 0.5):
            for y in (0.0, 0.1, -0.3):
                got = crps(gauss(0.0, sig), GRID, y)
                want = crps_gauss_closed(0.0, sig, y)
                self.assertAlmostEqual(got, want, places=6, msg=f'sig={sig} y={y}')

    def test_centred_gaussian_is_0p2337_sigma(self):
        for sig in (0.05, 0.2, 0.4):
            self.assertAlmostEqual(crps(gauss(0, sig), GRID, 0.0) / sig,
                                   0.23370, places=4)

    def test_point_mass_reduces_to_absolute_error(self):
        """CRPS -> |mu - y| as the density narrows.

        A Gaussian is not a point mass: for |mu-y| >> sigma the closed form is
        exactly |mu-y| - sigma/sqrt(pi), so the limit is approached from BELOW
        at a known rate. Assert that rate rather than a bare tolerance.
        """
        mu, y = 0.10, 0.0
        for sig in (4e-3, 2e-3, 1e-3):
            got = crps(gauss(mu, sig), GRID, y)
            want = crps_gauss_closed(mu, sig, y)
            # These sigmas are only 2-8 grid cells wide (dz=5e-4), so the
            # density itself is under-resolved; 1e-4 is the honest tolerance.
            self.assertAlmostEqual(got, want, places=4, msg=f'sig={sig}')
            # closed-form gap to |mu-y| is sigma/sqrt(pi) = 0.5642*sigma
            self.assertAlmostEqual(abs(mu - y) - want, sig / np.sqrt(np.pi),
                                   places=6, msg=f'sig={sig}')
        # and it is monotonically approaching |mu-y|
        vals = [crps(gauss(mu, sg), GRID, y) for sg in (4e-3, 2e-3, 1e-3)]
        self.assertLess(vals[0], vals[1])
        self.assertLess(vals[1], vals[2])
        self.assertLess(vals[2], abs(mu - y))

    def test_scales_linearly_with_y_scale(self):
        base = query_metrics(gauss(0, 0.2), GRID, 0.05, y_scale=1.0)['crps']
        scaled = query_metrics(gauss(0, 0.2), GRID, 0.05, y_scale=7.0)['crps']
        self.assertAlmostEqual(scaled, 7.0 * base, places=9)


class TestEqualTailed(unittest.TestCase):
    def test_matches_gaussian_quantiles(self):
        z = {0.50: 0.674489750, 0.80: 1.281551566,
             0.90: 1.644853627, 0.95: 1.959963985}
        sig = 0.3
        for a, zz in [(0.5, z[0.50]), (0.2, z[0.80]), (0.1, z[0.90]), (0.05, z[0.95])]:
            lo, hi, cens = interval_equal_tailed(gauss(0.0, sig), GRID, a)
            self.assertFalse(cens)
            self.assertAlmostEqual(lo, -zz * sig, places=5)
            self.assertAlmostEqual(hi, +zz * sig, places=5)

    def test_censoring_detected_when_tails_leave_grid(self):
        # sigma=2 on a [-3,3] grid: the 95% interval needs +/-3.92
        _, _, cens = interval_equal_tailed(gauss(0.0, 2.0), GRID, 0.05)
        self.assertTrue(cens)


class TestHPD(unittest.TestCase):
    def test_symmetric_unimodal_matches_equal_tailed(self):
        p = gauss(0.0, 0.3)
        for a in (0.5, 0.2, 0.05):
            lo_e, hi_e, _ = interval_equal_tailed(p, GRID, a)
            lo_h, hi_h, _, disj, meas, _ = interval_hpd(p, GRID, a)
            self.assertFalse(disj)
            self.assertAlmostEqual(lo_h, lo_e, places=2)
            self.assertAlmostEqual(hi_h, hi_e, places=2)
            self.assertAlmostEqual(meas, hi_e - lo_e, places=2)

    def test_hpd_no_longer_than_equal_tailed_when_skewed(self):
        p = 0.7 * gauss(-0.2, 0.10) + 0.3 * gauss(0.6, 0.35)
        p /= np.trapezoid(p, GRID) if hasattr(np, 'trapezoid') else np.trapz(p, GRID)
        for a in (0.2, 0.1):
            lo_e, hi_e, _ = interval_equal_tailed(p, GRID, a)
            _, _, _, _, meas, _ = interval_hpd(p, GRID, a)
            # the MEASURE is what must be no longer; the hull may exceed it
            # when the region is disjoint (that is what `disjoint` is for)
            self.assertLessEqual(meas, (hi_e - lo_e) + 1e-6)

    def test_bimodal_flagged_disjoint(self):
        p = 0.5 * gauss(-1.0, 0.08) + 0.5 * gauss(1.0, 0.08)
        p /= np.trapezoid(p, GRID) if hasattr(np, 'trapezoid') else np.trapz(p, GRID)
        _, _, _, disj, _, _ = interval_hpd(p, GRID, 0.2)
        self.assertTrue(disj)


class TestWinkler(unittest.TestCase):
    def test_inside_interval_is_just_length(self):
        self.assertAlmostEqual(winkler(-1.0, 1.0, 0.5, 0.05), 2.0)

    def test_miss_penalty_formula(self):
        # y=1.5 is 0.5 above hi=1.0, alpha=0.05 -> 2 + (2/0.05)*0.5 = 22
        self.assertAlmostEqual(winkler(-1.0, 1.0, 1.5, 0.05), 22.0)
        self.assertAlmostEqual(winkler(-1.0, 1.0, -1.5, 0.05), 22.0)

    def test_properness_widening_does_not_pay_when_already_covering(self):
        narrow = winkler(-0.5, 0.5, 0.0, 0.1)
        wide = winkler(-5.0, 5.0, 0.0, 0.1)
        self.assertLess(narrow, wide)

    def test_properness_truth_minimises_expected_score(self):
        # honest N(0,sigma) interval should beat both an over- and under-wide one
        rng = np.random.default_rng(0)
        y = rng.normal(0.0, 0.3, 20000)
        a = 0.1
        z = 1.644853627
        def mean_w(scale):
            lo, hi = -z * 0.3 * scale, z * 0.3 * scale
            return np.mean([winkler(lo, hi, v, a) for v in y])
        self.assertLess(mean_w(1.0), mean_w(0.5))
        self.assertLess(mean_w(1.0), mean_w(2.0))


class TestCoverageEndToEnd(unittest.TestCase):
    def test_calibrated_forecast_attains_nominal_coverage(self):
        rng = np.random.default_rng(7)
        sig = 0.25
        per_q = []
        for _ in range(1500):
            mu = rng.normal(0, 0.4)
            y = rng.normal(mu, sig)              # truth FROM the forecast law
            if abs(y) > 2.5:
                continue
            per_q.append(query_metrics(gauss(mu, sig), GRID, y))
        s = summarize(per_q)
        for a in (0.5, 0.2, 0.1, 0.05):
            d = s['levels'][a]
            self.assertLess(abs(d['coverage_pct'] - d['nominal_pct']),
                            4.0 * max(d['coverage_se_pct'], 0.5),
                            msg=f"level {d['nominal_pct']}: got {d['coverage_pct']:.1f}")

    def test_overconfident_forecast_undercovers(self):
        rng = np.random.default_rng(11)
        sig = 0.25
        per_q = [query_metrics(gauss(0.0, sig / 3), GRID, rng.normal(0, sig))
                 for _ in range(800)]
        s = summarize(per_q)
        self.assertLess(s['levels'][0.05]['coverage_pct'], 90.0)

    def test_diffuse_forecast_overcovers_and_is_long(self):
        rng = np.random.default_rng(13)
        sig = 0.25
        tight = [query_metrics(gauss(0.0, sig), GRID, rng.normal(0, sig))
                 for _ in range(800)]
        rng = np.random.default_rng(13)
        loose = [query_metrics(gauss(0.0, sig * 3), GRID, rng.normal(0, sig))
                 for _ in range(800)]
        st, sl = summarize(tight), summarize(loose)
        self.assertGreater(sl['levels'][0.05]['coverage_pct'],
                           st['levels'][0.05]['coverage_pct'])
        self.assertGreater(sl['levels'][0.05]['length_mean'],
                           st['levels'][0.05]['length_mean'])
        # ... and Winkler + CRPS both catch the over-width
        self.assertGreater(sl['levels'][0.05]['winkler_mean'],
                           st['levels'][0.05]['winkler_mean'])
        self.assertGreater(sl['crps_mean'], st['crps_mean'])


class TestIndependenceInflation(unittest.TestCase):
    def test_sqrt2_inflation_costs_sqrt2_crps(self):
        """Y0 _|_ Y1 inflates sd by sqrt(2); CRPS should degrade by sqrt(2)."""
        s0 = 0.15
        joint = crps(gauss(0.0, s0), GRID, 0.0)
        conv = crps(gauss(0.0, s0 * np.sqrt(2)), GRID, 0.0)
        self.assertAlmostEqual(conv / joint, np.sqrt(2), places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
