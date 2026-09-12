"""Validation for density_cpfn.py. Run: python -m unittest test_density_cpfn -v"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from density_common import joint_tau_density, uwyk_tau_density      # noqa: E402
from density_cpfn import (                                          # noqa: E402
    CPFN1D, CPFN2D, cpfn1d_tau_density, cpfn2d_tau_density,
    load_dump, iter_queries, tau_support, support_diagnostics,
)

_TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
J = 32
EDGES = np.linspace(-2.0, 2.0, J + 1)
TAU = np.linspace(-3.0, 3.0, 12001)          # real TAU_CENTERS geometry
# tau support for edges spanning [-2, 2] is +/-4, i.e. WIDER than TAU_CENTERS.
# Exactness properties (mass, mean) are asserted on a grid that contains the
# whole support; TAU is used where the point is to exercise the real geometry.
TAU_FULL = np.linspace(-4.5, 4.5, 18001)


def _gauss_hist(mu, sig, edges=EDGES):
    c = 0.5 * (edges[:-1] + edges[1:])
    p = np.exp(-0.5 * ((c - mu) / sig) ** 2)
    return p / p.sum()


def _corr_joint(mu0, mu1, s0, s1, rho, edges=EDGES):
    """Discretised bivariate normal with correlation rho."""
    c = 0.5 * (edges[:-1] + edges[1:])
    z0 = (c[:, None] - mu0) / s0
    z1 = (c[None, :] - mu1) / s1
    q = (z0 ** 2 - 2 * rho * z0 * z1 + z1 ** 2) / (1 - rho ** 2)
    p = np.exp(-0.5 * q)
    return p / p.sum()


class TestEquivalenceWithGenericPath(unittest.TestCase):
    """The dedicated exact routines must equal density_common's generic
    interior+quadrature functions when the tails are empty. This validates the
    new code against machinery that is already tested upstream."""

    def test_2d_matches_joint_tau_density(self):
        for rho in (-0.6, 0.0, 0.5):
            jt = CPFN2D.from_dump(_corr_joint(0.1, 0.4, 0.5, 0.6, rho), EDGES)
            mine = cpfn2d_tau_density(jt, TAU)
            generic = joint_tau_density(jt.to_joint2d(), TAU)
            self.assertLess(float(np.abs(mine - generic).max()), 1e-9,
                            msg=f'rho={rho}')

    def test_1d_matches_uwyk_tau_density(self):
        f0 = CPFN1D.from_dump(_gauss_hist(-0.2, 0.45), EDGES)
        f1 = CPFN1D.from_dump(_gauss_hist(0.35, 0.5), EDGES)
        mine = cpfn1d_tau_density(f0, f1, TAU)
        generic = uwyk_tau_density(f0.to_uwyk1d(), f1.to_uwyk1d(), TAU)
        self.assertLess(float(np.abs(mine - generic).max()), 1e-9)


class TestDensityProperties(unittest.TestCase):
    def test_tau_density_integrates_to_one(self):
        jt = CPFN2D.from_dump(_corr_joint(0.0, 0.3, 0.5, 0.5, 0.4), EDGES)
        self.assertAlmostEqual(_TRAPZ(cpfn2d_tau_density(jt, TAU_FULL), TAU_FULL),
                               1.0, places=9)
        f0 = CPFN1D.from_dump(_gauss_hist(0.0, 0.4), EDGES)
        f1 = CPFN1D.from_dump(_gauss_hist(0.3, 0.4), EDGES)
        self.assertAlmostEqual(
            _TRAPZ(cpfn1d_tau_density(f0, f1, TAU_FULL), TAU_FULL), 1.0, places=9)

    def test_tau_density_mean_equals_cate(self):
        jt = CPFN2D.from_dump(_corr_joint(-0.1, 0.5, 0.5, 0.55, 0.3), EDGES)
        p = cpfn2d_tau_density(jt, TAU_FULL)
        self.assertAlmostEqual(float(_TRAPZ(p * TAU_FULL, TAU_FULL)), jt.cate(),
                               places=8)

    def test_compact_support_is_exactly_zero_outside(self):
        jt = CPFN2D.from_dump(_corr_joint(0.0, 0.0, 0.4, 0.4, 0.0), EDGES)
        lo, hi = tau_support(EDGES)
        self.assertAlmostEqual(lo, -4.0); self.assertAlmostEqual(hi, 4.0)
        far = np.array([-5.0, -4.5, 4.5, 5.0])
        self.assertTrue(np.all(cpfn2d_tau_density(jt, far) == 0.0))

    def test_product_joint_equals_independent_arms(self):
        p0, p1 = _gauss_hist(-0.1, 0.4), _gauss_hist(0.3, 0.5)
        jt = CPFN2D.from_dump(np.outer(p0, p1), EDGES)
        a = cpfn2d_tau_density(jt, TAU)
        b = cpfn1d_tau_density(CPFN1D.from_dump(p0, EDGES),
                               CPFN1D.from_dump(p1, EDGES), TAU)
        self.assertLess(float(np.abs(a - b).max()), 1e-12)


class TestGridTruncation(unittest.TestCase):
    """CausalPFN's tau support (+/- the y-range span) can EXCEED TAU_CENTERS'
    [-3, 3]. That truncation biases mass, mean, length and CRPS downward, so it
    must be detectable rather than silent."""

    def test_tau_support_exceeds_tau_centers(self):
        lo, hi = tau_support(EDGES)
        self.assertLess(lo, TAU[0])
        self.assertGreater(hi, TAU[-1])

    def test_truncation_loses_mass_on_the_real_grid(self):
        # a deliberately wide prediction: most mass beyond |tau| = 3
        jt = CPFN2D.from_dump(_corr_joint(0.0, 0.0, 1.4, 1.4, -0.8), EDGES)
        m_full = _TRAPZ(cpfn2d_tau_density(jt, TAU_FULL), TAU_FULL)
        m_cut = _TRAPZ(cpfn2d_tau_density(jt, TAU), TAU)
        self.assertAlmostEqual(m_full, 1.0, places=8)
        self.assertLess(m_cut, 0.999)          # visibly short of 1
        # interval_metrics reports exactly this as `mass`, so it is catchable
        self.assertGreater(m_full - m_cut, 1e-3)


class TestIndependenceCost(unittest.TestCase):
    """The scientific claim: discarding a POSITIVE coupling inflates p(tau).
    Var(tau) = v0 + v1 - 2*rho*sqrt(v0 v1)."""

    def _sd(self, p, tau):
        m = float(_TRAPZ(p * tau, tau))
        return float(np.sqrt(_TRAPZ(p * (tau - m) ** 2, tau)))

    def test_positive_rho_joint_is_sharper_than_independent(self):
        for rho in (0.3, 0.6, 0.85):
            jt = CPFN2D.from_dump(_corr_joint(0.0, 0.4, 0.5, 0.5, rho), EDGES)
            f0, f1 = jt.independent()
            sd_j = self._sd(cpfn2d_tau_density(jt, TAU), TAU)
            sd_i = self._sd(cpfn1d_tau_density(f0, f1, TAU), TAU)
            self.assertLess(sd_j, sd_i, msg=f'rho={rho}')
            # sd ratio should track sqrt(1-rho) for equal-variance arms
            self.assertAlmostEqual(sd_j / sd_i, np.sqrt(1.0 - rho), places=1,
                                   msg=f'rho={rho}')

    def test_negative_rho_joint_is_wider(self):
        jt = CPFN2D.from_dump(_corr_joint(0.0, 0.4, 0.5, 0.5, -0.5), EDGES)
        f0, f1 = jt.independent()
        self.assertGreater(self._sd(cpfn2d_tau_density(jt, TAU), TAU),
                           self._sd(cpfn1d_tau_density(f0, f1, TAU), TAU))

    def test_marginals_are_preserved_by_independent(self):
        jt = CPFN2D.from_dump(_corr_joint(0.1, 0.4, 0.5, 0.6, 0.5), EDGES)
        f0, f1 = jt.independent()
        m0, m1 = jt.mean()
        self.assertAlmostEqual(f0.mean(), m0, places=12)
        self.assertAlmostEqual(f1.mean(), m1, places=12)
        # ... so the CATE point estimate is IDENTICAL; only the width changes
        self.assertAlmostEqual(f1.mean() - f0.mean(), jt.cate(), places=12)


class TestRho(unittest.TestCase):
    def test_recovers_correlation(self):
        for rho in (-0.7, -0.2, 0.0, 0.4, 0.8):
            jt = CPFN2D.from_dump(_corr_joint(0.0, 0.0, 0.5, 0.5, rho), EDGES)
            self.assertAlmostEqual(jt.rho(), rho, places=1, msg=f'rho={rho}')


class TestValidation(unittest.TestCase):
    def test_rejects_non_uniform_edges(self):
        bad = np.concatenate([np.linspace(-2, 0, 17), np.linspace(0.5, 2, 16)])
        with self.assertRaises(ValueError):
            CPFN1D.from_dump(np.ones(len(bad) - 1) / (len(bad) - 1), bad)

    def test_rejects_unnormalised(self):
        with self.assertRaises(ValueError):
            CPFN1D.from_dump(np.full(J, 2.0 / J), EDGES)

    def test_rejects_negative(self):
        p = _gauss_hist(0, 0.4).copy(); p[0] = -1e-6
        with self.assertRaises(ValueError):
            CPFN1D.from_dump(p, EDGES)

    def test_rejects_bin_count_mismatch(self):
        with self.assertRaises(ValueError):
            CPFN1D.from_dump(_gauss_hist(0, 0.4)[:-1] * 0 + 1.0 / (J - 1), EDGES)

    def test_rejects_non_square_joint(self):
        with self.assertRaises(ValueError):
            CPFN2D.from_dump(np.ones((J, J - 1)) / (J * (J - 1)), EDGES)


class TestDumpIO(unittest.TestCase):
    def _write(self, kind, n_q=5):
        d = tempfile.mkdtemp()
        p = os.path.join(d, f'{kind}.npz')
        p0 = np.stack([_gauss_hist(-0.1 * i, 0.45) for i in range(n_q)])
        p1 = np.stack([_gauss_hist(0.1 * i, 0.5) for i in range(n_q)])
        payload = dict(edges=EDGES.astype(np.float32),
                       p_y0_scaled=p0.astype(np.float32),
                       p_y1_scaled=p1.astype(np.float32),
                       y_shift=np.float32(0.0), y_scale=np.float32(2.5))
        if kind == '2d':
            payload['p_joint_scaled'] = np.stack(
                [_corr_joint(-0.1 * i, 0.1 * i, 0.5, 0.5, 0.4)
                 for i in range(n_q)]).astype(np.float32)
            payload['true_cate_per_query'] = np.zeros(n_q, np.float32)
        np.savez(p, **payload)
        return p

    def test_loads_1d_and_2d(self):
        for kind in ('1d', '2d'):
            dump = load_dump(self._write(kind))
            self.assertEqual(dump['kind'], kind)
            self.assertAlmostEqual(dump['y_scale'], 2.5, places=5)

    def test_iter_queries_yields_valid_densities(self):
        for kind in ('1d', '2d'):
            dump = load_dump(self._write(kind))
            seen = 0
            for q, dens, extras in iter_queries(dump):
                p = dens(TAU_FULL)
                self.assertAlmostEqual(_TRAPZ(p, TAU_FULL), 1.0, places=8)
                self.assertIn('cate_scaled', extras)
                if kind == '2d':
                    self.assertGreater(extras['rho'], 0.2)
                    self.assertAlmostEqual(
                        _TRAPZ(extras['indep_density'](TAU_FULL), TAU_FULL),
                        1.0, places=8)
                seen += 1
            self.assertEqual(seen, 5)

    def test_missing_keys_reported(self):
        d = tempfile.mkdtemp(); p = os.path.join(d, 'bad.npz')
        np.savez(p, edges=EDGES, p_y0_scaled=np.zeros((2, J)))
        with self.assertRaises(KeyError):
            load_dump(p)


class TestSupportDiagnostics(unittest.TestCase):
    def test_counts_truths_outside_support(self):
        t = np.array([0.0, 1.0, -3.9, 4.1, -5.0])      # support is +/-4
        d = support_diagnostics(EDGES, t)
        self.assertAlmostEqual(d['tau_support_lo'], -4.0)
        self.assertAlmostEqual(d['outside_support_frac'], 2 / 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)


class TestPerArmScaling(unittest.TestCase):
    """STD_MODE=per_arm: arms carry their own (shift, scale), so their RAW bin
    grids differ in both width and offset."""

    def _raw(self, edges, shift, scale):
        from density_cpfn import raw_edges
        return raw_edges(edges, shift, scale)

    def test_reduces_to_pooled_when_arms_share_scaling(self):
        from density_cpfn import cpfn1d_tau_density_raw
        p0, p1 = _gauss_hist(-0.2, 0.45), _gauss_hist(0.35, 0.5)
        shift, scale = 1.7, 2.5
        e = self._raw(EDGES, shift, scale)
        raw = cpfn1d_tau_density_raw(p0, e, p1, e, TAU_FULL * scale)
        # pooled path on the scaled axis, converted to raw: p_raw(t) = p(t/s)/s
        scaled = cpfn1d_tau_density(CPFN1D.from_dump(p0, EDGES),
                                    CPFN1D.from_dump(p1, EDGES), TAU_FULL) / scale
        self.assertLess(float(np.abs(raw - scaled).max()), 1e-9)

    def test_integrates_to_one_with_different_arm_scales(self):
        from density_cpfn import cpfn1d_tau_density_raw, tau_support_raw
        p0, p1 = _gauss_hist(-0.1, 0.5), _gauss_hist(0.3, 0.45)
        e0 = self._raw(EDGES, 0.5, 2.0)
        e1 = self._raw(EDGES, -0.3, 3.5)          # different shift AND scale
        lo, hi = tau_support_raw(e0, e1)
        tau = np.linspace(lo - 0.5, hi + 0.5, 40001)
        p = cpfn1d_tau_density_raw(p0, e0, p1, e1, tau)
        self.assertAlmostEqual(float(_TRAPZ(p, tau)), 1.0, places=6)

    def test_mean_equals_raw_cate(self):
        from density_cpfn import cpfn1d_tau_density_raw, tau_support_raw
        p0, p1 = _gauss_hist(-0.1, 0.5), _gauss_hist(0.3, 0.45)
        e0 = self._raw(EDGES, 0.5, 2.0)
        e1 = self._raw(EDGES, -0.3, 3.5)
        c0 = 0.5 * (e0[:-1] + e0[1:]); c1 = 0.5 * (e1[:-1] + e1[1:])
        cate_raw = float((p1 / p1.sum()) @ c1 - (p0 / p0.sum()) @ c0)
        lo, hi = tau_support_raw(e0, e1)
        tau = np.linspace(lo - 0.5, hi + 0.5, 40001)
        p = cpfn1d_tau_density_raw(p0, e0, p1, e1, tau)
        self.assertAlmostEqual(float(_TRAPZ(p * tau, tau)), cate_raw, places=5)

    def test_zero_outside_support(self):
        from density_cpfn import cpfn1d_tau_density_raw, tau_support_raw
        p0, p1 = _gauss_hist(0.0, 0.5), _gauss_hist(0.0, 0.5)
        e0 = self._raw(EDGES, 0.5, 2.0); e1 = self._raw(EDGES, -0.3, 3.5)
        lo, hi = tau_support_raw(e0, e1)
        far = np.array([lo - 1e-6, lo - 1.0, hi + 1e-6, hi + 1.0])
        self.assertTrue(np.all(cpfn1d_tau_density_raw(p0, e0, p1, e1, far) == 0.0))

    def test_matches_brute_force_convolution(self):
        """Independent check against a direct 2D sum over bin pairs."""
        from density_cpfn import cpfn1d_tau_density_raw
        rng = np.random.default_rng(3)
        Js = 8
        ed = np.linspace(-1.0, 1.0, Js + 1)
        p0 = rng.random(Js); p0 /= p0.sum()
        p1 = rng.random(Js); p1 /= p1.sum()
        e0 = ed * 2.0 + 0.4
        e1 = ed * 1.3 - 0.2
        tau = np.linspace(-6, 6, 4001)
        mine = cpfn1d_tau_density_raw(p0, e0, p1, e1, tau)
        # brute force: sample-free numeric convolution on a fine y grid
        y = np.linspace(min(e0[0], e1[0]) - 8, max(e0[-1], e1[-1]) + 8, 200001)
        def step(p, e, x):
            out = np.zeros_like(x)
            m = (x >= e[0]) & (x < e[-1])
            k = np.clip(np.searchsorted(e[1:-1], x[m], side='right'), 0, len(p) - 1)
            out[m] = p[k] / np.diff(e)[k]
            return out
        f0 = step(p0, e0, y)
        brute = np.array([_TRAPZ(f0 * step(p1, e1, y + t), y) for t in tau[::200]])
        self.assertLess(float(np.abs(mine[::200] - brute).max()), 2e-3)
