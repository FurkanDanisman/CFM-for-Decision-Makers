"""CPU-only tests: python -m unittest discover -s benchmarks/eval_graph2d
-p 'test_density_point_metrics.py'. No torch or model checkpoints required.
"""
import math
import unittest
from dataclasses import replace

import numpy as np

from density_common import Joint2D, UWYK1D, point_metrics
from summarize_density_tauC import point_table


def quadrature(edges, left_scale, right_scale):
    """Integrate each bar and tail separately to avoid discontinuity bias."""
    nodes, weights = np.polynomial.legendre.leggauss(100)
    boundaries = [edges[0] - 10 * left_scale, *edges, edges[-1] + 10 * right_scale]
    xs, ws = [], []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        xs.append((lo + hi) / 2 + nodes * (hi - lo) / 2)
        ws.append(weights * (hi - lo) / 2)
    return np.concatenate(xs), np.concatenate(ws)


class DensityPointMetricsTest(unittest.TestCase):
    def test_joint_means_match_integration_in_every_region(self):
        rng = np.random.default_rng(15)
        J = 3
        jt = Joint2D.from_pred(rng.normal(size=J * J + 13), J,
                               np.linspace(-1, 1, J + 1))
        x, wx = quadrature(jt.edges, jt.sL0, jt.sR0)
        y, wy = quadrature(jt.edges, jt.sL1, jt.sR1)
        for rho in (-0.65, 0.0, 0.65):
            for region in range(9):
                with self.subTest(rho=rho, region=region):
                    w = np.eye(9)[region]
                    model = replace(jt, w=w, rho=rho)
                    weighted = model.density(x[:, None], y[None, :]) * wx[:, None] * wy
                    self.assertAlmostEqual(float(weighted.sum()), 1.0, places=7)
                    numeric = ((weighted * x[:, None]).sum(), (weighted * y).sum())
                    np.testing.assert_allclose(model.mean(), numeric, atol=1e-7, rtol=1e-7)

    def test_inner_mean_is_point_runner_center_of_mass(self):
        J = 4
        edges = np.linspace(-1, 1, J + 1)
        logits = np.arange(J * J + 13, dtype=float) / 10
        jt = Joint2D.from_pred(logits, J, edges)
        centers = (edges[:-1] + edges[1:]) / 2
        expected = ((jt.p_mat.sum(axis=1) * centers).sum(),
                    (jt.p_mat.sum(axis=0) * centers).sum())
        np.testing.assert_allclose(jt.inner_mean(), expected)
        # With tails removed, the full mean must reduce to the original raw mean.
        np.testing.assert_allclose(replace(jt, w=np.eye(9)[0]).mean(), expected)

    def test_uwyk_mean_matches_density_integration(self):
        edges = np.linspace(-1, 1, 5)
        f = UWYK1D.from_pred(np.array([1, -2, 0, 2, -1, 0.5, 1, -0.5]),
                            edges, np.diff(edges), 0.5, 0.5)
        x, w = quadrature(edges, f.sL, f.sR)
        self.assertAlmostEqual(float(f.density(x) @ w), 1.0, places=10)
        self.assertAlmostEqual(f.mean(), float((x * f.density(x)) @ w), places=10)

    def test_full_mean_does_not_truncate_large_tails(self):
        f = UWYK1D(log_pL=-800, log_pBars=np.full(4, -800.0), log_pR=0,
                   sL=1, sR=5, edges=np.linspace(-1, 1, 5), widths=np.full(4, 0.5))
        self.assertAlmostEqual(f.mean(), 1 + 5 * math.sqrt(2 / math.pi))
        self.assertGreater(f.mean(), 3)  # Outside the density evaluation's tau grid.

    def test_point_metrics_use_original_units_and_root_of_mean_square(self):
        result = point_metrics([1, 3], [1, 9], y_scale=2)
        self.assertAlmostEqual(result['pehe'], math.sqrt(5))
        self.assertEqual(result['cate_l1'], 2)
        self.assertEqual(result['ate_abs_err'], 1)
        with self.assertRaises(ValueError):
            point_metrics([1], [1, 2], 2)

    def test_summary_accepts_old_and_new_shards(self):
        self.assertIsNone(point_table([{'nll_joint': 1.0}]))
        row = {f'{metric}_{method}': 1.0
               for metric in ('pehe', 'cate_l1', 'ate_abs_err')
               for method in ('uwyk_native', 'uwyk_matched', 'joint', 'joint_inner')}
        rendered = point_table([row, row])
        self.assertIn('Joint-2D interior mean (raw)', rendered)
        self.assertIn('1.0000±0.0000', rendered)
        self.assertIn('--', point_table([row, {}]))


if __name__ == '__main__':
    unittest.main()
