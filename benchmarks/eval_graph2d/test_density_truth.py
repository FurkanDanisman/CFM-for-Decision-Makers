"""CPU regressions for Tier-C generator truth; no checkpoints or downloads.

Run: python -m unittest discover -s benchmarks/eval_graph2d -p test_density_truth.py
"""
import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np

from density_common import truth_tau_density
from density_truth import harness_y_affine, load_density_truth
from summarize_density_tauC import truth_summary


class DensityTruthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the ACTUAL harness scaler without importing its model/GPU
        # dependencies or triggering its import-time environment reads.
        path = Path(__file__).with_name('eval_graph2d_realcause.py')
        tree = ast.parse(path.read_text())
        scale_y = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == '_scale_y')
        cls.scaler_code = compile(ast.Module(body=[scale_y], type_ignores=[]),
                                  str(path), 'exec')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ihdp = self.root / 'benchmarks' / 'IHDP'
        self.ihdp.mkdir(parents=True)
        self.acic = self.root / 'acic_cache'
        self.acic.mkdir()

    def transform(self, context, mode, std_target=0.3):
        namespace = dict(np=np, Y_SCALING=mode, STD_TARGET=std_target)
        exec(self.scaler_code, namespace)
        scaled, offset, span = namespace['_scale_y'](np.asarray(context, dtype=np.float32))
        shift, scale = harness_y_affine(offset, span, mode)
        np.testing.assert_allclose((np.asarray(context) - shift) / scale, scaled,
                                   atol=1e-7, rtol=1e-7)
        return shift, scale

    def load(self, dataset, shift, scale):
        return load_density_truth(dataset, 1, y_shift=shift, y_scale=scale,
                                  causalpfn_dir=str(self.root),
                                  acic_cache_dir=str(self.acic))

    def write_ihdp(self, residual_factor=1):
        train_t = np.array([0, 1, 1, 0])
        train_m0 = np.array([10., 20., 30., 40.])
        train_m1 = train_m0 + [2, 4, 6, 8]
        residuals = residual_factor * np.array([-3., -1., 1., 3.])
        train_yf = np.where(train_t == 1, train_m1, train_m0) + residuals
        # Deliberately distinct realizations to catch selecting the wrong r.
        def realizations(x):
            return np.stack([np.zeros_like(x), x], axis=-1)
        np.savez(self.ihdp / 'ihdp_npci_1-100.train.npz',
                 t=realizations(train_t), yf=realizations(train_yf),
                 mu0=realizations(train_m0), mu1=realizations(train_m1))
        t = np.array([1, 0, 1])
        mu0, mu1 = np.array([13., 21., 35.]), np.array([17., 28., 44.])
        y0, y1 = mu0 + [-1., 0.5, 2.], mu1 + [0.5, -2., 1.]
        np.savez(self.ihdp / 'ihdp_npci_1-100.test.npz',
                 t=realizations(t), mu0=realizations(mu0), mu1=realizations(mu1),
                 yf=realizations(np.where(t == 1, y1, y0)),
                 ycf=realizations(np.where(t == 1, y0, y1)))
        return mu0, mu1, y1 - y0, residuals, train_yf[[0, 2]]

    def write_acic(self, residual_factor=1):
        n = 30
        t = np.arange(n) % 2
        mu0 = 10. + np.arange(n)
        mu1 = mu0 + 1. + np.arange(n) / 10
        residuals = np.arange(n) / 3 - 4
        y0, y1 = mu0 + residuals, mu1 - 2 * residuals
        perm = np.random.default_rng(43).permutation(n)
        train_idx, test_idx = perm[:27], perm[27:]
        # Change only training outcomes: the oracle and test NLL targets must
        # be independent of this change in the residual diagnostic.
        y0[train_idx] = mu0[train_idx] + residual_factor * residuals[train_idx]
        y1[train_idx] = mu1[train_idx] - 2 * residual_factor * residuals[train_idx]
        np.savetxt(self.acic / 'zymu_2.csv', np.column_stack([t, y0, y1, mu0, mu1]),
                   delimiter=',', header='"z","y0","y1","mu0","mu1"', comments='')
        factual_y = np.where(t == 1, y1, y0)
        factual_residuals = np.where(t == 1, y1 - mu1, y0 - mu0)[train_idx]
        return (mu0[test_idx], mu1[test_idx], (y1 - y0)[test_idx],
                factual_residuals, factual_y[train_idx[:5]])

    def test_oracle_and_targets_match_context_transform_for_both_datasets(self):
        for dataset, writer in [('IHDP', self.write_ihdp), ('ACIC', self.write_acic)]:
            mu0, mu1, tau, residuals, context = writer()
            for mode in ('minmax', 'std'):
                with self.subTest(dataset=dataset, mode=mode):
                    shift, scale = self.transform(context, mode)
                    truth = self.load(dataset, shift, scale)
                    np.testing.assert_allclose(truth.mu0_scaled * scale + shift, mu0)
                    np.testing.assert_allclose(truth.mu1_scaled * scale + shift, mu1)
                    np.testing.assert_allclose(truth.tau_star_scaled * scale, tau)
                    self.assertEqual(truth.sigma_raw, 1.0)
                    self.assertAlmostEqual(truth.sigma_scaled * scale, 1.0)
                    self.assertAlmostEqual(truth.sigma_residual_raw,
                                           np.std(residuals, ddof=1))
                    self.assertAlmostEqual(truth.sigma_residual_scaled * scale,
                                           np.std(residuals, ddof=1))
                    # The transformed true Gaussian has the required density
                    # Jacobian; it remains centered at the supplied CATE.
                    raw_tau = mu1[0] - mu0[0] + np.array([-2., 0., 2.])
                    actual = truth_tau_density(truth.mu0_scaled[0], truth.mu1_scaled[0],
                                               truth.sigma_scaled, raw_tau / scale)
                    expected = scale * np.exp(-np.array([1., 0., 1.])) / np.sqrt(4*np.pi)
                    np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_residual_changes_do_not_change_reference_or_nll_targets(self):
        for dataset, writer in [('IHDP', self.write_ihdp), ('ACIC', self.write_acic)]:
            with self.subTest(dataset=dataset):
                writer()
                before = self.load(dataset, 15., 4.)
                writer(residual_factor=7)
                after = self.load(dataset, 15., 4.)
                for field in ('mu0_scaled', 'mu1_scaled', 'tau_star_scaled',
                              'sigma_raw', 'sigma_scaled'):
                    np.testing.assert_array_equal(getattr(before, field), getattr(after, field))
                self.assertAlmostEqual(after.sigma_residual_raw,
                                       7 * before.sigma_residual_raw)

    def test_scaler_handles_nonzero_std_shift_and_clamped_context_range(self):
        for context in ([11., 13., 21.], [7., 7., 7.]):
            for mode in ('minmax', 'std'):
                with self.subTest(context=context, mode=mode):
                    shift, scale = self.transform(context, mode, std_target=0.7)
                    self.assertGreater(scale, 0)
                    if mode == 'std':
                        self.assertAlmostEqual(shift, np.mean(np.array(context, dtype=np.float32)))

    def test_noise_metadata_survives_npz_and_summary_labels_the_reference(self):
        self.write_ihdp()
        truth = self.load('IHDP', 15., 4.)
        path = self.root / 'result.npz'
        np.savez(path, **truth.noise_metadata())
        with np.load(path) as saved:
            row = dict(saved)
        self.assertEqual(str(row['truth_noise_source']), 'generator')
        self.assertEqual(float(row['sigma_raw']), 1.)
        self.assertEqual(float(row['sigma_scaled']), .25)
        self.assertIn('generator Gaussian, raw sigma=1', truth_summary([row]))
        self.assertIn('diagnostic', truth_summary([row]))
        self.assertIn('legacy', truth_summary([{}]))
        with self.assertRaisesRegex(ValueError, 'Mixed density truth sources'):
            truth_summary([{}, row])


if __name__ == '__main__':
    unittest.main()
