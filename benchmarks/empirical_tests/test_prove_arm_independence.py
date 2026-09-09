"""CPU regressions for the saved-outcome dependence audit.

python -m unittest discover -s benchmarks/empirical_tests -p test_prove_arm_independence.py
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import stats

import prove_arm_independence as audit


class ArmIndependenceTest(unittest.TestCase):
    def setUp(self):
        self.config = patch.multiple(audit, N_PERM=99, N_BINS=4, POWER_RHO=[0.5])
        self.config.start()
        self.addCleanup(self.config.stop)
        self.rng = np.random.default_rng(73)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def independent_known_means(self, scale=1):
        # A Cartesian product has exactly independent empirical marginals.
        q = stats.norm.ppf((np.arange(32) + 0.5) / 32)
        # Binary fractions keep tied residuals exact after adding/removing mu.
        q = np.round(q * 1024) / 1024
        e0, e1 = np.repeat(q, 32) * scale, np.tile(q, 32) * scale
        mu0 = self.rng.integers(-10000, 10000, e0.size) / 64
        mu1 = 2 * mu0 + 5
        return audit.analyse_known_means('IHDP', mu0 + e0, mu1 + e1, mu0, mu1,
                                         np.zeros(e0.size, dtype=int), self.rng)

    def test_raw_correlation_is_separate_from_conditional_noise(self):
        result = self.independent_known_means()
        self.assertGreater(result['raw_outcomes']['pearson_r'], 0.99)
        self.assertAlmostEqual(result['pearson_r'], 0, places=12)
        self.assertAlmostEqual(result['spearman_r'], 0, places=3)
        self.assertFalse(result['dependence_detected'])
        self.assertEqual(result['generator_truth']['sigma_raw'], 1)
        self.assertEqual(result['generator_truth']['conditional_rho'], 0)
        self.assertLess(result['power']['0.5']['copula_p'], 0.05)
        self.assertNotIn('independent', result)
        self.assertNotIn('mi_bound_nats', result)
        self.assertNotIn('ci_lo', result['raw_outcomes'])
        json.dumps(result, allow_nan=False)

    def test_wrong_noise_scale_is_not_fitted_away(self):
        result = self.independent_known_means(scale=4)
        for key in ('noise0', 'noise1'):
            self.assertGreater(result[key]['sd_raw'], 3.8)
            self.assertEqual(result[key]['expected_sd_raw'], 1)
            self.assertLess(result[key]['normal_ks_p'], 1e-20)
        self.assertEqual(result['expected_residual_tau_variance'], 2)
        self.assertGreater(result['residual_tau_variance'], 25)

    def test_nonlinear_dependence_is_detected_with_zero_pearson(self):
        e0 = np.linspace(-3, 3, 1600)
        e1 = e0**2 - np.mean(e0**2)
        zero = np.zeros(e0.size)
        result = audit.analyse_known_means('ACIC', e0, e1, zero, zero,
                                           np.zeros(e0.size, dtype=int), self.rng)
        self.assertAlmostEqual(result['pearson_r'], 0, places=12)
        self.assertTrue(result['dependence_detected'])
        self.assertIn('pooled_g', result['rejected_tests'])

    def test_replicates_detect_opposing_correlations_and_handle_no_zeros(self):
        noise = self.rng.normal(size=(160, 20))
        mu = np.linspace(20, 1000, 20)
        y0 = noise + mu
        y1 = noise * np.tile([-1, 1], 10) + mu
        # Include a constant unit: retain it in raw summaries, exclude its rho.
        y0 = np.column_stack([y0, np.full(160, 2000)])
        y1 = np.column_stack([y1, np.full(160, 3000)])
        result = audit.analyse_replicates('CPS', y0, y1, True, self.rng)
        self.assertGreater(result['raw_outcomes']['pearson_r'], 0.9)
        self.assertAlmostEqual(result['rho_common'], 0, places=12)
        self.assertTrue(result['dependence_detected'])
        self.assertIn('heterogeneous_correlation', result['rejected_tests'])
        self.assertEqual(result['n_units_constant'], 1)
        self.assertEqual(result['raw_outcomes']['n_pairs'], 160 * 21)
        self.assertEqual(result['zero_p'], 1)
        self.assertEqual(result['zero_z'], 0)
        json.dumps(result, allow_nan=False)

    def test_upper_tail_and_degenerate_permutation_pvalues(self):
        null = np.arange(1., 100.)
        self.assertEqual(audit.calibrated(0, null)[1], 1)
        self.assertEqual(audit.calibrated(101, null)[1], 0.01)
        self.assertEqual(audit.calibrated(0, np.zeros(99))[:2], (0, 1))
        self.assertEqual(audit.calibrated(-101, null, 'two-sided')[1], 0.01)

    def test_realization_permutations_preserve_each_block(self):
        groups = np.array([0, 1, 0, 2, 2, 1, 2, 2])
        labels = np.arange(len(groups))[:, None]
        def check(shuffled):
            for group in np.unique(groups):
                np.testing.assert_array_equal(np.sort(shuffled[groups == group], axis=0),
                                              labels[groups == group])
            return float(shuffled[0, 0])
        audit.permutation_null(check, labels, self.rng, 20, groups=groups)

    def test_grouped_g_keeps_dependence_that_cancels_in_pooled_table(self):
        a = np.tile([0, 0, 1, 1], 100)
        b = np.tile([0, 0, 1, 1], 100)
        b[200:] = 1 - b[200:]
        groups = np.repeat([0, 1], 200)
        self.assertAlmostEqual(audit.pooled_g(a[:, None], b[:, None], 2), 0)
        self.assertGreater(audit.grouped_g(a, b, groups, 2, 2), 500)

    def test_ihdp_reconstructs_both_arms_without_repairing_pairs(self):
        root = self.root / 'benchmarks' / 'IHDP'
        root.mkdir(parents=True)
        expected = [[] for _ in range(5)]
        for split, n in [('train', 4), ('test', 2)]:
            t = (np.arange(n * 3).reshape(n, 3) % 2).astype(float)
            mu0 = self.rng.normal(size=(n, 3))
            mu1 = mu0 + 5
            y0 = mu0 + self.rng.normal(size=(n, 3))
            y1 = mu1 + self.rng.normal(size=(n, 3))
            np.savez(root / f'ihdp_npci_1-100.{split}.npz', t=t, mu0=mu0, mu1=mu1,
                     yf=np.where(t == 1, y1, y0), ycf=np.where(t == 1, y0, y1))
            for values, a in zip(expected[:4], (y0, y1, mu0, mu1)):
                values.append(a.ravel(order='F'))
            expected[4].append(np.repeat(np.arange(3), n))
        with patch.object(audit, 'CAUSALPFN', str(self.root)):
            actual = audit.load_ihdp_outcomes()
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, np.concatenate(b))

    def test_acic_cache_reads_named_columns_even_if_reordered(self):
        import pandas as pd
        for r in range(10):
            pd.DataFrame(dict(mu1=np.arange(4) + 2, y1=np.arange(4) + 3,
                              z=[0, 1, 0, 1], mu0=np.arange(4),
                              y0=np.arange(4) - 1)).to_csv(self.root / f'zymu_{r+1}.csv',
                                                          index=False)
        with patch.object(audit, 'ACIC_CACHE_DIR', str(self.root)):
            y0, y1, mu0, mu1, tags = audit.load_acic_outcomes()
        np.testing.assert_array_equal(y0 - mu0, -np.ones(40))
        np.testing.assert_array_equal(y1 - mu1, np.ones(40))
        np.testing.assert_array_equal(tags, np.repeat(np.arange(10), 4))

    def test_replicate_covariate_mismatch_stops_conditional_analysis(self):
        import pandas as pd
        root = self.root / 'benchmarks' / 'realcause_datasets'
        root.mkdir(parents=True)
        for r in range(4):
            pd.DataFrame(dict(x=np.roll(np.arange(4), r), y0=np.arange(4) + r,
                              y1=np.arange(4) - r)).to_csv(root / f'lalonde_cps_sample{r}.csv',
                                                          index=False)
        with patch.object(audit, 'CAUSALPFN', str(self.root)):
            args = audit.load_realcause_replicates('lalonde_cps', n_samples=4)
        self.assertFalse(args[2])
        with self.assertRaisesRegex(ValueError, 'covariates differ'):
            audit.analyse_replicates('CPS', *args, self.rng)

    def test_invalid_pairs_are_rejected_and_constant_correlations_are_null(self):
        with self.assertRaisesRegex(ValueError, 'finite'):
            audit.correlation_summary([0, 1, 2, np.nan], [0, 1, 2, 3])
        self.assertIsNone(audit.correlation_summary(np.ones(4), np.arange(4))['pearson_r'])
        with self.assertRaisesRegex(ValueError, 'nonconstant'):
            audit.analyse_replicates('PSID', np.ones((4, 4)), np.ones((4, 4)), True,
                                     self.rng)

    def test_cli_reports_partial_failure_and_writes_reviewable_json(self):
        result = self.independent_known_means()
        output = self.root / 'result.json'
        with patch.multiple(audit, DATASETS=['IHDP', 'MISSING'], OUT_JSON=str(output)), \
             patch.object(audit, 'load_ihdp_outcomes', return_value=()), \
             patch.object(audit, 'analyse_known_means', return_value=result), \
             redirect_stdout(io.StringIO()) as console:
            code = audit.main()
        self.assertEqual(code, 1)
        self.assertIn('NOT REJECTED', console.getvalue())
        self.assertIn('RAW SAMPLED OUTCOMES', console.getvalue())
        self.assertNotIn('INDEPENDENT', console.getvalue())
        self.assertIn('per_realization', json.loads(output.read_text())[0])


if __name__ == '__main__':
    unittest.main()
