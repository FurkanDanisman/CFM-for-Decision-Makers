"""CPU-only checks for the optional MALC-T ATE pass; no model inference."""
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import summarize_density_malcT as summary

with patch.dict(os.environ, {'DUMPS': 'unused-in-unit-tests'}):
    import eval_density_tauC_malcT as malc


class MalcAteTest(unittest.TestCase):
    def setUp(self):
        self.means = np.array([-0.2, 0.1, 0.4])
        self.sigma = 0.12
        self.d = dict(mu0_scaled=np.zeros(3), mu1_scaled=self.means,
                      sigma_scaled=self.sigma, tau_star_scaled=self.means)

    def results(self, shift=0):
        # Independent analytic Gaussian densities; their W2 barycenter has
        # the averaged mean and the SAME variance, not variance / n_queries.
        sd = math.sqrt(2) * self.sigma
        return [(q, {'model': {'density': np.exp(
            -0.5 * ((malc.TAU_CENTERS - mu - shift) / sd) ** 2)
            / (sd * math.sqrt(2 * math.pi))}}) for q, mu in enumerate(self.means)]

    def test_gaussian_barycenter_keeps_outcome_noise_variance(self):
        row = malc.ate_metrics(self.d, self.results(), ['model'])
        expected_nll = math.log(math.sqrt(4 * math.pi) * self.sigma)
        self.assertAlmostEqual(row['nll_bary_model'], expected_nll, delta=1e-3)
        self.assertLess(row['l2_bary_model'], 1e-10)
        self.assertGreater(row['l2_mix_model'], 0.1)
        self.assertEqual(row['nll_bary_model'], row['nll_mix_model'])
        self.assertAlmostEqual(row['mass_bary_model'], 1.0, places=12)

    def test_supplied_density_shift_changes_ate_mean(self):
        row = malc.ate_metrics(self.d, self.results(shift=0.2), ['model'])
        self.assertAlmostEqual(row['ate_err_bary_model'], 0.2, delta=1e-4)
        self.assertGreater(row['l2_bary_model'], 0.1)

    def test_query_retains_raw_grid_after_failed_fit(self):
        def raw(t):
            return np.exp(-np.asarray(t) ** 2 / 2) / math.sqrt(2 * math.pi)
        bundle = {'S': np.ones(3) / 3, 'bw': 1., 'w0': 1.,
                  'raw': raw, 'raw_mean': lambda: 0.}
        payload = dict(d=self.d, methods=['model'], r=0)
        with patch.object(malc, '_G', payload), patch.object(malc, 'COMPUTE_ATE', True), \
                patch.object(malc, 'build_query', return_value={'model': bundle}), \
                patch.object(malc, 'smooth_interior', return_value=(None, np.nan, np.nan, np.nan, 0., False)):
            q, rec = malc.run_query(0)
        self.assertEqual(q, 0)
        self.assertEqual(rec['model']['fallback'], 1)
        np.testing.assert_array_equal(rec['model']['density'], raw(malc.TAU_CENTERS))

    def test_resume_rejects_cate_only_and_nonfinite_ate_files(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(malc, 'COMPUTE_ATE', True):
            path = Path(tmp) / 'score.npz'
            row = dict(realization=0, methods=['model'])
            np.savez(path, **row)
            self.assertFalse(malc._shard_ok(path))
            row.update(self.d)
            row.update(malc.ate_metrics(self.d, self.results(), ['model']))
            np.savez(path, **row)
            self.assertTrue(malc._shard_ok(path))
            row['nll_bary_model'] = np.inf
            np.savez(path, **row)
            self.assertFalse(malc._shard_ok(path))

    def test_reference_rejects_changed_fallback_mask(self):
        row = dict(dataset='IHDP', realization=0, malc_variant='T', malc_B=1000,
                   malc_K=1, malc_seed=0, malc_n_tau=12001, n_y0=4096, n_queries=3,
                   malc_fallback_model=np.zeros(3))
        row.update({f'{m}_model': 0.5 for m in ('nll', 'l2', 'kl_fwd', 'kl_rev', 'mass')})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'reference.npz'
            np.savez(path, **row)
            malc.check_reference(row, ['model'], path)
            row['malc_fallback_model'] = np.array([1, 0, 0])
            with self.assertRaisesRegex(ValueError, 'fallback mask changed'):
                malc.check_reference(row, ['model'], path)

    def test_summary_switches_both_tiers_only_after_complete_refit(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(summary, 'ROWS', [('Example', 'run', 'shard', 'model')]), \
                patch.object(summary, 'DATASETS', {'IHDP': 2}):
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                old_root, new_root = Path('old'), Path('new')
                refits = []
                for r in range(2):
                    name = f'IHDP_r{r:03d}.npz'
                    pred = dict(mu0_scaled=np.zeros(2), true_cate=np.zeros(2),
                                tau_star_scaled=np.zeros(2), y_scale=2., y_shift=0.,
                                sigma_scaled=0.1, n_y0=4096, n_context=2)
                    row = dict(pred, dataset='IHDP', realization=r, n_queries=2,
                               source_dump=f'/repo/results_density_tauC/shard/IHDP/predictions/{name}',
                               nll_model=1., l2_model=2., mass_model=1.,
                               malc_fallback_model=np.zeros(2, dtype=int),
                               n_fallback_model=0, malc_fallback_query_model=np.array([], dtype=int))
                    row.update(summary.CONFIG)
                    marginal = dict(row, nll_y0_model=0.1, nll_y1_model=0.2)
                    for path, values in [
                            (Path('results_density_tauC/shard/IHDP/predictions') / name, pred),
                            (old_root / 'shard/IHDP' / name, row),
                            (Path('results_density_marginals/run/IHDP') / name, marginal)]:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez(path, **values)
                    refits.append(dict(row, nll_model=5., nll_bary_model=-0.8,
                                       nll_mix_model=-0.8, ate_density_source='malcT_post_fallback',
                                       malc_fallback_model=np.array([1, 0]), n_fallback_model=1,
                                       malc_fallback_query_model=np.array([0])))
                (new_root / 'shard/IHDP').mkdir(parents=True)
                np.savez(new_root / 'shard/IHDP/IHDP_r000.npz', **refits[0])
                partial, _ = summary.collect(old_root, new_root)
                np.testing.assert_array_equal(partial['IHDP', 'Example']['nll'], [1., 1.])
                self.assertFalse(partial['IHDP', 'Example']['refit'])
                self.assertEqual(len(partial['IHDP', 'Example']['ate']), 1)
                np.savez(new_root / 'shard/IHDP/IHDP_r001.npz', **refits[1])
                complete, _ = summary.collect(old_root, new_root)
                np.testing.assert_array_equal(complete['IHDP', 'Example']['nll'], [5., 5.])
                self.assertTrue(complete['IHDP', 'Example']['refit'])
                self.assertEqual(complete['IHDP', 'Example']['fallback'], 2)
                self.assertEqual(len(complete['IHDP', 'Example']['ate']), 2)
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
