"""Numerical and inference-contract tests; no pretrained artifacts required."""
import ast
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad

from density_common import DoPFN1D, dopfn_tau_density, mass
from density_truth import DensityTruth, harness_y_affine
from summarize_density_tauC import point_table


class DoPFNDensityTest(unittest.TestCase):
    def setUp(self):
        self.f0 = DoPFN1D.from_pred([0.5, -0.5, 0.8, 1.2, -0.7],
                                   [-1.8, -1.3, -0.5, 0.1, 1.1, 1.4])
        self.f1 = DoPFN1D.from_pred([-0.2, 0.7, -0.4, 0.3],
                                   [-0.9, -0.7, 0.5, 1.2, 1.8])

    def test_density_and_mean_integrate_with_tails(self):
        for f in (self.f0, self.f1):
            edges = [-np.inf, *f.edges, np.inf]
            for moment, target in ((0, 1.0), (1, f.mean())):
                actual = sum(quad(lambda x: x ** moment * f.density(x), a, b,
                                  epsabs=1e-11)[0] for a, b in zip(edges[:-1], edges[1:]))
                self.assertAlmostEqual(actual, target, places=9)

    def test_exact_tau_matches_independent_adaptive_quadrature(self):
        tau = np.array([-5.0, -1.23, -0.1, 0.0, 0.27, 1.4, 5.0])
        expected = []
        for t in tau:
            cuts = [-np.inf, *np.unique(np.r_[self.f0.edges, self.f1.edges - t]), np.inf]
            expected.append(sum(quad(lambda x: self.f0.density(x) * self.f1.density(x+t),
                                     a, b, epsabs=1e-11)[0]
                                for a, b in zip(cuts[:-1], cuts[1:])))
        np.testing.assert_allclose(dopfn_tau_density(self.f0, self.f1, tau), expected,
                                   rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(dopfn_tau_density(self.f1, self.f0, -tau), expected,
                                   rtol=1e-8, atol=1e-10)

    def test_tau_mass_mean_and_affine_change_of_units(self):
        grid = np.linspace(-15, 15, 30001)
        density = dopfn_tau_density(self.f0, self.f1, grid)
        self.assertAlmostEqual(mass(density, grid), 1.0, places=6)
        self.assertAlmostEqual(mass(grid * density, grid), self.f1.mean()-self.f0.mean(), places=6)
        raw_borders = np.array([-2, -1, -0.3, 0.8, 2])
        raw = DoPFN1D.from_pred([0.2, 1, -1, 0.3], raw_borders)
        scaled = DoPFN1D.from_pred([0.2, 1, -1, 0.3], raw_borders, y_shift=7, y_scale=3)
        xs = np.linspace(-5, 5, 100)
        np.testing.assert_allclose(scaled.density((xs-7)/3), 3*raw.density(xs), rtol=1e-12)
        np.testing.assert_allclose(dopfn_tau_density(scaled, scaled, xs/3),
                                   3*dopfn_tau_density(raw, raw, xs), atol=1e-12)
        self.assertAlmostEqual(scaled.mean()*3+7, raw.mean())

    def test_tail_only_cases_do_not_lose_mass(self):
        grid = np.linspace(-15, 15, 30001)
        for i in (0, -1):
            for j in (0, -1):
                p0, p1 = np.full(4, -800.0), np.full(4, -800.0)
                p0[i], p1[j] = 0.0, 0.0
                f0 = DoPFN1D.from_pred(p0, [-2, -1, 0, 1, 2])
                f1 = DoPFN1D.from_pred(p1, [-2, -1, 0, 1, 2])
                self.assertAlmostEqual(mass(dopfn_tau_density(f0, f1, grid), grid), 1, places=6)

    def test_rejects_invalid_borders(self):
        with self.assertRaises(ValueError):
            DoPFN1D.from_pred([0, 0, 0], [-1, 0, 0, 1])

    def test_optional_upstream_reference(self):
        root = os.environ.get('DOPFN_ROOT')
        if not root:
            self.skipTest('Set DOPFN_ROOT for comparison against upstream FullSupportBarDistribution')
        import torch
        spec = importlib.util.spec_from_file_location(
            '_dopfn_bar_reference', Path(root) / 'model/bar_distribution.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'utils': SimpleNamespace(print_once=lambda *a: None)}):
            spec.loader.exec_module(module)
        borders = torch.tensor([-2, -1, -0.3, 0.8, 2], dtype=torch.float64)
        criterion = module.FullSupportBarDistribution(borders)
        logits = torch.tensor([0.2, 1, -1, 0.3], dtype=torch.float64)
        scales = [float(criterion.halfnormal_with_p_weight_before(w).scale)
                  for w in torch.diff(borders)[[0, -1]]]
        f = DoPFN1D.from_pred(logits, borders, tail_scales=scales)
        y = torch.linspace(-8, 8, 500, dtype=torch.float64)
        expected = torch.exp(-criterion(logits.repeat(len(y), 1), y)).numpy()
        np.testing.assert_allclose(f.density(y.numpy()), expected, rtol=1e-9, atol=1e-12)
        self.assertAlmostEqual(f.mean(), float(criterion.mean(logits)), places=10)


class DoPFNAdapterTest(unittest.TestCase):
    def test_chunking_treatment_columns_all_features_and_stale_widths(self):
        import torch
        from density_dopfn import DoPFNDensityModels

        calls = []

        class Regressor:
            def fit(self, x, y):
                calls.append(('fit', x.numpy(), y.numpy()))

            def predict_full(self, x):
                calls.append(('predict', x.numpy()))
                borders = torch.tensor([-4, -2, 0, 1, 4], dtype=torch.float64)
                criterion = SimpleNamespace(
                    borders=borders, bucket_widths=torch.ones(4),
                    halfnormal_with_p_weight_before=lambda w: torch.distributions.HalfNormal(w / 0.67448975))
                return dict(logits=torch.zeros((len(x), 4)), criterion=criterion)

        class Joint:
            def __call__(self, **kw):
                calls.append(('joint', kw))
                return {'predictions': torch.zeros(1, kw['X_query'].shape[1], 17)}

        models = object.__new__(DoPFNDensityModels)
        models.device, models.query_chunk = torch.device('cpu'), 2
        models.environment = nullcontext
        models.regressor_class, models.joint = Regressor, Joint()
        models.J, models.edges = 2, np.linspace(-1, 1, 3)
        models.root, models.checkpoint = '/fake/dopfn', '/fake/joint.pt'
        x, xt = np.arange(4*58).reshape(4, 58), np.arange(5*58).reshape(5, 58)
        arms, logits, dump = models.predict(x, [0, 1, 0, 1], [3, 4, 5, 6], xt,
                                             x/10, [-1, -0.3, 0.3, 1], xt/10,
                                             y_shift=1, y_scale=2)
        self.assertEqual(logits.shape, (5, 17))
        self.assertEqual([len(a) for a in arms], [5, 5])
        np.testing.assert_array_equal(calls[0][1][:, 1:], x)
        predicts = [c[1] for c in calls if c[0] == 'predict']
        self.assertEqual([len(c) for c in predicts], [2, 2, 1, 2, 2, 1])
        for arm in (0, 1):
            combined = np.concatenate(predicts[arm*3:arm*3+3])
            np.testing.assert_array_equal(combined[:, 0], arm)
            np.testing.assert_array_equal(combined[:, 1:], xt)
        self.assertEqual(dump['dopfn_n_features'], 58)
        np.testing.assert_allclose(dump['dopfn_tail_scales0_raw'], np.array([2, 3])/0.67448975)
        joint_calls = [c[1] for c in calls if c[0] == 'joint']
        np.testing.assert_allclose(joint_calls[0]['X_context'][0], x/10, rtol=1e-6)

    def test_environment_restores_modules_paths_and_cwd_on_error(self):
        from density_dopfn import DoPFNDensityModels

        models = object.__new__(DoPFNDensityModels)
        old_cwd, old_path = os.getcwd(), sys.path[:]
        sentinel, native = SimpleNamespace(), SimpleNamespace()
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, {'utils': sentinel}):
            models.root, models._modules = root, {'utils': native}
            with self.assertRaisesRegex(RuntimeError, 'test'):
                with models.environment():
                    self.assertIs(sys.modules['utils'], native)
                    raise RuntimeError('test')
            self.assertIs(sys.modules['utils'], sentinel)
        self.assertEqual(os.getcwd(), old_cwd)
        self.assertEqual(sys.path, old_path)

    def test_dopfn_only_summary(self):
        row = {f'{metric}_{method}': 1.0
               for metric in ('pehe', 'cate_l1', 'ate_abs_err')
               for method in ('dopfn_native', 'dopfn_joint', 'dopfn_joint_inner')}
        table = point_table([row, row])
        self.assertIn('DoPFN (x)indep native', table)
        self.assertIn('DoPFN Joint-2D interior mean', table)
        self.assertNotIn('UWYK', table)

    def test_runner_families_share_context_and_save_replayable_predictions(self):
        """Run the actual evaluator with deterministic model outputs and truth.

        Extract its functions to avoid importing the external dataset/model
        packages. Preprocessing is taken from the actual shared harness too.
        """
        import torch
        import density_common as common
        here = Path(__file__).parent

        def functions_from(filename, names, namespace):
            path = here / filename
            body = [n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name in names]
            exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)

        rng = np.random.default_rng(3)
        cate = SimpleNamespace(X_train=rng.normal(size=(9, 58)).astype(np.float32),
                               t_train=np.arange(9) % 2, y_train=np.arange(9, dtype=np.float32),
                               X_test=rng.normal(size=(2, 58)).astype(np.float32),
                               true_cate=np.array([1., 2.]))
        ds = [(cate, None)]
        for family in ('uwyk', 'dopfn', 'all'):
            for dataset, mode in (('IHDP', 'minmax'), ('ACIC', 'std')):
                with self.subTest(family=family, dataset=dataset), tempfile.TemporaryDirectory() as out:
                    scaling = dict(np=np, Y_SCALING=mode, STD_TARGET=0.3, X_CLIP_QUANTILE='')
                    functions_from('eval_graph2d_realcause.py',
                                   ('_scale_y', '_standardize_train_test', '_pad_features'), scaling)
                    J, edges = 2, np.linspace(-1, 1, 3)
                    logits = np.zeros((2, J*J+13)); logits[:, J*J] = 8
                    observed_contexts = []

                    def forward(model, x, t, y, xt, adj, j):
                        observed_contexts.append((x.copy(), t.copy(), y.copy()))
                        return None, None, logits, None

                    harness = SimpleNamespace(
                        EVAL_MAX_CONTEXT='4', EVAL_CONTEXT_SEED=7, Y_SCALING=mode,
                        STD_TARGET=0.3, X_CLIP_QUANTILE='', CAUSALPFN='/unused',
                        CKPT='/fake/uwyk-joint.pt', BIAS_EDGE_SCALE=1, T_INTV_OVERRIDE='',
                        _scale_y=scaling['_scale_y'],
                        _standardize_train_test=scaling['_standardize_train_test'],
                        _pad_features=scaling['_pad_features'],
                        build_mode_list=lambda *a: [('v6a', np.zeros((5, 5), dtype=np.float32))],
                        marginals_from_forward=forward)

                    def truth(dataset, r, *, y_shift, y_scale, **kw):
                        return DensityTruth(np.zeros(2), cate.true_cate/y_scale,
                                            cate.true_cate/y_scale, 1, 1/y_scale, 1, 1/y_scale)

                    def predict_native_and_joint(x, t, y, xt, xs, ys, xts, **affine):
                        idx = np.random.default_rng(7).choice(9, 4, replace=False)
                        np.testing.assert_array_equal(x, cate.X_train[idx])
                        np.testing.assert_array_equal(t, cate.t_train[idx])
                        np.testing.assert_array_equal(y, cate.y_train[idx])
                        self.assertEqual(xs.shape, (4, 58))
                        if observed_contexts:
                            np.testing.assert_array_equal(observed_contexts[0][0], xs[:, :3])
                            np.testing.assert_array_equal(observed_contexts[0][2].ravel(), ys)
                        f = DoPFN1D.from_pred([0, 1, 0], [-2, -1, 1, 2], **affine)
                        return [[f, f], [f, f]], logits, dict(dopfn_joint_logits=logits)

                    dopfn = (SimpleNamespace(predict=predict_native_and_joint, J=J, edges=edges)
                              if family != 'uwyk' else None)
                    bd = SimpleNamespace(edges=torch.tensor(edges), widths=torch.tensor(np.diff(edges)),
                                         base_s_left=1, base_s_right=1)
                    uwyk = (SimpleNamespace(bar_distribution=bd, device='cpu',
                                             _preprocess_adjacency_matrix=lambda adj: adj)
                            if family != 'dopfn' else None)
                    namespace = dict(vars(common), np=np, torch=torch, os=os, H=harness,
                                     DATASET=dataset, ANC_TAG='v6a', MODEL_FAMILY=family,
                                     UWYK_CKPT='/fake/uwyk.pt', UWYK_CFG='/fake/config.yaml',
                                     ACIC_CACHE='', SAVE_PREDICTIONS=True, OUT=out, N_Y0=128,
                                     TAU_CENTERS=np.linspace(-3, 3, 121),
                                     harness_y_affine=harness_y_affine, load_density_truth=truth,
                                     uwyk_preds_chunked=lambda *a: np.zeros((2, 6)))
                    functions_from('eval_density_tauC.py', ('score', 'evaluate'), namespace)
                    row = namespace['evaluate'](0, ds, object(), J, edges, uwyk, 3, dopfn)
                    self.assertEqual(row['n_context'], 4)
                    self.assertEqual('nll_dopfn_native' in row, dopfn is not None)
                    self.assertEqual('nll_joint' in row, uwyk is not None)
                    for method in row['methods']:
                        self.assertTrue(np.isfinite(row[f'nll_{method}']))
                        self.assertEqual(row[f'cate_pred_{method}'].shape, (2,))
                    with np.load(Path(out) / 'predictions' / f'{dataset}_r000.npz') as dump:
                        self.assertEqual('joint_logits' in dump, uwyk is not None)
                        self.assertEqual('adj_joint' in dump, uwyk is not None)
                        self.assertEqual('dopfn_joint_logits' in dump, dopfn is not None)
                        self.assertEqual(str(dump['model_family']), family)


if __name__ == '__main__':
    unittest.main()
