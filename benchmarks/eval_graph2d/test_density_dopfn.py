"""Numerical and inference-contract tests; no pretrained artifacts required."""
import ast
from contextlib import nullcontext, redirect_stdout
import importlib.util
import inspect
import io
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad

from density_common import DoPFN1D, dopfn_tau_density, mass
from density_truth import DensityTruth, harness_y_affine
from summarize_density_tauC import point_table, render_family


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
    def test_predict_patches_live_validation_after_module_restore_and_constructor(self):
        try:
            import sklearn.utils.validation as validation
        except ImportError:
            self.skipTest('Install sklearn to exercise its real validation API')
        import torch
        from density_dopfn import DoPFNDensityModels

        original = inspect.unwrap(validation.check_array)
        name = 'scripts.transformer_prediction_interface.base'
        detached = ModuleType(name)
        detached.__dict__.update(np=np, torch=torch, SimpleNamespace=SimpleNamespace,
                                 original=original, check_array=original)
        exec('''
class Base:
    @staticmethod
    def check_training_data(clf, x, y):
        return (check_array(x, dtype=np.float32, force_all_finite=False),
                check_array(y, dtype=np.float32, ensure_2d=False, force_all_finite=False))

    def fit(self, x, y):
        self.x, self.y = self.check_training_data(self, x, y)

    def predict_common_setup(self, x):
        return check_array(x.numpy(), dtype=np.float32, force_all_finite=False)

    def predict_full(self, x):
        x = self.predict_common_setup(x)
        criterion = SimpleNamespace(
            borders=torch.tensor([-2., -1., 1., 2.]),
            halfnormal_with_p_weight_before=lambda w: torch.distributions.HalfNormal(w / 0.67448975))
        return dict(logits=np.zeros((len(x), 3)), criterion=criterion)

class DoPFNRegressor(Base):
    def __init__(self):
        # An import/pickle during construction can rebind the old reference.
        global check_array
        check_array = original
''', detached.__dict__)
        replacement = ModuleType(name)
        replacement.check_array = original
        cached_dopfn = ModuleType('dopfn')
        cached_dopfn._repatch_dopfn_check_array = lambda: None
        models = object.__new__(DoPFNDensityModels)
        models.device, models.query_chunk = torch.device('cpu'), 1
        models.regressor_class = detached.DoPFNRegressor
        models.joint = lambda **kw: {'predictions': torch.zeros(1, kw['X_query'].shape[1], 17)}
        models.J, models.edges = 2, np.linspace(-1, 1, 3)
        models.checkpoint = '/fake/joint.pt'
        models._modules = {name: replacement}
        x = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as root, redirect_stdout(output), patch.dict(
                sys.modules, {'dopfn': cached_dopfn}):
            models.root = root
            # Without the direct binding this fails on modern sklearn even
            # if a sys.modules scan successfully patches the replacement.
            if 'force_all_finite' not in inspect.signature(original).parameters:
                reg = models.regressor_class()
                with self.assertRaisesRegex(TypeError, 'force_all_finite'):
                    reg.fit(x, [0., 1.])
            for _ in range(2):
                arms, logits, _ = models.predict(x, [0., 1.], [0., 1.], x,
                                                 x, [-1., 1.], x, y_shift=0., y_scale=1.)
                self.assertEqual(logits.shape, (2, 17))
                self.assertEqual(len(arms[0]), 2)
            self.assertIs(sys.modules['dopfn'], cached_dopfn)
        self.assertEqual(output.getvalue().count('[dopfn-compat]'), 1)
        self.assertIn('validated=check_training_data,predict_common_setup', output.getvalue())
        self.assertEqual(Path(models._compat_shim.__file__).resolve(),
                         Path(__file__).resolve().parents[1] / 'methods/dopfn.py')

    def test_outdated_shim_reports_the_exact_file_to_sync(self):
        from density_dopfn import _load_check_array_compat

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'benchmarks/methods/dopfn.py'
            path.parent.mkdir(parents=True)
            path.write_text('def _repatch_dopfn_check_array(): pass\n')
            with patch('density_dopfn._REPO', Path(root)):
                with self.assertRaisesRegex(RuntimeError, 'Outdated DoPFN compatibility shim') as error:
                    _load_check_array_compat()
            self.assertIn(str(path), str(error.exception))

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

    def test_summary_separates_model_families_and_graph_headers(self):
        row = dict(n_queries=2, frac_tau_outside_grid=0.0, anc_tag='v6a',
                   truth_noise_source='generator', sigma_raw=1.0,
                   sigma_residual_raw=1.0)
        for method in ('uwyk_native', 'uwyk_matched', 'joint',
                       'dopfn_native', 'dopfn_joint'):
            for metric in ('nll', 'l2', 'kl_fwd', 'kl_rev', 'mass'):
                row[f'{metric}_{method}'] = 1.0

        outputs = {}
        for model in ('uwyk', 'dopfn'):
            stream = io.StringIO()
            with redirect_stdout(stream):
                render_family('IHDP', [row, row], model)
            outputs[model] = stream.getvalue()

        self.assertIn('### IHDP — UWYK / g4cfm', outputs['uwyk'])
        self.assertIn('graph=v6a', outputs['uwyk'])
        self.assertNotIn('DoPFN', outputs['uwyk'])
        self.assertIn('### IHDP — DoPFN', outputs['dopfn'])
        self.assertIn('graph=none', outputs['dopfn'])
        self.assertNotIn('UWYK', outputs['dopfn'])

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
        for family in ('uwyk', 'dopfn', 'causalpfn', 'all'):
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

                    def predict_dopfn(x, t, y, xt, xs, ys, xts, **affine):
                        idx = np.random.default_rng(7).choice(9, 4, replace=False)
                        np.testing.assert_array_equal(x, cate.X_train[idx])
                        np.testing.assert_array_equal(t, cate.t_train[idx])
                        np.testing.assert_array_equal(y, cate.y_train[idx])
                        self.assertEqual(xs.shape, (4, 58))
                        if observed_contexts:
                            np.testing.assert_array_equal(observed_contexts[0][0], xs[:, :3])
                            np.testing.assert_array_equal(observed_contexts[0][2].ravel(), ys)
                        f = DoPFN1D.from_pred([0, 1, 0], [-2, -1, 1, 2], **affine)
                        wide = DoPFN1D.from_pred([0, 1, 0], [-6, -3, 3, 6], **affine)
                        jt = common.Joint2D.from_pred(logits[0], J, edges)
                        return {'dopfn_native': ('1d', [[f, f], [f, f]]),
                                'dopfn_wide': ('1d', [[wide, wide], [wide, wide]]),
                                'dopfn_repro_joint2d': ('joint', [jt, jt])}, dict(
                            dopfn_repro_joint2d_logits=logits)

                    dopfn = (SimpleNamespace(predict=predict_dopfn, sources={
                        'dopfn_native': 'library', 'dopfn_wide': '/fake/wide.pt',
                        'dopfn_repro_joint2d': '/fake/joint.pt'})
                        if family in ('dopfn', 'all') else None)

                    def predict_causalpfn(xs, t, y, xts, **affine):
                        idx = np.random.default_rng(7).choice(9, 4, replace=False)
                        np.testing.assert_array_equal(y, cate.y_train[idx])
                        np.testing.assert_array_equal(t, cate.t_train[idx])
                        self.assertEqual(xs.shape, (4, 58))
                        # Include an out-of-support observation to exercise
                        # native +inf NLL through the actual scoring runner.
                        native_edges = edges * (0.01 if dataset == 'ACIC' else 1)
                        f = common.CausalPFN1D.from_pred([0, 1], native_edges)
                        jt = common.Joint2D.from_pred(logits[0], J, edges)
                        # {method: (kind, dens)}, dump -- the CAUSALPFN_MODELS
                        # layout CausalPFNModelSet.predict actually returns.
                        # This fixture used to return (arms, joints, dump), the
                        # pre-list two-row API, and the evaluator unpacks two
                        # values -- so every causalpfn/all case here died with
                        # "too many values to unpack (expected 2)". The DoPFN
                        # half of this test was migrated to the model list and
                        # this half was not.
                        return ({'causalpfn_native': ('1d', [[f, f], [f, f]]),
                                 'causalpfn_joint': ('joint', [jt, jt])},
                                dict(causalpfn_joint_logits=logits,
                                     causalpfn_edges2d=edges))

                    # `sources` as well as `predict`: evaluate() reads
                    # owner.sources[method] for every model it scored, and the
                    # dopfn fake below has carried that since the model-list
                    # migration. Those two attributes are the COMPLETE surface
                    # evaluate() touches on a model set -- .models / .describe
                    # are used by main(), which this test does not call.
                    causalpfn = (SimpleNamespace(
                        predict=predict_causalpfn,
                        sources={'causalpfn_native': '/fake/cpfn_native.pt',
                                 'causalpfn_joint': '/fake/cpfn_joint.pt'})
                        if family in ('causalpfn', 'all') else None)
                    bd = SimpleNamespace(edges=torch.tensor(edges), widths=torch.tensor(np.diff(edges)),
                                         base_s_left=1, base_s_right=1)
                    uwyk = (SimpleNamespace(bar_distribution=bd, device='cpu',
                                             _preprocess_adjacency_matrix=lambda adj: adj)
                            if family in ('uwyk', 'all') else None)
                    namespace = dict(vars(common), np=np, torch=torch, os=os, H=harness,
                                     DATASET=dataset, ANC_TAG='v6a', ANC_FAMILY='v6a_only',
                                     MODEL_FAMILY=family,
                                     UWYK_CKPT='/fake/uwyk.pt', UWYK_CFG='/fake/config.yaml',
                                     ACIC_CACHE='', SAVE_PREDICTIONS=True, OUT=out, N_Y0=128,
                                     TAU_CENTERS=np.linspace(-3, 3, 121),
                                     harness_y_affine=harness_y_affine, load_density_truth=truth,
                                     uwyk_preds_chunked=lambda *a: np.zeros((2, 6)))
                    functions_from('eval_density_tauC.py', ('score', 'evaluate'), namespace)
                    row = namespace['evaluate'](0, ds, object(), J, edges, uwyk, 3,
                                                dopfn, causalpfn)
                    self.assertEqual(row['n_context'], 4)
                    self.assertEqual('nll_dopfn_native' in row, dopfn is not None)
                    if dopfn is not None:
                        self.assertEqual(
                            [m for m in row['methods'] if m.startswith('dopfn_')],
                            ['dopfn_native', 'dopfn_wide', 'dopfn_repro_joint2d'])
                        # Each 1D row must score its own arms, not the last model's.
                        self.assertNotAlmostEqual(row['nll_dopfn_native'],
                                                  row['nll_dopfn_wide'])
                        self.assertIn('pehe_dopfn_repro_joint2d_inner', row)
                        self.assertNotIn('pehe_dopfn_wide_inner', row)
                        self.assertEqual(str(row['kind_dopfn_repro_joint2d']), 'joint')
                        self.assertEqual(str(row['kind_dopfn_wide']), '1d')
                        self.assertEqual(str(row['source_dopfn_wide']), '/fake/wide.pt')
                    self.assertEqual('nll_joint' in row, uwyk is not None)
                    self.assertEqual('nll_causalpfn_native' in row, causalpfn is not None)
                    self.assertEqual('pehe_causalpfn_joint_inner' in row, causalpfn is not None)
                    for method in row['methods']:
                        if method == 'causalpfn_native' and dataset == 'ACIC':
                            self.assertEqual(row[f'nll_{method}'], np.inf)
                            self.assertEqual(row[f'frac_zero_density_{method}'], 1.0)
                        else:
                            self.assertTrue(np.isfinite(row[f'nll_{method}']))
                        self.assertEqual(row[f'cate_pred_{method}'].shape, (2,))
                    with np.load(Path(out) / 'predictions' / f'{dataset}_r000.npz') as dump:
                        self.assertEqual('joint_logits' in dump, uwyk is not None)
                        self.assertEqual('adj_joint' in dump, uwyk is not None)
                        self.assertEqual('dopfn_repro_joint2d_logits' in dump,
                                         dopfn is not None)
                        self.assertEqual('causalpfn_joint_logits' in dump, causalpfn is not None)
                        self.assertEqual(str(dump['model_family']), family)


class DoPFNModelListTest(unittest.TestCase):
    """DOPFN_MODELS parsing, checkpoint loading and each model's input contract."""

    def test_parse_model_list(self):
        from density_dopfn import parse_model_list

        self.assertEqual(parse_model_list(' native ,\n a_1=/x/a.pt,b = rel/b.pt,, '),
                         [('native', None), ('a_1', '/x/a.pt'), ('b', 'rel/b.pt')])
        for bad, message in (('native=/x.pt', 'takes no checkpoint'),
                             ('a=/x.pt,a=/y.pt', 'twice'),
                             ('/x/a.pt', 'letters, digits'),
                             ('a-b=/x.pt', 'letters, digits'),
                             ('a', 'expected name=checkpoint'),
                             ('a=', 'expected name=checkpoint'),
                             (' , ', 'no models')):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, message):
                parse_model_list(bad)

    @staticmethod
    def _model_set(models, query_chunk=2):
        import torch
        from density_dopfn import DoPFNModelSet

        s = object.__new__(DoPFNModelSet)
        s.device, s.query_chunk, s.root = torch.device('cpu'), query_chunk, '/fake/dopfn'
        s.environment, s.models = nullcontext, models
        return s

    @staticmethod
    def _net(n_out, calls):
        import torch

        def net(x_ctx, y_ctx, xq, only_return_standard_out):
            assert only_return_standard_out
            calls.append((x_ctx.clone(), y_ctx.clone(), xq.clone()))
            # Depends on the query row, so chunk order and arm are both visible.
            return (torch.linspace(-1, 1, n_out)
                    + 0.1 * xq[:, :, 1:2] + 0.3 * torch.nan_to_num(xq[:, :, :1]))
        return net

    def _inputs(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(6, 3)).astype(np.float32)
        T = np.array([0, 1, 0, 1, 1, 0], dtype=np.float32)
        y = np.array([1, 2, 4, 7, 11, 16], dtype=np.float32)
        Xt = rng.normal(size=(5, 3)).astype(np.float32)
        return X, T, y, Xt

    def test_model_set_feeds_each_model_its_training_inputs(self):
        import torch

        X, T, y, Xt = self._inputs()
        calls = {'r1': [], 'rj': [], 'bb': []}
        borders = np.array([-2.0, -1.0, 0.0, 1.5, 3.0])

        def bb(**kw):
            calls['bb'].append(kw)
            return {'predictions': torch.zeros(1, kw['X_query'].shape[1], 17)}

        models = {
            'dopfn_r1': SimpleNamespace(kind='repro_1d', net=self._net(4, calls['r1']),
                                        borders=borders, source='/fake/r1.pt'),
            'dopfn_rj': SimpleNamespace(kind='repro_joint', net=self._net(17, calls['rj']),
                                        J=2, edges=np.linspace(-1, 1, 3),
                                        query_fill=float('nan'), source='/fake/rj.pt'),
            'dopfn_bb': SimpleNamespace(kind='bb_joint', net=bb, J=2,
                                        edges=np.linspace(-1, 1, 3), source='/fake/bb.pt'),
        }
        out, dump = self._model_set(models).predict(
            X, T, y, Xt, X / 10, np.linspace(-1, 1, 6), Xt / 10, y_shift=0.5, y_scale=2.0)

        self.assertEqual(list(out), ['dopfn_r1', 'dopfn_rj', 'dopfn_bb'])
        self.assertEqual([kind for kind, _ in out.values()], ['1d', 'joint', 'joint'])
        self.assertEqual([len(out['dopfn_r1'][1][arm]) for arm in (0, 1)], [5, 5])
        self.assertEqual(len(out['dopfn_rj'][1]), 5)
        z = (y - y.mean()) / y.std(ddof=1)
        for name, n_calls in (('r1', 6), ('rj', 3)):
            self.assertEqual(len(calls[name]), n_calls)
            self.assertEqual([len(c[2]) for c in calls[name]], [2, 2, 1] * (n_calls // 3))
            for x_ctx, y_ctx, _ in calls[name]:
                # Raw covariates, factual T in column 0, context-z-scored y.
                np.testing.assert_array_equal(x_ctx[:, 0].numpy(),
                                              np.column_stack((T, X)))
                np.testing.assert_allclose(y_ctx[:, 0].numpy(), z, rtol=1e-6)
            queries = [c[2][:, 0].numpy() for c in calls[name]]
            np.testing.assert_array_equal(np.concatenate(queries[:3])[:, 1:], Xt)
        for arm in (0, 1):
            np.testing.assert_array_equal(
                np.concatenate([c[2][:, 0, 0].numpy() for c in calls['r1'][3*arm:3*arm+3]]),
                np.full(5, arm))
        self.assertTrue(all(np.isnan(c[2][:, 0, 0].numpy()).all() for c in calls['rj']))
        # The legacy joint keeps the harness inputs it was trained on.
        np.testing.assert_allclose(calls['bb'][0]['X_context'][0].numpy(), X / 10)
        np.testing.assert_allclose(calls['bb'][0]['Y_context'][0, :, 0].numpy(),
                                   np.linspace(-1, 1, 6))
        self.assertEqual(list(dump['dopfn_methods']), list(models))
        self.assertEqual(list(dump['dopfn_kinds']), ['repro_1d', 'repro_joint', 'bb_joint'])
        self.assertAlmostEqual(dump['dopfn_r1_y_scale'], float(y.std(ddof=1)), places=5)

    def test_model_set_maps_repro_outputs_to_harness_axis(self):
        import density_common as common

        X, T, y, Xt = self._inputs()
        J, edges = 2, np.linspace(-1, 1, 3)
        borders = np.array([-2.0, -1.0, 0.0, 1.5, 3.0])
        models = {
            'dopfn_r1': SimpleNamespace(kind='repro_1d', net=self._net(4, []),
                                        borders=borders, source='a'),
            'dopfn_rj': SimpleNamespace(kind='repro_joint', net=self._net(J*J+13, []), J=J,
                                        edges=edges, query_fill=float('nan'), source='b'),
        }
        y_shift, y_scale = 0.5, 2.0
        out, dump = self._model_set(models).predict(
            X, T, y, Xt, X, y, Xt, y_shift=y_shift, y_scale=y_scale)
        # The z-score actually fed to the nets (float32 torch statistics).
        mu, sd = dump['dopfn_r1_y_shift'], dump['dopfn_r1_y_scale']
        self.assertEqual((dump['dopfn_rj_y_shift'], dump['dopfn_rj_y_scale']), (mu, sd))
        self.assertAlmostEqual(sd, float(y.std(ddof=1)), places=5)
        raw = np.linspace(-30, 50, 41)
        h, zz = (raw - y_shift) / y_scale, (raw - mu) / sd
        for q in range(len(Xt)):
            for arm in (0, 1):
                native = DoPFN1D.from_pred(dump[f'dopfn_r1_pred{arm}'][q], borders)
                mapped = out['dopfn_r1'][1][arm][q]
                np.testing.assert_allclose(mapped.density(h), native.density(zz) * y_scale / sd,
                                           rtol=1e-9, atol=1e-15)
                self.assertAlmostEqual(mapped.mean() * y_scale + y_shift,
                                       native.mean() * sd + mu, places=9)
            native = common.Joint2D.from_pred(dump['dopfn_rj_logits'][q], J, edges)
            mapped = out['dopfn_rj'][1][q]
            h0, h1 = np.meshgrid(h, h[::-1])
            z0, z1 = np.meshgrid(zz, zz[::-1])
            np.testing.assert_allclose(mapped.density(h0, h1),
                                       native.density(z0, z1) * (y_scale / sd) ** 2,
                                       rtol=1e-9, atol=1e-15)
            for m, n in zip(mapped.mean(), native.mean()):
                self.assertAlmostEqual(m * y_scale + y_shift, n * sd + mu, places=9)

    @staticmethod
    def _repro_checkpoint(variant, n_out, **batch_cfg):
        import torch

        state = {'encoder.weight': torch.randn(4, 3), 'encoder.bias': torch.randn(4),
                 'decoder_dict.standard.0.weight': torch.randn(8, 4),
                 'decoder_dict.standard.0.bias': torch.randn(8),
                 'decoder_dict.standard.2.weight': torch.randn(n_out, 8),
                 'decoder_dict.standard.2.bias': torch.randn(n_out),
                 # dopfn_1d swaps in its own criterion; its size differs from the pickle's
                 'criterion.borders': torch.linspace(-3, 3, n_out + 1)}
        cfg = {'y_space': 'zscore_ctx', 'query_treatment': 'nan', **batch_cfg}
        return {'model': state, 'provenance': dict(variant=variant, spec={'j_2d': 2},
                                                   batch_cfg=cfg),
                'edges': torch.linspace(-1, 1, 3) if variant == 'joint_2d' else None}

    def test_repro_checkpoint_loading(self):
        import torch

        class Criterion(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('borders', torch.zeros(101))

        class Architecture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(3, 4)
                self.decoder_dict = torch.nn.ModuleDict({'standard': torch.nn.Linear(4, 100)})
                self.criterion = Criterion()

        loader = self._model_set({})
        with patch('density_dopfn._dopfn_architecture', Architecture):
            ck = self._repro_checkpoint('dopfn_1d', 10)
            e = loader._load_repro(ck, 'one.pt')
            self.assertEqual(e.kind, 'repro_1d')
            np.testing.assert_allclose(e.borders, np.linspace(-3, 3, 11), atol=1e-6)
            self.assertTrue(torch.equal(e.net.decoder_dict['standard'][2].weight,
                                        ck['model']['decoder_dict.standard.2.weight']))
            self.assertFalse(e.net.training)

            # dopfn_1d_botharms differs from dopfn_1d only in its training
            # query block, so the loader must treat it as the same 1-D head.
            botharms = loader._load_repro(
                self._repro_checkpoint('dopfn_1d_botharms', 10), 'botharms.pt')
            self.assertEqual(botharms.kind, 'repro_1d')
            np.testing.assert_allclose(botharms.borders, np.linspace(-3, 3, 11),
                                       atol=1e-6)

            e = loader._load_repro(self._repro_checkpoint('joint_2d', 17), 'joint.pt')
            self.assertEqual((e.kind, e.J), ('repro_joint', 2))
            self.assertTrue(np.isnan(e.query_fill))
            np.testing.assert_allclose(e.edges, [-1, 0, 1])
            e = loader._load_repro(
                self._repro_checkpoint('joint_2d', 17, query_treatment='zero'), 'z.pt')
            self.assertEqual(e.query_fill, 0.0)

            for ck, error, message in (
                    (self._repro_checkpoint('joint_2d', 18), ValueError, 'J\\^2\\+13'),
                    (self._repro_checkpoint('dopfn_1d', 10, y_space='raw'),
                     ValueError, 'zscore_ctx'),
                    (self._repro_checkpoint('other', 10), ValueError, 'variant'),
                    (self._repro_checkpoint('joint_2d', 17, query_treatment='x'),
                     ValueError, 'query_treatment')):
                with self.subTest(message=message), self.assertRaisesRegex(error, message):
                    loader._load_repro(ck, 'bad.pt')
            ck = self._repro_checkpoint('dopfn_1d', 10)
            ck['model']['stray.weight'] = torch.zeros(1)
            with self.assertRaisesRegex(RuntimeError, 'stray.weight'):
                loader._load_repro(ck, 'bad.pt')

    def test_released_repro_checkpoints_parse(self):
        """The real files, with a stub architecture (no Do-PFN checkout needed)."""
        paths = sorted((Path(__file__).resolve().parents[2]
                        / 'Required_checkpoints/new').glob('dopfn_repro_*.pt'))
        if not paths:
            self.skipTest('no Required_checkpoints/new/dopfn_repro_*.pt here')

        class Stub:
            def __init__(self):
                self.decoder_dict = {}

            def load_state_dict(self, state, strict):
                self.state = state
                return [], []

            def to(self, device):
                return self

            def eval(self):
                return self

        loader = self._model_set({})
        with patch('density_dopfn._dopfn_architecture', Stub):
            for path in paths:
                with self.subTest(path=path.name):
                    e = loader._load_checkpoint(str(path))
                    self.assertFalse(any(k.startswith('criterion.') for k in e.net.state))
                    n_out = e.net.state['decoder_dict.standard.2.bias'].shape[0]
                    if e.kind == 'repro_1d':
                        self.assertEqual(len(e.borders), n_out + 1)
                    else:
                        self.assertEqual(e.kind, 'repro_joint')
                        self.assertEqual(e.J ** 2 + 13, n_out)
                        self.assertTrue(np.isnan(e.query_fill))

    def test_summary_reads_dopfn_rows_from_shards(self):
        methods = ['dopfn_native', 'dopfn_repro_1d_J10', 'dopfn_repro_joint2d']
        row = dict(n_queries=2, frac_tau_outside_grid=0.0, anc_tag='none',
                   truth_noise_source='generator', sigma_raw=1.0, sigma_residual_raw=1.0,
                   methods=np.asarray(methods), kind_dopfn_native='1d',
                   kind_dopfn_repro_1d_J10='1d', kind_dopfn_repro_joint2d='joint',
                   source_dopfn_native='library', source_dopfn_repro_1d_J10='/c/j10.pt',
                   source_dopfn_repro_joint2d='/c/joint.pt')
        for i, method in enumerate(methods):
            for metric in ('nll', 'l2', 'kl_fwd', 'kl_rev', 'mass', 'pehe',
                           'cate_l1', 'ate_abs_err'):
                row[f'{metric}_{method}'] = 1.0 + i
        row['pehe_dopfn_repro_joint2d_inner'] = 0.5
        stream = io.StringIO()
        with redirect_stdout(stream):
            render_family('IHDP', [row, row], 'dopfn')
        text = stream.getvalue()
        for expected in ('DoPFN (x)indep native', 'DoPFN repro_1d_J10 (x)indep',
                         'DoPFN repro_joint2d Joint-2D',
                         'DoPFN repro_joint2d Joint-2D interior mean (raw)',
                         '(dopfn_native -> dopfn_repro_joint2d)',
                         '(dopfn_repro_1d_J10 -> dopfn_repro_joint2d)',
                         '`dopfn_repro_1d_J10`: /c/j10.pt'):
            self.assertIn(expected, text)
        self.assertNotIn('MIXED', text)
        other = dict(row, source_dopfn_repro_1d_J10='/c/other.pt')
        with redirect_stdout(io.StringIO()) as stream:
            render_family('IHDP', [row, other], 'dopfn')
        self.assertIn('MIXED CHECKPOINTS', stream.getvalue())
        with self.assertRaisesRegex(ValueError, 'reused'), redirect_stdout(io.StringIO()):
            render_family('IHDP', [row, dict(row, kind_dopfn_repro_1d_J10='joint')],
                          'dopfn')


if __name__ == '__main__':
    unittest.main()
