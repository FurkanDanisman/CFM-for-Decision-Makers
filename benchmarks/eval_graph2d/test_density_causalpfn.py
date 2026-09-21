"""CausalPFN numerical, checkpoint and inference contracts; no GPU required."""
from contextlib import contextmanager, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch

from density_common import (CausalPFN1D, Joint2D, causalpfn_tau_density,
                            joint_tau_density, mass)
from density_causalpfn import (CausalPFNDensityModels, CausalPFNModelSet,
                               checkpoint_kind, parse_model_list)
from summarize_density_tauC import mean_se, point_table, render_family


class CausalPFNDensityTest(unittest.TestCase):
    def test_uniform_histograms_give_triangular_difference(self):
        f = CausalPFN1D.from_pred(np.zeros(8), np.linspace(-2, 2, 9))
        grid = np.linspace(-5, 5, 10001)
        p = causalpfn_tau_density(f, f, grid)
        np.testing.assert_allclose(p, np.maximum(4 - np.abs(grid), 0) / 16,
                                   atol=1e-14)
        self.assertAlmostEqual(mass(p, grid), 1, places=12)
        self.assertEqual(float(p[0]), 0)

    def test_convolution_matches_bin_overlap_and_mean(self):
        edges = np.linspace(-2, 2, 9)
        f0 = CausalPFN1D.from_pred(np.arange(8) / 3, edges)
        f1 = CausalPFN1D.from_pred(-np.arange(8) / 2, edges)
        tau = np.array([-5, -2.07, -0.25, 0, 0.33, 1.9, 5])
        expected = []
        for t in tau:
            overlap = np.maximum(0, np.minimum(edges[1:, None], edges[None, 1:]-t)
                                 - np.maximum(edges[:-1, None], edges[None, :-1]-t))
            expected.append(np.sum(overlap * f0.p[:, None] * f1.p[None, :]) / f0.bw**2)
        np.testing.assert_allclose(causalpfn_tau_density(f0, f1, tau), expected,
                                   atol=1e-14)
        grid = np.linspace(-5, 5, 10001)
        density = causalpfn_tau_density(f0, f1, grid)
        self.assertAlmostEqual(mass(density, grid), 1, places=12)
        self.assertAlmostEqual(mass(grid*density, grid), f1.mean()-f0.mean(), places=12)

    def test_affine_conversion_preserves_joint_tails_and_native_density(self):
        factor, offset = 0.23, 0.7
        edges = np.linspace(-10, 10, 5)
        logits = np.random.default_rng(4).normal(size=4*4+13)
        raw = Joint2D.from_pred(logits, 4, edges)
        scaled = raw.affine(factor, offset)
        y0 = np.array([-20, -7, 3, 17, -13, 14])
        y1 = np.array([-13, 12, -3, 18, 15, -17])
        np.testing.assert_allclose(
            scaled.density(y0*factor+offset, y1*factor+offset),
            raw.density(y0, y1) / factor**2, rtol=1e-12)
        np.testing.assert_allclose(scaled.mean(), np.asarray(raw.mean())*factor+offset)
        # Point NLL includes the density Jacobian under a change of tau units.
        tau = np.array([-7., 0., 3.])
        np.testing.assert_allclose(
            joint_tau_density(scaled, tau*factor, n_y0=512),
            joint_tau_density(raw, tau, n_y0=512) / factor, rtol=1e-10)
        f = CausalPFN1D.from_pred([1, 2, 3, 4], edges)
        g = CausalPFN1D.from_pred([1, 2, 3, 4], edges*factor+offset)
        np.testing.assert_allclose(causalpfn_tau_density(g, g, tau*factor),
                                   causalpfn_tau_density(f, f, tau)/factor)
        self.assertEqual(raw.affine(1e-8, 100).rho, raw.rho)

    def test_invalid_histograms_fail(self):
        for logits, edges in (([0, 0], [0, 1, 3]), ([0], [0, 0]),
                              ([0], [0, np.nan]), ([np.nan], [0, 1])):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                CausalPFN1D.from_pred(logits, edges)

    def test_summary_reports_family_and_keeps_infinite_nll(self):
        self.assertEqual(mean_se([1, np.inf])[0], np.inf)
        self.assertEqual(mean_se([-np.inf, -2])[0], -np.inf)
        row = dict(n_queries=2, frac_tau_outside_grid=0, anc_tag='none',
                   truth_noise_source='generator', sigma_raw=1, sigma_residual_raw=2,
                   frac_zero_density_causalpfn_native=0.5)
        for method in ('causalpfn_native', 'causalpfn_joint'):
            for metric in ('nll', 'l2', 'kl_fwd', 'kl_rev', 'mass',
                           'pehe', 'cate_l1', 'ate_abs_err'):
                row[f'{metric}_{method}'] = 1.0
        row['nll_causalpfn_native'] = np.inf
        for metric in ('pehe', 'cate_l1', 'ate_abs_err'):
            row[f'{metric}_causalpfn_joint_inner'] = 2.0
        stream = io.StringIO()
        with redirect_stdout(stream):
            render_family('ACIC', [row, row], 'causalpfn')
        output = stream.getvalue()
        self.assertIn('ACIC — CausalPFN', output)
        self.assertIn('graph=none', output)
        self.assertIn('50.00%', output)
        self.assertIn('inf±nan', output)
        self.assertNotIn('UWYK', output)
        self.assertNotIn('DoPFN', output)
        self.assertIn('CausalPFN Joint-2D interior mean', point_table([row]))


class TinyNative(torch.nn.Module):
    """Small recording model with the real checkpoint's encoder/head layout."""
    def __init__(self, *, num_features, ninp, nhid, nbins, n_out, **kwargs):
        super().__init__()
        self.encoder = torch.nn.Linear(num_features, ninp, bias=False)
        self.head = torch.nn.Sequential(torch.nn.Linear(ninp, nhid, bias=False),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(nhid, n_out+nbins, bias=False))
        self.calls = []

    def forward(self, x, y):
        self.calls.append((x.clone(), y.clone()))
        return self.head(self.encoder(x[len(y):]))


class TinyJoint(torch.nn.Module):
    def __init__(self, *, J, num_features, edge_lo, edge_hi, **kwargs):
        super().__init__()
        self.backbone = TinyNative(num_features=num_features+1, nbins=J*J+13, **kwargs)
        self.model = self.backbone
        self.register_buffer('edges', torch.linspace(edge_lo, edge_hi, J+1))
        self.null_t_intv = torch.nn.Parameter(torch.zeros(1, 1, 1))
        self.J = J
        self.calls = []

    def _forward_logits(self, x, t, y, xq):
        self.calls.append((x.clone(), t.clone(), y.clone(), xq.clone()))
        xc = torch.cat((t[..., None], x), dim=-1)
        xt = torch.cat((self.null_t_intv.expand(1, xq.shape[1], 1), xq), dim=-1)
        p = self.backbone(torch.cat((xc, xt), dim=1).transpose(0, 1), y.transpose(0, 1))
        return p.transpose(0, 1)[..., -(self.J*self.J+13):]


def save_tiny_checkpoints(root):
    """The 1D and cpfn2d fakes, written under `root` -> (1d path, 2d path)."""
    cfg = dict(ninp=4, nhid=8, nhead=2, nlayers=1, n_out=2, dropout=0.)
    native = TinyNative(num_features=5, nbins=8, **cfg)
    joint = TinyJoint(J=2, num_features=4, edge_lo=-10., edge_hi=10., **cfg)
    native_sd = {'_orig_mod.model.' + k: v for k, v in native.state_dict().items()}
    native_sd['bin_edges'] = torch.linspace(-10, 10, 9)
    p1, p2 = Path(root) / '1d.pt', Path(root) / '2d.pt'
    torch.save(dict(model_state_dict=native_sd, model_config={'model': cfg}), p1)
    torch.save(dict(model_state_dict=joint.state_dict(), model_config={
        'model': dict(cfg, J=2, num_features=4),
        'y_scaling_mode': 'pooled_std'}), p2)
    return p1, p2


@contextmanager
def patched_causalpfn_modules():
    model_module = ModuleType('causalpfn.models.model')
    model_module.TabDPTLongContextModel = TinyNative
    joint_module = ModuleType('training_causalpfn2d.model_causalpfn_2d')
    joint_module.CausalPFN2DHead = TinyJoint
    with patch.dict(sys.modules, {
            'causalpfn.models.model': model_module,
            'training_causalpfn2d.model_causalpfn_2d': joint_module}):
        yield


class CausalPFNModelSetTest(unittest.TestCase):
    def test_parse_model_list(self):
        self.assertEqual(parse_model_list(' a_1=/x/a.pt,\n b = rel/b.pt,, '),
                         [('a_1', '/x/a.pt'), ('b', 'rel/b.pt')])
        # `native` has no meaning here: every CausalPFN row needs its own file.
        for bad, message in (('native', 'name=checkpoint'),
                             ('a=/x.pt,a=/y.pt', 'twice'),
                             ('/x/a.pt', 'letters, digits'),
                             (' ,, ', 'no models')):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, message):
                parse_model_list(bad)

    def test_head_is_read_from_the_file_not_the_name(self):
        self.assertEqual(checkpoint_kind({'config': {}, 'edges': [0., 1.]}), 'joint')
        self.assertEqual(checkpoint_kind({'model_config': {'model_type': 'cpfn2d'},
                                          'model_state_dict': {}}), 'joint')
        self.assertEqual(checkpoint_kind(
            {'model_state_dict': {'_orig_mod.null_t_intv': 0}}), 'joint')
        self.assertEqual(checkpoint_kind(
            {'model_state_dict': {'bin_edges': 0, 'model.encoder.weight': 0}}), '1d')
        with self.assertRaises(ValueError):
            checkpoint_kind({'model_state_dict': {'something.else': 0}})

    def test_listed_models_reproduce_the_fixed_pair_row_for_row(self):
        x, xq = np.arange(12).reshape(4, 3), np.arange(15).reshape(5, 3)
        t, y = np.array([0, 1, 0, 1]), np.array([10, 12, 18, 20])
        with tempfile.TemporaryDirectory() as root, patched_causalpfn_modules():
            p1, p2 = save_tiny_checkpoints(root)
            pair = CausalPFNDensityModels(root, p1, p2, 'cpu', query_chunk=2)
            arms, joints, pair_dump = pair.predict(x, t, y, xq, y_shift=10, y_scale=5)
            # Listed order is kept, and the joint file is recognised from its
            # contents even though this name says nothing about the head.
            models = CausalPFNModelSet(
                root, [('j32_2d', str(p2)), ('headrand_1d', str(p1))],
                'cpu', query_chunk=2)
            self.assertEqual(list(models.models),
                             ['causalpfn_j32_2d', 'causalpfn_headrand_1d'])
            self.assertEqual(models.kinds, {'causalpfn_j32_2d': 'joint',
                                            'causalpfn_headrand_1d': '1d'})
            self.assertEqual(models.sources['causalpfn_j32_2d'],
                             str(Path(p2).resolve()))
            self.assertIn('K=8', models.describe('causalpfn_headrand_1d'))
            out, dump = models.predict(x, t, y, xq, y_shift=10, y_scale=5)
            kind_1d, dens_1d = out['causalpfn_headrand_1d']
            kind_joint, dens_joint = out['causalpfn_j32_2d']
            self.assertEqual((kind_1d, kind_joint), ('1d', 'joint'))
            self.assertEqual([len(a) for a in dens_1d], [5, 5])
            self.assertEqual(len(dens_joint), 5)
            for q in range(5):
                for arm in (0, 1):
                    self.assertAlmostEqual(dens_1d[arm][q].mean(), arms[arm][q].mean())
                np.testing.assert_allclose(dens_joint[q].mean(), joints[q].mean())
            # Same numbers, namespaced per row so two models never collide.
            np.testing.assert_allclose(dump['causalpfn_headrand_1d_pred0'],
                                       pair_dump['causalpfn_pred0'])
            np.testing.assert_allclose(dump['causalpfn_headrand_1d_edges1d'],
                                       pair_dump['causalpfn_edges1d'])
            np.testing.assert_allclose(dump['causalpfn_j32_2d_logits'],
                                       pair_dump['causalpfn_joint_logits'])
            self.assertEqual(dump['causalpfn_y_scale'], pair_dump['causalpfn_y_scale'])
            self.assertEqual(list(dump['causalpfn_methods']),
                             ['causalpfn_j32_2d', 'causalpfn_headrand_1d'])
            self.assertEqual(list(dump['causalpfn_kinds']), ['joint', '1d'])
            with self.assertRaises(FileNotFoundError):
                CausalPFNModelSet(root, [('gone', str(Path(root) / 'nope.pt'))], 'cpu')

    def test_summary_labels_every_listed_model(self):
        names = ['causalpfn_j32_random_2d', 'causalpfn_botharms_1d']
        row = dict(n_queries=2, frac_tau_outside_grid=0, anc_tag='none',
                   truth_noise_source='generator', sigma_raw=1,
                   sigma_residual_raw=2, methods=np.asarray(names),
                   kind_causalpfn_j32_random_2d='joint',
                   kind_causalpfn_botharms_1d='1d',
                   source_causalpfn_j32_random_2d='/ckpt/cpfn2d_j32_random.pt',
                   source_causalpfn_botharms_1d='/ckpt/cpfn1d_botharms.pt')
        for method in names:
            for metric in ('nll', 'l2', 'kl_fwd', 'kl_rev', 'mass',
                           'pehe', 'cate_l1', 'ate_abs_err'):
                row[f'{metric}_{method}'] = 1.0
        stream = io.StringIO()
        with redirect_stdout(stream):
            render_family('IHDP', [row, row], 'causalpfn')
        output = stream.getvalue()
        self.assertIn('CausalPFN j32_random_2d Joint-2D', output)
        self.assertIn('CausalPFN botharms_1d (x)indep', output)
        self.assertIn('/ckpt/cpfn1d_botharms.pt', output)
        self.assertIn('causalpfn_botharms_1d -> causalpfn_j32_random_2d', output)


class CausalPFNAdapterTest(unittest.TestCase):
    def test_checkpoint_loading_chunking_padding_scaling_and_replay(self):
        cfg = dict(ninp=4, nhid=8, nhead=2, nlayers=1, n_out=2, dropout=0.)
        native = TinyNative(num_features=5, nbins=8, **cfg)
        joint = TinyJoint(J=2, num_features=4, edge_lo=-10., edge_hi=10., **cfg)
        native_sd = {'_orig_mod.model.'+k: v for k, v in native.state_dict().items()}
        native_sd['bin_edges'] = torch.linspace(-10, 10, 9)
        model_module = ModuleType('causalpfn.models.model')
        model_module.TabDPTLongContextModel = TinyNative
        joint_module = ModuleType('training_causalpfn2d.model_causalpfn_2d')
        joint_module.CausalPFN2DHead = TinyJoint
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, {
                'causalpfn.models.model': model_module,
                'training_causalpfn2d.model_causalpfn_2d': joint_module}):
            p1, p2 = Path(root)/'1d.pt', Path(root)/'2d.pt'
            torch.save(dict(model_state_dict=native_sd, model_config={'model': cfg}), p1)
            ck2 = dict(model_state_dict=joint.state_dict(), model_config={
                'model': dict(cfg, J=2, num_features=4), 'y_scaling_mode': 'pooled_std'})
            torch.save(ck2, p2)
            models = CausalPFNDensityModels(root, p1, p2, 'cpu', query_chunk=2)
            x, xq = np.arange(12).reshape(4, 3), np.arange(15).reshape(5, 3)
            t, y = np.array([0, 1, 0, 1]), np.array([10, 12, 18, 20])
            arms, joints, dump = models.predict(x, t, y, xq, y_shift=10, y_scale=5)
            self.assertEqual([len(a) for a in arms], [5, 5])
            self.assertEqual(len(joints), 5)
            self.assertEqual([c[0].shape[0]-4 for c in models.native.calls], [2, 2, 2, 2, 1, 1])
            scale = np.std(y, ddof=1)
            for i, (xs, ys) in enumerate(models.native.calls):
                np.testing.assert_allclose(xs[:4, 0, 0], t)
                np.testing.assert_allclose(xs[4:, 0, 0], i % 2)
                np.testing.assert_allclose(xs[:4, 0, 1:4], x)
                np.testing.assert_allclose(xs[:, 0, 4], 0)
                np.testing.assert_allclose(ys[:, 0], (y-y.mean())/scale, rtol=1e-6)
            np.testing.assert_allclose(dump['causalpfn_edges1d'],
                                       (models.edges1d*scale+y.mean()-10)/5, rtol=1e-6)
            for q in range(5):
                replay = Joint2D.from_pred(dump['causalpfn_joint_logits'][q], 2,
                                           dump['causalpfn_edges2d_native']).affine(
                    dump['causalpfn_y_scale']/5, (dump['causalpfn_y_shift']-10)/5)
                np.testing.assert_allclose(replay.mean(), joints[q].mean())
                f = CausalPFN1D.from_pred(dump['causalpfn_pred0'][q],
                                          dump['causalpfn_edges1d'])
                self.assertAlmostEqual(f.mean(), arms[0][q].mean())
            # Strict loading catches an incomplete head instead of silently
            # evaluating randomly initialized parameters.
            broken = dict(native_sd)
            del broken['_orig_mod.model.head.2.weight']
            torch.save(dict(model_state_dict=broken, model_config={'model': cfg}), p1)
            with self.assertRaises((KeyError, RuntimeError)):
                CausalPFNDensityModels(root, p1, p2, 'cpu')


if __name__ == '__main__':
    unittest.main()
