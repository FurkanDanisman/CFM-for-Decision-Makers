"""DoPFN inference adapters for Tier C; numerical densities live in density_common.

DoPFN and UWYK both import top-level utils/models packages. Keep DoPFN's
module cache and relative artifact paths scoped to its inference calls.

DoPFNModelSet scores every model listed in DOPFN_MODELS on one shared context.
DoPFNDensityModels is the older fixed pair (library model + one
training_dopfn_base joint); case_study/density_eval/eval_density_tauC.py still
imports it, so its interface is unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import inspect
import os
from pathlib import Path
import pickle
import re
import sys
from types import SimpleNamespace

import numpy as np
import torch

from density_common import DoPFN1D, Joint2D

_REPO = Path(__file__).resolve().parents[2]
_PREFIXES = ('utils', 'models', 'model', 'scripts', 'datasets', 'priors')
_ARTIFACTS = ('artifacts/dopfn_model.pkl', 'artifacts/dopfn_config.pkl',
              'artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt')
NATIVE = 'native'


def _collides(name):
    return name.split('.')[0] in _PREFIXES


def _load_check_array_compat():
    """Load this checkout's shim by filename, bypassing cached `import dopfn`."""
    path = _REPO / 'benchmarks/methods/dopfn.py'
    spec = importlib.util.spec_from_file_location('_tauc_dopfn_compat', path)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    if 'regressor' not in inspect.signature(shim._repatch_dopfn_check_array).parameters:
        raise RuntimeError(
            f'Outdated DoPFN compatibility shim: {path}. Sync '
            'benchmarks/methods/dopfn.py together with density_dopfn.py '
            'to the checkout used by this job, then start a new process.')
    return shim


def parse_model_list(spec):
    """DOPFN_MODELS -> [(name, checkpoint or None)], in the listed order.

    Entries are separated by commas or newlines. `native` is the DoPFN library
    regressor and takes no path; every other entry is `name=checkpoint`. Each
    entry becomes the density row `dopfn_<name>`.
    """
    entries, names = [], set()
    for item in re.split(r'[,\n]', spec):
        item = item.strip()
        if not item:
            continue
        name, has_path, path = (part.strip() for part in item.partition('='))
        if not re.fullmatch(r'[A-Za-z0-9_]+', name):
            raise ValueError(f'DOPFN_MODELS entry {item!r}: the name before "=" '
                             'may only use letters, digits and _')
        if name == NATIVE:
            if has_path:
                raise ValueError('DOPFN_MODELS: native is the DoPFN library model '
                                 'and takes no checkpoint path')
            path = None
        elif not path:
            raise ValueError(f'DOPFN_MODELS entry {item!r}: expected name=checkpoint, '
                             f'or {NATIVE} for the library model')
        if name in names:
            raise ValueError(f'DOPFN_MODELS lists {name!r} twice')
        names.add(name)
        entries.append((name, path))
    if not entries:
        raise ValueError('DOPFN_MODELS lists no models')
    return entries


class _DoPFNRuntime:
    """Scoped Do-PFN imports, plus the library regressor's per-arm outputs."""

    def _setup(self, root, device, query_chunk, checkpoints=()):
        self.root = str(Path(root).expanduser().resolve())
        self.device = torch.device(device)
        self.query_chunk = int(query_chunk)
        if self.query_chunk <= 0:
            raise ValueError('DOPFN_QUERY_CHUNK must be positive')
        for path in (*(Path(self.root) / a for a in _ARTIFACTS), *map(Path, checkpoints)):
            if not path.is_file():
                raise FileNotFoundError(f'Missing DoPFN input: {path}')
        self._modules = {}
        self._compat_shim = _load_check_array_compat()

    def _load_regressor(self):
        # Reuse the point benchmark's sklearn and dataset import shims.
        sys.path.insert(0, str(_REPO / 'benchmarks/empirical_tests'))
        from dopfn_helpers import load_dopfn
        return load_dopfn(SimpleNamespace(repo=str(_REPO), dopfn=self.root))

    def _load_bb_joint(self, ck):
        """training_dopfn_base checkpoint -> (DoPFNBackboneWith2DHead, J, edges)."""
        from training_dopfn_base.dopfn_backbone_head import DoPFNBackboneWith2DHead
        J = int(ck['config']['J'])
        edges = _uniform_edges(ck['edges'], J)
        joint = DoPFNBackboneWith2DHead(dopfn_root=self.root, K=J)
        state = {k.removeprefix('_orig_mod.'): v for k, v in ck['model_state_dict'].items()}
        joint.load_state_dict(state, strict=True)
        return joint.to(self.device).eval(), J, edges

    def _prepare_regressor(self, reg):
        # Bind immediately before fit, after restoring modules and constructing
        # the regressor. Constructor/pickle imports may retain an older alias.
        if not hasattr(self, '_compat_shim'):
            self._compat_shim = _load_check_array_compat()
        self._compat_shim._repatch_dopfn_check_array(reg)
        checked = []
        for name in ('check_training_data', 'predict_common_setup'):
            method = getattr(reg, name, None)
            function = getattr(method, '__func__', method)
            if function is None:
                continue
            function = inspect.unwrap(function)
            check = getattr(function, '__globals__', {}).get('check_array')
            if check is not None:
                # Exercise the actual callable used by DoPFN. This catches a
                # partial deployment before checkpoint loading inside fit().
                probe = np.array([[0.0, np.nan]], dtype=np.float32)
                check(probe, force_all_finite=False)
                check(probe, ensure_all_finite=False)
                checked.append(name)
        if checked and not getattr(self, '_compat_reported', False):
            import sklearn
            print(f'[dopfn-compat] sklearn={sklearn.__version__} '
                  f'shim={self._compat_shim.__file__} '
                  f'validated={",".join(checked)}', flush=True)
            self._compat_reported = True

    @contextmanager
    def environment(self):
        old_cwd, old_path = os.getcwd(), sys.path[:]
        previous = {k: sys.modules.pop(k) for k in list(sys.modules) if _collides(k)}
        sys.modules.update(self._modules)
        sys.path.insert(0, self.root)
        sys.path.insert(1, str(_REPO))
        try:
            os.chdir(self.root)
            yield
        finally:
            self._modules = {k: sys.modules.pop(k) for k in list(sys.modules) if _collides(k)}
            sys.modules.update(previous)
            sys.path[:] = old_path
            os.chdir(old_cwd)

    def _native_arms(self, X_raw, treatment, y_raw, X_test_raw):
        """DoPFNRegressor.predict_full per arm -> [(logits, borders, tail scales)].

        Raw features and outcomes; the library owns its preprocessing. Call
        inside environment().
        """
        reg = self.regressor_class()
        reg.device = str(self.device)
        self._prepare_regressor(reg)
        x_train = np.column_stack((treatment, X_raw)).astype(np.float32)
        reg.fit(torch.from_numpy(x_train), torch.as_tensor(y_raw, dtype=torch.float32))
        arms = []
        for arm in (0.0, 1.0):
            chunks, borders, scales = [], None, None
            for start in range(0, len(X_test_raw), self.query_chunk):
                x = X_test_raw[start:start + self.query_chunk]
                x = np.column_stack((np.full(len(x), arm), x)).astype(np.float32)
                full = reg.predict_full(torch.from_numpy(x))
                criterion = full['criterion']
                if not hasattr(criterion, 'halfnormal_with_p_weight_before'):
                    raise TypeError('Expected DoPFN FullSupportBarDistribution criterion')
                current = _numpy(criterion.borders)
                # Upstream may leave bucket_widths stale after undoing Y
                # normalization. Derive raw-unit tail widths from borders.
                widths = torch.diff(criterion.borders)
                current_scales = np.array([
                    float(criterion.halfnormal_with_p_weight_before(widths[i]).scale)
                    for i in (0, -1)])
                if borders is not None and (not np.array_equal(borders, current)
                                            or not np.array_equal(scales, current_scales)):
                    raise RuntimeError('DoPFN borders/scales changed between query chunks')
                borders, scales = current, current_scales
                logits = _numpy(full['logits'])
                if logits.shape != (len(x), len(borders) - 1):
                    raise ValueError(f'Unexpected DoPFN logits shape: {logits.shape}')
                chunks.append(logits)
            arms.append((np.concatenate(chunks), borders, scales))
        del reg
        return arms


class DoPFNDensityModels(_DoPFNRuntime):
    """Fixed pair: the library DoPFNRegressor + one training_dopfn_base joint."""

    def __init__(self, root, checkpoint, device='cpu', query_chunk=20):
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self._setup(root, device, query_chunk, checkpoints=(self.checkpoint,))
        with self.environment():
            self.regressor_class = self._load_regressor()
            ck = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
            self.joint, self.J, self.edges = self._load_bb_joint(ck)

    @torch.inference_mode()
    def predict(self, X_raw, treatment, y_raw, X_test_raw,
                X_standardized, y_scaled, X_test_standardized, *, y_shift, y_scale):
        """Same selected context for both models; native API owns its preprocessing.

        The joint checkpoint consumes standardized X and harness-scaled factual
        Y, matching its existing raw evaluation path. Neither model caps features.
        """
        with self.environment():
            arms = self._native_arms(X_raw, treatment, y_raw, X_test_raw)
            joint_logits = _bb_joint_logits(
                self.joint, X_standardized, treatment, y_scaled,
                X_test_standardized, self.query_chunk, self.device)
            if joint_logits.shape != (len(X_test_raw), self.J ** 2 + 13):
                raise ValueError(f'Unexpected joint DoPFN logits shape: {joint_logits.shape}')

        distributions, native_dump = _native_outputs(arms, y_shift, y_scale)
        dump = dict(dopfn_joint_logits=joint_logits,
                    dopfn_J=self.J, dopfn_edges2d=self.edges,
                    dopfn_joint_ckpt=self.checkpoint,
                    dopfn_root=self.root,
                    dopfn_n_features=X_raw.shape[1],
                    dopfn_query_chunk=self.query_chunk,
                    dopfn_tail_width_source='returned_borders', **native_dump)
        return distributions, joint_logits, dump


class DoPFNModelSet(_DoPFNRuntime):
    """Every model named in DOPFN_MODELS, scored on one shared context.

    A checkpoint's kind is read from the file, never from its list name:
      native       DoPFNRegressor from DOPFN_ROOT, 1D arms
      repro_1d     training_dopfn_repro dopfn_1d / dopfn_1d_botharms:
                   FullSupportBarDistribution arms
      repro_joint  training_dopfn_repro joint_2d: J x J BarDistribution2D joint
      bb_joint     training_dopfn_base DoPFNBackboneWith2DHead joint (dopfn_bb_*)

    Each model gets its own training contract. native and the repro kinds take
    raw covariates with the treatment in column 0; the transformer normalizes
    features internally. The repro kinds also take factual y z-scored with
    context statistics (training_dopfn_repro/batch.py::_apply_y_space). Their
    outputs are mapped affinely to the harness axis. bb_joint receives
    standardized covariates and harness-scaled y, exactly as
    DoPFNDensityModels passes them. No model caps features.
    """

    def __init__(self, root, entries, device='cpu', query_chunk=20):
        # Resolve before environment() changes cwd to the Do-PFN checkout.
        entries = [(name, None if path is None else str(Path(path).expanduser().resolve()))
                   for name, path in entries]
        self._setup(root, device, query_chunk,
                    checkpoints=[path for _, path in entries if path is not None])
        self.models = {}
        with self.environment():
            self.regressor_class = self._load_regressor()
            for name, path in entries:
                entry = (SimpleNamespace(kind=NATIVE) if path is None
                         else self._load_checkpoint(path))
                entry.source = path or 'library'
                self.models[f'dopfn_{name}'] = entry

    @property
    def kinds(self):
        """method -> '1d' (arms convolved under independence) or 'joint'."""
        return {m: 'joint' if e.kind.endswith('joint') else '1d'
                for m, e in self.models.items()}

    @property
    def sources(self):
        return {m: e.source for m, e in self.models.items()}

    def describe(self, method):
        e = self.models[method]
        size = (f' K={len(e.borders) - 1}' if e.kind == 'repro_1d'
                else f' J={e.J}' if e.kind.endswith('joint') else '')
        return f'{e.kind}{size} {e.source}'

    def _load_checkpoint(self, path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if 'model' in ck and 'provenance' in ck:
            return self._load_repro(ck, path)
        if 'model_state_dict' in ck and 'config' in ck:
            net, J, edges = self._load_bb_joint(ck)
            return SimpleNamespace(kind='bb_joint', net=net, J=J, edges=edges)
        raise ValueError(f'{path}: neither a training_dopfn_repro nor a '
                         f'training_dopfn_base checkpoint (keys: {sorted(ck)})')

    def _load_repro(self, ck, path):
        prov = ck['provenance']
        batch_cfg = prov.get('batch_cfg') or {}
        if batch_cfg.get('y_space') != 'zscore_ctx':
            raise ValueError(f'{path}: y_space={batch_cfg.get("y_space")!r}; only '
                             'zscore_ctx checkpoints are supported')
        state = {k.removeprefix('_orig_mod.'): v for k, v in ck['model'].items()}
        # Head support is read from here. The pickled architecture keeps its own
        # 100-bucket criterion, which neither forward pass uses.
        criterion = {k: state.pop(k) for k in list(state) if k.startswith('criterion.')}
        hidden, d_model = state['decoder_dict.standard.0.weight'].shape
        n_out = int(state['decoder_dict.standard.2.weight'].shape[0])
        net = _dopfn_architecture()
        # training_dopfn_repro/model.py::make_decoder
        net.decoder_dict['standard'] = torch.nn.Sequential(
            torch.nn.Linear(d_model, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, n_out))
        missing, unexpected = net.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith('criterion.')]
        if missing or unexpected:
            raise RuntimeError(f'{path}: state dict does not fit the Do-PFN architecture '
                               f'(missing {missing[:5]}, unexpected {unexpected[:5]})')
        net.to(self.device).eval()

        variant = prov.get('variant')
        # dopfn_1d and dopfn_1d_botharms are the SAME model at inference: one
        # FullSupportBarDistribution head over one set of borders, queried once
        # per arm. They differ only in how the training query block was built
        # (coin-flipped single arm vs both arms of every unit), which leaves no
        # trace in the checkpoint beyond this name. So they load identically --
        # and p(tau) still comes from the independence convolution either way,
        # because a 1-D head emits marginals, not a joint.
        if variant in ('dopfn_1d', 'dopfn_1d_botharms'):
            if 'criterion.borders' not in criterion:
                raise ValueError(f'{path}: {variant} checkpoint has no criterion.borders')
            borders = _numpy(criterion['criterion.borders'])
            if borders.shape != (n_out + 1,) or np.any(np.diff(borders) <= 0):
                raise ValueError(f'{path}: {n_out} logits need {n_out + 1} increasing borders, '
                                 f'got shape {borders.shape}')
            return SimpleNamespace(kind='repro_1d', net=net, borders=borders)
        if variant == 'joint_2d':
            J = int((prov.get('spec') or {}).get('j_2d', round((n_out - 13) ** 0.5)))
            if J * J + 13 != n_out:
                raise ValueError(f'{path}: head width {n_out} is not J^2+13 for J={J}')
            fill = {'nan': float('nan'), 'zero': 0.0}.get(
                batch_cfg.get('query_treatment', 'nan'))
            if fill is None:
                raise ValueError(f'{path}: unknown query_treatment '
                                 f'{batch_cfg.get("query_treatment")!r}')
            return SimpleNamespace(kind='repro_joint', net=net, J=J,
                                   edges=_uniform_edges(ck['edges'], J), query_fill=fill)
        raise ValueError(f'{path}: unknown training_dopfn_repro variant {variant!r}')

    def _repro_logits(self, net, X, treatment, y_z, X_query, fill):
        """Seq-first forward, as training_dopfn_repro/train.py::compute_loss calls it.

        Context column 0 is the factual treatment. Query column 0 is the
        intervened arm (1D) or the training placeholder (joint).
        """
        def seq(a):
            return torch.as_tensor(np.asarray(a, dtype=np.float32),
                                   device=self.device).unsqueeze(1)

        x_ctx = seq(np.column_stack((treatment, X)))                    # (N, 1, F+1)
        y_ctx = y_z.to(self.device).unsqueeze(1)                          # (N, 1)
        chunks = []
        for start in range(0, len(X_query), self.query_chunk):
            xq = X_query[start:start + self.query_chunk]
            xq = seq(np.column_stack((np.full(len(xq), fill), xq)))      # (M, 1, F+1)
            out = net(x_ctx, y_ctx, xq, only_return_standard_out=True)   # (M, 1, n_out)
            chunks.append(_numpy(out[:, 0]))
        return np.concatenate(chunks)

    @torch.inference_mode()
    def predict(self, X_raw, treatment, y_raw, X_test_raw,
                X_standardized, y_scaled, X_test_standardized, *, y_shift, y_scale):
        """-> ({method: ('1d', (arm0, arm1)) or ('joint', joints)}, dump).

        arm0/arm1 hold one DoPFN1D per query and joints one Joint2D per query,
        all on the harness axis y_scaled = (y_raw - y_shift) / y_scale.
        """
        if not np.isfinite(y_scale) or y_scale <= 0:
            raise ValueError('y_scale must be finite and positive')
        n = len(X_test_raw)
        out = {}
        dump = dict(dopfn_methods=np.asarray(list(self.models)),
                    dopfn_kinds=np.asarray([e.kind for e in self.models.values()]),
                    dopfn_sources=np.asarray([e.source for e in self.models.values()]),
                    dopfn_root=self.root, dopfn_n_features=X_raw.shape[1],
                    dopfn_query_chunk=self.query_chunk)
        with self.environment():
            context = None
            for method, e in self.models.items():
                if e.kind == NATIVE:
                    arms = self._native_arms(X_raw, treatment, y_raw, X_test_raw)
                    distributions, fields = _native_outputs(arms, y_shift, y_scale)
                    out[method] = ('1d', distributions)
                    dump.update(fields, dopfn_tail_width_source='returned_borders')
                    continue
                if e.kind == 'bb_joint':
                    logits = _bb_joint_logits(e.net, X_standardized, treatment, y_scaled,
                                              X_test_standardized, self.query_chunk,
                                              self.device)
                    _check_shape(method, logits, (n, e.J ** 2 + 13))
                    out[method] = ('joint', [Joint2D.from_pred(p, e.J, e.edges)
                                             for p in logits])
                    dump.update({f'{method}_logits': logits, f'{method}_J': e.J,
                                 f'{method}_edges2d': e.edges, f'{method}_ckpt': e.source})
                    continue

                if context is None:
                    context = _zscore_context(y_raw)
                y_z, shift, scale = context
                dump.update({f'{method}_y_shift': shift, f'{method}_y_scale': scale,
                             f'{method}_ckpt': e.source})
                if e.kind == 'repro_1d':
                    preds = []
                    for arm in (0.0, 1.0):
                        logits = self._repro_logits(e.net, X_raw, treatment, y_z,
                                                    X_test_raw, arm)
                        _check_shape(method, logits, (n, len(e.borders) - 1))
                        preds.append(logits)
                    # z -> raw is affine, so tail widths (and the half-normal
                    # scales DoPFN1D derives from them) scale with it.
                    borders_raw = shift + scale * e.borders
                    out[method] = ('1d', [[DoPFN1D.from_pred(p, borders_raw, y_shift=y_shift,
                                                             y_scale=y_scale) for p in logits]
                                          for logits in preds])
                    dump.update({f'{method}_pred0': preds[0], f'{method}_pred1': preds[1],
                                 f'{method}_borders_native': e.borders,
                                 f'{method}_borders_raw': borders_raw})
                else:
                    logits = self._repro_logits(e.net, X_raw, treatment, y_z,
                                                X_test_raw, e.query_fill)
                    _check_shape(method, logits, (n, e.J ** 2 + 13))
                    factor, offset = scale / y_scale, (shift - y_shift) / y_scale
                    out[method] = ('joint', [Joint2D.from_pred(p, e.J, e.edges).affine(
                        factor, offset) for p in logits])
                    dump.update({f'{method}_logits': logits, f'{method}_J': e.J,
                                 f'{method}_edges2d_native': e.edges,
                                 f'{method}_edges2d': e.edges * factor + offset})
        return out, dump


def _dopfn_architecture():
    """Do-PFN's PerFeatureTransformer, unpickled. cwd must be the Do-PFN root."""
    with open('artifacts/dopfn_model.pkl', 'rb') as fh:
        return pickle.load(fh)


def _zscore_context(y_raw):
    """training_dopfn_repro/batch.py::_apply_y_space('zscore_ctx') on the context.

    -> (z-scored y, mean, std); torch.std is the unbiased estimate, as in training.
    """
    y = torch.as_tensor(np.asarray(y_raw, dtype=np.float32).reshape(-1))
    if y.numel() < 2:
        raise ValueError('DoPFN repro models need at least two context rows')
    mean, std = y.mean(), y.std().clamp_min(1e-6)
    return (y - mean) / std, float(mean), float(std)


def _bb_joint_logits(joint, X, treatment, y_scaled, X_query, query_chunk, device):
    def tensor(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device).unsqueeze(0)

    Xc = tensor(X)
    Tc = tensor(np.asarray(treatment).reshape(-1, 1))
    Yc = tensor(np.asarray(y_scaled).reshape(-1, 1))
    chunks = []
    for start in range(0, len(X_query), query_chunk):
        Xq = tensor(X_query[start:start + query_chunk])
        pred = joint(X_context=Xc, T_context=Tc, Y_context=Yc, X_query=Xq)['predictions']
        chunks.append(_numpy(pred[0]))
    return np.concatenate(chunks)


def _native_outputs(arms, y_shift, y_scale):
    distributions, dump = [], {}
    for arm, (logits, borders, scales) in enumerate(arms):
        distributions.append([
            DoPFN1D.from_pred(p, borders, y_shift=y_shift, y_scale=y_scale,
                              tail_scales=scales) for p in logits])
        dump.update({f'dopfn_pred{arm}': logits,
                     f'dopfn_borders{arm}_raw': borders,
                     f'dopfn_tail_scales{arm}_raw': scales})
    return distributions, dump


def _uniform_edges(edges, J):
    edges = _numpy(edges)
    if (edges.shape != (J + 1,) or np.any(np.diff(edges) <= 0)
            or not np.allclose(np.diff(edges), np.diff(edges).mean(), rtol=1e-4)):
        raise ValueError('Joint DoPFN checkpoint must have J+1 uniform increasing edges')
    return edges


def _check_shape(method, logits, expected):
    if logits.shape != expected:
        raise ValueError(f'Unexpected {method} logits shape: {logits.shape}, '
                         f'expected {expected}')


def _numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().double().numpy()
    return np.asarray(value, dtype=np.float64)
