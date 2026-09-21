"""CausalPFN checkpoint inference for the shared RealCause density runner.

Both heads consume pooled-standardized factual outcomes. Their native output
axes are mapped back to the harness axis before scoring; Joint2D then retains
all region weights and tail scales, unlike the old interior-only density dump.

CausalPFNModelSet scores every model listed in CAUSALPFN_MODELS on one shared
context, the way DoPFNModelSet does for DOPFN_MODELS. It is what the eval
driver builds, including for a one-1D-plus-one-joint list.

CausalPFNDensityModels is the older fixed pair (one 1D head + one joint),
unchanged and still exercised by test_density_causalpfn.py, which pins the set
to it row for row. Nothing else in the tree constructs it now; keep it as the
reference implementation of the two forward contracts.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import torch

from density_common import CausalPFN1D, Joint2D


def _state_dict(checkpoint):
    return {k.replace('_orig_mod.', ''): v
            for k, v in checkpoint['model_state_dict'].items()}


def _pad_features(x, n_features):
    x = np.asarray(x, dtype=np.float32)
    if x.shape[1] > n_features:
        return x[:, :n_features]
    return np.pad(x, ((0, 0), (0, n_features - x.shape[1])))


def parse_model_list(spec):
    """CAUSALPFN_MODELS -> [(name, checkpoint)], in the listed order.

    Entries are separated by commas or newlines and every one is
    `name=checkpoint`: unlike DoPFN there is no library model to name on its
    own, so a bare word is an error rather than a silent no-checkpoint entry.
    Each entry becomes the density row `causalpfn_<name>`; whether it is a 1D
    or a joint head is read from the checkpoint, never from the name.
    """
    entries, names = [], set()
    for item in re.split(r'[,\n]', spec):
        item = item.strip()
        if not item:
            continue
        name, _, path = (part.strip() for part in item.partition('='))
        if not re.fullmatch(r'[A-Za-z0-9_]+', name):
            raise ValueError(f'CAUSALPFN_MODELS entry {item!r}: the name before "=" '
                             'may only use letters, digits and _')
        if not path:
            raise ValueError(f'CAUSALPFN_MODELS entry {item!r}: expected '
                             'name=checkpoint; every CausalPFN row needs its own file')
        if name in names:
            raise ValueError(f'CAUSALPFN_MODELS lists {name!r} twice')
        names.add(name)
        entries.append((name, path))
    if not entries:
        raise ValueError('CAUSALPFN_MODELS lists no models')
    return entries


def checkpoint_kind(ck):
    """'1d' or 'joint', decided by the file's own structure.

    A trained cpfn2d checkpoint carries model_type=cpfn2d (older ones a
    top-level config/edges pair) and registers `edges` / `null_t_intv` in its
    state dict; a 1D head carries the HL-Gauss support as `bin_edges` next to
    the `model.*` backbone. Names are not consulted: cpfn1d_botharms and
    cpfn1d_j32 differ from cpfn1d_j1024_headrand only in training, and a list
    name is free text either way.
    """
    if 'config' in ck and 'edges' in ck:
        return 'joint'
    if (ck.get('model_config') or {}).get('model_type') == 'cpfn2d':
        return 'joint'
    state = {k.replace('_orig_mod.', '') for k in (ck.get('model_state_dict') or {})}
    if 'edges' in state or 'null_t_intv' in state:
        return 'joint'
    if 'bin_edges' in state or 'model.encoder.weight' in state:
        return '1d'
    raise ValueError('Not a CausalPFN 1D or cpfn2d checkpoint (top-level keys: '
                     f'{sorted(ck)})')


def _check_shape(method, logits, expected):
    if logits.shape != expected:
        raise ValueError(f'Unexpected {method} logits shape: {logits.shape}, '
                         f'expected {expected}')


class _CausalPFNRuntime:
    """Checkpoint loading and the two forward contracts, shared by both classes."""

    def _setup(self, root, device, query_chunk, checkpoints=()):
        self.root = str(Path(root).resolve())
        self.device = torch.device(device)
        self.query_chunk = int(query_chunk)
        if self.query_chunk <= 0:
            raise ValueError('CAUSALPFN_QUERY_CHUNK must be positive')
        for path in checkpoints:
            if not Path(path).is_file():
                raise FileNotFoundError(f'Missing CausalPFN checkpoint: {path}')
        repo = Path(__file__).resolve().parents[2]
        for path in (repo, Path(self.root), Path(self.root) / 'src'):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

    def _tensor(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32,
                               device=self.device).unsqueeze(0)

    # -- loading ------------------------------------------------------------
    def _load_checkpoint(self, path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        kind = checkpoint_kind(ck)
        entry = self._load_native(ck) if kind == '1d' else self._load_joint(ck)
        del ck
        return entry

    def _load_native(self, ck):
        """CausalPFN 1D head -> (net, K bins, F features, native bin edges)."""
        from causalpfn.models.model import TabDPTLongContextModel

        state = _state_dict(ck)
        mc = ck.get('model_config', {})
        cfg = mc.get('model', mc)
        sd = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
        F = int(sd['encoder.weight'].shape[1]) - 1
        n_out = int(cfg.get('n_out', cfg.get('max_num_classes', 10)))
        K = int(sd['head.2.weight'].shape[0]) - n_out
        ninp = int(sd['encoder.weight'].shape[0])
        net = TabDPTLongContextModel(
            dropout=cfg.get('dropout', 0.0), n_out=n_out,
            nhead=cfg.get('nhead', 6),
            nhid=cfg.get('nhid', ninp * cfg.get('nhid_factor', 2)),
            ninp=ninp,
            nlayers=cfg.get('nlayers', 20), num_features=F + 1,
            nbins=K)
        net.load_state_dict(sd, strict=True)
        net.to(self.device).eval()
        # InContextModel stores the HL-Gauss support outside model.*.
        saved_edges = state.get('bin_edges')
        edges = (saved_edges.cpu().double().numpy().reshape(-1)
                 if saved_edges is not None else np.linspace(-10.0, 10.0, K + 1))
        CausalPFN1D.from_pred(np.zeros(K), edges)
        return SimpleNamespace(kind='1d', net=net, K=K, F=F, edges=edges)

    def _load_joint(self, ck):
        """training_causalpfn2d head -> (net, J, F, edges, y_scaling_mode)."""
        from training_causalpfn2d.model_causalpfn_2d import CausalPFN2DHead

        sd = _state_dict(ck)
        if 'config' in ck:
            cfg = dict(ck['config'])
            edges = ck['edges']
        else:
            mc = ck['model_config']
            cfg = dict(mc['model'])
            cfg.update({k: mc[k] for k in
                        ('y_scaling_mode', 'loss_type', 'hlgauss_sigma') if k in mc})
            edges = sd['edges']
        y_scaling_mode = cfg.get('y_scaling_mode', 'pooled_std')
        if y_scaling_mode != 'pooled_std':
            raise ValueError('CausalPFN density evaluation requires a pooled_std '
                             f'joint checkpoint; got {y_scaling_mode!r}')
        J = int(cfg['J'])
        F = int(cfg['num_features'])
        edges2d = np.asarray(torch.as_tensor(edges).cpu(), dtype=np.float64)
        if edges2d.shape != (J + 1,):
            raise ValueError('CausalPFN joint checkpoint edges disagree with J')
        net = CausalPFN2DHead(
            J=J, num_features=F,
            ninp=cfg['ninp'], nhid=cfg['nhid'], nhead=cfg['nhead'],
            nlayers=cfg['nlayers'], dropout=cfg.get('dropout', 0.0),
            n_out=cfg.get('n_out', 10), y_scaling_mode=y_scaling_mode,
            loss_type=cfg.get('loss_type', 'density'),
            hlgauss_sigma=cfg.get('hlgauss_sigma', 0.2),
            edge_lo=float(edges2d[0]), edge_hi=float(edges2d[-1]))
        # Older training checkpoints may omit the registered edge buffer and
        # model.* aliases of backbone.*. Fill only these deterministic aliases.
        sd.setdefault('edges', torch.as_tensor(edges))
        for key in net.state_dict():
            if key.startswith('model.'):
                source = 'backbone.' + key[len('model.'):]
                if key not in sd and source in sd:
                    sd[key] = sd[source]
        net.load_state_dict(sd, strict=True)
        net.to(self.device).eval()
        return SimpleNamespace(kind='joint', net=net, J=J, F=F, edges=edges2d,
                               y_scaling_mode=y_scaling_mode)

    # -- forwards -----------------------------------------------------------
    def _pooled_context(self, y_tr_raw, y_scale):
        """Pooled standardization of the SELECTED context, as cpfn2d trained.

        -> (standardized y, mean, std); torch.std is the unbiased estimate.
        A common positive affine transform changes edges and tail scales by
        the same factor, which is what maps both heads onto the harness axis.
        """
        y = self._tensor(np.asarray(y_tr_raw).reshape(-1))
        if y.shape[1] < 2 or not np.isfinite(y_scale) or y_scale <= 0:
            raise ValueError('CausalPFN needs at least two context rows and positive y_scale')
        shift = float(y.mean())
        scale = float(y.std().clamp(min=1e-6))
        return (y - shift) / scale, shift, scale

    def _native_logits(self, entry, xc, tc, yc, xq):
        """Per-arm (n_query, K) head logits; the arm goes in query column 0."""
        xt = torch.cat((tc.unsqueeze(-1), xc), dim=-1)
        y = yc.transpose(0, 1).contiguous()
        predictions = [[], []]
        for start in range(0, xq.shape[1], self.query_chunk):
            query = xq[:, start:start + self.query_chunk]
            for arm in (0, 1):
                qt = torch.cat((torch.full_like(query[..., :1], arm), query), dim=-1)
                pred = entry.net(torch.cat((xt, qt), dim=1).transpose(0, 1).contiguous(), y)
                predictions[arm].append(pred[:, 0, -entry.K:].double().cpu().numpy())
        return [np.concatenate(p) for p in predictions]

    def _joint_logits(self, entry, xc, tc, yc, xq):
        """(n_query, J^2 + 13) joint head logits."""
        chunks = []
        for start in range(0, xq.shape[1], self.query_chunk):
            logits = entry.net._forward_logits(
                xc, tc, yc, xq[:, start:start + self.query_chunk])
            chunks.append(logits[0].double().cpu().numpy())
        return np.concatenate(chunks)


class CausalPFNDensityModels(_CausalPFNRuntime):
    """Fixed pair: one CausalPFN 1D head + one training_causalpfn2d joint."""

    def __init__(self, root, checkpoint, joint_checkpoint, device, query_chunk=512):
        self.checkpoint = str(Path(checkpoint).resolve())
        self.joint_checkpoint = str(Path(joint_checkpoint).resolve())
        self._setup(root, device, query_chunk,
                    checkpoints=(self.checkpoint, self.joint_checkpoint))
        # Positional, not sniffed: this pair's roles are fixed by the caller.
        ck = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
        native = self._load_native(ck)
        del ck
        self.native, self.K, self.F, self.edges1d = (
            native.net, native.K, native.F, native.edges)
        ck = torch.load(self.joint_checkpoint, map_location='cpu', weights_only=False)
        joint = self._load_joint(ck)
        del ck
        self.joint, self.J, self.joint_F, self.edges2d = (
            joint.net, joint.J, joint.F, joint.edges)
        self.y_scaling_mode = joint.y_scaling_mode
        self._native_entry, self._joint_entry = native, joint

    @torch.inference_mode()
    def predict(self, X_tr_std, T_tr, y_tr_raw, X_te_std, *, y_shift, y_scale):
        """Return native arms and full joint densities on the harness axis.

        Use the selected context for pooled statistics, matching the joint
        checkpoint's training (torch.std, correction=1). A common positive
        affine transform changes edges and tail scales by the same factor.
        """
        yc, shift, scale = self._pooled_context(y_tr_raw, y_scale)
        tc = self._tensor(np.asarray(T_tr).reshape(-1))
        xc = self._tensor(_pad_features(X_tr_std, self.F))
        xq = self._tensor(_pad_features(X_te_std, self.F))
        jxc = self._tensor(_pad_features(X_tr_std, self.joint_F))
        jxq = self._tensor(_pad_features(X_te_std, self.joint_F))
        pred0, pred1 = self._native_logits(self._native_entry, xc, tc, yc, xq)
        joint_logits = self._joint_logits(self._joint_entry, jxc, tc, yc, jxq)
        edges1d = (self.edges1d * scale + shift - y_shift) / y_scale
        edges2d = (self.edges2d * scale + shift - y_shift) / y_scale
        arms = [[CausalPFN1D.from_pred(p, edges1d) for p in pred]
                for pred in (pred0, pred1)]
        joints = [Joint2D.from_pred(p, self.J, self.edges2d).affine(
            scale / y_scale, (shift - y_shift) / y_scale) for p in joint_logits]
        dump = dict(
            causalpfn_pred0=pred0, causalpfn_pred1=pred1,
            causalpfn_joint_logits=joint_logits, causalpfn_J=self.J,
            causalpfn_edges1d=edges1d, causalpfn_edges2d=edges2d,
            causalpfn_edges1d_native=self.edges1d, causalpfn_edges2d_native=self.edges2d,
            causalpfn_y_shift=shift, causalpfn_y_scale=scale,
            causalpfn_std_mode='pooled', causalpfn_y_std_correction=1,
            causalpfn_ckpt=self.checkpoint, causalpfn_joint_ckpt=self.joint_checkpoint,
            causalpfn_root=self.root, causalpfn_num_features=self.F,
            causalpfn_joint_num_features=self.joint_F,
            causalpfn_query_chunk=self.query_chunk)
        return arms, joints, dump


class CausalPFNModelSet(_CausalPFNRuntime):
    """Every model named in CAUSALPFN_MODELS, scored on one shared context.

    A checkpoint's head is read from the file, never from its list name:
      1d     a CausalPFN 1D head (bin_edges + model.*). Finite K-bin arm
             histograms, convolved under independence -- no invented tails.
      joint  a training_causalpfn2d CausalPFN2DHead. J x J joint with the 8
             tail regions, diagonal-integrated.

    Every model sees the same selected context and the same pooled
    standardization (context mean, torch.std correction=1), so the rows differ
    only by checkpoint. Covariates are padded or truncated to each model's own
    feature width; outputs are mapped affinely onto the harness axis
    y_scaled = (y_raw - y_shift) / y_scale.
    """

    def __init__(self, root, entries, device='cpu', query_chunk=512):
        entries = [(name, str(Path(path).expanduser().resolve()))
                   for name, path in entries]
        self._setup(root, device, query_chunk,
                    checkpoints=[path for _, path in entries])
        self.models = {}
        for name, path in entries:
            entry = self._load_checkpoint(path)
            entry.source = path
            self.models[f'causalpfn_{name}'] = entry

    @property
    def kinds(self):
        """method -> '1d' (arms convolved under independence) or 'joint'."""
        return {m: e.kind for m, e in self.models.items()}

    @property
    def sources(self):
        return {m: e.source for m, e in self.models.items()}

    def describe(self, method):
        e = self.models[method]
        size = f'K={e.K}' if e.kind == '1d' else f'J={e.J}'
        return f'{e.kind} {size} F={e.F} {e.source}'

    @torch.inference_mode()
    def predict(self, X_tr_std, T_tr, y_tr_raw, X_te_std, *, y_shift, y_scale):
        """-> ({method: ('1d', (arm0, arm1)) or ('joint', joints)}, dump).

        arm0/arm1 hold one CausalPFN1D per query and joints one Joint2D per
        query, all on the harness axis.
        """
        yc, shift, scale = self._pooled_context(y_tr_raw, y_scale)
        tc = self._tensor(np.asarray(T_tr).reshape(-1))
        n = len(X_te_std)
        # One padded copy per feature width, not per model: the checkpoints
        # here all cap at 99, and the pad is pure preprocessing either way.
        context, queries = {}, {}
        out = {}
        dump = dict(causalpfn_methods=np.asarray(list(self.models)),
                    causalpfn_kinds=np.asarray([e.kind for e in self.models.values()]),
                    causalpfn_sources=np.asarray([e.source for e in self.models.values()]),
                    causalpfn_root=self.root, causalpfn_y_shift=shift,
                    causalpfn_y_scale=scale, causalpfn_std_mode='pooled',
                    causalpfn_y_std_correction=1,
                    causalpfn_query_chunk=self.query_chunk)
        for method, e in self.models.items():
            if e.F not in context:
                context[e.F] = self._tensor(_pad_features(X_tr_std, e.F))
                queries[e.F] = self._tensor(_pad_features(X_te_std, e.F))
            xc, xq = context[e.F], queries[e.F]
            dump[f'{method}_ckpt'] = e.source
            dump[f'{method}_num_features'] = e.F
            if e.kind == '1d':
                pred0, pred1 = self._native_logits(e, xc, tc, yc, xq)
                _check_shape(method, pred0, (n, e.K))
                edges = (e.edges * scale + shift - y_shift) / y_scale
                out[method] = ('1d', [[CausalPFN1D.from_pred(p, edges) for p in pred]
                                      for pred in (pred0, pred1)])
                dump.update({f'{method}_pred0': pred0, f'{method}_pred1': pred1,
                             f'{method}_edges1d': edges,
                             f'{method}_edges1d_native': e.edges})
            else:
                logits = self._joint_logits(e, xc, tc, yc, xq)
                _check_shape(method, logits, (n, e.J ** 2 + 13))
                factor, offset = scale / y_scale, (shift - y_shift) / y_scale
                out[method] = ('joint', [Joint2D.from_pred(p, e.J, e.edges).affine(
                    factor, offset) for p in logits])
                dump.update({f'{method}_logits': logits, f'{method}_J': e.J,
                             f'{method}_edges2d_native': e.edges,
                             f'{method}_edges2d': e.edges * factor + offset})
        return out, dump
