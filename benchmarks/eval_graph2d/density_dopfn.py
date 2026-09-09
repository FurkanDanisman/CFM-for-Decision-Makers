"""DoPFN inference adapters for Tier C; numerical densities live in density_common.

DoPFN and UWYK both import top-level utils/models packages. Keep DoPFN's
module cache and relative artifact paths scoped to its inference calls.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from density_common import DoPFN1D

_REPO = Path(__file__).resolve().parents[2]
_PREFIXES = ('utils', 'models', 'model', 'scripts', 'datasets', 'priors')


def _collides(name):
    return name.split('.')[0] in _PREFIXES


class DoPFNDensityModels:
    def __init__(self, root, checkpoint, device='cpu', query_chunk=20):
        self.root = str(Path(root).expanduser().resolve())
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.device = torch.device(device)
        self.query_chunk = int(query_chunk)
        if self.query_chunk <= 0:
            raise ValueError('DOPFN_QUERY_CHUNK must be positive')
        for path in (Path(self.root) / 'artifacts/dopfn_model.pkl',
                     Path(self.root) / 'artifacts/dopfn_config.pkl',
                     Path(self.root) / 'artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt',
                     Path(self.checkpoint)):
            if not path.is_file():
                raise FileNotFoundError(f'Missing DoPFN input: {path}')
        self._modules = {}
        with self.environment():
            # Reuse the point benchmark's sklearn and dataset import shims.
            sys.path.insert(0, str(_REPO / 'benchmarks/empirical_tests'))
            from dopfn_helpers import load_dopfn
            self.regressor_class = load_dopfn(SimpleNamespace(repo=str(_REPO), dopfn=self.root))
            from training_dopfn_base.dopfn_backbone_head import DoPFNBackboneWith2DHead
            ck = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
            self.J = int(ck['config']['J'])
            self.edges = _numpy(ck['edges'])
            if (self.edges.shape != (self.J + 1,) or np.any(np.diff(self.edges) <= 0)
                    or not np.allclose(np.diff(self.edges), np.diff(self.edges).mean(), rtol=1e-4)):
                raise ValueError('Joint DoPFN checkpoint must have J+1 uniform increasing edges')
            self.joint = DoPFNBackboneWith2DHead(dopfn_root=self.root, K=self.J)
            state = {k.removeprefix('_orig_mod.'): v for k, v in ck['model_state_dict'].items()}
            self.joint.load_state_dict(state, strict=True)
            self.joint.to(self.device).eval()

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

    @torch.inference_mode()
    def predict(self, X_raw, treatment, y_raw, X_test_raw,
                X_standardized, y_scaled, X_test_standardized, *, y_shift, y_scale):
        """Same selected context for both models; native API owns its preprocessing.

        The joint checkpoint consumes standardized X and harness-scaled factual
        Y, matching its existing raw evaluation path. Neither model caps features.
        """
        with self.environment():
            reg = self.regressor_class()
            reg.device = str(self.device)
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

            def tensor(a):
                return torch.as_tensor(a, dtype=torch.float32, device=self.device).unsqueeze(0)

            Xc = tensor(X_standardized)
            Tc = tensor(np.asarray(treatment).reshape(-1, 1))
            Yc = tensor(np.asarray(y_scaled).reshape(-1, 1))
            joint_logits = []
            for start in range(0, len(X_test_standardized), self.query_chunk):
                Xq = tensor(X_test_standardized[start:start + self.query_chunk])
                pred = self.joint(X_context=Xc, T_context=Tc, Y_context=Yc,
                                  X_query=Xq)['predictions']
                joint_logits.append(_numpy(pred[0]))
            joint_logits = np.concatenate(joint_logits)
            if joint_logits.shape != (len(X_test_raw), self.J ** 2 + 13):
                raise ValueError(f'Unexpected joint DoPFN logits shape: {joint_logits.shape}')

        distributions, dump = [], dict(dopfn_joint_logits=joint_logits,
                                       dopfn_J=self.J, dopfn_edges2d=self.edges,
                                       dopfn_joint_ckpt=self.checkpoint,
                                       dopfn_root=self.root,
                                       dopfn_n_features=X_raw.shape[1],
                                       dopfn_query_chunk=self.query_chunk,
                                       dopfn_tail_width_source='returned_borders')
        for arm, (logits, borders, scales) in enumerate(arms):
            distributions.append([
                DoPFN1D.from_pred(p, borders, y_shift=y_shift, y_scale=y_scale,
                                 tail_scales=scales) for p in logits])
            dump.update({f'dopfn_pred{arm}': logits,
                         f'dopfn_borders{arm}_raw': borders,
                         f'dopfn_tail_scales{arm}_raw': scales})
        return distributions, joint_logits, dump


def _numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().double().numpy()
    return np.asarray(value, dtype=np.float64)
