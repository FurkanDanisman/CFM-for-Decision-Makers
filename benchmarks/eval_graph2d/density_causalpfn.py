"""CausalPFN checkpoint inference for the shared RealCause density runner.

Both heads consume pooled-standardized factual outcomes. Their native output
axes are mapped back to the harness axis before scoring; Joint2D then retains
all region weights and tail scales, unlike the old interior-only density dump.
"""
from __future__ import annotations

import sys
from pathlib import Path

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


class CausalPFNDensityModels:
    def __init__(self, root, checkpoint, joint_checkpoint, device, query_chunk=512):
        self.root = str(Path(root).resolve())
        self.checkpoint = str(Path(checkpoint).resolve())
        self.joint_checkpoint = str(Path(joint_checkpoint).resolve())
        self.device = torch.device(device)
        self.query_chunk = int(query_chunk)
        if self.query_chunk <= 0:
            raise ValueError('CAUSALPFN_QUERY_CHUNK must be positive')
        repo = Path(__file__).resolve().parents[2]
        for path in (repo, Path(self.root), Path(self.root) / 'src'):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from causalpfn.models.model import TabDPTLongContextModel
        from training_causalpfn2d.model_causalpfn_2d import CausalPFN2DHead

        ck = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
        mc = ck.get('model_config', {})
        cfg = mc.get('model', mc)
        sd = {k[len('model.'):]: v for k, v in _state_dict(ck).items()
              if k.startswith('model.')}
        self.F = int(sd['encoder.weight'].shape[1]) - 1
        n_out = int(cfg.get('n_out', cfg.get('max_num_classes', 10)))
        self.K = int(sd['head.2.weight'].shape[0]) - n_out
        ninp = int(sd['encoder.weight'].shape[0])
        self.native = TabDPTLongContextModel(
            dropout=cfg.get('dropout', 0.0), n_out=n_out,
            nhead=cfg.get('nhead', 6),
            nhid=cfg.get('nhid', ninp * cfg.get('nhid_factor', 2)),
            ninp=ninp,
            nlayers=cfg.get('nlayers', 20), num_features=self.F + 1,
            nbins=self.K)
        self.native.load_state_dict(sd, strict=True)
        self.native.to(self.device).eval()
        # InContextModel stores the HL-Gauss support outside model.*.
        saved_edges = _state_dict(ck).get('bin_edges')
        self.edges1d = (saved_edges.cpu().double().numpy().reshape(-1)
                        if saved_edges is not None else
                        np.linspace(-10.0, 10.0, self.K + 1))
        CausalPFN1D.from_pred(np.zeros(self.K), self.edges1d)
        del ck, sd

        ck = torch.load(self.joint_checkpoint, map_location='cpu', weights_only=False)
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
        self.y_scaling_mode = cfg.get('y_scaling_mode', 'pooled_std')
        if self.y_scaling_mode != 'pooled_std':
            raise ValueError('CausalPFN density evaluation requires a pooled_std '
                             f'joint checkpoint; got {self.y_scaling_mode!r}')
        self.J = int(cfg['J'])
        self.joint_F = int(cfg['num_features'])
        self.edges2d = np.asarray(torch.as_tensor(edges).cpu(), dtype=np.float64)
        if self.edges2d.shape != (self.J + 1,):
            raise ValueError('CausalPFN joint checkpoint edges disagree with J')
        self.joint = CausalPFN2DHead(
            J=self.J, num_features=self.joint_F,
            ninp=cfg['ninp'], nhid=cfg['nhid'], nhead=cfg['nhead'],
            nlayers=cfg['nlayers'], dropout=cfg.get('dropout', 0.0),
            n_out=cfg.get('n_out', 10), y_scaling_mode=self.y_scaling_mode,
            loss_type=cfg.get('loss_type', 'density'),
            hlgauss_sigma=cfg.get('hlgauss_sigma', 0.2),
            edge_lo=float(self.edges2d[0]), edge_hi=float(self.edges2d[-1]))
        # Older training checkpoints may omit the registered edge buffer and
        # model.* aliases of backbone.*. Fill only these deterministic aliases.
        sd.setdefault('edges', torch.as_tensor(edges))
        for key in self.joint.state_dict():
            if key.startswith('model.'):
                source = 'backbone.' + key[len('model.'):]
                if key not in sd and source in sd:
                    sd[key] = sd[source]
        self.joint.load_state_dict(sd, strict=True)
        self.joint.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, X_tr_std, T_tr, y_tr_raw, X_te_std, *, y_shift, y_scale):
        """Return native arms and full joint densities on the harness axis.

        Use the selected context for pooled statistics, matching the joint
        checkpoint's training (torch.std, correction=1). A common positive
        affine transform changes edges and tail scales by the same factor.
        """
        def tensor(x):
            return torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

        y = tensor(np.asarray(y_tr_raw).reshape(-1))
        if y.shape[1] < 2 or not np.isfinite(y_scale) or y_scale <= 0:
            raise ValueError('CausalPFN needs at least two context rows and positive y_scale')
        shift = float(y.mean())
        scale = float(y.std().clamp(min=1e-6))
        yc = (y - shift) / scale
        tc = tensor(np.asarray(T_tr).reshape(-1))
        xc = tensor(_pad_features(X_tr_std, self.F))
        xq = tensor(_pad_features(X_te_std, self.F))
        jxc = tensor(_pad_features(X_tr_std, self.joint_F))
        jxq = tensor(_pad_features(X_te_std, self.joint_F))
        predictions = [[], []]
        joint_predictions = []
        for start in range(0, xq.shape[1], self.query_chunk):
            stop = start + self.query_chunk
            query = xq[:, start:stop]
            for arm in (0, 1):
                xt = torch.cat((tc.unsqueeze(-1), xc), dim=-1)
                qt = torch.cat((torch.full_like(query[..., :1], arm), query), dim=-1)
                pred = self.native(torch.cat((xt, qt), dim=1).transpose(0, 1).contiguous(),
                                   yc.transpose(0, 1).contiguous())
                predictions[arm].append(pred[:, 0, -self.K:].double().cpu().numpy())
            logits = self.joint._forward_logits(jxc, tc, yc, jxq[:, start:stop])
            joint_predictions.append(logits[0].double().cpu().numpy())
        pred0, pred1 = [np.concatenate(p) for p in predictions]
        joint_logits = np.concatenate(joint_predictions)
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
