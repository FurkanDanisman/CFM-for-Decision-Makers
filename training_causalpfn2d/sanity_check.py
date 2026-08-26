"""One-batch smoke test for the CausalPFN 2D-head training pipeline.

Uses CausalPFN's own BackdoorDGPMetaDataset (not our PairedInterventionalDataset).
Runs a tiny model config at J=10 for a few-second smoke.
"""
from __future__ import annotations
import os
import sys
import time
import torch

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_SRC)

from training_causalpfn2d.model_causalpfn_2d import CausalPFN2DHead, _wire_causalpfn_paths
_wire_causalpfn_paths()

from losses.BarDistribution2D import fit_edges_2d, total_params
from omegaconf import OmegaConf
from hydra.utils import instantiate


def main():
    torch.manual_seed(0)
    J = int(os.environ.get('SANITY_J', 10))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'── sanity check (CausalPFN 2D-head) ── device={device}  J={J}')

    print('[1/4] one batch from BackdoorDGPMetaDataset (via CausalPFN hydra config)…')
    t0 = time.time()
    yaml_path = os.path.join(
        os.environ.get('CAUSALPFN_ROOT',
                        '/scratch/furkanbd/rpfn_bench_kit/external/causalpfn'),
        'conf', 'meta_dataset', 'synthetic_backdoor.yaml',
    )
    cfg = OmegaConf.load(yaml_path)
    # Tiny overrides for a fast smoke test
    cfg.n_samples = 64
    cfg.max_n_covariates = 8
    cfg.post_padding_n_cols = 50
    meta = instantiate(cfg)
    loader = torch.utils.data.DataLoader(meta, batch_size=2, num_workers=0)
    it = iter(loader)
    b = next(it)
    print(f'      done in {time.time()-t0:.1f}s')
    for k in ('X','t','y','E_y0','E_y1'):
        v = b[k]
        print(f'      {k:>6}: shape={tuple(v.shape)}  dtype={v.dtype}  range=[{v.float().min():.2f}, {v.float().max():.2f}]')

    print('[2/4] fitting edges from standardised y_context…')
    warmup = []
    for _ in range(2):
        bb = next(it)
        for i in range(2):
            y = bb['y'][i].float()
            y_std = (y - y.mean()) / (y.std() + 1e-6)
            warmup.append({'Y_obs': y_std})
    edges = fit_edges_2d(warmup, J).to(device)
    print(f'      edges: {tuple(edges.shape)}   range=[{edges.min():.2f}, {edges.max():.2f}]')

    print('[3/4] build model + forward + loss…')
    model = CausalPFN2DHead(J=J, num_features=50,
                             ninp=32, nhid=64, nhead=4, nlayers=1,
                             dropout=0.0, n_out=2).to(device)
    print(f'      params: {sum(p.numel() for p in model.parameters()):,}')

    X = b['X'].to(device)
    t = b['t'].to(device)
    y = b['y'].to(device)
    E_y0 = b['E_y0'].to(device); E_y1 = b['E_y1'].to(device)
    split = X.shape[1] // 2
    losses = model(
        X_context=X[:, :split], t_context=t[:, :split], y_context=y[:, :split],
        X_query=X[:, split:], E_y0_query=E_y0[:, split:], E_y1_query=E_y1[:, split:],
        edges=edges,
    )
    print(f'      per-task losses: {losses.tolist()}')
    assert torch.isfinite(losses).all()

    print('[4/4] backward + grad flow check…')
    losses.mean().backward()
    ng = model.null_t_intv.grad
    print(f'      null_t_intv grad: {ng.item() if ng is not None else None}')
    body_l1 = sum(p.grad.abs().sum().item()
                   for name, p in model.backbone.named_parameters()
                   if p.grad is not None)
    print(f'      backbone total-grad L1: {body_l1:.4f}')
    assert body_l1 > 0
    assert ng is not None
    print('\nAll sanity checks passed.')


if __name__ == '__main__':
    main()
