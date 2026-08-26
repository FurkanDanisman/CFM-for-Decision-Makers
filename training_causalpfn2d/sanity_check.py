"""One-batch smoke test for the CausalPFN 2D-head training pipeline.

Exercises:
  1. Streaming loader emits paired (Y_do0, Y_do1) batches.
  2. Model forward produces (B, M, K**2 + 9 + 4) logits.
  3. neg_log_prob_2d returns a finite scalar.
  4. Backward populates grads on both the 2D head and the transformer body.

Runs in seconds at J=10, ninp=32, nlayers=1.
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_SRC)

from losses.BarDistribution2D import fit_edges_2d, neg_log_prob_2d, total_params
from training.data.PairedInterventionalDataset import make_streaming_loader
from training_causalpfn2d.model_causalpfn_2d import CausalPFN2DHead


def main():
    torch.manual_seed(0)
    J = int(os.environ.get('SANITY_J', 10))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'── sanity check (CausalPFN 2D-head) ──')
    print(f'device: {device}   J: {J}')

    print('[1/4] streaming a small batch…')
    t0 = time.time()
    loader = make_streaming_loader(batch_size=2, num_workers=0, seed_base=1)
    b = next(iter(loader))
    print(f'      done in {time.time()-t0:.1f}s')
    for k, v in b.items():
        print(f'      {k:>12}: {tuple(v.shape)}   dtype={v.dtype}')

    print('[2/4] fitting edges…')
    samples = [{k: v[i] for k, v in b.items()} for i in range(2)]
    edges = fit_edges_2d(samples, J).to(device)
    print(f'      edges: {tuple(edges.shape)}')

    print('[3/4] building model + forward + loss…')
    model = CausalPFN2DHead(
        J=J, num_features=50,
        ninp=32, nhid=64, nhead=4, nlayers=1, dropout=0.0, n_out=2,
    ).to(device)
    print(f'      params: {sum(p.numel() for p in model.parameters()):,}')

    X_obs  = b['X_obs'].to(device)
    T_obs  = b['T_obs'].to(device)
    Y_obs  = b['Y_obs'][..., 0].to(device)
    X_intv = b['X_intv'].to(device)
    Y_do0  = b['Y_do0'][..., 0].to(device)
    Y_do1  = b['Y_do1'][..., 0].to(device)

    logits = model(X_obs, T_obs, Y_obs, X_intv)
    B, M = X_intv.shape[0], X_intv.shape[1]
    exp = (B, M, total_params(J))
    print(f'      logits: {tuple(logits.shape)}  (expected {exp})')
    assert tuple(logits.shape) == exp

    loss = neg_log_prob_2d(logits.float(), Y_do0, Y_do1, J, edges)
    print(f'      loss: {loss.item():.4f}')
    assert torch.isfinite(loss)

    print('[4/4] backward + grad-flow check…')
    loss.backward()

    # 2D-head gets grad via the backbone's Linear at the end of `head`.
    head_last = None
    for name, m in model.backbone.head.named_modules():
        if isinstance(m, torch.nn.Linear):
            head_last = (name, m)   # keep the final Linear
    assert head_last is not None
    n, m = head_last
    g = m.weight.grad
    assert g is not None and g.abs().sum().item() > 0, \
        f'final head Linear ({n}) received no gradient'
    print(f'      head Linear "{n}" weight-grad L1: {g.abs().sum().item():.4f}')

    # Transformer body: verify at least one layer's parameters received gradient.
    body_grad_l1 = 0.0
    body_params  = 0
    for name, p in model.backbone.named_parameters():
        if name.startswith('head'):
            continue
        if p.grad is None:
            continue
        body_grad_l1 += p.grad.abs().sum().item()
        body_params  += 1
    assert body_grad_l1 > 0, 'no gradient reached the transformer body'
    print(f'      backbone body grad L1 (across {body_params} tensors): {body_grad_l1:.4f}')

    # Verify null_t_intv is trained.
    ng = model.null_t_intv.grad
    print(f'      null_t_intv grad: {ng.item() if ng is not None else None}')
    assert ng is not None, 'null_t_intv did not receive a gradient'

    print('\nAll sanity checks passed.')


if __name__ == '__main__':
    main()
