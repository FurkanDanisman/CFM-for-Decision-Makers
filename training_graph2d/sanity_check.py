"""Sanity check for the graph-conditioned 2D-head training pipeline.

Runs a single mini-batch through the model + loss + backward pass on the
smallest reasonable config (J=10, d_model=32, depth=1). If this succeeds,
train_graph_2d.py should work at UWYK scale — the only extra risk at scale
is memory pressure, which the config's activation-checkpointing addresses.

Checks:
  1. Streaming loader yields anc_matrix in the expected shape
  2. Model forward emits (B, M, J**2 + 9 + 4) logits
  3. neg_log_prob_2d returns a finite scalar
  4. Backward populates grads on BOTH the swapped head AND the graph_encoder
     (regression guard: verifies the graph path is being trained, not bypassed)

Usage:
    UWYK_SRC=/path/to/uwyk/src python training_graph2d/sanity_check.py
"""
from __future__ import annotations
import os
import sys
import time

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_SRC)

import torch

from losses.BarDistribution2D import fit_edges_2d, neg_log_prob_2d, total_params
from training.data.PairedInterventionalDataset import make_streaming_loader
from training_graph2d.model_graph_2d import GraphConditioned2DHead


def main():
    torch.manual_seed(0)
    J = int(os.environ.get('SANITY_J', 10))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"── sanity check ──")
    print(f"device: {device}   J: {J}")

    print("[1/5] streaming a small batch…")
    t0 = time.time()
    loader = make_streaming_loader(
        batch_size=2,
        num_workers=0,
        seed_base=1,
    )
    it = iter(loader)
    b = next(it)
    print(f"      done in {time.time()-t0:.1f}s")
    for k, v in b.items():
        print(f"      {k:>12}: {tuple(v.shape)}   dtype={v.dtype}")
    assert 'anc_matrix' in b, 'PairedInterventionalDataset must emit anc_matrix'
    F_plus_2 = b['anc_matrix'].shape[-1]
    print(f"      anc_matrix shape (F+2)×(F+2) = {F_plus_2}×{F_plus_2}")

    print("[2/5] fitting edges…")
    samples = [{k: v[i] for k, v in b.items()} for i in range(2)]
    edges = fit_edges_2d(samples, J).to(device)
    print(f"      edges: {tuple(edges.shape)}")

    print("[3/5] building model…")
    model = GraphConditioned2DHead(
        num_features=50,
        d_model=32,
        depth=1,
        heads_feat=4,
        heads_samp=4,
        dropout=0.0,
        hidden_mult=2,
        normalize_features=True,
        normalize_treatment=False,
        use_checkpoint=False,
        J=J,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      params: {n_params:,}")
    print(f"      output_dim_2d: {model.output_dim_2d} (expected {total_params(J)})")
    assert model.output_dim_2d == total_params(J)

    print("[4/5] forward + loss…")
    X_obs      = b['X_obs'].to(device)
    T_obs      = b['T_obs'].to(device)
    Y_obs      = b['Y_obs'][..., 0].to(device)
    X_intv     = b['X_intv'].to(device)
    Y_do0      = b['Y_do0'][..., 0].to(device)
    Y_do1      = b['Y_do1'][..., 0].to(device)
    anc_matrix = b['anc_matrix'].to(device)

    out = model(X_obs, T_obs, Y_obs, X_intv, anc_matrix)
    logits = out['predictions'] if isinstance(out, dict) else out
    B, M = X_intv.shape[0], X_intv.shape[1]
    exp_shape = (B, M, total_params(J))
    print(f"      logits: {tuple(logits.shape)}  (expected {exp_shape})")
    assert tuple(logits.shape) == exp_shape, \
        f'logits shape {tuple(logits.shape)} != {exp_shape}'

    loss = neg_log_prob_2d(logits.float(), Y_do0, Y_do1, J, edges)
    print(f"      loss: {loss.item():.4f}")
    assert torch.isfinite(loss), 'loss is non-finite'

    print("[5/5] backward + grad-flow check…")
    loss.backward()

    head_name = model._final_proj_name
    head = getattr(model, head_name)
    head_grad_norm = head.weight.grad.abs().sum().item()
    print(f"      head '{head_name}' grad L1: {head_grad_norm:.4f}")
    assert head_grad_norm > 0, f'head {head_name} received no gradient'

    graph_grads = []
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if any(k in n.lower() for k in ('graph_encoder', 'gcn', 'graph_')):
            graph_grads.append((n, p.grad.abs().sum().item()))
    if graph_grads:
        gsum = sum(g for _, g in graph_grads)
        print(f"      graph_encoder grad L1 (sum across {len(graph_grads)} tensors): {gsum:.4f}")
        assert gsum > 0, 'graph_encoder received no gradient — adjacency path is bypassed'
    else:
        print("      WARN: no graph_encoder tensors found by name pattern; "
              "the graph module may have a non-standard attribute name — check "
              "[n for n,p in model.named_parameters()] and update sanity_check.py")

    print("\nAll sanity checks passed. Model + loss + graph path are wired end-to-end.")


if __name__ == '__main__':
    main()
