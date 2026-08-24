"""Sanity check: does the UWYK 'Predictive' checkpoint share the graph-conditioning
architecture with the 'FullCond' checkpoint, or is it a stripped-down predictor?

Usage:
    python benchmarks/l2_ihdp/check_uwyk_predictive.py
"""
import os
import torch

D = os.environ.get('DEPLOY_ROOT', '/scratch/furkanbd/rpfn_bench_kit')
D = f'{D}/external/uwyk/experiments/checkpoints'

pred = torch.load(f'{D}/no_graph_conditioning/unconditional/best_model.pt',
                   map_location='cpu', weights_only=False)
full = torch.load(f'{D}/full_conditioned_model/final_earlytest_full_conditioning_16773252.0/best_model.pt',
                   map_location='cpu', weights_only=False)


def _tensors(x):
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if hasattr(vv, 'numel'):
                        yield f'{k}.{kk}', vv
            elif hasattr(v, 'numel'):
                yield k, v


psd = dict(_tensors(pred))
fsd = dict(_tensors(full))

print(f'Predictive: {len(psd)} tensors, {sum(v.numel() for v in psd.values()):,} params')
print(f'FullCond:   {len(fsd)} tensors, {sum(v.numel() for v in fsd.values()):,} params')

print('top-level keys Predictive:',
      list(pred.keys())[:8] if isinstance(pred, dict) else type(pred))
print('top-level keys FullCond:  ',
      list(full.keys())[:8] if isinstance(full, dict) else type(full))


def _graph(sd):
    return sorted({k for k in sd if any(t in k.lower()
                                          for t in ('gcn', 'graph', 'adjacency', 'adj'))})[:8]


print('graph keys FullCond:  ', _graph(fsd))
print('graph keys Predictive:', _graph(psd))

# Extra: overlap analysis
common = set(psd) & set(fsd)
only_full = set(fsd) - set(psd)
only_pred = set(psd) - set(fsd)
print(f'\nShared param keys: {len(common)}')
print(f'Only in FullCond:  {len(only_full)}   (examples: {sorted(only_full)[:5]})')
print(f'Only in Predictive: {len(only_pred)}   (examples: {sorted(only_pred)[:5]})')
