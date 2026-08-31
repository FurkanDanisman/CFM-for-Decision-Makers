"""Inspect a graph2d checkpoint's state_dict vs what the current model
architecture expects. Prints:
  - keys in ckpt but not in model (silently dropped at load)
  - keys in model but not in ckpt (random-init at eval)
  - shape mismatches (also silently dropped at load, unless caught)
  - bias_edge/bias_no_edge/GCN param presence on both sides

Point: if our training config didn't produce the same keys that our
current eval model expects, some parameters are running on random init
without complaint. That would break anc mode (which uses graph-derived
biases and GCN embeddings) while leaving noanc mostly intact.

Env: CKPT, UWYK, REPO
"""
from __future__ import annotations
import os, sys
import torch


CKPT = os.environ['CKPT']
UWYK = os.environ['UWYK']
REPO = os.environ.get('REPO', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

sys.path.insert(0, REPO)
sys.path.insert(0, UWYK); sys.path.insert(0, UWYK + '/src')

from training_graph2d.model_graph_2d import GraphConditioned2DHead  # noqa: E402


ck = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = ck['model_state_dict']
if any('_orig_mod.' in k for k in sd):
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}


def _sink_count(prefix):
    for suffix in ('_x', '_y'):
        k = prefix + suffix
        if k not in sd or sd[k].dim() < 2:
            continue
        t = sd[k]
        return int(t.shape[1] if t.shape[0] == 1 else t.shape[0])
    return 0


cfg = ck['config']
model = GraphConditioned2DHead(
    num_features=cfg['num_features'], d_model=cfg['d_model'],
    depth=cfg['depth'], heads_feat=cfg['heads'], heads_samp=cfg['heads'],
    dropout=0.0, hidden_mult=cfg['hidden_mult'], normalize_features=True,
    J=cfg['J'],
    n_sample_attention_sink_rows=_sink_count('sink_rows'),
    n_feature_attention_sink_cols=_sink_count('sink_cols'),
)
ref = model.state_dict()

in_ckpt_not_model = [k for k in sd if k not in ref]
in_model_not_ckpt = [k for k in ref if k not in sd]
shape_mismatch    = [k for k in sd if k in ref and ref[k].shape != sd[k].shape]

print(f'== ckpt: {len(sd)} keys  |  model: {len(ref)} keys ==')

print(f'\n[In ckpt but NOT in model — silently dropped]: {len(in_ckpt_not_model)}')
for k in in_ckpt_not_model:
    print(f'  {k}: shape={tuple(sd[k].shape)}')

print(f'\n[In model but NOT in ckpt — RANDOM INIT at eval]: {len(in_model_not_ckpt)}')
for k in in_model_not_ckpt:
    print(f'  {k}: shape={tuple(ref[k].shape)}')

print(f'\n[Shape mismatches — silently dropped]: {len(shape_mismatch)}')
for k in shape_mismatch:
    print(f'  {k}: ckpt={tuple(sd[k].shape)}  model={tuple(ref[k].shape)}')

print(f'\n-- soft attention bias params --')
found = False
for k in sd:
    if 'bias_edge' in k or 'bias_no_edge' in k:
        print(f'  [ckpt]  {k}: shape={tuple(sd[k].shape)}, values={sd[k].detach().cpu().numpy().tolist()}')
        found = True
for k in ref:
    if 'bias_edge' in k or 'bias_no_edge' in k:
        print(f'  [model] {k}: shape={tuple(ref[k].shape)}')
        found = True
if not found:
    print('  NONE found in either — soft-attention-bias mode is not wired up')

print(f'\n-- GCN / graph_encoder params --')
gcn_ckpt = [k for k in sd if 'gcn' in k.lower() or 'graph_enc' in k.lower()]
gcn_ref  = [k for k in ref if 'gcn' in k.lower() or 'graph_enc' in k.lower()]
print(f'  in ckpt : {len(gcn_ckpt)} keys')
for k in gcn_ckpt[:10]: print(f'    {k}: shape={tuple(sd[k].shape)}')
print(f'  in model: {len(gcn_ref)} keys')
for k in gcn_ref[:10]: print(f'    {k}: shape={tuple(ref[k].shape)}')

print(f'\n-- AdaLN params --')
adaln_ckpt = [k for k in sd if 'adaln' in k.lower() or 'ada_ln' in k.lower() or 'ln_feat' in k.lower()]
adaln_ref  = [k for k in ref if 'adaln' in k.lower() or 'ada_ln' in k.lower() or 'ln_feat' in k.lower()]
print(f'  in ckpt : {len(adaln_ckpt)} keys')
for k in adaln_ckpt[:10]: print(f'    {k}: shape={tuple(sd[k].shape)}')
print(f'  in model: {len(adaln_ref)} keys')
for k in adaln_ref[:10]: print(f'    {k}: shape={tuple(ref[k].shape)}')
