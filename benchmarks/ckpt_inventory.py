#!/usr/bin/env python3
"""Read grid resolution, head output dim and deduplicated parameter count
straight out of checkpoints, for tab:train_cost.

    python R-PFN/benchmarks/ckpt_inventory.py CKPT.pt [CKPT.pt ...] [--latex]

Parameter counts DEDUPLICATE shared storage: tied embeddings and weight-shared
heads otherwise get counted twice. Two tensors share parameters when they share
an untyped storage, so we key on (storage_ptr, nbytes) rather than on shape.

The head row is whichever final projection the model actually has; the three
families name it differently (head_2d.*, regression_head.*, criterion/decoder),
so we search a candidate list and report the layer name we used, never a guess.
"""
import argparse, os, re, sys

def _load(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)

def _state_dict(ck):
    for k in ('state_dict', 'model_state_dict', 'model', 'sd'):
        if isinstance(ck, dict) and k in ck and isinstance(ck[k], dict):
            inner = ck[k]
            if any(hasattr(v, 'shape') for v in inner.values()):
                return inner
    if isinstance(ck, dict) and any(hasattr(v, 'shape') for v in ck.values()):
        return ck
    raise SystemExit(f'no state_dict found; top-level keys: {list(ck)[:12]}')

def _dedup_params(sd):
    """Sum numel over DISTINCT storages, so shared/tied weights count once."""
    seen, total, shared = set(), 0, 0
    for name, t in sd.items():
        if not hasattr(t, 'numel'):
            continue
        try:
            key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        except Exception:
            key = (id(t), t.numel())
        if key in seen:
            shared += t.numel(); continue
        seen.add(key); total += t.numel()
    return total, shared

HEAD_PAT = re.compile(
    r'(head_2d\.\d+|\bhead\.\d+|regression_head|decoder_dict\.\w+\.\d+|'
    r'criterion\.|final_layer)', re.I)
# Attention output projections are named out_proj and are NOT heads; including
# them made every transformer layer a candidate.
ATTN_PAT = re.compile(r'out_proj|self_attn|feat_attn|samp_attn', re.I)

def _head(sd):
    """The head is TERMINAL, so take the last head-ish 2-D weight in dict order.

    Picking the widest instead is wrong: Do-PFN's head is head_2d.0 (768,192)
    -> head_2d.2 (113,768), so the widest matrix is the hidden layer and the
    real output dim (113 = K^2+13) is the narrower one.
    """
    cands = [(k, tuple(v.shape)) for k, v in sd.items()
             if k.endswith('.weight') and hasattr(v, 'shape') and v.ndim == 2
             and not ATTN_PAT.search(k)]
    named = [(k, sh) for k, sh in cands if HEAD_PAT.search(k)]
    pool = named or cands
    return pool[-1] if pool else (None, None)

def _grid(out_dim):
    """out_dim is J+tail (1D) or K^2+tail (2D). Report both readings."""
    import math
    reads = []
    for tail in (0, 4, 10, 13, 23):
        j = out_dim - tail
        if j > 0:
            reads.append(f'J={j}(+{tail})')
        r = out_dim - tail
        if r > 0:
            k = int(round(math.isqrt(r)))
            if k * k == r:
                reads.append(f'K={k}(K^2+{tail})')
    return ', '.join(reads)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--latex', action='store_true')
    a = ap.parse_args()
    rows = []
    for p in a.ckpts:
        if not os.path.isfile(p):
            print(f'{os.path.basename(p):42s}  MISSING', file=sys.stderr); continue
        try:
            ck = _load(p)
        except Exception as e:
            print(f'{os.path.basename(p):42s}  LOAD FAILED: {e}', file=sys.stderr); continue
        sd = _state_dict(ck)
        tot, shared = _dedup_params(sd)
        hk, hs = _head(sd)
        cfg = ck.get('config') if isinstance(ck, dict) else None
        cfgbits = ''
        if isinstance(cfg, dict):
            keep = {k: cfg[k] for k in
                    ('J', 'K', 'N_OUT', 'num_bars', 'n_out', 'nbins', 'steps',
                     'max_steps', 'lr', 'batch_size', 'seq_len')
                    if k in cfg}
            cfgbits = ' '.join(f'{k}={v}' for k, v in keep.items())
        step = ck.get('step') or ck.get('actual_step') or ck.get('global_step') \
               if isinstance(ck, dict) else None
        rows.append(dict(name=os.path.basename(p), total=tot, shared=shared,
                         head=hk, shape=hs, grid=_grid(hs[0]) if hs else '',
                         cfg=cfgbits, step=step))
        print(f'{os.path.basename(p)}')
        print(f'   total params (dedup) : {tot:,}   (shared, not counted: {shared:,})')
        print(f'   head layer           : {hk}  shape={hs}')
        print(f'   head output dim      : {hs[0] if hs else "?"}   -> {_grid(hs[0]) if hs else ""}')
        print(f'   step in ckpt         : {step}')
        print(f'   config keys          : {cfgbits or "<no config in ckpt>"}')
        print()
    if a.latex and rows:
        print('% name & head dim & total params')
        for r in rows:
            print(f'{r["name"]:44s} & ${r["shape"][0] if r["shape"] else 0}$ '
                  f'& ${r["total"]:,}$ \\\\'.replace(',', '{,}'))

if __name__ == '__main__':
    main()
