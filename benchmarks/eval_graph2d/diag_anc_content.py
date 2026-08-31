"""Diagnostic: probe what our graph2d checkpoint does with different anc
contents on a single ACIC realization.

For each of five adjacency-matrix variants we
  (A) print the attn_mask that reaches layer 0 feature-attention (values,
      distribution over {-1, 0, +1}, dominant positions),
  (B) run the full forward and report per-query CATE + realization PEHE,
  (C) capture and summarise the actual attention weights at each layer
      (monkey-patched MultiheadAttention.forward with need_weights=True).

Variants:
  none        — build_anc_none(F, n_real)                (baseline all-zeros)
  full        — build_anc_full(F, n_real)                (all forward edges)
  only_TY     — only A[T, Y] = 1                          (isolates T→Y contribution)
  only_XT     — only A[X_i, T] = 1                        (isolates X→T contribution)
  only_XY     — only A[X_i, Y] = 1                        (isolates X→Y contribution)

Padded slots keep the -1 mask in every variant so the shape+dtype+mask
convention matches what our normal eval sends.

Env:
  CKPT, UWYK, CAUSALPFN, OUT (optional)
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch
import torch.nn as nn


CKPT      = os.environ['CKPT']
UWYK      = os.environ['UWYK']
CAUSALPFN = os.environ['CAUSALPFN']
OUT       = os.environ.get('OUT', './results_diag_anc_content')
DATASET   = os.environ.get('DATASET', 'ACIC')
REAL      = int(os.environ.get('REAL', 0))

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, UWYK); sys.path.insert(0, UWYK + '/src')
sys.path.insert(0, CAUSALPFN); sys.path.insert(0, CAUSALPFN + '/src')

from benchmarks import ACIC2016Dataset, IHDPDataset  # noqa: E402
from training_graph2d.model_graph_2d import GraphConditioned2DHead  # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Adjacency builders ──────────────────────────────────────────────────
def _pad_slots(A, F, n_real):
    feat_off = 2
    for i in range(n_real, F):
        A[feat_off + i, :] = -1.0
        A[:, feat_off + i] = -1.0
        A[feat_off + i, feat_off + i] = -1.0
    return A


def anc_none(F, n_real):
    return _pad_slots(np.zeros((F + 2, F + 2), dtype=np.float32), F, n_real)


def anc_full(F, n_real):
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    A[0, 1] = 1.0
    for i in range(n_real):
        A[2 + i, 0] = 1.0
        A[2 + i, 1] = 1.0
    return _pad_slots(A, F, n_real)


def anc_only_TY(F, n_real):
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    A[0, 1] = 1.0
    return _pad_slots(A, F, n_real)


def anc_only_XT(F, n_real):
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    for i in range(n_real):
        A[2 + i, 0] = 1.0
    return _pad_slots(A, F, n_real)


def anc_only_XY(F, n_real):
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    for i in range(n_real):
        A[2 + i, 1] = 1.0
    return _pad_slots(A, F, n_real)


# ── Model load (mirrors eval_graph2d_realcause.py::load_model) ──────────
def load_model():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    cfg = ck['config']
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

    model = GraphConditioned2DHead(
        num_features=cfg['num_features'],
        d_model=cfg['d_model'],
        depth=cfg['depth'],
        heads_feat=cfg['heads'],
        heads_samp=cfg['heads'],
        dropout=0.0,
        hidden_mult=cfg['hidden_mult'],
        normalize_features=True,
        J=cfg['J'],
        n_sample_attention_sink_rows=_sink_count('sink_rows'),
        n_feature_attention_sink_cols=_sink_count('sink_cols'),
    ).to(DEVICE)

    ref = model.state_dict()
    kept = {k: v for k, v in sd.items() if k in ref and ref[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(kept, strict=False)
    print(f'[load_model] missing={len(missing)} unexpected={len(unexpected)}  '
          f'loaded={len(kept)}/{len(ref)}  step={ck.get("step")}', flush=True)
    model.eval()
    return model, cfg


# ── Attention capture (Option A + C) ────────────────────────────────────
# We monkey-patch each nn.MultiheadAttention.forward that lives on a
# `feat_attn` attribute so we can force need_weights=True and record the
# attn_mask that reaches it.
def register_capture(model):
    captured: dict = {}
    handles = []

    def make_wrapped(orig, name):
        def wrapped(query, key, value, attn_mask=None, **kw):
            kw['need_weights'] = True
            kw['average_attn_weights'] = False
            out, w = orig(query, key, value, attn_mask=attn_mask, **kw)
            entry = captured.setdefault(name, {})
            # Overwrite each forward with the last call (we do one forward
            # per anc mode, one at a time — no accumulation).
            entry['mask'] = attn_mask.detach().cpu() if attn_mask is not None else None
            entry['weights'] = w.detach().cpu() if w is not None else None
            return out, None
        return wrapped

    for full_name, module in model.named_modules():
        if not full_name.endswith('feat_attn'):
            continue
        if not isinstance(module, nn.MultiheadAttention):
            continue
        # Depth index from name like 'backbone.blocks.0.feat_attn'.
        idx = full_name.split('.')
        depth_idx = next((int(x) for x in idx if x.isdigit()), -1)
        key = f'layer_{depth_idx}'
        module.forward = make_wrapped(module.forward, key)
        handles.append((full_name, key))
    print(f'[capture] wrapped {len(handles)} feature-attention modules: '
          f'{[k for _, k in handles]}', flush=True)
    return captured


# ── Data prep (mirrors eval_graph2d_realcause.py) ───────────────────────
def _pad_features(X, F):
    if X.shape[1] == F: return X
    if X.shape[1] > F:  return X[:, :F]
    return np.hstack([X, np.zeros((X.shape[0], F - X.shape[1]), dtype=X.dtype)])


def _standardize_train_test(X_train, X_test, eps=1e-8):
    mu = X_train.mean(0, keepdims=True); sd = X_train.std(0, keepdims=True)
    sd = np.where(sd < eps, eps, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd


def _scale_y(y):
    ymin = float(y.min()); ymax = float(y.max())
    yr = max(ymax - ymin, 1e-9)
    return (2.0 * (y - ymin) / yr - 1.0).astype(np.float32), ymin, yr


# ── Forward + CATE (matches our normal eval) ────────────────────────────
@torch.no_grad()
def cate_and_capture(model, X_tr, T_tr, Y_scaled, X_te, adj, J, F):
    X_obs = torch.from_numpy(X_tr).unsqueeze(0).to(DEVICE)
    T_obs = torch.from_numpy(T_tr).reshape(1, -1, 1).to(DEVICE)
    Y_obs = torch.from_numpy(Y_scaled).unsqueeze(0).to(DEVICE)
    X_intv = torch.from_numpy(X_te).unsqueeze(0).to(DEVICE)
    adj_t  = torch.from_numpy(adj).unsqueeze(0).to(DEVICE)

    out = model(X_obs, T_obs, Y_obs, X_intv, adj_t)
    logits = out['predictions'] if isinstance(out, dict) else out
    interior = logits[..., : J * J]
    p = torch.softmax(interior, dim=-1).reshape(1, -1, J, J)

    centres = torch.linspace(-1.0 + 1.0 / J, 1.0 - 1.0 / J, J, device=logits.device)
    p_y0 = p.sum(dim=-1); p_y1 = p.sum(dim=-2)
    e_y0 = (p_y0 * centres.view(1, 1, J)).sum(dim=-1)
    e_y1 = (p_y1 * centres.view(1, 1, J)).sum(dim=-1)
    return (e_y1 - e_y0).squeeze(0).cpu().numpy()


def summarise_mask(name, mask):
    if mask is None:
        print(f'  {name}: mask=None'); return
    m = mask.float().cpu().numpy()
    # If mask is float (soft-bias mode), stats are over its raw values.
    # If bool (hard mask), .float() converts to 0/1.
    flat = m.reshape(-1)
    unique, counts = np.unique(np.round(flat, 4), return_counts=True)
    dist = {float(u): int(c) for u, c in zip(unique, counts)}
    print(f'  {name}: shape={tuple(mask.shape)} dtype={mask.dtype}  '
          f'min={flat.min():.4f} max={flat.max():.4f}  values={dist}')


def summarise_weights(name, w, F):
    if w is None:
        print(f'  {name}: weights=None'); return
    # w shape: (B*S, num_heads, tokens, tokens)
    # tokens = n_sink_cols + F+2 (features + T + Y), no sinks here → tokens = F+2
    W = w.float()
    # Average across "batch*samples" and heads to get a token-token heatmap.
    W_avg = W.mean(dim=(0, 1))                     # (tokens, tokens)
    T_idx = -2                                      # after reorder to [X..., T, Y]
    Y_idx = -1
    L = F                                           # feature slots 0..L-1
    # How much does Y attend to X, to T, to itself?
    y_row = W_avg[Y_idx, :]                         # (tokens,)
    print(f'  {name}: shape={tuple(w.shape)}  '
          f'Y→X_mean={y_row[:L].mean().item():.4f}  '
          f'Y→T={y_row[T_idx].item():.4f}  '
          f'Y→Y={y_row[Y_idx].item():.4f}  '
          f'T→X_mean={W_avg[T_idx, :L].mean().item():.4f}  '
          f'T→Y={W_avg[T_idx, Y_idx].item():.4f}')


# ── Main ────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)
    print(f'[bootstrap] ckpt={CKPT}  dataset={DATASET}  realization={REAL}', flush=True)
    model, cfg = load_model()
    captured = register_capture(model)

    J = cfg['J']; F = cfg['num_features']
    print(f'[bootstrap] J={J} F={F}', flush=True)

    ds = ACIC2016Dataset() if DATASET == 'ACIC' else IHDPDataset()
    cate_ds = ds[REAL][0]
    X_tr = np.asarray(cate_ds.X_train, dtype=np.float32)
    T_tr = np.asarray(cate_ds.t_train, dtype=np.float32).reshape(-1)
    y_tr = np.asarray(cate_ds.y_train, dtype=np.float32).reshape(-1)
    X_te = np.asarray(cate_ds.X_test, dtype=np.float32)
    true_cate = np.asarray(cate_ds.true_cate, dtype=np.float32)
    true_ate = float(true_cate.mean())

    n_real = min(X_tr.shape[1], F)
    X_tr_std, X_te_std = _standardize_train_test(X_tr, X_te)
    X_tr_p = _pad_features(X_tr_std, F); X_te_p = _pad_features(X_te_std, F)
    y_scaled, ymin, yrange = _scale_y(y_tr)
    Y_obs = y_scaled.reshape(-1, 1)

    print(f'[data] n_context={X_tr.shape[0]}  n_query={X_te.shape[0]}  n_real={n_real}',
          flush=True)

    variants = [
        ('none',    anc_none),
        ('full',    anc_full),
        ('only_TY', anc_only_TY),
        ('only_XT', anc_only_XT),
        ('only_XY', anc_only_XY),
    ]

    results = {}
    for name, builder in variants:
        adj = builder(F, n_real)
        # ── (A) adjacency itself ─────────────
        real_sub = adj[:2 + n_real, :2 + n_real]
        vals, cnts = np.unique(np.round(real_sub, 4), return_counts=True)
        print(f'\n── mode={name} ─────────────────────────────────────', flush=True)
        print(f'  adj real-submatrix ({2+n_real}x{2+n_real}) values: '
              f'{dict(zip(vals.tolist(), cnts.tolist()))}', flush=True)
        print(f'  adj shape={adj.shape}  |A|_1={np.abs(adj).sum():.0f}  '
              f'#(+1)={int((adj == 1).sum())}  #(-1)={int((adj == -1).sum())}  '
              f'#(0)={int((adj == 0).sum())}', flush=True)

        # ── (B) forward + CATE ───────────────
        cate_scaled = cate_and_capture(model, X_tr_p, T_tr, Y_obs, X_te_p, adj, J, F)
        cate = cate_scaled * yrange / 2.0
        pehe = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
        ate_hat = float(cate.mean())
        err = abs(ate_hat - true_ate) / max(abs(true_ate), 1e-9)
        print(f'  CATE: pehe={pehe:.4f}  err_ate={err:.4f}  ate_hat={ate_hat:+.4f}',
              flush=True)

        # ── (A + C) attention capture ─────────
        print(f'  [layer 0 attn_mask]')
        entry = captured.get('layer_0', {})
        summarise_mask('    mask', entry.get('mask'))
        print(f'  [layer 0 attn_weights]')
        summarise_weights('    weights', entry.get('weights'), F)

        results[name] = {'pehe': pehe, 'err': err, 'ate': ate_hat,
                         'cate': cate}

    np.savez(os.path.join(OUT, f'diag_{DATASET}_r{REAL:03d}.npz'),
             **{k: results[k]['cate'] for k in results},
             true_cate=true_cate)

    print('\n══ summary ══', flush=True)
    print(f'  true_ate = {true_ate:+.4f}')
    for name in ('none', 'full', 'only_TY', 'only_XT', 'only_XY'):
        r = results[name]
        print(f'  {name:10s}  pehe={r["pehe"]:.4f}  err={r["err"]:.4f}  ate_hat={r["ate"]:+.4f}',
              flush=True)


if __name__ == '__main__':
    main()
