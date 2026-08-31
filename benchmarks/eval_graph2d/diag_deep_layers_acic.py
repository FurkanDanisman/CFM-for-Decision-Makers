"""Deep-layer attention capture on ACIC r=0.

Runs the graph2d 50k ckpt through TWO adjacency variants:
  - noanc:  all-zero real submatrix (baseline)
  - anc:    sparse +1 at (X_i, T), (X_i, Y), (T, Y)  — matches
            reproduce-branch build_adjacency_matrix (NO propagate,
            NO indep-features)

For EACH of the 6 feature-attention layers, captures + reports:
  - attn_mask value distribution
  - full attention-weight matrix statistics (averaged across
    batch*samples*heads)
  - Specifically: how much Y and T tokens attend to X_i, T, Y, and how
    that distribution SHIFTS between anc and noanc modes at each layer.

Point: on ACIC our model doesn't gain from anc; the shift MUST corrupt
something in the deep layers. Layer-by-layer capture pinpoints WHERE
the pattern goes wrong (early? late? some specific block?).

Env: CKPT, UWYK, CAUSALPFN, DATASET (default ACIC), REAL (default 0)
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
import torch.nn as nn


CKPT      = os.environ['CKPT']
UWYK      = os.environ['UWYK']
CAUSALPFN = os.environ['CAUSALPFN']
DATASET   = os.environ.get('DATASET', 'ACIC')
REAL      = int(os.environ.get('REAL', 0))
OUT       = os.environ.get('OUT', './results_diag_deep_layers')

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, UWYK); sys.path.insert(0, UWYK + '/src')
sys.path.insert(0, CAUSALPFN); sys.path.insert(0, CAUSALPFN + '/src')

from benchmarks import ACIC2016Dataset, IHDPDataset  # noqa: E402
from training_graph2d.model_graph_2d import GraphConditioned2DHead  # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Sparse anc (matches reproduce-branch build_adjacency_matrix) ────────
def _pad_slots(A, F, n_real):
    feat_off = 2
    for i in range(n_real, F):
        A[feat_off + i, :] = -1.0
        A[:, feat_off + i] = -1.0
        A[feat_off + i, feat_off + i] = -1.0
    return A


def anc_none(F, n_real):
    return _pad_slots(np.zeros((F + 2, F + 2), dtype=np.float32), F, n_real)


def anc_sparse(F, n_real):
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    A[0, 1] = 1.0
    for i in range(n_real):
        A[2 + i, 0] = 1.0
        A[2 + i, 1] = 1.0
    return _pad_slots(A, F, n_real)


# ── Model load ──────────────────────────────────────────────────────────
def load_model():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    cfg = ck['config']
    sd = ck['model_state_dict']
    if any('_orig_mod.' in k for k in sd):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}

    def _sink_count(prefix):
        for suffix in ('_x', '_y'):
            k = prefix + suffix
            if k not in sd or sd[k].dim() < 2: continue
            t = sd[k]
            return int(t.shape[1] if t.shape[0] == 1 else t.shape[0])
        return 0

    model = GraphConditioned2DHead(
        num_features=cfg['num_features'], d_model=cfg['d_model'],
        depth=cfg['depth'], heads_feat=cfg['heads'], heads_samp=cfg['heads'],
        dropout=0.0, hidden_mult=cfg['hidden_mult'], normalize_features=True,
        J=cfg['J'],
        n_sample_attention_sink_rows=_sink_count('sink_rows'),
        n_feature_attention_sink_cols=_sink_count('sink_cols'),
    ).to(DEVICE)
    ref = model.state_dict()
    kept = {k: v for k, v in sd.items() if k in ref and ref[k].shape == v.shape}
    model.load_state_dict(kept, strict=False)
    model.eval()
    return model, cfg


# ── Attention capture across ALL layers ─────────────────────────────────
def register_capture_all(model):
    captured: dict = {}

    def make_wrapped(orig, key):
        def wrapped(query, key_arg, value, attn_mask=None, **kw):
            kw['need_weights'] = True
            kw['average_attn_weights'] = False
            out, w = orig(query, key_arg, value, attn_mask=attn_mask, **kw)
            entry = captured.setdefault(key, {})
            entry['mask']    = attn_mask.detach().cpu() if attn_mask is not None else None
            entry['weights'] = w.detach().cpu() if w is not None else None
            return out, None
        return wrapped

    for full_name, module in model.named_modules():
        if not full_name.endswith('feat_attn'): continue
        if not isinstance(module, nn.MultiheadAttention): continue
        idx = full_name.split('.')
        depth_idx = next((int(x) for x in idx if x.isdigit()), -1)
        key = f'layer_{depth_idx}'
        module.forward = make_wrapped(module.forward, key)
    print(f'[capture] wrapped {sum(1 for k in captured or [""])} — will fill during forward')
    return captured


# ── Data prep ───────────────────────────────────────────────────────────
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


@torch.no_grad()
def forward_and_cate(model, X_tr, T_tr, Y_scaled, X_te, adj, J, F):
    X_obs = torch.from_numpy(X_tr).unsqueeze(0).to(DEVICE)
    T_obs = torch.from_numpy(T_tr).reshape(1, -1, 1).to(DEVICE)
    Y_obs = torch.from_numpy(Y_scaled).unsqueeze(0).to(DEVICE)
    X_intv = torch.from_numpy(X_te).unsqueeze(0).to(DEVICE)
    adj_t = torch.from_numpy(adj).unsqueeze(0).to(DEVICE)
    out = model(X_obs, T_obs, Y_obs, X_intv, adj_t)
    logits = out['predictions'] if isinstance(out, dict) else out
    interior = logits[..., : J * J]
    p = torch.softmax(interior, dim=-1).reshape(1, -1, J, J)
    centres = torch.linspace(-1.0 + 1.0/J, 1.0 - 1.0/J, J, device=logits.device)
    p_y0 = p.sum(dim=-1); p_y1 = p.sum(dim=-2)
    e_y0 = (p_y0 * centres.view(1, 1, J)).sum(dim=-1)
    e_y1 = (p_y1 * centres.view(1, 1, J)).sum(dim=-1)
    return (e_y1 - e_y0).squeeze(0).cpu().numpy()


def summarise_layer_weights(name, w, F, n_real):
    if w is None:
        print(f'    {name}: (no weights captured)'); return None
    # w shape: (B*S, num_heads, tokens, tokens)
    # After model reorder: tokens = [X_0, ..., X_{L-1}, T, Y]  →  T at -2, Y at -1
    W = w.float().mean(dim=(0, 1))            # (tokens, tokens)
    tokens = W.shape[0]
    T_idx = tokens - 2
    Y_idx = tokens - 1
    L = F                                     # feature slots 0..L-1
    real_L = n_real                           # real feature slots 0..n_real-1

    y_row = W[Y_idx, :]
    t_row = W[T_idx, :]
    return {
        'Y→X_real_sum':   float(y_row[:real_L].sum().item()),   # mass on real X
        'Y→X_padded_sum': float(y_row[real_L:L].sum().item()),  # mass on padded X
        'Y→T':            float(y_row[T_idx].item()),
        'Y→Y_self':       float(y_row[Y_idx].item()),
        'T→X_real_sum':   float(t_row[:real_L].sum().item()),
        'T→X_padded_sum': float(t_row[real_L:L].sum().item()),
        'T→Y':            float(t_row[Y_idx].item()),
        'T→T_self':       float(t_row[T_idx].item()),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f'[bootstrap] ckpt={CKPT}  dataset={DATASET}  r={REAL}', flush=True)
    model, cfg = load_model()
    captured = register_capture_all(model)
    J = cfg['J']; F = cfg['num_features']

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

    results = {}
    for mode_name, builder in (('none', anc_none), ('sparse', anc_sparse)):
        adj = builder(F, n_real)
        cate_scaled = forward_and_cate(model, X_tr_p, T_tr, Y_obs, X_te_p, adj, J, F)
        cate = cate_scaled * yrange / 2.0
        pehe = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
        ate_hat = float(cate.mean())
        err = abs(ate_hat - true_ate) / max(abs(true_ate), 1e-9)

        layer_summaries = {}
        for layer_key in sorted(captured.keys()):
            layer_summaries[layer_key] = summarise_layer_weights(
                layer_key, captured[layer_key].get('weights'), F, n_real,
            )
        results[mode_name] = {'pehe': pehe, 'err': err, 'ate': ate_hat,
                              'layers': layer_summaries}

    # ── Print comparison ──────────────────────────────────────
    print(f'\n══ {DATASET} r={REAL}  (n_real={n_real}, F={F}) ══')
    print(f'  true_ate={true_ate:+.4f}')
    for m in ('none', 'sparse'):
        r = results[m]
        print(f'  mode={m:6s}  pehe={r["pehe"]:.4f}  err={r["err"]:.4f}  ate_hat={r["ate"]:+.4f}')

    print(f'\n── Per-layer attention weight summaries (avg across B*S*heads) ──')
    layer_keys = sorted(set(results['none']['layers'].keys()) & set(results['sparse']['layers'].keys()))
    keys_to_show = ('Y→X_real_sum', 'Y→X_padded_sum', 'Y→T', 'Y→Y_self',
                    'T→X_real_sum', 'T→X_padded_sum', 'T→Y', 'T→T_self')
    for lk in layer_keys:
        print(f'\n  {lk}:')
        for k in keys_to_show:
            v_none = results['none']['layers'][lk][k]
            v_anc  = results['sparse']['layers'][lk][k]
            delta  = v_anc - v_none
            print(f'    {k:18s}  none={v_none:+.4f}  anc={v_anc:+.4f}  Δ={delta:+.4f}')


if __name__ == '__main__':
    main()
