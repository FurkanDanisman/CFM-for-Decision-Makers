"""Side-by-side per-query E[Y0] and E[Y1] from 1D CausalPFN vs our 2D cpfn2d.

Diagnostic: on the same IHDP realizations and same test queries, extract:
  1D:  E[Y | do(T=0)] and E[Y | do(T=1)] from CausalPFN's 1D BarDist head
       (pure ICL, per-arm Y standardization, un-standardised back to raw Y).
  2D:  E[Y_do0] and E[Y_do1] from marginals of our joint p_mat, un-standardised
       by pooled Y std.
  Truth: IHDPDataset provides true_cate; if it also exposes mu_0, mu_1 we
         print those too (per-query counterfactual means).

Env:
  CKPT_1D          1D CausalPFN ckpt
  CKPT_2D          our cpfn2d ckpt
  CAUSALPFN        external/causalpfn root
  REALIZATIONS     comma-separated realization indices (default '0,8,12,27,83')
  N_QUERY          per realization, cap test queries printed (default 5)
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch


CKPT_1D   = os.environ['CKPT_1D']
CKPT_2D   = os.environ['CKPT_2D']
CAUSALPFN = os.environ['CAUSALPFN']
REALIZATIONS = tuple(int(x) for x in os.environ.get('REALIZATIONS', '0,8,12,27,83').split(','))
N_QUERY      = int(os.environ.get('N_QUERY', '5'))

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, CAUSALPFN)
sys.path.insert(0, CAUSALPFN + '/src')

from benchmarks import IHDPDataset  # noqa: E402
from causalpfn.models.model import TabDPTLongContextModel  # noqa: E402
from training_causalpfn2d.model_causalpfn_2d import CausalPFN2DHead  # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VMIN, VMAX = -10.0, 10.0


def _strip_prefix(sd, prefix, drop_no_prefix=False):
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix): out[k[len(prefix):]] = v
        elif not drop_no_prefix: out[k] = v
    return out


def _pad(X, F):
    if X.shape[1] == F: return X
    if X.shape[1] > F:  return X[:, :F]
    return np.hstack([X, np.zeros((X.shape[0], F - X.shape[1]), dtype=X.dtype)])


def _std_X(Xtr, Xte, eps=1e-8):
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + eps
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _per_arm_shift_scale(t, y, eps=1e-6):
    t = t.reshape(-1); y = y.reshape(-1)
    y0 = y[t < 0.5]; y1 = y[t > 0.5]
    y0s, y0sc = (float(y0.mean()), float(y0.std() + eps)) if y0.size else (0.0, 1.0)
    y1s, y1sc = (float(y1.mean()), float(y1.std() + eps)) if y1.size else (0.0, 1.0)
    return y0s, y0sc, y1s, y1sc


# ── 1D model (CausalPFN) ────────────────────────────────────────────────
def load_1d(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ck.get('model_config', {})
    sd = _strip_prefix(ck['model_state_dict'], 'model.', drop_no_prefix=True)
    enc_w = sd.get('encoder.weight')
    F_plus_T = enc_w.shape[1] if enc_w is not None else 101
    F = F_plus_T - 1
    ninp     = cfg.get('ninp',    enc_w.shape[0] if enc_w is not None else 384)
    nhid     = cfg.get('nhid',    768)
    nhead    = cfg.get('nhead',   6)
    nlayers  = cfg.get('nlayers', 20)
    n_out    = cfg.get('n_out',   10)
    nbins = cfg.get('nbins') or cfg.get('model', {}).get('nbins')
    if nbins is None:
        head_w = sd.get('head.2.weight')
        nbins = head_w.shape[0] - n_out if head_w is not None else 1024
    model = TabDPTLongContextModel(
        dropout=0.0, n_out=n_out, nhead=nhead, nhid=nhid, ninp=ninp,
        nlayers=nlayers, num_features=F_plus_T, nbins=nbins,
    ).to(DEVICE)
    model.load_state_dict(sd, strict=False)
    model.eval()
    edges = np.linspace(VMIN, VMAX, nbins + 1)
    return model, F, nbins, edges


@torch.no_grad()
def arm_means_1d(model, X_ctx, T_ctx, y_ctx_raw, X_q, F, nbins, edges):
    y0s, y0sc, y1s, y1sc = _per_arm_shift_scale(T_ctx, y_ctx_raw)
    y_ctx_std = np.where(T_ctx.reshape(-1) > 0.5,
                          (y_ctx_raw - y1s) / y1sc,
                          (y_ctx_raw - y0s) / y0sc).astype(np.float32)
    X_ctx_t = torch.from_numpy(X_ctx.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_ctx_t = torch.from_numpy(T_ctx.astype(np.float32)).unsqueeze(0).to(DEVICE)
    Y_ctx_t = torch.from_numpy(y_ctx_std).unsqueeze(0).to(DEVICE)
    X_q_t   = torch.from_numpy(X_q.astype(np.float32)).unsqueeze(0).to(DEVICE)
    N_q = X_q.shape[0]; N_ctx = X_ctx.shape[0]
    centers = 0.5 * (edges[:-1] + edges[1:])

    def _fwd(t_val):
        t_ctx_col = T_ctx_t.reshape(1, N_ctx, 1)
        t_q_col   = torch.full((1, N_q, 1), float(t_val), dtype=X_q_t.dtype, device=DEVICE)
        xt_ctx = torch.cat([t_ctx_col, X_ctx_t], dim=-1)
        xt_q   = torch.cat([t_q_col,   X_q_t],   dim=-1)
        x_all  = torch.cat([xt_ctx, xt_q], dim=1).transpose(0, 1).contiguous()
        y_src  = Y_ctx_t.transpose(0, 1).contiguous()
        pred = model(x_all, y_src).transpose(0, 1).contiguous()
        return pred[..., -nbins:].squeeze(0).cpu().numpy()

    p0 = np.exp(np.log(np.exp(_fwd(0.0)) / np.exp(_fwd(0.0)).sum(-1, keepdims=True)))  # softmax
    p1 = np.exp(np.log(np.exp(_fwd(1.0)) / np.exp(_fwd(1.0)).sum(-1, keepdims=True)))
    # Cleaner: torch softmax
    logits0 = _fwd(0.0); logits1 = _fwd(1.0)
    p0 = np.exp(logits0 - logits0.max(-1, keepdims=True))
    p0 /= p0.sum(-1, keepdims=True)
    p1 = np.exp(logits1 - logits1.max(-1, keepdims=True))
    p1 /= p1.sum(-1, keepdims=True)

    e_y0_std = (p0 * centers).sum(axis=-1)
    e_y1_std = (p1 * centers).sum(axis=-1)
    e_y0 = e_y0_std * y0sc + y0s
    e_y1 = e_y1_std * y1sc + y1s
    return e_y0, e_y1


# ── 2D model (our cpfn2d) ───────────────────────────────────────────────
def load_2d(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'config' in ck:
        cfg = ck['config']; edges = ck['edges']
    else:
        mc = ck['model_config']
        cfg = dict(mc['model'])
        cfg['y_scaling_mode'] = mc.get('y_scaling_mode', 'pooled_std')
        cfg['loss_type']      = mc.get('loss_type',      'density')
        cfg['hlgauss_sigma']  = float(mc.get('hlgauss_sigma', 0.2))
        edges = ck['model_state_dict']['edges']
    y_scaling = cfg.get('y_scaling_mode', 'pooled_std')
    loss_type = cfg.get('loss_type', 'density')
    hlg_sigma = float(cfg.get('hlgauss_sigma', 0.2))
    model = CausalPFN2DHead(
        J=cfg['J'], num_features=cfg['num_features'],
        ninp=cfg['ninp'], nhid=cfg['nhid'], nhead=cfg['nhead'],
        nlayers=cfg['nlayers'], dropout=cfg.get('dropout', 0.0),
        n_out=cfg.get('n_out', 10),
        y_scaling_mode=y_scaling,
        loss_type=loss_type,
        hlgauss_sigma=hlg_sigma,
    ).to(DEVICE)
    sd = ck['model_state_dict']
    if any('_orig_mod.' in k for k in sd):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    ref = model.state_dict()
    kept = {k: v for k, v in sd.items() if k in ref and ref[k].shape == v.shape}
    model.load_state_dict(kept, strict=False)
    model.eval()
    edges_np = edges.detach().cpu().numpy() if hasattr(edges, 'detach') else np.asarray(edges)
    return model, cfg['num_features'], cfg['J'], edges_np, y_scaling


@torch.no_grad()
def arm_means_2d(model, X_ctx, T_ctx, y_ctx_raw, X_q, F, J, edges_np, y_scaling_mode):
    X_ctx_t = torch.from_numpy(X_ctx.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_ctx_t = torch.from_numpy(T_ctx.astype(np.float32)).unsqueeze(0).to(DEVICE)
    # y_context must be (B, N_ctx) 2D — _forward_logits does `y_src = y.transpose(0,1)`
    # and the backbone expects (N_ctx, B). A (B, N_ctx, 1) tensor breaks it.
    Y_ctx_t = torch.from_numpy(y_ctx_raw.astype(np.float32)).unsqueeze(0).to(DEVICE)  # (1, N_ctx)
    X_q_t   = torch.from_numpy(X_q.astype(np.float32)).unsqueeze(0).to(DEVICE)

    if y_scaling_mode == 'uwyk_minmax':
        y_lo = Y_ctx_t.amin(dim=1, keepdim=True); y_hi = Y_ctx_t.amax(dim=1, keepdim=True)
        sh = 0.5 * (y_lo + y_hi); sc = (0.5 * (y_hi - y_lo)).clamp(min=1e-6)
    else:
        sh = Y_ctx_t.mean(dim=1, keepdim=True); sc = Y_ctx_t.std(dim=1, keepdim=True).clamp(min=1e-6)
    y_std = (Y_ctx_t - sh) / sc  # (1, N_ctx)
    scale = float(sc.item()); shift = float(sh.item())

    logits = model._forward_logits(X_ctx_t, T_ctx_t, y_std, X_q_t)
    interior = logits[..., : J * J]
    p_mats = torch.softmax(interior, dim=-1).reshape(1, -1, J, J).squeeze(0).cpu().numpy()

    centers = 0.5 * (edges_np[:-1] + edges_np[1:])
    N_q = X_q.shape[0]
    e_y0_std = np.empty(N_q); e_y1_std = np.empty(N_q)
    for q in range(N_q):
        pm = p_mats[q]
        p0 = pm.sum(axis=1); p1 = pm.sum(axis=0)
        p0 /= max(p0.sum(), 1e-12); p1 /= max(p1.sum(), 1e-12)
        e_y0_std[q] = float((centers * p0).sum())
        e_y1_std[q] = float((centers * p1).sum())
    # Both arms un-standardised by the SAME pooled (shift, scale) — matches training
    e_y0 = e_y0_std * scale + shift
    e_y1 = e_y1_std * scale + shift
    return e_y0, e_y1


def main():
    print(f'[bootstrap] loading 1D from {CKPT_1D}', flush=True)
    model_1d, F_1d, nbins_1d, edges_1d = load_1d(CKPT_1D)
    print(f'  1D: F={F_1d} nbins={nbins_1d} edges=[{edges_1d[0]:.2f},{edges_1d[-1]:.2f}]', flush=True)

    print(f'[bootstrap] loading 2D from {CKPT_2D}', flush=True)
    model_2d, F_2d, J, edges_2d, y_scaling_2d = load_2d(CKPT_2D)
    print(f'  2D: F={F_2d} J={J} edges=[{edges_2d[0]:.2f},{edges_2d[-1]:.2f}] '
          f'y_scaling={y_scaling_2d}', flush=True)

    ds = IHDPDataset()

    for r in REALIZATIONS:
        print(f'\n══════ IHDP realization r={r} ══════')
        cate_ds = ds[r][0]
        X_tr = np.asarray(cate_ds.X_train, dtype=np.float32)
        T_tr = np.asarray(cate_ds.t_train, dtype=np.float32).reshape(-1)
        y_tr = np.asarray(cate_ds.y_train, dtype=np.float32).reshape(-1)
        X_te = np.asarray(cate_ds.X_test,  dtype=np.float32)
        true_cate = np.asarray(cate_ds.true_cate, dtype=np.float32)
        true_ate  = float(true_cate.mean())
        # Try to extract per-query counterfactual means
        mu0 = getattr(cate_ds, 'mu_0_test', None)
        mu1 = getattr(cate_ds, 'mu_1_test', None)
        if mu0 is None: mu0 = getattr(cate_ds, 'mu_0', None)
        if mu1 is None: mu1 = getattr(cate_ds, 'mu_1', None)

        print(f'  N_ctx={X_tr.shape[0]}  N_q={X_te.shape[0]}  true_ATE={true_ate:+.3f}  '
              f'y_ctx: range=[{y_tr.min():.2f},{y_tr.max():.2f}] mean={y_tr.mean():+.2f} std={y_tr.std():.2f}')

        # Preprocess for each model separately (different F)
        X_tr_1d_std, X_te_1d_std = _std_X(X_tr, X_te)
        X_tr_1d = _pad(X_tr_1d_std, F_1d); X_te_1d = _pad(X_te_1d_std, F_1d)
        X_tr_2d_std, X_te_2d_std = _std_X(X_tr, X_te)
        X_tr_2d = _pad(X_tr_2d_std, F_2d); X_te_2d = _pad(X_te_2d_std, F_2d)

        e0_1d, e1_1d = arm_means_1d(model_1d, X_tr_1d, T_tr, y_tr, X_te_1d,
                                     F_1d, nbins_1d, edges_1d)
        e0_2d, e1_2d = arm_means_2d(model_2d, X_tr_2d, T_tr, y_tr, X_te_2d,
                                     F_2d, J, edges_2d, y_scaling_2d)

        cate_1d = e1_1d - e0_1d
        cate_2d = e1_2d - e0_2d
        pehe_1d = float(np.sqrt(np.mean((cate_1d - true_cate) ** 2)))
        pehe_2d = float(np.sqrt(np.mean((cate_2d - true_cate) ** 2)))

        n_show = min(N_QUERY, X_te.shape[0])
        header = f'{"q":>3}  |  {"E[Y0] 1D":>10} {"E[Y0] 2D":>10}'
        if mu0 is not None: header += f' {"mu0_true":>10}'
        header += f'  |  {"E[Y1] 1D":>10} {"E[Y1] 2D":>10}'
        if mu1 is not None: header += f' {"mu1_true":>10}'
        header += f'  |  {"CATE 1D":>9} {"CATE 2D":>9} {"true":>9}'
        print(header)
        for q in range(n_show):
            row = f'{q:>3}  |  {e0_1d[q]:>10.3f} {e0_2d[q]:>10.3f}'
            if mu0 is not None: row += f' {float(mu0[q]):>10.3f}'
            row += f'  |  {e1_1d[q]:>10.3f} {e1_2d[q]:>10.3f}'
            if mu1 is not None: row += f' {float(mu1[q]):>10.3f}'
            row += f'  |  {cate_1d[q]:>+9.3f} {cate_2d[q]:>+9.3f} {true_cate[q]:>+9.3f}'
            print(row)
        print(f'  ─── realization {r} PEHE:  1D={pehe_1d:.3f}   2D={pehe_2d:.3f}   '
              f'(2D − 1D = {pehe_2d - pehe_1d:+.3f})')


if __name__ == '__main__':
    main()
