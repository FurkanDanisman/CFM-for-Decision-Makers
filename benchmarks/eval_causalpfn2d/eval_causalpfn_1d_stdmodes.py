"""1D CausalPFN IHDP eval with configurable Y-standardization mode.

Mirror of eval_cpfn2d_ihdp_stdmodes.py for the 1D model.  Only difference
in the model interface: the 1D head takes (X_all, Y_ctx) with T appended
to X, forwards once per arm (t=0 then t=1), and returns nbins bin-logits
per query.

STD_MODEs (identical semantics to 2D):
  pooled   per_arm   winsor   log   recursive   asinh   quantile   log_winsor

K_NN env var: if >0, per-query retrieval + local ctx (same as 2D script).

Env: CKPT, OUT, CAUSALPFN, STD_MODE, [K_NN], [MAX_REAL]
"""
from __future__ import annotations
import os, sys, time, numpy as np, torch

CKPT      = os.environ['CKPT']
OUT       = os.environ['OUT']
CAUSALPFN = os.environ['CAUSALPFN']
STD_MODE  = os.environ.get('STD_MODE', 'pooled')
K_NN      = int(os.environ.get('K_NN', '0'))
MAX_REAL  = os.environ.get('MAX_REAL', '')

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, CAUSALPFN); sys.path.insert(0, CAUSALPFN + '/src')

from benchmarks import IHDPDataset  # noqa: E402
from causalpfn.models.model import TabDPTLongContextModel  # noqa: E402
from scipy.stats import norm  # noqa: E402


def _em_mean_1d(p, edges, sigma, start, max_step=1000, eps2=1e-10, eps1=1e-5):
    pn = p / p.sum(); mu = float(start)
    for _ in range(max_step):
        a = (edges - mu) / sigma
        G1 = norm.cdf(a); G2 = norm.pdf(a)
        temp = (np.diff(G2) + eps2) / (np.diff(G1) + eps2)
        mu_new = mu - sigma * float(np.sum(pn * temp))
        if abs(mu_new - mu) < eps1: return float(mu_new)
        mu = mu_new
    return float(mu)


def _marginal_stats(p, edges):
    delta = edges[1] - edges[0]
    centres = 0.5 * (edges[:-1] + edges[1:])
    mu_low = float(np.sum(p * edges[:-1]))
    mu_mid = 0.5 * (mu_low + float(np.sum(p * edges[1:])))
    sigma = float(np.sqrt(np.sum(p * (centres - mu_mid) ** 2) + delta**2 / 12.0))
    if not np.isfinite(sigma) or sigma <= 0: sigma = float(delta)
    return mu_mid, sigma


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VMIN, VMAX = -10.0, 10.0
VALID_MODES = ('pooled', 'per_arm', 'winsor', 'log', 'recursive',
               'asinh', 'quantile', 'log_winsor', 'log_per_arm')
if STD_MODE not in VALID_MODES:
    raise SystemExit(f'STD_MODE={STD_MODE!r} invalid; choose {VALID_MODES}')


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


def _load_model_1d(ckpt_path):
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
    nbins    = cfg.get('nbins') or cfg.get('model', {}).get('nbins')
    if nbins is None:
        head_w = sd.get('head.2.weight')
        nbins = head_w.shape[0] - n_out if head_w is not None else 1024
    model = TabDPTLongContextModel(
        dropout=0.0, n_out=n_out, nhead=nhead, nhid=nhid, ninp=ninp,
        nlayers=nlayers, num_features=F_plus_T, nbins=nbins,
    ).to(DEVICE)
    model.load_state_dict(sd, strict=False)
    model.eval()
    edges_np = np.linspace(VMIN, VMAX, nbins + 1)
    return model, F, nbins, edges_np


# ── STD modes ────────────────────────────────────────────────────────────
def _stats_pooled(Y):
    return float(Y.mean()), float(max(Y.std(), 1e-6))


def _apply_std(Y_ctx, T_ctx, edges_np):
    """Return (y_ctx_std, arm0_stats, arm1_stats, y_transform).
    arm_stats is (mean, scale). y_transform is None or (kind, aux).
    For pooled/log/etc the two arm_stats are equal; only per_arm differs."""
    T = T_ctx.reshape(-1); Y = Y_ctx.reshape(-1).astype(np.float64)

    if STD_MODE == 'pooled':
        m, s = _stats_pooled(Y)
        z = ((Y - m) / s).astype(np.float32)
        return z, (m, s), (m, s), None

    if STD_MODE == 'per_arm':
        y0 = Y[T < 0.5]; y1 = Y[T > 0.5]
        m0, s0 = _stats_pooled(y0) if y0.size else (0.0, 1.0)
        m1, s1 = _stats_pooled(y1) if y1.size else (0.0, 1.0)
        z = np.where(T > 0.5, (Y - m1) / s1, (Y - m0) / s0).astype(np.float32)
        return z, (m0, s0), (m1, s1), None

    if STD_MODE == 'winsor':
        lo, hi = np.quantile(Y, [0.01, 0.99])
        Yw = np.clip(Y, lo, hi)
        m, s = _stats_pooled(Yw)
        edge_lo, edge_hi = float(edges_np[0]), float(edges_np[-1])
        z = np.clip((Y - m) / s, edge_lo, edge_hi).astype(np.float32)
        return z, (m, s), (m, s), None

    if STD_MODE == 'log':
        y_min = float(Y.min())
        Y_l = np.log1p(Y - y_min)
        m, s = _stats_pooled(Y_l)
        z = ((Y_l - m) / s).astype(np.float32)
        return z, (m, s), (m, s), ('log', y_min)

    if STD_MODE == 'log_winsor':
        y_min = float(Y.min())
        Y_l = np.log1p(Y - y_min)
        lo, hi = np.quantile(Y_l, [0.01, 0.99])
        Yw = np.clip(Y_l, lo, hi)
        m, s = _stats_pooled(Yw)
        edge_lo, edge_hi = float(edges_np[0]), float(edges_np[-1])
        z = np.clip((Y_l - m) / s, edge_lo, edge_hi).astype(np.float32)
        return z, (m, s), (m, s), ('log', y_min)

    if STD_MODE == 'log_per_arm':
        y_min = float(Y.min())
        Y_l = np.log1p(Y - y_min)
        y0 = Y_l[T < 0.5]; y1 = Y_l[T > 0.5]
        m0, s0 = _stats_pooled(y0) if y0.size else (0.0, 1.0)
        m1, s1 = _stats_pooled(y1) if y1.size else (0.0, 1.0)
        z = np.where(T > 0.5, (Y_l - m1) / s1, (Y_l - m0) / s0).astype(np.float32)
        return z, (m0, s0), (m1, s1), ('log', y_min)

    if STD_MODE == 'recursive':
        Yc = Y.copy()
        for _ in range(3):
            m, s = _stats_pooled(Yc)
            mask = np.abs((Yc - m) / s) < 3.0
            if mask.sum() < 5: break
            Yc = Yc[mask]
        m, s = _stats_pooled(Yc)
        z = ((Y - m) / s).astype(np.float32)
        return z, (m, s), (m, s), None

    if STD_MODE == 'asinh':
        scale = float(1.4826 * np.median(np.abs(Y - np.median(Y))) + 1e-6)
        Y_a = np.arcsinh(Y / scale)
        m, s = _stats_pooled(Y_a)
        z = ((Y_a - m) / s).astype(np.float32)
        return z, (m, s), (m, s), ('asinh', scale)

    if STD_MODE == 'quantile':
        from scipy.stats import norm
        N = len(Y); order = np.argsort(Y)
        ranks = np.empty(N, dtype=np.float64); ranks[order] = np.arange(N)
        Y_qn = norm.ppf((ranks + 0.5) / N)
        m, s = _stats_pooled(Y_qn)
        z = ((Y_qn - m) / s).astype(np.float32)
        Y_sorted = np.sort(Y).astype(np.float64)
        return z, (m, s), (m, s), ('quantile', Y_sorted)

    raise SystemExit(f'unknown STD_MODE {STD_MODE}')


def _unstd(e_std, mean_scale, y_transform):
    m, s = mean_scale
    e_pre = e_std * s + m
    if y_transform is None:
        return e_pre
    kind, aux = y_transform
    if kind == 'log':
        return np.expm1(e_pre) + aux
    if kind == 'asinh':
        return np.sinh(e_pre) * aux
    if kind == 'quantile':
        from scipy.stats import norm
        Y_sorted = aux
        u = np.clip(norm.cdf(e_pre), 1.0 / (2 * len(Y_sorted)),
                                     1.0 - 1.0 / (2 * len(Y_sorted)))
        pos = u * len(Y_sorted) - 0.5
        lo = np.clip(np.floor(pos).astype(np.int64), 0, len(Y_sorted) - 1)
        hi = np.clip(lo + 1, 0, len(Y_sorted) - 1)
        w = (pos - lo).astype(np.float64)
        return (1 - w) * Y_sorted[lo] + w * Y_sorted[hi]
    return e_pre


# ── Forward + CATE ────────────────────────────────────────────────────────
@torch.no_grad()
def _forward_one_arm(model, X_ctx, T_ctx, Y_std_ctx, X_q, t_val, nbins):
    B, N_ctx, _ = X_ctx.shape
    N_q = X_q.shape[1]
    t_ctx_col = T_ctx.reshape(B, N_ctx, 1)
    t_q_col   = torch.full((B, N_q, 1), float(t_val), dtype=X_q.dtype, device=X_q.device)
    xt_ctx = torch.cat([t_ctx_col, X_ctx], dim=-1)
    xt_q   = torch.cat([t_q_col,   X_q],   dim=-1)
    x_all = torch.cat([xt_ctx, xt_q], dim=1).transpose(0, 1).contiguous()
    y_src = Y_std_ctx.transpose(0, 1).contiguous()
    pred = model(x_all, y_src).transpose(0, 1).contiguous()
    return pred[..., -nbins:]


@torch.no_grad()
def _predict_arms(model, X_ctx, T_ctx, Y_ctx_raw, X_q, nbins, edges_np):
    y_ctx_std, s0, s1, y_transform = _apply_std(Y_ctx_raw, T_ctx, edges_np)
    X_ctx_t = torch.from_numpy(X_ctx.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_ctx_t = torch.from_numpy(T_ctx.astype(np.float32)).unsqueeze(0).to(DEVICE)
    Y_ctx_t = torch.from_numpy(y_ctx_std).unsqueeze(0).to(DEVICE)
    X_q_t   = torch.from_numpy(X_q.astype(np.float32)).unsqueeze(0).to(DEVICE)
    centers = 0.5 * (edges_np[:-1] + edges_np[1:])

    l0 = _forward_one_arm(model, X_ctx_t, T_ctx_t, Y_ctx_t, X_q_t, 0.0, nbins)
    l1 = _forward_one_arm(model, X_ctx_t, T_ctx_t, Y_ctx_t, X_q_t, 1.0, nbins)
    p0 = torch.softmax(l0.float(), dim=-1).squeeze(0).cpu().numpy()
    p1 = torch.softmax(l1.float(), dim=-1).squeeze(0).cpu().numpy()
    e0_std = (p0 * centers).sum(axis=-1)
    e1_std = (p1 * centers).sum(axis=-1)

    N_q = p0.shape[0]; edges_f64 = edges_np.astype(np.float64)
    e0_em_std = np.empty(N_q); e1_em_std = np.empty(N_q)
    for q in range(N_q):
        mu0, sig0 = _marginal_stats(p0[q], edges_f64)
        mu1, sig1 = _marginal_stats(p1[q], edges_f64)
        e0_em_std[q] = _em_mean_1d(p0[q], edges_f64, sig0, mu0)
        e1_em_std[q] = _em_mean_1d(p1[q], edges_f64, sig1, mu1)

    e0_raw = _unstd(e0_std, s0, y_transform)
    e1_raw = _unstd(e1_std, s1, y_transform)
    e0_em  = _unstd(e0_em_std, s0, y_transform)
    e1_em  = _unstd(e1_em_std, s1, y_transform)

    frac_in = float(((y_ctx_std >= edges_np[0]) & (y_ctx_std <= edges_np[-1])).mean())
    return e0_raw, e1_raw, e0_em, e1_em, frac_in


@torch.no_grad()
def _predict_arms_knn(model, X_ctx, T_ctx, Y_ctx_raw, X_q, nbins, edges_np, k):
    N_q = X_q.shape[0]
    e0 = np.empty(N_q); e1 = np.empty(N_q)
    e0m = np.empty(N_q); e1m = np.empty(N_q); fr = np.empty(N_q)
    for q in range(N_q):
        d = ((X_ctx - X_q[q:q+1]) ** 2).sum(axis=1)
        idx = np.argpartition(d, min(k - 1, len(d) - 1))[:k]
        r0, r1, em0, em1, fi = _predict_arms(model, X_ctx[idx], T_ctx[idx],
                                              Y_ctx_raw[idx], X_q[q:q+1], nbins, edges_np)
        e0[q] = r0[0]; e1[q] = r1[0]; e0m[q] = em0[0]; e1m[q] = em1[0]; fr[q] = fi
    return e0, e1, e0m, e1m, float(fr.mean())


def evaluate(realization, model, edges_np, nbins, F):
    ds = IHDPDataset(); cate_ds = ds[realization][0]
    X_tr = cate_ds.X_train.astype(np.float32)
    T_tr = cate_ds.t_train.astype(np.float32).reshape(-1)
    y_tr = cate_ds.y_train.astype(np.float32).reshape(-1)
    X_te = cate_ds.X_test.astype(np.float32)
    true_cate = cate_ds.true_cate.astype(np.float32)
    true_ate = float(true_cate.mean())

    Xs, Xq = _std_X(X_tr, X_te); Xs = _pad(Xs, F); Xq = _pad(Xq, F)
    if K_NN > 0:
        k = min(K_NN, Xs.shape[0])
        e0, e1, e0m, e1m, frac_in = _predict_arms_knn(model, Xs, T_tr, y_tr, Xq,
                                                       nbins, edges_np, k)
    else:
        e0, e1, e0m, e1m, frac_in = _predict_arms(model, Xs, T_tr, y_tr, Xq,
                                                   nbins, edges_np)

    def _pe(cate):
        pehe = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
        ate  = float(cate.mean()); err = abs(ate - true_ate) / max(abs(true_ate), 0.1)
        return pehe, err, ate
    p_raw, e_raw, a_raw = _pe((e1 - e0).astype(np.float32))
    p_em,  e_em,  a_em  = _pe((e1m - e0m).astype(np.float32))
    return {'dataset': 'IHDP', 'realization': realization, 'true_ate': true_ate,
            'std_mode': STD_MODE, 'k_nn': K_NN, 'frac_ctx_in_edges': frac_in,
            'pehe_raw': p_raw, 'err_raw': e_raw, 'ate_raw': a_raw,
            'pehe_em':  p_em,  'err_em':  e_em,  'ate_em':  a_em}


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f'[bootstrap] 1D ckpt={CKPT}', flush=True)
    print(f'[bootstrap] STD_MODE={STD_MODE}  K_NN={K_NN}', flush=True)
    model, F, nbins, edges_np = _load_model_1d(CKPT)
    print(f'[bootstrap] nbins={nbins} F={F} edges=[{edges_np[0]:.2f},{edges_np[-1]:.2f}]', flush=True)

    ds = IHDPDataset()
    n = ds.n_tables if not MAX_REAL else min(ds.n_tables, int(MAX_REAL))

    rows = []; t0 = time.time()
    for r in range(n):
        row = evaluate(r, model, edges_np, nbins, F)
        np.savez(os.path.join(OUT, f'r{r:03d}.npz'), **{k: np.array(v) for k, v in row.items()})
        rows.append(row)
        print(f'r={r:03d}  in={row["frac_ctx_in_edges"]:.3f}  '
              f'raw={row["pehe_raw"]:6.3f}  em={row["pehe_em"]:6.3f}  '
              f'(ate={row["true_ate"]:+5.2f}, {time.time()-t0:.0f}s)', flush=True)

    def _ms(k):
        v = np.array([r[k] for r in rows if np.isfinite(r[k])])
        return v.mean(), v.std(ddof=1) / np.sqrt(max(len(v), 1)), int(v.size)
    print(f'\n══ IHDP  STD_MODE={STD_MODE}  K_NN={K_NN}  n={len(rows)}  '
          f'edges=[{edges_np[0]:.1f},{edges_np[-1]:.1f}] ══')
    for k in ('frac_ctx_in_edges', 'pehe_raw', 'err_raw', 'pehe_em', 'err_em'):
        m, s, _ = _ms(k)
        print(f'  {k:22s} = {m:8.3f} ± {s:6.3f}')


if __name__ == '__main__':
    main()
