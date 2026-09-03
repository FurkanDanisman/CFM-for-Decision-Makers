"""cpfn2d IHDP eval with configurable Y-standardization mode.

Motivation: for the tight-edge model (edges [-4,4]) an outlier context Y that
maps to |y_std_val| > 4 gets its mass saturated to the edge bin. Pooled
(mean/std) on heavy-tailed IHDP realizations (r=8, r=12 have Y up to ~190)
gives huge y_std → the middle of the distribution fills a tiny slice of the
grid AND the outliers still poke out the edges. Different y-standardization
recipes trade these two failure modes.

Modes (STD_MODE env var):
  pooled       — Y_std = std(Y_ctx)                    (baseline)
  per_arm      — separate (mean, std) for T=0 vs T=1   (1D CausalPFN style)
  winsor       — clip Y_ctx to [Q1, Q99] BEFORE computing mean/std, then
                 clip standardised Y_ctx to model edges. Guarantees no saturation.
  log          — Y' = log1p(Y - min(Y_ctx)) then pooled std. Compresses heavy tail.
  recursive    — pooled, then drop |z|>3, recompute, iterate 3× (trimmed std).

Every mode uses the SAME forward pass code path — only the (y_mean, y_std)
choice and any Y-value transform changes.

Reports pehe_raw + pehe_full for each mode; no EM (keeps output compact).

Env vars: CKPT, OUT, CAUSALPFN, STD_MODE, [MAX_REAL]
"""
from __future__ import annotations
import os, sys, time, numpy as np, torch


CKPT      = os.environ['CKPT']
OUT       = os.environ['OUT']
CAUSALPFN = os.environ['CAUSALPFN']
STD_MODE  = os.environ.get('STD_MODE', 'pooled')
K_NN      = int(os.environ.get('K_NN', '0'))    # 0 = full context; >0 = per-query retrieval
MAX_REAL  = os.environ.get('MAX_REAL', '')

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, CAUSALPFN); sys.path.insert(0, CAUSALPFN + '/src')
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path: sys.path.insert(0, _here)

from benchmarks import IHDPDataset  # noqa: E402
from training_causalpfn2d.model_causalpfn_2d import CausalPFN2DHead  # noqa: E402
from full_mixture_mean import full_mixture_mean  # noqa: E402
from scipy.stats import norm  # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VALID_MODES = ('pooled', 'per_arm', 'winsor', 'log', 'recursive',
               'asinh', 'quantile', 'log_winsor', 'log_per_arm')
if STD_MODE not in VALID_MODES:
    raise SystemExit(f'STD_MODE={STD_MODE!r} invalid; choose {VALID_MODES}')


def _std_X(Xtr, Xte, eps=1e-8):
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + eps
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _pad(X, F):
    if X.shape[1] == F: return X
    if X.shape[1] > F:  return X[:, :F]
    return np.hstack([X, np.zeros((X.shape[0], F - X.shape[1]), dtype=X.dtype)])


def _load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if 'config' in ck:
        cfg = ck['config']; edges = ck['edges']
    else:
        mc = ck['model_config']; cfg = dict(mc['model'])
        cfg['y_scaling_mode'] = mc.get('y_scaling_mode', 'pooled_std')
        cfg['loss_type']      = mc.get('loss_type',      'density')
        cfg['hlgauss_sigma']  = float(mc.get('hlgauss_sigma', 0.2))
        edges = ck['model_state_dict']['edges']
    model = CausalPFN2DHead(
        J=cfg['J'], num_features=cfg['num_features'],
        ninp=cfg['ninp'], nhid=cfg['nhid'], nhead=cfg['nhead'],
        nlayers=cfg['nlayers'], dropout=cfg.get('dropout', 0.0),
        n_out=cfg.get('n_out', 10),
        y_scaling_mode=cfg.get('y_scaling_mode', 'pooled_std'),
        loss_type=cfg.get('loss_type', 'density'),
        hlgauss_sigma=float(cfg.get('hlgauss_sigma', 0.2)),
    ).to(DEVICE)
    sd = ck['model_state_dict']
    if any('_orig_mod.' in k for k in sd):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    ref = model.state_dict()
    kept = {k: v for k, v in sd.items() if k in ref and ref[k].shape == v.shape}
    model.load_state_dict(kept, strict=False)
    model.eval()
    edges_np = edges.detach().cpu().numpy() if hasattr(edges, 'detach') else np.asarray(edges)
    return model, cfg, edges_np


# ── EM-mean helpers (MALC's Gaussian-bin correction) ─────────────────────
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


# ── Standardisation modes ─────────────────────────────────────────────────
def _std_pooled(Y_ctx, T_ctx, edges_np):
    y_mean = float(Y_ctx.mean()); y_std = float(max(Y_ctx.std(), 1e-6))
    y_ctx_std = ((Y_ctx - y_mean) / y_std).astype(np.float32)
    return y_ctx_std, y_mean, y_std, y_mean, y_std   # (ctx_std, mean_y0, std_y0, mean_y1, std_y1)


def _std_per_arm(Y_ctx, T_ctx, edges_np):
    T = T_ctx.reshape(-1); Y = Y_ctx.reshape(-1)
    y0 = Y[T < 0.5]; y1 = Y[T > 0.5]
    m0, s0 = (float(y0.mean()), float(max(y0.std(), 1e-6))) if y0.size else (0.0, 1.0)
    m1, s1 = (float(y1.mean()), float(max(y1.std(), 1e-6))) if y1.size else (0.0, 1.0)
    y_ctx_std = np.where(T > 0.5, (Y - m1) / s1, (Y - m0) / s0).astype(np.float32)
    return y_ctx_std, m0, s0, m1, s1


def _std_winsor(Y_ctx, T_ctx, edges_np, lo_q=0.01, hi_q=0.99):
    """Clip Y to [Q1, Q99] BEFORE computing stats. Then clip std-Y to model
    edges to guarantee zero saturation regardless of edge range."""
    q_lo, q_hi = np.quantile(Y_ctx, [lo_q, hi_q])
    Y_win = np.clip(Y_ctx, q_lo, q_hi)
    y_mean = float(Y_win.mean()); y_std = float(max(Y_win.std(), 1e-6))
    # Standardise ORIGINAL Y then clip to model edges (matches _forward saturation)
    edge_lo, edge_hi = float(edges_np[0]), float(edges_np[-1])
    y_std_val = (Y_ctx - y_mean) / y_std
    y_ctx_std = np.clip(y_std_val, edge_lo, edge_hi).astype(np.float32)
    return y_ctx_std, y_mean, y_std, y_mean, y_std


def _std_log(Y_ctx, T_ctx, edges_np):
    """Y' = log1p(Y - min(Y_ctx)), then pooled std on Y'.
    Compresses heavy positive tails (IHDP outliers) into a manageable range."""
    y_min = float(Y_ctx.min())
    Y_log = np.log1p(Y_ctx - y_min)                # ≥0, heavy tail flattened
    y_mean = float(Y_log.mean()); y_std = float(max(Y_log.std(), 1e-6))
    y_ctx_std = ((Y_log - y_mean) / y_std).astype(np.float32)
    # NB: y_mean/y_std here are in LOG space — un-standardising must be
    # log_inv → shift back by y_min. Handled downstream via `y_transform`.
    return y_ctx_std, y_mean, y_std, y_mean, y_std, ('log', y_min)


def _std_asinh(Y_ctx, T_ctx, edges_np):
    """Y' = asinh(Y/scale) with scale = MAD*1.4826. Symmetric log — handles
    negative Y natively, no y_min shift. asinh(y) ≈ log(2y) for large |y|
    and ≈ y for small |y|, so it compresses tails without distorting the
    core. Un-standardise via sinh."""
    scale = float(1.4826 * np.median(np.abs(Y_ctx - np.median(Y_ctx))) + 1e-6)
    Y_tr = np.arcsinh(Y_ctx / scale)
    y_mean = float(Y_tr.mean()); y_std = float(max(Y_tr.std(), 1e-6))
    y_ctx_std = ((Y_tr - y_mean) / y_std).astype(np.float32)
    return y_ctx_std, y_mean, y_std, y_mean, y_std, ('asinh', scale)


def _std_quantile(Y_ctx, T_ctx, edges_np):
    """Rank-normal transform: map each Y to Φ⁻¹((rank + 0.5) / N).
    Removes ALL tail effects — the ctx Y distribution becomes exactly N(0,1)
    after transform. Un-standardise by inverse: recover ctx Y from ctx rank
    via linear interpolation."""
    from scipy.stats import norm
    N = len(Y_ctx)
    order = np.argsort(Y_ctx)
    ranks = np.empty(N, dtype=np.float64); ranks[order] = np.arange(N)
    Y_qn = norm.ppf((ranks + 0.5) / N)               # ≈ N(0,1)
    y_mean = float(Y_qn.mean()); y_std = float(max(Y_qn.std(), 1e-6))
    y_ctx_std = ((Y_qn - y_mean) / y_std).astype(np.float32)
    # Store sorted Y_ctx for inverse mapping (mean-space → raw Y)
    Y_sorted = np.sort(Y_ctx).astype(np.float64)
    return y_ctx_std, y_mean, y_std, y_mean, y_std, ('quantile', (Y_sorted, y_mean, y_std))


def _std_log_per_arm(Y_ctx, T_ctx, edges_np):
    """log-transform first, then per-arm (mean, std) inside log space.
    Combines log's tail-crushing with per-arm's separate scales for T=0/T=1
    (useful when treated arm has narrower Y distribution than control)."""
    T = T_ctx.reshape(-1); Y = Y_ctx.reshape(-1)
    y_min = float(Y.min())
    Y_log = np.log1p(Y - y_min)
    y0 = Y_log[T < 0.5]; y1 = Y_log[T > 0.5]
    m0, s0 = (float(y0.mean()), float(max(y0.std(), 1e-6))) if y0.size else (0.0, 1.0)
    m1, s1 = (float(y1.mean()), float(max(y1.std(), 1e-6))) if y1.size else (0.0, 1.0)
    z = np.where(T > 0.5, (Y_log - m1) / s1, (Y_log - m0) / s0).astype(np.float32)
    return z, m0, s0, m1, s1, ('log', y_min)


def _std_log_winsor(Y_ctx, T_ctx, edges_np, lo_q=0.01, hi_q=0.99):
    """log-transform, then winsorise in log space (belt + suspenders)."""
    y_min = float(Y_ctx.min())
    Y_log = np.log1p(Y_ctx - y_min)
    q_lo, q_hi = np.quantile(Y_log, [lo_q, hi_q])
    Y_win = np.clip(Y_log, q_lo, q_hi)
    y_mean = float(Y_win.mean()); y_std = float(max(Y_win.std(), 1e-6))
    edge_lo, edge_hi = float(edges_np[0]), float(edges_np[-1])
    y_std_val = np.clip((Y_log - y_mean) / y_std, edge_lo, edge_hi)
    return y_std_val.astype(np.float32), y_mean, y_std, y_mean, y_std, ('log', y_min)


def _std_recursive(Y_ctx, T_ctx, edges_np, z_thresh=3.0, n_iter=3):
    Y = Y_ctx.copy()
    for _ in range(n_iter):
        mu = float(Y.mean()); sd = float(max(Y.std(), 1e-6))
        z = np.abs((Y - mu) / sd)
        mask = z < z_thresh
        if mask.sum() < 5: break
        Y = Y[mask]
    y_mean = float(Y.mean()); y_std = float(max(Y.std(), 1e-6))
    y_ctx_std = ((Y_ctx - y_mean) / y_std).astype(np.float32)
    return y_ctx_std, y_mean, y_std, y_mean, y_std


def _apply_std(Y_ctx, T_ctx, edges_np):
    """Return (y_ctx_std, m0, s0, m1, s1, y_transform).
    y_transform is None (identity) or ('log', y_min)."""
    if STD_MODE == 'pooled':
        r = _std_pooled(Y_ctx, T_ctx, edges_np);   return (*r, None)
    if STD_MODE == 'per_arm':
        r = _std_per_arm(Y_ctx, T_ctx, edges_np);  return (*r, None)
    if STD_MODE == 'winsor':
        r = _std_winsor(Y_ctx, T_ctx, edges_np);   return (*r, None)
    if STD_MODE == 'log':
        return _std_log(Y_ctx, T_ctx, edges_np)
    if STD_MODE == 'recursive':
        r = _std_recursive(Y_ctx, T_ctx, edges_np);return (*r, None)
    if STD_MODE == 'asinh':
        return _std_asinh(Y_ctx, T_ctx, edges_np)
    if STD_MODE == 'quantile':
        return _std_quantile(Y_ctx, T_ctx, edges_np)
    if STD_MODE == 'log_winsor':
        return _std_log_winsor(Y_ctx, T_ctx, edges_np)
    if STD_MODE == 'log_per_arm':
        return _std_log_per_arm(Y_ctx, T_ctx, edges_np)
    raise SystemExit(f'unknown STD_MODE {STD_MODE}')


def _unstd_arm(e_std_arm, m_arm, s_arm, y_transform):
    """Un-standardise a std-space E[Y] per arm back to raw Y."""
    e_pre = e_std_arm * s_arm + m_arm          # back into transform-space
    if y_transform is None:
        return e_pre
    kind, aux = y_transform
    if kind == 'log':
        return np.expm1(e_pre) + aux            # inverse of log1p(y - y_min)
    if kind == 'asinh':
        return np.sinh(e_pre) * aux             # inverse of asinh(y/scale)
    if kind == 'quantile':
        # e_pre is a z-score in the rank-normal domain; map back via inverse
        # empirical CDF (linear interp on sorted ctx Y).
        Y_sorted, y_mean_qn, y_std_qn = aux
        z = e_pre                                     # already de-standardised
        from scipy.stats import norm
        u = np.clip(norm.cdf(z), 1.0 / (2 * len(Y_sorted)),
                                 1.0 - 1.0 / (2 * len(Y_sorted)))
        # position = u * N - 0.5, linear interp on sorted Y
        pos = u * len(Y_sorted) - 0.5
        lo = np.clip(np.floor(pos).astype(np.int64), 0, len(Y_sorted) - 1)
        hi = np.clip(lo + 1, 0, len(Y_sorted) - 1)
        w = (pos - lo).astype(np.float64)
        return (1 - w) * Y_sorted[lo] + w * Y_sorted[hi]
    return e_pre


# ── Forward + CATE ────────────────────────────────────────────────────────
@torch.no_grad()
def _predict_arms(model, X_ctx, T_ctx, Y_ctx_raw, X_q, J, edges_np):
    y_ctx_std, m0, s0, m1, s1, y_transform = _apply_std(Y_ctx_raw, T_ctx, edges_np)

    X_ctx_t = torch.from_numpy(X_ctx.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_ctx_t = torch.from_numpy(T_ctx.astype(np.float32)).unsqueeze(0).to(DEVICE)
    Y_ctx_t = torch.from_numpy(y_ctx_std).unsqueeze(0).to(DEVICE)
    X_q_t   = torch.from_numpy(X_q.astype(np.float32)).unsqueeze(0).to(DEVICE)

    logits = model._forward_logits(X_ctx_t, T_ctx_t, Y_ctx_t, X_q_t)  # (1, N_q, J²+9+4)
    logits_np = logits.squeeze(0).float().cpu().numpy()

    centers = 0.5 * (edges_np[:-1] + edges_np[1:])
    interior = logits_np[..., : J * J]
    p_max = interior.max(axis=-1, keepdims=True)
    p_mat = np.exp(interior - p_max)
    p_mat = p_mat / p_mat.sum(axis=-1, keepdims=True)
    p_mat = p_mat.reshape(-1, J, J)
    p_y0 = p_mat.sum(axis=-1); p_y0 /= p_y0.sum(axis=-1, keepdims=True)
    p_y1 = p_mat.sum(axis=-2); p_y1 /= p_y1.sum(axis=-1, keepdims=True)

    e0_raw_std = (p_y0 * centers).sum(axis=-1)
    e1_raw_std = (p_y1 * centers).sum(axis=-1)
    e0_full_std, e1_full_std = full_mixture_mean(logits_np, J, edges_np)

    # EM-mean (MALC Gaussian-bin correction) — per-query fixed-point iterate
    N_q = p_y0.shape[0]
    e0_em_std = np.empty(N_q); e1_em_std = np.empty(N_q)
    edges_f64 = edges_np.astype(np.float64)
    for q in range(N_q):
        mu0, sig0 = _marginal_stats(p_y0[q], edges_f64)
        mu1, sig1 = _marginal_stats(p_y1[q], edges_f64)
        e0_em_std[q] = _em_mean_1d(p_y0[q], edges_f64, sig0, mu0)
        e1_em_std[q] = _em_mean_1d(p_y1[q], edges_f64, sig1, mu1)

    e0_raw  = _unstd_arm(e0_raw_std,  m0, s0, y_transform)
    e1_raw  = _unstd_arm(e1_raw_std,  m1, s1, y_transform)
    e0_full = _unstd_arm(e0_full_std, m0, s0, y_transform)
    e1_full = _unstd_arm(e1_full_std, m1, s1, y_transform)
    e0_em   = _unstd_arm(e0_em_std,   m0, s0, y_transform)
    e1_em   = _unstd_arm(e1_em_std,   m1, s1, y_transform)

    frac_in = float(((y_ctx_std >= edges_np[0]) & (y_ctx_std <= edges_np[-1])).mean())
    return e0_raw, e1_raw, e0_full, e1_full, e0_em, e1_em, frac_in


@torch.no_grad()
def _predict_arms_knn(model, X_ctx, T_ctx, Y_ctx_raw, X_q, J, edges_np, k):
    """Per-query k-NN retrieval + chosen STD_MODE inside the local context."""
    N_q = X_q.shape[0]
    centers = 0.5 * (edges_np[:-1] + edges_np[1:])
    e0_raw = np.empty(N_q); e1_raw = np.empty(N_q)
    e0_full = np.empty(N_q); e1_full = np.empty(N_q)
    e0_em = np.empty(N_q); e1_em = np.empty(N_q)
    fracs = np.empty(N_q)
    for q in range(N_q):
        d = ((X_ctx - X_q[q:q+1]) ** 2).sum(axis=1)
        idx = np.argpartition(d, min(k - 1, len(d) - 1))[:k]
        e0, e1, ef0, ef1, em0, em1, fi = _predict_arms(
            model, X_ctx[idx], T_ctx[idx], Y_ctx_raw[idx], X_q[q:q+1], J, edges_np,
        )
        e0_raw[q] = e0[0]; e1_raw[q] = e1[0]
        e0_full[q] = ef0[0]; e1_full[q] = ef1[0]
        e0_em[q]  = em0[0]; e1_em[q]  = em1[0]
        fracs[q] = fi
    return e0_raw, e1_raw, e0_full, e1_full, e0_em, e1_em, float(fracs.mean())


def evaluate(realization, model, edges_np, J, F):
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
        e0_raw, e1_raw, e0_full, e1_full, e0_em, e1_em, frac_in = _predict_arms_knn(
            model, Xs, T_tr, y_tr, Xq, J, edges_np, k,
        )
    else:
        e0_raw, e1_raw, e0_full, e1_full, e0_em, e1_em, frac_in = _predict_arms(
            model, Xs, T_tr, y_tr, Xq, J, edges_np,
        )
    cate_raw  = (e1_raw  - e0_raw ).astype(np.float32)
    cate_full = (e1_full - e0_full).astype(np.float32)
    cate_em   = (e1_em   - e0_em  ).astype(np.float32)

    def _pe(cate):
        pehe = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
        ate  = float(cate.mean()); err = abs(ate - true_ate) / max(abs(true_ate), 1e-9)
        return pehe, err, ate
    p_raw,  e_raw,  a_raw  = _pe(cate_raw)
    p_full, e_full, a_full = _pe(cate_full)
    p_em,   e_em,   a_em   = _pe(cate_em)
    return {'dataset': 'IHDP', 'realization': realization, 'true_ate': true_ate,
            'std_mode': STD_MODE, 'k_nn': K_NN, 'frac_ctx_in_edges': frac_in,
            'pehe_raw':  p_raw,  'err_raw':  e_raw,  'ate_raw':  a_raw,
            'pehe_full': p_full, 'err_full': e_full, 'ate_full': a_full,
            'pehe_em':   p_em,   'err_em':   e_em,   'ate_em':   a_em}


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f'[bootstrap] ckpt={CKPT}', flush=True)
    print(f'[bootstrap] STD_MODE={STD_MODE}  K_NN={K_NN}', flush=True)
    model, cfg, edges_np = _load_model(CKPT)
    J = cfg['J']; F = cfg['num_features']
    print(f'[bootstrap] J={J} F={F} edges=[{edges_np[0]:.2f},{edges_np[-1]:.2f}]  '
          f'bw_std={(edges_np[1]-edges_np[0]):.4f}', flush=True)

    ds = IHDPDataset()
    n = ds.n_tables if not MAX_REAL else min(ds.n_tables, int(MAX_REAL))

    rows = []; t0 = time.time()
    for r in range(n):
        row = evaluate(r, model, edges_np, J, F)
        np.savez(os.path.join(OUT, f'r{r:03d}.npz'), **{k: np.array(v) for k, v in row.items()})
        rows.append(row)
        print(f'r={r:03d}  in={row["frac_ctx_in_edges"]:.3f}  '
              f'raw={row["pehe_raw"]:6.3f}  full={row["pehe_full"]:6.3f}  '
              f'em={row["pehe_em"]:6.3f}  '
              f'(ate={row["true_ate"]:+5.2f}, {time.time()-t0:.0f}s)', flush=True)

    def _ms(k):
        v = np.array([r[k] for r in rows if np.isfinite(r[k])])
        return v.mean(), v.std(ddof=1) / np.sqrt(max(len(v), 1)), int(v.size)
    print(f'\n══ IHDP  STD_MODE={STD_MODE}  n={len(rows)}  edges=[{edges_np[0]:.1f},{edges_np[-1]:.1f}] ══')
    for k in ('frac_ctx_in_edges', 'pehe_raw', 'err_raw', 'pehe_full', 'err_full',
              'pehe_em', 'err_em'):
        m, s, _ = _ms(k)
        print(f'  {k:22s} = {m:8.3f} ± {s:6.3f}')


if __name__ == '__main__':
    main()
