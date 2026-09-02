"""IHDP-only diagnostic: 4 CATE estimators per realization × 2 anc modes (v6a, noanc).

Identical pipeline to eval_graph2d_malc_mean_acic.py but on IHDP with the
current v6a anc encoding (no +1 edges; all -1s from unconfoundedness +
diagonal -1).

Estimators per (realization, mode):
  - raw:      E[Y] = Σ centres · p_marg
  - em:       fixed-point Gaussian correction on p_marg
  - malc_1d:  fit 1D log-concave MLE to each marginal, mean of smoothed
  - malc_2d:  fit 2D MALC to p_mat, diagonal-integrate → p(τ), mean of p(τ)

Env vars:
    CKPT              (required) graph2d checkpoint
    OUT               (required) output dir
    UWYK              (required) UWYK repo root
    CAUSALPFN         (required) CausalPFN repo root
    MALC_B            default 1000
    MALC_MAX_K        default 3
    MALC_N_EVAL       default 101
    EVAL_MAX_CONTEXT  default '' — context subsample cap
    EVAL_CONTEXT_SEED default '42' — context subsample seed
    MAX_REAL          default '' — cap on n_tables (for smoke tests)
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch


CKPT      = os.environ['CKPT']
OUT       = os.environ['OUT']
UWYK      = os.environ['UWYK']
CAUSALPFN = os.environ['CAUSALPFN']
MALC_B      = int(os.environ.get('MALC_B', '1000'))
MALC_MAX_K  = int(os.environ.get('MALC_MAX_K', '3'))
MALC_N_EVAL = int(os.environ.get('MALC_N_EVAL', '101'))
EVAL_MAX_CONTEXT  = os.environ.get('EVAL_MAX_CONTEXT', '')
EVAL_CONTEXT_SEED = int(os.environ.get('EVAL_CONTEXT_SEED', '42'))
MAX_REAL          = os.environ.get('MAX_REAL', '')

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MALC_DIR = os.path.join(REPO_SRC, 'MALC')
L2_IHDP_DIR = os.path.join(REPO_SRC, 'benchmarks', 'l2_ihdp')
for p in (REPO_SRC, MALC_DIR, L2_IHDP_DIR, UWYK, UWYK + '/src', CAUSALPFN, CAUSALPFN + '/src'):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from benchmarks import IHDPDataset  # noqa: E402  (resolves to CausalPFN's benchmarks)
from training_graph2d.model_graph_2d import GraphConditioned2DHead  # noqa: E402
from losses.BarDistribution2D import fit_malc_inner  # noqa: E402
from malc_2d import dmalc_2d  # noqa: E402
from methods_densities import malc_1d_cvxpy  # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Anc matrix builders ─────────────────────────────────────────────────
def _padded_neg1_only(F, n_real):
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    for i in range(n_real, F):
        A[2 + i, :] = -1.0
        A[:, 2 + i] = -1.0
        A[2 + i, 2 + i] = -1.0
    return A


def build_anc_v6a(F, n_real):
    """v6a: no +1 edges. -1 on diagonal, Y→T, Y→X_i, T→X_i."""
    A = _padded_neg1_only(F, n_real)
    real_n = 2 + n_real
    for i in range(real_n):
        A[i, i] = -1.0
    A[1, 0] = -1.0
    for i in range(n_real):
        A[1, 2 + i] = -1.0
        A[0, 2 + i] = -1.0
    return A


def build_anc_noanc(F, n_real):
    """No unconfoundedness knowledge; padded rows/cols still -1."""
    return _padded_neg1_only(F, n_real)


# ── Preprocessing ────────────────────────────────────────────────────────
def _pad_features(X, F):
    if X.shape[1] >= F:
        return X[:, :F]
    pad = np.zeros((X.shape[0], F - X.shape[1]), dtype=X.dtype)
    return np.hstack([X, pad])


def _standardize_train_test(X_train, X_test, eps=1e-8):
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, eps, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd


def _scale_y(y):
    ymin = float(y.min()); ymax = float(y.max())
    yr = max(ymax - ymin, 1e-9)
    return (2.0 * (y - ymin) / yr - 1.0).astype(np.float32), ymin, yr


def _marginal_stats(p, edges):
    centers = 0.5 * (edges[:-1] + edges[1:])
    mu = float((p * centers).sum())
    var = float((p * (centers - mu) ** 2).sum())
    return mu, max(np.sqrt(var), 1e-6)


def _em_mean_1d(p, edges, sigma, mu_init, n_iter=20):
    mu = mu_init
    centers = 0.5 * (edges[:-1] + edges[1:])
    for _ in range(n_iter):
        w = np.exp(-0.5 * ((centers - mu) / sigma) ** 2)
        wp = w * p
        s = wp.sum()
        if s <= 0:
            break
        mu = float((wp * centers).sum() / s)
    return mu


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ck['config']
    model = GraphConditioned2DHead(
        num_features=cfg['num_features'],
        d_model=cfg['d_model'], depth=cfg['depth'],
        heads_feat=cfg['heads'], heads_samp=cfg['heads'],
        dropout=0.0, hidden_mult=cfg['hidden_mult'],
        normalize_features=True,
        n_sample_attention_sink_rows=10,
        n_feature_attention_sink_cols=0,
        J=cfg['J'],
    ).to(DEVICE)
    sd = ck['model_state_dict']
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, cfg


def forward_pmat(model, X_train, T_train, Y_train_scaled, X_test, adj, J):
    X_obs = torch.from_numpy(X_train.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_obs = torch.from_numpy(T_train.astype(np.float32)).reshape(1, -1, 1).to(DEVICE)
    Y_obs = torch.from_numpy(Y_train_scaled).unsqueeze(0).to(DEVICE)
    X_intv = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(0).to(DEVICE)
    adj_t = torch.from_numpy(adj).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(X_obs, T_obs, Y_obs, X_intv, adj_t)
    logits = out['predictions'] if isinstance(out, dict) else out
    interior = logits[..., : J * J]
    p_mat = torch.softmax(interior, dim=-1).reshape(1, -1, J, J).squeeze(0).cpu().numpy()
    return p_mat.astype(np.float64)


def cate_estimators(p_mat, J, edges_scaled):
    """(N_q, J, J) → 4 CATE arrays each length N_q (on scaled y)."""
    N_q = p_mat.shape[0]
    centres = 0.5 * (edges_scaled[:-1] + edges_scaled[1:])

    p_y0_all = p_mat.sum(axis=2)
    p_y1_all = p_mat.sum(axis=1)
    p_y0_all /= np.clip(p_y0_all.sum(axis=1, keepdims=True), 1e-12, None)
    p_y1_all /= np.clip(p_y1_all.sum(axis=1, keepdims=True), 1e-12, None)

    cate_raw = (p_y1_all * centres).sum(axis=1) - (p_y0_all * centres).sum(axis=1)

    cate_em = np.empty(N_q); cate_malc1d = np.empty(N_q)
    for q in range(N_q):
        mu0, s0 = _marginal_stats(p_y0_all[q], edges_scaled)
        mu1, s1 = _marginal_stats(p_y1_all[q], edges_scaled)
        e0 = _em_mean_1d(p_y0_all[q], edges_scaled, s0, mu0)
        e1 = _em_mean_1d(p_y1_all[q], edges_scaled, s1, mu1)
        cate_em[q] = e1 - e0
        p0_malc = malc_1d_cvxpy(p_y0_all[q])
        p1_malc = malc_1d_cvxpy(p_y1_all[q])
        e0_m = float((p0_malc * centres).sum())
        e1_m = float((p1_malc * centres).sum())
        cate_malc1d[q] = e1_m - e0_m

    xs = np.linspace(edges_scaled[0], edges_scaled[-1], MALC_N_EVAL)
    ys = np.linspace(edges_scaled[0], edges_scaled[-1], MALC_N_EVAL)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    eval_pts = np.column_stack([XX.ravel(), YY.ravel()])
    dy0 = xs[1] - xs[0]
    tau_grid = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], 401)
    dtau = tau_grid[1] - tau_grid[0]

    cate_malc2d = np.empty(N_q)
    for q in range(N_q):
        try:
            fit = fit_malc_inner(
                p_mat[q].T, edges_scaled, edges_scaled,
                B_fit=MALC_B, B_select=MALC_B, max_K=MALC_MAX_K,
                seed=(q * 31 + 17) % (10 ** 8), parallel=False,
            )
            density = dmalc_2d(fit, eval_pts).reshape(len(xs), len(ys))
            p_tau = np.zeros_like(tau_grid)
            for k, t in enumerate(tau_grid):
                y1 = xs + t
                v = (y1 >= ys[0]) & (y1 <= ys[-1])
                if not np.any(v):
                    continue
                col = np.clip(np.searchsorted(xs, xs[v]), 0, len(xs) - 1)
                rf = (y1[v] - ys[0]) / (ys[1] - ys[0])
                rlo = np.clip(np.floor(rf).astype(int), 0, len(ys) - 2)
                rhi = rlo + 1
                whi = rf - rlo; wlo = 1.0 - whi
                f = wlo * density[rlo, col] + whi * density[rhi, col]
                p_tau[k] = float(f.sum()) * dy0
            s = p_tau.sum() * dtau
            if s > 0:
                p_tau /= s
                cate_malc2d[q] = float((p_tau * tau_grid).sum() * dtau)
            else:
                cate_malc2d[q] = float('nan')
        except Exception as e:
            print(f'  [MALC2D fail r={q}: {e}]', flush=True)
            cate_malc2d[q] = float('nan')

    return cate_raw, cate_em, cate_malc1d, cate_malc2d


def evaluate_realization(r, ds, model, J, F):
    cate_ds = ds[r][0]
    X_tr_raw = np.asarray(cate_ds.X_train, dtype=np.float32)
    T_tr     = np.asarray(cate_ds.t_train, dtype=np.float32).reshape(-1)
    y_tr_raw = np.asarray(cate_ds.y_train, dtype=np.float32).reshape(-1)
    X_te_raw = np.asarray(cate_ds.X_test,  dtype=np.float32)
    true_cate = np.asarray(cate_ds.true_cate, dtype=np.float32)
    true_ate  = float(true_cate.mean())

    if EVAL_MAX_CONTEXT:
        cap = int(EVAL_MAX_CONTEXT)
        if X_tr_raw.shape[0] > cap:
            rng = np.random.default_rng(EVAL_CONTEXT_SEED + r)
            idx = rng.choice(X_tr_raw.shape[0], cap, replace=False)
            X_tr_raw = X_tr_raw[idx]; T_tr = T_tr[idx]; y_tr_raw = y_tr_raw[idx]

    n_real = min(X_tr_raw.shape[1], F)
    X_tr_std, X_te_std = _standardize_train_test(X_tr_raw, X_te_raw)
    X_tr = _pad_features(X_tr_std, F)
    X_te = _pad_features(X_te_std, F)
    y_scaled, ymin, yrange = _scale_y(y_tr_raw)
    Y_obs = y_scaled.reshape(-1, 1)
    edges_scaled = np.linspace(-1.0, 1.0, J + 1)

    row = {'realization': r, 'true_ate': true_ate}
    for mode, adj in (('v6a',   build_anc_v6a(F, n_real)),
                      ('noanc', build_anc_noanc(F, n_real))):
        p_mat = forward_pmat(model, X_tr, T_tr, Y_obs, X_te, adj, J)
        c_raw, c_em, c_malc1d, c_malc2d = cate_estimators(p_mat, J, edges_scaled)
        for name, cate_scaled in (('raw', c_raw), ('em', c_em),
                                    ('malc1d', c_malc1d), ('malc2d', c_malc2d)):
            cate = cate_scaled * yrange / 2.0
            pehe = float(np.sqrt(np.nanmean((cate - true_cate) ** 2)))
            ate_hat = float(np.nanmean(cate))
            err = abs(ate_hat - true_ate) / max(abs(true_ate), 1e-9)
            row[f'pehe_{name}_{mode}'] = pehe
            row[f'err_{name}_{mode}']  = err
    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f'[bootstrap] ckpt={CKPT}  out={OUT}  MALC_B={MALC_B}  seed={EVAL_CONTEXT_SEED}', flush=True)
    model, cfg = load_model(CKPT)
    J = cfg['J']; F = cfg['num_features']
    ds = IHDPDataset()
    n = ds.n_tables if not MAX_REAL else min(ds.n_tables, int(MAX_REAL))
    print(f'[bootstrap] IHDP n_tables={ds.n_tables}  running={n}  J={J}  F={F}', flush=True)
    rows = []
    t0 = time.time()
    for r in range(n):
        row = evaluate_realization(r, ds, model, J, F)
        rows.append(row)
        np.savez(os.path.join(OUT, f'IHDP_r{r:03d}.npz'),
                 **{k: np.array(v) for k, v in row.items()})
        parts = []
        for est in ('raw', 'em', 'malc1d', 'malc2d'):
            for mode in ('v6a', 'noanc'):
                parts.append(f'{est[:6]}-{mode[:5]}={row[f"pehe_{est}_{mode}"]:6.3f}')
        print(f'r={r:03d}  ' + '  '.join(parts) + f'  ({time.time()-t0:.0f}s)', flush=True)

    def ms(k):
        v = np.array([r[k] for r in rows if np.isfinite(r[k])])
        return (float(v.mean()), float(v.std(ddof=1) / np.sqrt(max(len(v), 1))),
                int(v.size)) if v.size else (float('nan'), float('nan'), 0)

    print(f'\n══ IHDP summary  (n={len(rows)}, MALC_B={MALC_B}, seed={EVAL_CONTEXT_SEED}) ══')
    print(f'{"est":<8} {"mode":<6} {"pehe":>9} {"err":>7} {"n":>3}')
    for est in ('raw', 'em', 'malc1d', 'malc2d'):
        for mode in ('v6a', 'noanc'):
            m_pehe, se_pehe, n_ = ms(f'pehe_{est}_{mode}')
            m_err,  se_err,  _  = ms(f'err_{est}_{mode}')
            print(f'{est:<8} {mode:<6} {m_pehe:>9.3f} {m_err:>7.3f} {n_:>3d}')


if __name__ == '__main__':
    main()
