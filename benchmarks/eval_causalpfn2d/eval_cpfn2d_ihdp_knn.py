"""IHDP cpfn2d eval with k-NN context retrieval — per-query.

For each test query x_q:
  1. Compute L2 distance to every training-context row on standardised X.
  2. Take the K nearest as the LOCAL context for this query.
  3. Standardise Y using only the LOCAL context's stats (much tighter than global).
  4. Forward the 2D model with (LOCAL_context, single query).
  5. Marginalise the joint p_mat, take raw center-of-mass and full-mixture mean.
  6. Un-standardise per query with the LOCAL Y scale.

Purpose: shrink the local Y std so the model's marginal bins cover a tighter
raw-Y range, improving effective resolution. This is the trick CausalPFN's
paper table uses to publish PEHE 0.58 on IHDP.

Env vars:
  CKPT              (required) cpfn2d checkpoint
  OUT               (required) output dir
  CAUSALPFN         (required) external/causalpfn root
  K_NN              default 200  — nearest-neighbour count per query
  MAX_REAL          default '' — cap on realizations for smoke tests

Output schema matches eval_cpfn2d_ihdp_em.py so aggregators can share code.
Reports pehe_raw / pehe_full only (no em variant to keep this focused).
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch


CKPT      = os.environ['CKPT']
OUT       = os.environ['OUT']
CAUSALPFN = os.environ['CAUSALPFN']
K_NN      = int(os.environ.get('K_NN', '200'))
MAX_REAL  = os.environ.get('MAX_REAL', '')

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, CAUSALPFN)
sys.path.insert(0, CAUSALPFN + '/src')

# Ensure full_mixture_mean is importable from this dir
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path: sys.path.insert(0, _here)

from benchmarks import IHDPDataset  # noqa: E402
from training_causalpfn2d.model_causalpfn_2d import CausalPFN2DHead  # noqa: E402
from full_mixture_mean import full_mixture_mean  # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _standardize_train_test(X_tr, X_te, eps=1e-8):
    mu = X_tr.mean(0, keepdims=True); sd = X_tr.std(0, keepdims=True) + eps
    return (X_tr - mu) / sd, (X_te - mu) / sd


def _pad_features(X, F):
    if X.shape[1] == F: return X
    if X.shape[1] > F:  return X[:, :F]
    return np.hstack([X, np.zeros((X.shape[0], F - X.shape[1]), dtype=X.dtype)])


def _load_state_dict_safe(model, sd):
    if any('_orig_mod.' in k for k in sd):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    ref = model.state_dict()
    kept = {k: v for k, v in sd.items() if k in ref and ref[k].shape == v.shape}
    model.load_state_dict(kept, strict=False)


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    # Two ckpt layouts (custom trainer vs CausalPFN step-ckpt)
    if 'config' in ck:
        cfg = ck['config']; edges = ck['edges']
    else:
        mc = ck['model_config']
        cfg = dict(mc['model'])
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
    _load_state_dict_safe(model, ck['model_state_dict'])
    model.eval()
    return model, cfg, edges


@torch.no_grad()
def _forward_one_query(model, X_ctx_local, T_ctx_local, y_ctx_raw_local, x_q, J, edges_np):
    """Forward with (N_local ctx, 1 query). Return (p_mat, logits) both on device."""
    # Pool y stats from LOCAL ctx (that's the whole point of k-NN retrieval)
    y_mean = float(y_ctx_raw_local.mean())
    y_std  = float(max(y_ctx_raw_local.std(), 1e-6))
    y_ctx_std = ((y_ctx_raw_local - y_mean) / y_std).astype(np.float32)

    X_ctx_t = torch.from_numpy(X_ctx_local.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_ctx_t = torch.from_numpy(T_ctx_local.astype(np.float32)).unsqueeze(0).to(DEVICE)
    Y_ctx_t = torch.from_numpy(y_ctx_std).unsqueeze(0).to(DEVICE)          # (1, K)
    X_q_t   = torch.from_numpy(x_q.astype(np.float32)).reshape(1, 1, -1).to(DEVICE)

    logits = model._forward_logits(X_ctx_t, T_ctx_t, Y_ctx_t, X_q_t)       # (1, 1, J²+9+4)
    return logits.squeeze(0).float().cpu().numpy(), y_mean, y_std


def cate_knn(model, X_tr_s, T_tr, y_tr_raw, X_te_s, J, edges_np, k):
    """Per-query k-NN retrieval + raw + full-mixture CATE, in raw Y units."""
    N_q = X_te_s.shape[0]
    centers = 0.5 * (edges_np[:-1] + edges_np[1:])

    cate_raw  = np.empty(N_q, dtype=np.float32)
    cate_full = np.empty(N_q, dtype=np.float32)
    for q in range(N_q):
        # k-NN by L2 on standardised X
        d = ((X_tr_s - X_te_s[q:q+1]) ** 2).sum(axis=1)
        idx = np.argpartition(d, min(k - 1, len(d) - 1))[:k]
        X_ctx_local  = X_tr_s[idx]
        T_ctx_local  = T_tr[idx].reshape(-1)
        y_ctx_local  = y_tr_raw[idx]

        logits_q, y_mean_q, y_std_q = _forward_one_query(
            model, X_ctx_local, T_ctx_local, y_ctx_local, X_te_s[q], J, edges_np,
        )                                                # (1, J²+9+4)

        # raw: inner-only center-of-mass on marginals
        interior = logits_q[..., : J * J]
        p_max = interior.max(axis=-1, keepdims=True)
        p_mat = np.exp(interior - p_max)
        p_mat = p_mat / p_mat.sum(axis=-1, keepdims=True)
        p_mat = p_mat.reshape(1, J, J)                    # (1, J, J)
        p_y0 = p_mat.sum(axis=-1); p_y0 /= p_y0.sum(axis=-1, keepdims=True)
        p_y1 = p_mat.sum(axis=-2); p_y1 /= p_y1.sum(axis=-1, keepdims=True)
        e0_raw = float((centers * p_y0[0]).sum())
        e1_raw = float((centers * p_y1[0]).sum())

        # full 9-region mixture mean
        e0_full, e1_full = full_mixture_mean(logits_q, J, edges_np)
        e0_full = float(e0_full[0]); e1_full = float(e1_full[0])

        # Un-standardise with LOCAL stats
        cate_raw[q]  = (e1_raw  - e0_raw ) * y_std_q
        cate_full[q] = (e1_full - e0_full) * y_std_q

    return cate_raw, cate_full


def evaluate(realization, model, edges, J, F):
    ds = IHDPDataset()
    cate_ds = ds[realization][0]
    X_tr = cate_ds.X_train.astype(np.float32)
    T_tr = cate_ds.t_train.astype(np.float32).reshape(-1)
    y_tr = cate_ds.y_train.astype(np.float32).reshape(-1)
    X_te = cate_ds.X_test.astype(np.float32)
    true_cate = cate_ds.true_cate.astype(np.float32)
    true_ate  = float(true_cate.mean())

    X_tr_s, X_te_s = _standardize_train_test(X_tr, X_te)
    X_tr_p = _pad_features(X_tr_s, F)
    X_te_p = _pad_features(X_te_s, F)

    edges_np = edges.detach().cpu().numpy().astype(np.float64) if hasattr(edges, 'detach') else np.asarray(edges, dtype=np.float64)

    # K = min(K_NN, N_train) — can't retrieve more than we have
    k = min(K_NN, X_tr_p.shape[0])

    cate_raw, cate_full = cate_knn(model, X_tr_p, T_tr, y_tr, X_te_p, J, edges_np, k)

    def _pehe_err(cate):
        pehe = float(np.sqrt(np.nanmean((cate - true_cate) ** 2)))
        ate  = float(np.nanmean(cate))
        err  = abs(ate - true_ate) / max(abs(true_ate), 0.1)
        return pehe, err, ate

    p_raw, e_raw, a_raw = _pehe_err(cate_raw)
    p_full, e_full, a_full = _pehe_err(cate_full)
    return {
        'dataset': 'IHDP', 'realization': realization, 'true_ate': true_ate,
        'K_NN': k,
        'pehe_raw':  p_raw,  'err_raw':  e_raw,  'ate_raw':  a_raw,
        'pehe_full': p_full, 'err_full': e_full, 'ate_full': a_full,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f'[bootstrap] ckpt={CKPT}  K_NN={K_NN}', flush=True)
    model, cfg, edges = load_model(CKPT)
    J = cfg['J']; F = cfg['num_features']
    ds = IHDPDataset()
    n = ds.n_tables if not MAX_REAL else min(ds.n_tables, int(MAX_REAL))
    print(f'[bootstrap] IHDP n={n}  J={J}  F={F}', flush=True)

    rows = []
    t0 = time.time()
    for r in range(n):
        row = evaluate(r, model, edges, J, F)
        rows.append(row)
        np.savez(os.path.join(OUT, f'IHDP_r{r:03d}.npz'),
                 **{k: np.array(v) for k, v in row.items()})
        print(f'r={r:03d}  K={row["K_NN"]:>3}  raw: pehe={row["pehe_raw"]:6.3f} err={row["err_raw"]:5.3f}  |  '
              f'full: pehe={row["pehe_full"]:6.3f} err={row["err_full"]:5.3f}  '
              f'(true_ate={row["true_ate"]:+6.3f}, {time.time()-t0:.0f}s)', flush=True)

    def _ms(k):
        v = np.array([r[k] for r in rows if np.isfinite(r[k])])
        return (v.mean(), v.std(ddof=1) / np.sqrt(max(len(v), 1)), int(v.size)) if v.size \
            else (float('nan'), float('nan'), 0)

    print(f'\n══ IHDP summary  (n={len(rows)}, K_NN={K_NN}) ══')
    for k in ('pehe_raw', 'err_raw', 'pehe_full', 'err_full'):
        m, s, _ = _ms(k)
        print(f'  {k:12s} = {m:8.3f} ± {s:6.3f}')


if __name__ == '__main__':
    main()
