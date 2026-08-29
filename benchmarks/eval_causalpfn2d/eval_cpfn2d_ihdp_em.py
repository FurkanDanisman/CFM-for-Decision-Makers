"""Evaluate cpfn2d on IHDP with BOTH raw-mean and MALC EM-mean CATE.

Both methods take the same input: the K*K interior softmax reshaped to
(J, J), marginalised along each axis into 1D bin probabilities.

  raw mean:  E[Y] = sum_j centre[j] * p[j]
  EM mean:   iterative Gaussian-bin correction from MALC._em_mean_2d
             — fixed-point on mu using pdf/cdf-diff ratios; assumes each
             bin's mass is Gaussian around the true mean.

Un-standardised back to raw Y with per-task pooled (y_mean, y_std).

Env vars: CKPT, OUT (per-realization NPZ dir), CAUSALPFN
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch
from scipy.stats import norm


CKPT      = os.environ['CKPT']
OUT       = os.environ.get('OUT', './results_cpfn2d_ihdp_em')
CAUSALPFN = os.environ['CAUSALPFN']

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, CAUSALPFN)
sys.path.insert(0, CAUSALPFN + '/src')

from benchmarks import IHDPDataset  # noqa: E402
from training_causalpfn2d.model_causalpfn_2d import CausalPFN2DHead  # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _em_mean_1d(
    props: np.ndarray,
    grid: np.ndarray,
    sigma: float,
    start: float,
    max_step: int = 1000,
    eps2: float = 1e-10,
    eps1: float = 1e-5,
) -> float:
    """Copy of MALC/malc_2d.py::_em_mean_2d — a 1D EM-corrected marginal mean.

    Fixed-point iteration for the Gaussian-bin-corrected mean, given bin
    proportions and grid edges. Not a 2D function — the '2d' in the source
    name refers to being a helper of MALC_2D, not to dimensionality.
    """
    pn = props / props.sum()
    mu = start
    for _ in range(max_step):
        a = (grid - mu) / sigma
        G1 = norm.cdf(a)
        G2 = norm.pdf(a)
        temp = (np.diff(G2) + eps2) / (np.diff(G1) + eps2)
        mu_new = mu - sigma * float(np.sum(pn * temp))
        if abs(mu_new - mu) < eps1:
            return float(mu_new)
        mu = mu_new
    return float(mu)


def _marginal_stats(p: np.ndarray, grid: np.ndarray):
    """Given 1D bin probs and (J+1,) edges, return (mu_mid, sigma) used
    to seed the EM iteration. Matches _fit_component_2d's derivation
    (malc_2d.py:104-114)."""
    delta = grid[1] - grid[0]
    centres = 0.5 * (grid[:-1] + grid[1:])
    mu_low = float(np.sum(p * grid[:-1]))
    mu_mid = 0.5 * (mu_low + float(np.sum(p * grid[1:])))
    sigma = float(np.sqrt(np.sum(p * (centres - mu_mid) ** 2) + delta**2 / 12.0))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = delta
    return mu_mid, sigma


def _standardize_train_test(X_tr, X_te):
    mu = X_tr.mean(0, keepdims=True); sd = X_tr.std(0, keepdims=True) + 1e-8
    return (X_tr - mu) / sd, (X_te - mu) / sd


def _pad_features(X: np.ndarray, F: int) -> np.ndarray:
    if X.shape[1] == F: return X
    if X.shape[1] > F: return X[:, :F]
    return np.hstack([X, np.zeros((X.shape[0], F - X.shape[1]), dtype=X.dtype)])


@torch.no_grad()
def cate_raw_and_em(model, X_train, T_train, Y_train_raw, X_test, edges, J):
    """Return (cate_raw, cate_em) per query, in raw Y units.

    Both share the same forward + interior marginalisation; they only
    differ in how the 1D expected value is derived from p_y0 / p_y1.
    """
    X_ctx = torch.from_numpy(X_train.astype(np.float32)).unsqueeze(0).to(DEVICE)
    t_ctx = torch.from_numpy(T_train.astype(np.float32)).unsqueeze(0).to(DEVICE)
    y_ctx_raw = torch.from_numpy(Y_train_raw.astype(np.float32)).unsqueeze(0).to(DEVICE)
    X_q = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(0).to(DEVICE)

    y_mean = y_ctx_raw.mean(dim=1, keepdim=True)
    y_std  = y_ctx_raw.std(dim=1, keepdim=True).clamp(min=1e-6)
    y_ctx_std = (y_ctx_raw - y_mean) / y_std

    logits = model._forward_logits(X_ctx, t_ctx, y_ctx_std, X_q)
    interior = logits[..., : J * J]
    p = torch.softmax(interior, dim=-1).reshape(1, -1, J, J)      # (1, N_q, J, J)

    # 1D marginals
    p_y0 = p.sum(dim=-1).squeeze(0).cpu().numpy()                 # (N_q, J)
    p_y1 = p.sum(dim=-2).squeeze(0).cpu().numpy()                 # (N_q, J)

    edges_np = edges.cpu().numpy().astype(np.float64)
    centres = 0.5 * (edges_np[:-1] + edges_np[1:])                # (J,)

    N_q = p_y0.shape[0]
    y_mean_scalar = float(y_mean.item())
    y_std_scalar  = float(y_std.item())

    e0_raw = (p_y0 * centres[None, :]).sum(axis=-1)               # (N_q,)
    e1_raw = (p_y1 * centres[None, :]).sum(axis=-1)

    e0_em = np.empty(N_q); e1_em = np.empty(N_q)
    for q in range(N_q):
        mu0_mid, sig0 = _marginal_stats(p_y0[q], edges_np)
        mu1_mid, sig1 = _marginal_stats(p_y1[q], edges_np)
        e0_em[q] = _em_mean_1d(p_y0[q], edges_np, sig0, mu0_mid)
        e1_em[q] = _em_mean_1d(p_y1[q], edges_np, sig1, mu1_mid)

    # Un-standardise both back to raw Y units.
    cate_raw = (e1_raw - e0_raw) * y_std_scalar                   # (N_q,)  (mean cancels)
    cate_em  = (e1_em  - e0_em ) * y_std_scalar
    return cate_raw.astype(np.float32), cate_em.astype(np.float32)


def evaluate(realization: int, model, edges, J, F):
    ds = IHDPDataset()
    cate_ds = ds[realization][0]
    X_tr = cate_ds.X_train.astype(np.float32)
    T_tr = cate_ds.t_train.astype(np.float32).reshape(-1)
    y_tr = cate_ds.y_train.astype(np.float32).reshape(-1)
    X_te = cate_ds.X_test.astype(np.float32)
    true_cate = cate_ds.true_cate.astype(np.float32)
    true_ate = float(true_cate.mean())

    X_tr_std, X_te_std = _standardize_train_test(X_tr, X_te)
    X_tr_p = _pad_features(X_tr_std, F)
    X_te_p = _pad_features(X_te_std, F)

    cate_raw, cate_em = cate_raw_and_em(model, X_tr_p, T_tr, y_tr, X_te_p, edges, J)

    def _pehe_err(cate):
        pehe = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
        ate  = float(cate.mean())
        err  = abs(ate - true_ate) / max(abs(true_ate), 1e-9)
        return pehe, err, ate

    pehe_raw, err_raw, ate_raw = _pehe_err(cate_raw)
    pehe_em,  err_em,  ate_em  = _pehe_err(cate_em)

    return {
        'dataset': 'IHDP',
        'realization': realization,
        'true_ate': true_ate,
        'pehe_raw':  pehe_raw,  'err_raw':  err_raw,  'ate_raw':  ate_raw,
        'pehe_em':   pehe_em,   'err_em':   err_em,   'ate_em':   ate_em,
    }


def _load_state_dict_safe(model, sd):
    if any('_orig_mod.' in k for k in sd):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        print('[eval] stripped _orig_mod. (global) from state_dict keys', flush=True)
    ref = model.state_dict()
    kept = {k: v for k, v in sd.items() if k in ref and ref[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(kept, strict=False)
    print(f'[eval] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}  '
          f'loaded={len(kept)}/{len(ref)}', flush=True)
    if len(missing) > 20:
        raise RuntimeError(
            f'[eval] ABORT: {len(missing)} missing keys — refusing to eval random-init model.'
        )


def main():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    cfg = ck['config']; edges = ck['edges']; step = ck.get('step', '?')
    print(f'[bootstrap] ckpt={CKPT}')
    print(f'[bootstrap] step={step}  J={cfg["J"]}  num_features={cfg["num_features"]}')
    print(f'[bootstrap] edges: [{edges[0].item():.3f}, {edges[-1].item():.3f}]  '
          f'bw={((edges[-1]-edges[0])/cfg["J"]).item():.4f}')

    model = CausalPFN2DHead(
        J=cfg['J'], num_features=cfg['num_features'],
        ninp=cfg['ninp'], nhid=cfg['nhid'], nhead=cfg['nhead'],
        nlayers=cfg['nlayers'], dropout=cfg.get('dropout', 0.0),
        n_out=cfg.get('n_out', 10),
    ).to(DEVICE)
    _load_state_dict_safe(model, ck['model_state_dict'])
    model.eval()

    J = cfg['J']; F = cfg['num_features']
    os.makedirs(OUT, exist_ok=True)
    all_rows = []
    t0 = time.time()
    for r in range(100):
        row = evaluate(r, model, edges, J, F)
        np.savez(os.path.join(OUT, f'r{r:03d}.npz'), **{k: np.array(v) for k, v in row.items()})
        all_rows.append(row)
        print(
            f'r={r:03d}  '
            f'raw: pehe={row["pehe_raw"]:6.3f} err={row["err_raw"]:5.3f}  |  '
            f'em:  pehe={row["pehe_em"]:6.3f} err={row["err_em"]:5.3f}  '
            f'(true_ate={row["true_ate"]:+5.2f}, {time.time()-t0:.0f}s)',
            flush=True,
        )

    def _ms(k):
        v = np.array([r[k] for r in all_rows]); return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

    print(f'\n══ IHDP summary (n={len(all_rows)}, step={step}) ══')
    for k in ('pehe_raw', 'err_raw', 'pehe_em', 'err_em'):
        m, s = _ms(k)
        print(f'  {k:12s} = {m:8.3f} ± {s:6.3f}')


if __name__ == '__main__':
    main()
