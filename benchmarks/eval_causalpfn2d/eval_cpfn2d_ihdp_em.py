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
def cate_raw_and_em(model, X_train, T_train, Y_train_raw, X_test, edges, J,
                     y_scaling_mode='pooled_std'):
    """Return (cate_raw, cate_em, cate_full) per query, in raw Y units.

    All three share the same forward pass, they only differ in mean recipe:
      - raw:  inner-only marginal, Σ centers · p_marg          (inner mass renormalised to 1)
      - em:   Gaussian-corrected inner marginal mean
      - full: mean under the FULL 9-region mixture (inner + edges + corners)
              using w_reg + tail_scales alongside the inner grid.

    y_scaling_mode must match how the model was trained.
    """
    from full_mixture_mean import full_mixture_mean  # local import to avoid path fuss
    X_ctx = torch.from_numpy(X_train.astype(np.float32)).unsqueeze(0).to(DEVICE)
    t_ctx = torch.from_numpy(T_train.astype(np.float32)).unsqueeze(0).to(DEVICE)
    y_ctx_raw = torch.from_numpy(Y_train_raw.astype(np.float32)).unsqueeze(0).to(DEVICE)
    X_q = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(0).to(DEVICE)

    if y_scaling_mode == 'uwyk_minmax':
        y_lo = y_ctx_raw.amin(dim=1, keepdim=True)
        y_hi = y_ctx_raw.amax(dim=1, keepdim=True)
        y_mean = 0.5 * (y_lo + y_hi)
        y_std  = (0.5 * (y_hi - y_lo)).clamp(min=1e-6)
    else:  # pooled_std
        y_mean = y_ctx_raw.mean(dim=1, keepdim=True)
        y_std  = y_ctx_raw.std(dim=1, keepdim=True).clamp(min=1e-6)
    y_ctx_std = (y_ctx_raw - y_mean) / y_std

    # Full head output (J² + 9 + 4). Keep it — full-mixture mean needs all of it.
    logits_all = model._forward_logits(X_ctx, t_ctx, y_ctx_std, X_q)  # (1, N_q, J²+9+4)
    logits_np  = logits_all.squeeze(0).float().cpu().numpy()          # (N_q, J²+9+4)

    interior = logits_all[..., : J * J]
    p = torch.softmax(interior, dim=-1).reshape(1, -1, J, J)          # (1, N_q, J, J)
    p_y0 = p.sum(dim=-1).squeeze(0).cpu().numpy()                     # (N_q, J)
    p_y1 = p.sum(dim=-2).squeeze(0).cpu().numpy()                     # (N_q, J)

    edges_np = edges.cpu().numpy().astype(np.float64)
    centres = 0.5 * (edges_np[:-1] + edges_np[1:])                    # (J,)

    N_q = p_y0.shape[0]
    y_mean_scalar = float(y_mean.item())
    y_std_scalar  = float(y_std.item())

    # ── raw: inner-only ───────────────────────────────────────────────
    e0_raw = (p_y0 * centres[None, :]).sum(axis=-1)
    e1_raw = (p_y1 * centres[None, :]).sum(axis=-1)

    # ── em: Gaussian-corrected inner marginal ─────────────────────────
    e0_em = np.empty(N_q); e1_em = np.empty(N_q)
    for q in range(N_q):
        mu0_mid, sig0 = _marginal_stats(p_y0[q], edges_np)
        mu1_mid, sig1 = _marginal_stats(p_y1[q], edges_np)
        e0_em[q] = _em_mean_1d(p_y0[q], edges_np, sig0, mu0_mid)
        e1_em[q] = _em_mean_1d(p_y1[q], edges_np, sig1, mu1_mid)

    # ── full: 9-region mixture mean over ℝ² ────────────────────────────
    e0_full, e1_full = full_mixture_mean(logits_np, J, edges_np)

    # ── parab: sub-bin parabolic-peak-interpolation mean ──────────────
    # For a Gaussian-shaped softmax, log-probs around the peak bin are
    # approximately quadratic in bin index. Fitting a parabola through
    # (peak-1, peak, peak+1) log-probs and taking its vertex gives an
    # ~bin_width/10 estimate of the underlying continuous mean — vs the
    # ~bin_width/2 discretization error of center-of-mass. Falls back to
    # raw for edge-bin peaks.
    bw = float(edges_np[1] - edges_np[0])
    def _parab_mean(p_row):
        i = int(np.argmax(p_row))
        if i == 0 or i == len(p_row) - 1:
            return float((centres * p_row).sum())
        L = np.log(np.clip(p_row[i-1:i+2], 1e-45, None))
        denom = L[0] - 2 * L[1] + L[2]
        if abs(denom) < 1e-12:
            return centres[i]
        offset = 0.5 * (L[0] - L[2]) / denom          # in units of bins
        return centres[i] + offset * bw
    e0_parab = np.array([_parab_mean(p_y0[q]) for q in range(N_q)])
    e1_parab = np.array([_parab_mean(p_y1[q]) for q in range(N_q)])

    # Un-standardise (shift cancels in cate).
    cate_raw   = (e1_raw   - e0_raw  ) * y_std_scalar
    cate_em    = (e1_em    - e0_em   ) * y_std_scalar
    cate_full  = (e1_full  - e0_full ) * y_std_scalar
    cate_parab = (e1_parab - e0_parab) * y_std_scalar
    return (cate_raw.astype(np.float32),
            cate_em.astype(np.float32),
            cate_full.astype(np.float32),
            cate_parab.astype(np.float32))


def evaluate(realization: int, model, edges, J, F, y_scaling_mode='pooled_std'):
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

    cate_raw, cate_em, cate_full, cate_parab = cate_raw_and_em(
        model, X_tr_p, T_tr, y_tr, X_te_p, edges, J,
        y_scaling_mode=y_scaling_mode,
    )

    def _pehe_err(cate):
        pehe = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
        ate  = float(cate.mean())
        err  = abs(ate - true_ate) / max(abs(true_ate), 0.1)
        return pehe, err, ate

    pehe_raw,   err_raw,   ate_raw   = _pehe_err(cate_raw)
    pehe_em,    err_em,    ate_em    = _pehe_err(cate_em)
    pehe_full,  err_full,  ate_full  = _pehe_err(cate_full)
    pehe_parab, err_parab, ate_parab = _pehe_err(cate_parab)

    return {
        'dataset': 'IHDP',
        'realization': realization,
        'true_ate': true_ate,
        'pehe_raw':   pehe_raw,   'err_raw':   err_raw,   'ate_raw':   ate_raw,
        'pehe_em':    pehe_em,    'err_em':    err_em,    'ate_em':    ate_em,
        'pehe_full':  pehe_full,  'err_full':  err_full,  'ate_full':  ate_full,
        'pehe_parab': pehe_parab, 'err_parab': err_parab, 'ate_parab': ate_parab,
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
    # Two checkpoint layouts:
    #   (a) custom trainer  → ck['config'] (flat), ck['edges'] (tensor), ck['step']
    #   (b) CausalPFN step-ckpt → ck['model_config'] (nested), edges lives
    #       in ck['model_state_dict']['edges'], step in ck['actual_step']
    if 'config' in ck:
        cfg = ck['config']; edges = ck['edges']
        step = ck.get('step', '?')
    else:
        mc = ck['model_config']
        cfg = dict(mc['model'])
        cfg['y_scaling_mode'] = mc.get('y_scaling_mode', 'pooled_std')
        cfg['loss_type']      = mc.get('loss_type',      'density')
        cfg['hlgauss_sigma']  = float(mc.get('hlgauss_sigma', 0.2))
        edges = ck['model_state_dict']['edges']
        step = ck.get('actual_step', '?')
    y_scaling_mode = cfg.get('y_scaling_mode', 'pooled_std')
    loss_type      = cfg.get('loss_type',      'density')
    hlgauss_sigma  = float(cfg.get('hlgauss_sigma', 0.2))
    print(f'[bootstrap] ckpt={CKPT}')
    print(f'[bootstrap] step={step}  J={cfg["J"]}  num_features={cfg["num_features"]}  '
          f'y_scaling_mode={y_scaling_mode}  loss_type={loss_type}'
          f'{"  σ=" + str(hlgauss_sigma) if loss_type == "hlgauss" else ""}')
    print(f'[bootstrap] edges: [{edges[0].item():.3f}, {edges[-1].item():.3f}]  '
          f'bw={((edges[-1]-edges[0])/cfg["J"]).item():.4f}')

    model = CausalPFN2DHead(
        J=cfg['J'], num_features=cfg['num_features'],
        ninp=cfg['ninp'], nhid=cfg['nhid'], nhead=cfg['nhead'],
        nlayers=cfg['nlayers'], dropout=cfg.get('dropout', 0.0),
        n_out=cfg.get('n_out', 10),
        y_scaling_mode=y_scaling_mode,
        loss_type=loss_type,
        hlgauss_sigma=hlgauss_sigma,
    ).to(DEVICE)
    _load_state_dict_safe(model, ck['model_state_dict'])
    model.eval()

    J = cfg['J']; F = cfg['num_features']
    os.makedirs(OUT, exist_ok=True)
    all_rows = []
    t0 = time.time()
    for r in range(100):
        row = evaluate(r, model, edges, J, F, y_scaling_mode=y_scaling_mode)
        np.savez(os.path.join(OUT, f'r{r:03d}.npz'), **{k: np.array(v) for k, v in row.items()})
        all_rows.append(row)
        print(
            f'r={r:03d}  '
            f'raw:  pehe={row["pehe_raw"]:6.3f} err={row["err_raw"]:5.3f}  |  '
            f'em:   pehe={row["pehe_em"]:6.3f} err={row["err_em"]:5.3f}  |  '
            f'full: pehe={row["pehe_full"]:6.3f} err={row["err_full"]:5.3f}  |  '
            f'parab: pehe={row["pehe_parab"]:6.3f} err={row["err_parab"]:5.3f}  '
            f'(true_ate={row["true_ate"]:+5.2f}, {time.time()-t0:.0f}s)',
            flush=True,
        )

    def _ms(k):
        v = np.array([r[k] for r in all_rows]); return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

    print(f'\n══ IHDP summary (n={len(all_rows)}, step={step}) ══')
    for k in ('pehe_raw', 'err_raw', 'pehe_em', 'err_em',
              'pehe_full', 'err_full', 'pehe_parab', 'err_parab'):
        m, s = _ms(k)
        print(f'  {k:12s} = {m:8.3f} ± {s:6.3f}')


if __name__ == '__main__':
    main()
