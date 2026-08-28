"""Diagnostic: load a cpfn2d ckpt, run one IHDP realization, inspect the raw
model output. Are the two marginals actually different? Is mass in tails?

Prints:
  - Region mixture weights (softmax of the 9 region logits)
  - Per-query interior marginal means (standardised space + raw)
  - Difference |E[Y1] - E[Y0]| distribution over queries
  - A few sample marginal histograms (as ASCII sparklines)

Env vars: CKPT, CAUSALPFN, REAL (default 0)
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

CKPT      = os.environ['CKPT']
CAUSALPFN = os.environ['CAUSALPFN']
REAL      = int(os.environ.get('REAL', '0'))

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, CAUSALPFN)
sys.path.insert(0, CAUSALPFN + '/src')

from benchmarks import IHDPDataset  # noqa: E402
from training_causalpfn2d.model_causalpfn_2d import CausalPFN2DHead  # noqa: E402
from losses.BarDistribution2D import (  # noqa: E402
    total_params, N_REGIONS, N_TAIL_PARAMS, R_INNER,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_sd(model, sd):
    # Global replace — the trainer compiled only the backbone, so keys look
    # like `backbone._orig_mod.encoder.weight` with the prefix MID-PATH.
    # Prior startswith-based strip silently dropped 148/149 backbone keys.
    if any('_orig_mod.' in k for k in sd):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        print('[load] stripped _orig_mod. (global) from state_dict keys', flush=True)
    ref = model.state_dict()
    kept = {k: v for k, v in sd.items() if k in ref and ref[k].shape == v.shape}
    m, u = model.load_state_dict(kept, strict=False)
    print(f'[load] missing={len(m)} unexpected={len(u)}  loaded={len(kept)}/{len(ref)}', flush=True)
    if len(m) > 20:
        raise RuntimeError(
            f'[load] ABORT: {len(m)} missing keys after load — refusing to '
            f'diagnose a random-init model. First missing: {list(m)[:8]}'
        )


def _std_train_test(Xtr, Xte):
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _pad(X, F_dim):
    if X.shape[1] == F_dim: return X
    if X.shape[1] > F_dim: return X[:, :F_dim]
    return np.hstack([X, np.zeros((X.shape[0], F_dim - X.shape[1]), dtype=X.dtype)])


def sparkline(v, w=40):
    """ASCII sparkline for a 1D histogram."""
    v = np.asarray(v)
    v = v / max(v.max(), 1e-9)
    idx = np.linspace(0, len(v) - 1, w).astype(int)
    bars = ' ▁▂▃▄▅▆▇█'
    return ''.join(bars[min(int(x * 8), 8)] for x in v[idx])


@torch.no_grad()
def main():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    cfg = ck['config']; edges = ck['edges']; step = ck.get('step', '?')
    print(f'[bootstrap] ckpt={CKPT}  step={step}')
    print(f'[bootstrap] J={cfg["J"]}  num_features={cfg["num_features"]}')
    print(f'[bootstrap] edges std-space: [{edges[0]:.3f}, {edges[-1]:.3f}]  '
          f'J={cfg["J"]}  bw={((edges[-1]-edges[0])/cfg["J"]).item():.4f}')

    model = CausalPFN2DHead(
        J=cfg['J'], num_features=cfg['num_features'],
        ninp=cfg['ninp'], nhid=cfg['nhid'], nhead=cfg['nhead'],
        nlayers=cfg['nlayers'], dropout=cfg.get('dropout', 0.0),
        n_out=cfg.get('n_out', 10),
    ).to(DEVICE)
    _load_sd(model, ck['model_state_dict'])
    model.eval()

    # ── Diagnostic 1: what does null_t_intv converge to? ─────────────────
    ntv = float(model.null_t_intv.detach().cpu())
    print(f'\n[null_t_intv] = {ntv:+.6f}   '
          f'(context T is in {{0, 1}}; 0.5 = "unknown"; 0 or 1 = anchored to one arm)')

    # ── IHDP realization ────────────────────────────────────────────────
    ds = IHDPDataset()
    cate_ds = ds[REAL][0]
    Xtr = cate_ds.X_train.astype(np.float32)
    Ttr = cate_ds.t_train.astype(np.float32).reshape(-1)
    ytr = cate_ds.y_train.astype(np.float32).reshape(-1)
    Xte = cate_ds.X_test.astype(np.float32)
    true_cate = cate_ds.true_cate.astype(np.float32)
    true_ate = float(true_cate.mean())

    print(f'\n[data] realization={REAL}  N_ctx={len(Xtr)}  N_q={len(Xte)}  '
          f'F={Xtr.shape[1]}  true_ATE={true_ate:+.3f}  '
          f'T_ctx: {Ttr.sum():.0f} treated / {len(Ttr)-Ttr.sum():.0f} control')

    Xtr_s, Xte_s = _std_train_test(Xtr, Xte)
    Xtr_p = _pad(Xtr_s, cfg['num_features'])
    Xte_p = _pad(Xte_s, cfg['num_features'])

    # Per-task pooled standardisation (matches training exactly)
    y_raw = torch.from_numpy(ytr).unsqueeze(0).to(DEVICE)         # (1, N_ctx)
    y_mean = y_raw.mean(dim=1, keepdim=True)
    y_std  = y_raw.std(dim=1, keepdim=True).clamp(min=1e-6)
    y_std_ctx = (y_raw - y_mean) / y_std
    print(f'[y-std] pooled mean={y_mean.item():+.3f}  std={y_std.item():.3f}')

    X_ctx = torch.from_numpy(Xtr_p).unsqueeze(0).to(DEVICE)
    t_ctx = torch.from_numpy(Ttr).unsqueeze(0).to(DEVICE)
    X_q   = torch.from_numpy(Xte_p).unsqueeze(0).to(DEVICE)

    logits = model._forward_logits(X_ctx, t_ctx, y_std_ctx, X_q)   # (1, N_q, nbins)
    J = cfg['J']; nbins = total_params(J)
    assert logits.shape[-1] == nbins, (logits.shape, nbins)

    interior = logits[..., :J*J]                                    # (1, N_q, J*J)
    region   = logits[..., J*J : J*J + N_REGIONS]                   # (1, N_q, 9)
    tail_raw = logits[..., J*J + N_REGIONS:]                        # (1, N_q, 4)

    p        = F.softmax(interior, dim=-1).reshape(1, -1, J, J)     # (1, N_q, J, J)
    w_region = F.softmax(region, dim=-1)                            # (1, N_q, 9)

    N_q = p.shape[1]
    centres = 0.5 * (edges[:-1] + edges[1:]).to(DEVICE)              # (J,)

    p_y0 = p.sum(dim=-1)                                             # (1, N_q, J)
    p_y1 = p.sum(dim=-2)

    e_y0_std = (p_y0 * centres.view(1, 1, J)).sum(dim=-1)            # (1, N_q)
    e_y1_std = (p_y1 * centres.view(1, 1, J)).sum(dim=-1)
    diff_std = (e_y1_std - e_y0_std).squeeze(0).cpu().numpy()

    e_y0 = (e_y0_std * y_std.squeeze(-1) + y_mean.squeeze(-1)).squeeze(0).cpu().numpy()
    e_y1 = (e_y1_std * y_std.squeeze(-1) + y_mean.squeeze(-1)).squeeze(0).cpu().numpy()
    cate = e_y1 - e_y0

    # ── Region weights (averaged across queries) ────────────────────────
    print('\n──── Region mixture weights (mean over queries) ────')
    labels = ['INNER', 'L0', 'R0', 'L1', 'R1', 'L0L1', 'L0R1', 'R0L1', 'R0R1']
    w_mean = w_region.squeeze(0).mean(dim=0).cpu().numpy()
    for k, lab in enumerate(labels):
        print(f'  {lab:6s} = {w_mean[k]:.4f}')
    print(f'  → interior mass carries {w_mean[R_INNER]*100:.1f}% of the density')

    # ── Marginal-mean stats ─────────────────────────────────────────────
    print('\n──── Marginal means (standardised space, over queries) ────')
    e0s = e_y0_std.squeeze(0).cpu().numpy()
    e1s = e_y1_std.squeeze(0).cpu().numpy()
    print(f'  E[Y0 | x] mean={e0s.mean():+.4f}  std={e0s.std():.4f}  '
          f'min={e0s.min():+.3f}  max={e0s.max():+.3f}')
    print(f'  E[Y1 | x] mean={e1s.mean():+.4f}  std={e1s.std():.4f}  '
          f'min={e1s.min():+.3f}  max={e1s.max():+.3f}')
    print(f'  diff (std) mean={diff_std.mean():+.4f}  std={diff_std.std():.4f}  '
          f'|diff|.mean={np.abs(diff_std).mean():.4f}')

    print('\n──── CATE in raw Y units ────')
    print(f'  E[Y0] raw   mean={e_y0.mean():+.3f}  std={e_y0.std():.3f}')
    print(f'  E[Y1] raw   mean={e_y1.mean():+.3f}  std={e_y1.std():.3f}')
    print(f'  ate_hat = mean(cate) = {cate.mean():+.4f}   (true_ATE = {true_ate:+.3f})')
    print(f'  PEHE    = {float(np.sqrt(((cate - true_cate)**2).mean())):.4f}')

    # ── Sparklines for a few queries: are the two marginals different? ──
    print('\n──── Sample query marginals (indices 0, 25, 50, 75) ────')
    py0 = p_y0.squeeze(0).cpu().numpy()
    py1 = p_y1.squeeze(0).cpu().numpy()
    for idx in (0, min(25, N_q-1), min(50, N_q-1), min(75, N_q-1)):
        print(f'  q={idx:3d}  Y0: {sparkline(py0[idx])}   sum={py0[idx].sum():.3f}')
        print(f'         Y1: {sparkline(py1[idx])}   sum={py1[idx].sum():.3f}')
        print(f'         E0={e_y0[idx]:+.3f}  E1={e_y1[idx]:+.3f}  '
              f'diff={e_y1[idx]-e_y0[idx]:+.3f}  true={true_cate[idx]:+.3f}')

    # ── Sanity: symmetry check ──────────────────────────────────────────
    p_mat = p.squeeze(0).cpu().numpy()                              # (N_q, J, J)
    sym_err = np.abs(p_mat - p_mat.transpose(0, 2, 1)).max(axis=(1,2))
    print(f'\n──── Diagonal symmetry check ────')
    print(f'  If model treats Y0 and Y1 identically, p_mat is symmetric.')
    print(f'  max|p[i,j] - p[j,i]| over queries: '
          f'mean={sym_err.mean():.4f}  max={sym_err.max():.4f}')


if __name__ == '__main__':
    main()
