"""Diagnostic: what fraction of 2D-model per-query mean estimates fall within
one bin-width of the true per-query mu_t on IHDP?

If |Ê[Y_do_t](q) - μ_t(q)| < δ (bin_width in RAW-Y units) for most queries,
the residual error is quantization — bounded below by (δ/2)² × N and no
mean-recipe can fix it. If most errors are ≫ δ, the model's predicted
marginals are systematically wrong, not just discretized.

Reports per realization + overall:
  - mean |error| for Y_do0 and Y_do1 (raw + full-mixture means)
  - fraction of queries with |error| < c·δ for c ∈ {0.5, 1.0, 2.0}
  - bin_width in raw Y (varies per realization since Y is standardised)

Env:
  CKPT      (required) cpfn2d checkpoint
  CAUSALPFN (required)
  MAX_REAL  default '' (100)
"""
from __future__ import annotations
import os, sys, numpy as np, torch

CKPT      = os.environ['CKPT']
CAUSALPFN = os.environ['CAUSALPFN']
MAX_REAL  = os.environ.get('MAX_REAL', '')

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, CAUSALPFN); sys.path.insert(0, CAUSALPFN + '/src')
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path: sys.path.insert(0, _here)

from benchmarks import IHDPDataset  # noqa: E402
from training_causalpfn2d.model_causalpfn_2d import CausalPFN2DHead  # noqa: E402
from full_mixture_mean import full_mixture_mean  # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _pad(X, F):
    if X.shape[1] == F: return X
    if X.shape[1] > F:  return X[:, :F]
    return np.hstack([X, np.zeros((X.shape[0], F - X.shape[1]), dtype=X.dtype)])


def _std_X(Xtr, Xte, eps=1e-8):
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + eps
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
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
    sd = ck['model_state_dict']
    if any('_orig_mod.' in k for k in sd):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    ref = model.state_dict()
    kept = {k: v for k, v in sd.items() if k in ref and ref[k].shape == v.shape}
    model.load_state_dict(kept, strict=False)
    model.eval()
    edges_np = edges.detach().cpu().numpy() if hasattr(edges, 'detach') else np.asarray(edges)
    return model, cfg, edges_np


@torch.no_grad()
def _predict_arms(model, X_ctx_std_padded, T_ctx, y_ctx_raw, X_q_std_padded, J, edges_np):
    """Returns e0_raw, e1_raw, e0_full, e1_full IN RAW Y UNITS, plus bin_width_raw."""
    Y = y_ctx_raw.reshape(-1)
    y_mean = float(Y.mean()); y_std = float(max(Y.std(), 1e-6))
    y_ctx_std = ((Y - y_mean) / y_std).astype(np.float32)

    X_ctx_t = torch.from_numpy(X_ctx_std_padded.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_ctx_t = torch.from_numpy(T_ctx.astype(np.float32)).unsqueeze(0).to(DEVICE)
    Y_ctx_t = torch.from_numpy(y_ctx_std).unsqueeze(0).to(DEVICE)
    X_q_t   = torch.from_numpy(X_q_std_padded.astype(np.float32)).unsqueeze(0).to(DEVICE)

    logits = model._forward_logits(X_ctx_t, T_ctx_t, Y_ctx_t, X_q_t)   # (1, N_q, J²+9+4)
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

    # Un-standardise (raw units)
    e0_raw  = e0_raw_std  * y_std + y_mean
    e1_raw  = e1_raw_std  * y_std + y_mean
    e0_full = e0_full_std * y_std + y_mean
    e1_full = e1_full_std * y_std + y_mean

    bw_std = float(edges_np[1] - edges_np[0])
    bw_raw = bw_std * y_std
    return e0_raw, e1_raw, e0_full, e1_full, bw_raw, y_std


def _extract_true_mu(cate_ds):
    """IHDPDataset per-query counterfactual means.  Different releases use
    different attribute names; try a bunch."""
    mu0 = mu1 = None
    for name in ('mu_0_test', 'mu0_test', 'y0_test_mean', 'mu_0'):
        if hasattr(cate_ds, name):
            mu0 = np.asarray(getattr(cate_ds, name)).reshape(-1); break
    for name in ('mu_1_test', 'mu1_test', 'y1_test_mean', 'mu_1'):
        if hasattr(cate_ds, name):
            mu1 = np.asarray(getattr(cate_ds, name)).reshape(-1); break
    return mu0, mu1


def main():
    print(f'[bootstrap] {CKPT}', flush=True)
    model, cfg, edges_np = _load_model(CKPT)
    J = cfg['J']; F = cfg['num_features']
    ds = IHDPDataset()
    n = ds.n_tables if not MAX_REAL else min(ds.n_tables, int(MAX_REAL))

    # Overall accumulators per estimator × arm
    all_bw = []
    thresholds = (0.5, 1.0, 2.0)
    fracs = {(est, arm, c): [] for est in ('raw', 'full') for arm in ('y0', 'y1') for c in thresholds}
    mean_err = {(est, arm): [] for est in ('raw', 'full') for arm in ('y0', 'y1')}
    n_realizations_with_truth = 0

    for r in range(n):
        cate_ds = ds[r][0]
        mu0_true, mu1_true = _extract_true_mu(cate_ds)
        X_tr = cate_ds.X_train.astype(np.float32)
        T_tr = cate_ds.t_train.astype(np.float32).reshape(-1)
        y_tr = cate_ds.y_train.astype(np.float32).reshape(-1)
        X_te = cate_ds.X_test.astype(np.float32)
        Xs, Xq = _std_X(X_tr, X_te); Xs = _pad(Xs, F); Xq = _pad(Xq, F)

        e0_raw, e1_raw, e0_full, e1_full, bw_raw, y_std = _predict_arms(
            model, Xs, T_tr, y_tr, Xq, J, edges_np)
        all_bw.append(bw_raw)

        if mu0_true is None or mu1_true is None:
            if r == 0:
                print(f'  (r=0)  no truth mu_0_test/mu_1_test on IHDPDataset — reporting bin-width stats only', flush=True)
            continue
        n_realizations_with_truth += 1
        err = {
            ('raw',  'y0'): np.abs(e0_raw  - mu0_true),
            ('raw',  'y1'): np.abs(e1_raw  - mu1_true),
            ('full', 'y0'): np.abs(e0_full - mu0_true),
            ('full', 'y1'): np.abs(e1_full - mu1_true),
        }
        for k, e in err.items():
            mean_err[k].append(float(e.mean()))
            for c in thresholds:
                fracs[(*k, c)].append(float((e < c * bw_raw).mean()))

    print()
    print(f'══ IHDP  bin-width stats  n_real={n}  cpfn2d J={J}  edges=[{edges_np[0]:.1f},{edges_np[-1]:.1f}] ══')
    bw = np.array(all_bw)
    print(f'  bin_width_raw : mean={bw.mean():.3f}   min={bw.min():.3f}   max={bw.max():.3f}   median={np.median(bw):.3f}')

    if n_realizations_with_truth == 0:
        print('  ⚠ no realizations exposed mu_0_test / mu_1_test — cannot compare Ê[Y_do_t] vs truth')
        return

    print(f'\n══ |Ê - μ_true|  (raw Y units)  averaged over {n_realizations_with_truth} realizations ══')
    print(f'{"estimator":<8} {"arm":<3} {"mean |err|":>11} {"|err|<0.5δ":>12} {"|err|<δ":>10} {"|err|<2δ":>10}')
    for est in ('raw', 'full'):
        for arm in ('y0', 'y1'):
            me   = float(np.mean(mean_err[(est, arm)]))
            f_h  = 100 * float(np.mean(fracs[(est, arm, 0.5)]))
            f_1  = 100 * float(np.mean(fracs[(est, arm, 1.0)]))
            f_2  = 100 * float(np.mean(fracs[(est, arm, 2.0)]))
            print(f'{est:<8} {arm:<3} {me:>11.3f} {f_h:>11.1f}% {f_1:>9.1f}% {f_2:>9.1f}%')

    print(f'\nInterpretation:')
    print(f'  If most errors < δ  →  quantization is dominant; finer bins / sub-bin recipes have room.')
    print(f'  If most errors > 2δ →  systematic prediction bias; no mean-recipe fixes it.')


if __name__ == '__main__':
    main()
