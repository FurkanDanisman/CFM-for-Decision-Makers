"""Same diagnostic as diag_within_binwidth.py, but for the 1D CausalPFN head.

For each IHDP query, extract Ê[Y_do_t] per arm from the 1D BarDist head
(1024 bins over [-10, +10] std space, per-arm Y standardisation), un-standardise
per arm, and compare to the true per-query μ_t. Report the same
|error| < c·δ fractions so we can compare quantization-vs-bias between
the 1024-bin 1D head and the 32-bin 2D head.

Env:
  CKPT       (required) 1D CausalPFN checkpoint (step-ckpt or model_config-format)
  CAUSALPFN  (required)
  MAX_REAL   default '' (100)
"""
from __future__ import annotations
import os, sys, numpy as np, torch

CKPT      = os.environ['CKPT']
CAUSALPFN = os.environ['CAUSALPFN']
MAX_REAL  = os.environ.get('MAX_REAL', '')

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, CAUSALPFN); sys.path.insert(0, CAUSALPFN + '/src')

from benchmarks import IHDPDataset  # noqa: E402
from causalpfn.models.model import TabDPTLongContextModel  # noqa: E402


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
    y0_shift, y0_scale = (float(y0.mean()), float(y0.std() + eps)) if y0.size else (0.0, 1.0)
    y1_shift, y1_scale = (float(y1.mean()), float(y1.std() + eps)) if y1.size else (0.0, 1.0)
    return y0_shift, y0_scale, y1_shift, y1_scale


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
    edges_std = np.linspace(VMIN, VMAX, nbins + 1)
    return model, F, nbins, edges_std


@torch.no_grad()
def _forward_one_arm(model, X_ctx, T_ctx, Y_std_ctx, X_q, t_val, F, nbins):
    B, N_ctx, Fpt1 = X_ctx.shape
    N_q = X_q.shape[1]
    t_ctx_col = T_ctx.reshape(B, N_ctx, 1)
    t_q_col   = torch.full((B, N_q, 1), float(t_val), dtype=X_q.dtype, device=X_q.device)
    xt_ctx = torch.cat([t_ctx_col, X_ctx], dim=-1)
    xt_q   = torch.cat([t_q_col,   X_q],   dim=-1)
    x_all  = torch.cat([xt_ctx, xt_q], dim=1).transpose(0, 1).contiguous()
    y_src  = Y_std_ctx.transpose(0, 1).contiguous()
    pred = model(x_all, y_src).transpose(0, 1).contiguous()
    return pred[..., -nbins:]                                 # (B, N_q, nbins)


@torch.no_grad()
def _predict_arms_1d(model, X_ctx, T_ctx, y_ctx_raw, X_q, F, nbins, edges_std):
    y0s, y0sc, y1s, y1sc = _per_arm_shift_scale(T_ctx, y_ctx_raw)
    y_ctx_std = np.where(T_ctx.reshape(-1) > 0.5,
                          (y_ctx_raw - y1s) / y1sc,
                          (y_ctx_raw - y0s) / y0sc).astype(np.float32)
    X_ctx_t = torch.from_numpy(X_ctx.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_ctx_t = torch.from_numpy(T_ctx.astype(np.float32)).unsqueeze(0).to(DEVICE)
    Y_ctx_t = torch.from_numpy(y_ctx_std).unsqueeze(0).to(DEVICE)
    X_q_t   = torch.from_numpy(X_q.astype(np.float32)).unsqueeze(0).to(DEVICE)
    centers_std = 0.5 * (edges_std[:-1] + edges_std[1:])

    l0 = _forward_one_arm(model, X_ctx_t, T_ctx_t, Y_ctx_t, X_q_t, 0.0, F, nbins)
    l1 = _forward_one_arm(model, X_ctx_t, T_ctx_t, Y_ctx_t, X_q_t, 1.0, F, nbins)
    p0 = torch.softmax(l0.float(), dim=-1).squeeze(0).cpu().numpy()  # (N_q, nbins)
    p1 = torch.softmax(l1.float(), dim=-1).squeeze(0).cpu().numpy()

    e0_std = (p0 * centers_std).sum(axis=-1)
    e1_std = (p1 * centers_std).sum(axis=-1)

    # Un-standardise per arm (raw Y)
    e0_raw = e0_std * y0sc + y0s
    e1_raw = e1_std * y1sc + y1s

    # per-arm bin_widths in raw Y
    bw_std = float(edges_std[1] - edges_std[0])
    bw_raw_y0 = bw_std * y0sc
    bw_raw_y1 = bw_std * y1sc
    return e0_raw, e1_raw, bw_raw_y0, bw_raw_y1


def _extract_true_mu(cate_ds):
    mu0 = mu1 = None
    for name in ('mu_0_test', 'mu0_test', 'y0_test_mean', 'mu_0'):
        if hasattr(cate_ds, name):
            mu0 = np.asarray(getattr(cate_ds, name)).reshape(-1); break
    for name in ('mu_1_test', 'mu1_test', 'y1_test_mean', 'mu_1'):
        if hasattr(cate_ds, name):
            mu1 = np.asarray(getattr(cate_ds, name)).reshape(-1); break
    return mu0, mu1


def main():
    print(f'[bootstrap] 1D ckpt: {CKPT}', flush=True)
    model, F, nbins, edges_std = _load_model_1d(CKPT)
    ds = IHDPDataset()
    n = ds.n_tables if not MAX_REAL else min(ds.n_tables, int(MAX_REAL))

    all_bw_y0 = []; all_bw_y1 = []
    thresholds = (0.5, 1.0, 2.0)
    fracs = {(arm, c): [] for arm in ('y0', 'y1') for c in thresholds}
    mean_err = {'y0': [], 'y1': []}
    n_realizations_with_truth = 0

    for r in range(n):
        cate_ds = ds[r][0]
        mu0_true, mu1_true = _extract_true_mu(cate_ds)
        X_tr = cate_ds.X_train.astype(np.float32)
        T_tr = cate_ds.t_train.astype(np.float32).reshape(-1)
        y_tr = cate_ds.y_train.astype(np.float32).reshape(-1)
        X_te = cate_ds.X_test.astype(np.float32)
        Xs, Xq = _std_X(X_tr, X_te); Xs = _pad(Xs, F); Xq = _pad(Xq, F)

        e0_raw, e1_raw, bw_y0, bw_y1 = _predict_arms_1d(
            model, Xs, T_tr, y_tr, Xq, F, nbins, edges_std)
        all_bw_y0.append(bw_y0); all_bw_y1.append(bw_y1)

        if mu0_true is None or mu1_true is None:
            if r == 0:
                print('  no mu_0_test/mu_1_test on IHDPDataset — reporting bin-width stats only', flush=True)
            continue
        n_realizations_with_truth += 1
        err_y0 = np.abs(e0_raw - mu0_true)
        err_y1 = np.abs(e1_raw - mu1_true)
        mean_err['y0'].append(float(err_y0.mean()))
        mean_err['y1'].append(float(err_y1.mean()))
        for c in thresholds:
            fracs[('y0', c)].append(float((err_y0 < c * bw_y0).mean()))
            fracs[('y1', c)].append(float((err_y1 < c * bw_y1).mean()))

    print()
    print(f'══ IHDP  bin-width stats  n_real={n}  1D head nbins={nbins}  edges=[{edges_std[0]:.1f},{edges_std[-1]:.1f}] ══')
    for arm, bws in (('y0', all_bw_y0), ('y1', all_bw_y1)):
        bw = np.array(bws)
        print(f'  bin_width_raw ({arm}) : mean={bw.mean():.4f}   min={bw.min():.4f}   max={bw.max():.4f}   median={np.median(bw):.4f}')

    if n_realizations_with_truth == 0:
        print('  ⚠ no realizations exposed mu_*_test — cannot compare Ê vs truth')
        return

    print(f'\n══ |Ê - μ_true|  (raw Y units)  averaged over {n_realizations_with_truth} realizations ══')
    print(f'{"arm":<3} {"mean |err|":>11} {"|err|<0.5δ":>12} {"|err|<δ":>10} {"|err|<2δ":>10}')
    for arm in ('y0', 'y1'):
        me   = float(np.mean(mean_err[arm]))
        f_h  = 100 * float(np.mean(fracs[(arm, 0.5)]))
        f_1  = 100 * float(np.mean(fracs[(arm, 1.0)]))
        f_2  = 100 * float(np.mean(fracs[(arm, 2.0)]))
        print(f'{arm:<3} {me:>11.4f} {f_h:>11.1f}% {f_1:>9.1f}% {f_2:>9.1f}%')

    print(f'\nInterpretation:')
    print(f'  δ_1D ≈ {np.mean(all_bw_y0):.3f} (raw Y), ~30× smaller than δ_2D (~1.4).')
    print(f'  Very high |err|<δ means the 1D head is limited by bias not quantization — but its δ is TINY,')
    print(f'  so most errors are already many bin-widths, and the fraction just tells us relative model accuracy.')


if __name__ == '__main__':
    main()
