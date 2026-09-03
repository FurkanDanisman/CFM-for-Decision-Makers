"""Evaluate a cpfn2d checkpoint on the RealCause suite (IHDP, ACIC, CPS,
PSID, PSID_bal). Reports raw + em mean CATE per realization.

Same dataset loading + subsampling logic as eval_graph2d_realcause.py so
the two are apples-to-apples on the context side.

Env vars:
  CKPT              (required)
  OUT               (required) — per-dataset dir will be created as OUT/<DATASET>
  DATASET           IHDP | ACIC | CPS | PSID | PSID_bal   (default IHDP)
  CAUSALPFN         (required)
  EVAL_MAX_CONTEXT  cap for context subsampling  (default '' = no cap)
  EVAL_CONTEXT_SEED (default 1)
  PSID_BAL_SEED     (default 42) — only used for PSID_bal
  MAX_REAL          cap on n_tables (for smoke tests)
  Y_STD_MODE_EVAL   'pooled' (default) or 'per_arm'
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import numpy as np
import torch
from scipy.stats import norm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default=os.environ.get('DATASET', 'IHDP'),
                    choices=('IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal'))
args, _ = parser.parse_known_args()
DATASET = args.dataset

CKPT      = os.environ['CKPT']
OUT       = os.environ['OUT']
CAUSALPFN = os.environ['CAUSALPFN']
EVAL_MAX_CONTEXT  = os.environ.get('EVAL_MAX_CONTEXT', '')
EVAL_CONTEXT_SEED = int(os.environ.get('EVAL_CONTEXT_SEED', '1'))
PSID_BAL_SEED     = int(os.environ.get('PSID_BAL_SEED', '42'))
MAX_REAL          = os.environ.get('MAX_REAL', '')
Y_STD_MODE_EVAL   = os.environ.get('Y_STD_MODE_EVAL', 'pooled').lower()
STD_MODE          = os.environ.get('STD_MODE', '').lower()
# STD_MODE takes precedence over Y_STD_MODE_EVAL when set.
# Valid: '' (use Y_STD_MODE_EVAL), pooled, per_arm, log, log_per_arm, log_winsor
if STD_MODE:
    assert STD_MODE in ('pooled', 'per_arm', 'log', 'log_per_arm', 'log_winsor')
    Y_STD_MODE_EVAL = STD_MODE
else:
    assert Y_STD_MODE_EVAL in ('pooled', 'per_arm')

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, CAUSALPFN)
sys.path.insert(0, CAUSALPFN + '/src')

from benchmarks import (  # noqa: E402
    IHDPDataset, ACIC2016Dataset,
    RealCauseLalondeCPSDataset, RealCauseLalondePSIDDataset,
)
from training_causalpfn2d.model_causalpfn_2d import CausalPFN2DHead  # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_dataset(name):
    if name == 'IHDP':                return IHDPDataset()
    if name == 'ACIC':                return ACIC2016Dataset()
    if name == 'CPS':                 return RealCauseLalondeCPSDataset()
    if name in ('PSID', 'PSID_bal'):  return RealCauseLalondePSIDDataset()
    raise ValueError(name)


def _pad_features(X, F):
    if X.shape[1] == F: return X
    if X.shape[1] > F:  return X[:, :F]
    return np.hstack([X, np.zeros((X.shape[0], F - X.shape[1]), dtype=X.dtype)])


def _standardize_train_test(Xtr, Xte, eps=1e-8):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True); sd = np.where(sd < eps, eps, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd


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


def psid_balance_subsample(X, t, y):
    t_flat = t.reshape(-1)
    tr = (t_flat == 1); ct = (t_flat == 0)
    X_tr = X[tr]; t_tr = t[tr]; y_tr = y[tr]
    X_ct = X[ct]; t_ct = t[ct]; y_ct = y[ct]
    n_control = X_ct.shape[0]
    n_keep = min(500, n_control)
    if n_control > n_keep:
        np.random.seed(PSID_BAL_SEED)
        idx = np.random.choice(n_control, n_keep, replace=False)
        X_ct = X_ct[idx]; t_ct = t_ct[idx]; y_ct = y_ct[idx]
    X = np.vstack([X_tr, X_ct]); t = np.concatenate([t_tr, t_ct]); y = np.concatenate([y_tr, y_ct])
    perm = np.random.RandomState(PSID_BAL_SEED).permutation(X.shape[0])
    return X[perm], t[perm], y[perm]


def _load_state_dict_safe(model, sd):
    if any('_orig_mod.' in k for k in sd):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    ref = model.state_dict()
    kept = {k: v for k, v in sd.items() if k in ref and ref[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(kept, strict=False)
    print(f'[eval] load_state_dict: missing={len(missing)} unexpected={len(unexpected)} '
          f'loaded={len(kept)}/{len(ref)}', flush=True)
    if len(missing) > 20:
        raise RuntimeError(f'ABORT: {len(missing)} missing keys')


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
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
    return model, cfg, edges, step, y_scaling_mode


@torch.no_grad()
def forward_pmats(model, X_ctx, T_ctx, Y_ctx_raw, X_q, J,
                   y_scaling_mode, y_std_mode_eval):
    X_ctx_t = torch.from_numpy(X_ctx.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_ctx_t = torch.from_numpy(T_ctx.astype(np.float32)).unsqueeze(0).to(DEVICE)
    Y_ctx_r = torch.from_numpy(Y_ctx_raw.astype(np.float32)).unsqueeze(0).to(DEVICE)
    X_q_t   = torch.from_numpy(X_q.astype(np.float32)).unsqueeze(0).to(DEVICE)

    # --- log-family: apply log1p(Y - min_Y) FIRST, then pooled/per_arm on log-Y.
    y_min = None
    if y_std_mode_eval in ('log', 'log_per_arm', 'log_winsor'):
        y_min = float(Y_ctx_r.amin().item())
        Y_work = torch.log1p(Y_ctx_r - y_min)
    else:
        Y_work = Y_ctx_r

    if y_std_mode_eval in ('per_arm', 'log_per_arm'):
        tf = T_ctx_t.reshape(-1); yf = Y_work.reshape(-1)
        y0 = yf[tf < 0.5]; y1 = yf[tf > 0.5]
        y0s  = float(y0.mean().item()) if y0.numel() else 0.0
        y0sc = float(y0.std().clamp(min=1e-6).item()) if y0.numel() else 1.0
        y1s  = float(y1.mean().item()) if y1.numel() else 0.0
        y1sc = float(y1.std().clamp(min=1e-6).item()) if y1.numel() else 1.0
        y_std = torch.where(T_ctx_t > 0.5, (Y_work - y1s) / y1sc, (Y_work - y0s) / y0sc)
        stats = {'mode': 'per_arm', 'y0s': y0s, 'y0sc': y0sc, 'y1s': y1s, 'y1sc': y1sc,
                 'log_y_min': y_min}
    elif y_std_mode_eval == 'log_winsor':
        # winsor Q1/Q99 in log space, clip std-Y to model edges
        qlo = torch.quantile(Y_work, 0.01); qhi = torch.quantile(Y_work, 0.99)
        Yw = Y_work.clamp(min=qlo.item(), max=qhi.item())
        sh = Yw.mean(dim=1, keepdim=True); sc = Yw.std(dim=1, keepdim=True).clamp(min=1e-6)
        y_std = (Y_work - sh) / sc
        # edge-clip to prevent saturation
        # (edges come in via _forward_logits; approximate as ±(J/2)*bin_width — but
        # skip explicit clip since with log the range is naturally tight)
        stats = {'mode': 'pooled', 'shift': float(sh.item()), 'scale': float(sc.item()),
                 'log_y_min': y_min}
    elif y_scaling_mode == 'uwyk_minmax':
        y_lo = Y_ctx_r.amin(dim=1, keepdim=True); y_hi = Y_ctx_r.amax(dim=1, keepdim=True)
        sh = 0.5 * (y_lo + y_hi); sc = (0.5 * (y_hi - y_lo)).clamp(min=1e-6)
        y_std = (Y_ctx_r - sh) / sc
        stats = {'mode': 'pooled', 'shift': float(sh.item()), 'scale': float(sc.item()),
                 'log_y_min': y_min}
    else:
        sh = Y_work.mean(dim=1, keepdim=True); sc = Y_work.std(dim=1, keepdim=True).clamp(min=1e-6)
        y_std = (Y_work - sh) / sc
        stats = {'mode': 'pooled', 'shift': float(sh.item()), 'scale': float(sc.item()),
                 'log_y_min': y_min}

    logits = model._forward_logits(X_ctx_t, T_ctx_t, y_std, X_q_t)  # (1, N_q, J²+9+4)
    interior = logits[..., : J * J]
    p_mats = torch.softmax(interior, dim=-1).reshape(1, -1, J, J).squeeze(0).cpu().numpy()
    logits_np = logits.squeeze(0).float().cpu().numpy()             # (N_q, J²+9+4)
    return p_mats.astype(np.float64), logits_np, stats


def cate_raw_and_em(p_mats, edges_np):
    N_q = p_mats.shape[0]
    centers = 0.5 * (edges_np[:-1] + edges_np[1:])
    e_y0_raw = np.empty(N_q); e_y1_raw = np.empty(N_q)
    e_y0_em  = np.empty(N_q); e_y1_em  = np.empty(N_q)
    for q in range(N_q):
        pm = p_mats[q]
        p0 = pm.sum(axis=1); p1 = pm.sum(axis=0)
        p0 /= max(p0.sum(), 1e-12); p1 /= max(p1.sum(), 1e-12)
        e_y0_raw[q] = float((centers * p0).sum())
        e_y1_raw[q] = float((centers * p1).sum())
        mu0, s0 = _marginal_stats(p0, edges_np)
        mu1, s1 = _marginal_stats(p1, edges_np)
        e_y0_em[q] = _em_mean_1d(p0, edges_np, s0, mu0)
        e_y1_em[q] = _em_mean_1d(p1, edges_np, s1, mu1)
    return e_y0_raw, e_y1_raw, e_y0_em, e_y1_em


def evaluate(r, ds, model, J, F, edges_np, y_scaling_mode, apply_psid_balance):
    cate_ds = ds[r][0]
    X_tr = np.asarray(cate_ds.X_train, dtype=np.float32)
    T_tr = np.asarray(cate_ds.t_train, dtype=np.float32).reshape(-1)
    y_tr = np.asarray(cate_ds.y_train, dtype=np.float32).reshape(-1)
    X_te = np.asarray(cate_ds.X_test, dtype=np.float32)
    true_cate = np.asarray(cate_ds.true_cate, dtype=np.float32)
    true_ate  = float(true_cate.mean())

    if apply_psid_balance:
        X_tr, T_tr, y_tr = psid_balance_subsample(X_tr, T_tr, y_tr)

    if EVAL_MAX_CONTEXT:
        cap = int(EVAL_MAX_CONTEXT)
        n_ctx = X_tr.shape[0]
        if n_ctx > cap:
            rng = np.random.default_rng(EVAL_CONTEXT_SEED + r)
            idx = rng.choice(n_ctx, cap, replace=False)
            X_tr = X_tr[idx]; T_tr = T_tr[idx]; y_tr = y_tr[idx]

    X_tr_std, X_te_std = _standardize_train_test(X_tr, X_te)
    X_tr_p = _pad_features(X_tr_std, F); X_te_p = _pad_features(X_te_std, F)

    from full_mixture_mean import full_mixture_mean
    p_mats, logits_np, stats = forward_pmats(model, X_tr_p, T_tr, y_tr, X_te_p, J,
                                              y_scaling_mode, Y_STD_MODE_EVAL)
    e_y0_raw, e_y1_raw, e_y0_em, e_y1_em = cate_raw_and_em(p_mats, edges_np)
    e_y0_full, e_y1_full = full_mixture_mean(logits_np, J, edges_np)

    y_min = stats.get('log_y_min', None)   # None → not a log-family mode
    def _unlog(x):
        return np.expm1(x) + y_min if y_min is not None else x
    if stats['mode'] == 'per_arm':
        y0s, y0sc = stats['y0s'], stats['y0sc']; y1s, y1sc = stats['y1s'], stats['y1sc']
        # each arm de-standardises independently (log means shift doesn't cancel)
        e0_raw_r  = _unlog(e_y0_raw  * y0sc + y0s); e1_raw_r  = _unlog(e_y1_raw  * y1sc + y1s)
        e0_em_r   = _unlog(e_y0_em   * y0sc + y0s); e1_em_r   = _unlog(e_y1_em   * y1sc + y1s)
        e0_full_r = _unlog(e_y0_full * y0sc + y0s); e1_full_r = _unlog(e_y1_full * y1sc + y1s)
        cate_raw  = e1_raw_r  - e0_raw_r
        cate_em   = e1_em_r   - e0_em_r
        cate_full = e1_full_r - e0_full_r
    else:
        sh = stats.get('shift', 0.0); sc = stats['scale']
        if y_min is not None:
            # de-standardise each arm to log-Y, then expm1, then subtract
            e0r = _unlog(e_y0_raw  * sc + sh); e1r = _unlog(e_y1_raw  * sc + sh)
            e0e = _unlog(e_y0_em   * sc + sh); e1e = _unlog(e_y1_em   * sc + sh)
            e0f = _unlog(e_y0_full * sc + sh); e1f = _unlog(e_y1_full * sc + sh)
            cate_raw  = e1r - e0r
            cate_em   = e1e - e0e
            cate_full = e1f - e0f
        else:
            # non-log pooled: shift cancels, just scale the difference
            cate_raw  = (e_y1_raw  - e_y0_raw ) * sc
            cate_em   = (e_y1_em   - e_y0_em  ) * sc
            cate_full = (e_y1_full - e_y0_full) * sc

    def _pehe(cate):
        pehe = float(np.sqrt(np.nanmean((cate - true_cate) ** 2)))
        ate_hat = float(np.nanmean(cate))
        err = abs(ate_hat - true_ate) / max(abs(true_ate), 1e-9)
        return pehe, err, ate_hat

    p_r, e_r, a_r = _pehe(cate_raw)
    p_e, e_e, a_e = _pehe(cate_em)
    p_f, e_f, a_f = _pehe(cate_full)
    return {
        'dataset': DATASET, 'realization': r, 'true_ate': true_ate,
        'pehe_raw':  p_r, 'err_raw':  e_r, 'ate_raw':  a_r,
        'pehe_em':   p_e, 'err_em':   e_e, 'ate_em':   a_e,
        'pehe_full': p_f, 'err_full': e_f, 'ate_full': a_f,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = get_dataset(DATASET)
    apply_psid_balance = (DATASET == 'PSID_bal')
    model, cfg, edges, step, y_scaling_mode = load_model(CKPT)
    J = cfg['J']; F = cfg['num_features']
    edges_np = edges.detach().cpu().numpy().astype(np.float64) if hasattr(edges, 'detach') else np.asarray(edges, dtype=np.float64)

    n = ds.n_tables if not MAX_REAL else min(ds.n_tables, int(MAX_REAL))
    print(f'[bootstrap] {DATASET}  ckpt={CKPT}  step={step}  J={J}  F={F}  '
          f'edges=[{edges_np[0]:.2f},{edges_np[-1]:.2f}]  n={n}  '
          f'ctx_cap={EVAL_MAX_CONTEXT or "none"}  seed={EVAL_CONTEXT_SEED}  '
          f'y_std_mode_eval={Y_STD_MODE_EVAL}', flush=True)

    rows = []
    t0 = time.time()
    for r in range(n):
        row = evaluate(r, ds, model, J, F, edges_np, y_scaling_mode, apply_psid_balance)
        rows.append(row)
        tag = f'{DATASET}_r{r:03d}' if DATASET != 'ACIC' else f'{DATASET}_r{r:02d}'
        np.savez(os.path.join(OUT, f'{tag}.npz'),
                 **{k: np.array(v) for k, v in row.items()})
        print(f'r={r:03d}  raw: pehe={row["pehe_raw"]:6.3f} err={row["err_raw"]:5.3f}  |  '
              f'em: pehe={row["pehe_em"]:6.3f} err={row["err_em"]:5.3f}  |  '
              f'full: pehe={row["pehe_full"]:6.3f} err={row["err_full"]:5.3f}  '
              f'(true_ate={row["true_ate"]:+6.3f}, {time.time()-t0:.0f}s)', flush=True)

    def _ms(k):
        v = np.array([r[k] for r in rows if np.isfinite(r[k])])
        return (v.mean(), v.std(ddof=1) / np.sqrt(max(len(v), 1)), int(v.size)) if v.size \
            else (float('nan'), float('nan'), 0)

    print(f'\n══ {DATASET} summary  (n={len(rows)}) ══')
    for k in ('pehe_raw', 'err_raw', 'pehe_em', 'err_em', 'pehe_full', 'err_full'):
        m, s, _ = _ms(k)
        print(f'  {k:10s} = {m:8.3f} ± {s:6.3f}')


if __name__ == '__main__':
    main()
