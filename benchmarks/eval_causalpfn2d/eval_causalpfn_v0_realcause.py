"""Pure-ICL RealCause suite eval for a CausalPFN 1D checkpoint — raw + em.

Same dataset loading + subsampling logic as eval_graph2d_realcause.py
(and eval_cpfn2d_realcause.py). No faiss / no CATEEstimator.

Env: CKPT (or CPFN_V0_LOCAL), OUT, DATASET, CAUSALPFN,
     EVAL_MAX_CONTEXT, EVAL_CONTEXT_SEED, PSID_BAL_SEED, MAX_REAL
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
_SCM_CASES = ('Observed_Confounder', 'Observed_Mediator',
              'Observed_Mediator_and_Confounder', 'Unobserved_Confounder',
              'Frontdoor_Criterion', 'Backdoor_Criterion')
parser.add_argument('--dataset', type=str,
                    default=os.environ.get('DATASET', 'IHDP'),
                    choices=('IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal') + _SCM_CASES)
args, _ = parser.parse_known_args()
DATASET = args.dataset

CKPT      = os.environ.get('CKPT') or os.environ['CPFN_V0_LOCAL']
OUT       = os.environ['OUT']
CAUSALPFN = os.environ['CAUSALPFN']
EVAL_MAX_CONTEXT  = os.environ.get('EVAL_MAX_CONTEXT', '')
EVAL_CONTEXT_SEED = int(os.environ.get('EVAL_CONTEXT_SEED', '1'))
PSID_BAL_SEED     = int(os.environ.get('PSID_BAL_SEED', '42'))
MAX_REAL          = os.environ.get('MAX_REAL', '')
STD_MODE          = os.environ.get('STD_MODE', 'per_arm').lower()
assert STD_MODE in ('pooled', 'per_arm', 'log', 'log_per_arm', 'log_winsor'), STD_MODE

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, CAUSALPFN)
sys.path.insert(0, CAUSALPFN + '/src')

from benchmarks import (  # noqa: E402
    IHDPDataset, ACIC2016Dataset,
    RealCauseLalondeCPSDataset, RealCauseLalondePSIDDataset,
)
from causalpfn.models.model import TabDPTLongContextModel  # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VMIN, VMAX = -10.0, 10.0


def get_dataset(name):
    if name == 'IHDP':                return IHDPDataset()
    if name == 'ACIC':                return ACIC2016Dataset()
    if name == 'CPS':                 return RealCauseLalondeCPSDataset()
    if name in ('PSID', 'PSID_bal'):  return RealCauseLalondePSIDDataset()
    if name in _SCM_CASES:
        import sys as _sys
        _rp_bench = os.path.join(REPO_SRC, 'benchmarks')
        if _rp_bench not in _sys.path: _sys.path.insert(0, _rp_bench)
        from scm_case_study_dataset import SCMCaseStudyDataset
        return SCMCaseStudyDataset(name)
    raise ValueError(name)


def _strip_prefix(sd, prefix, drop_no_prefix=False):
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix): out[k[len(prefix):]] = v
        elif not drop_no_prefix: out[k] = v
    return out


def _pad_features(X, F):
    if X.shape[1] == F: return X
    if X.shape[1] > F:  return X[:, :F]
    return np.hstack([X, np.zeros((X.shape[0], F - X.shape[1]), dtype=X.dtype)])


def _standardize_train_test(Xtr, Xte, eps=1e-8):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + eps
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _per_arm_shift_scale(t, y, eps=1e-6):
    t = t.reshape(-1); y = y.reshape(-1)
    y0 = y[t < 0.5]; y1 = y[t > 0.5]
    y0_shift, y0_scale = (float(y0.mean()), float(y0.std() + eps)) if y0.size else (0.0, 1.0)
    y1_shift, y1_scale = (float(y1.mean()), float(y1.std() + eps)) if y1.size else (0.0, 1.0)
    return y0_shift, y0_scale, y1_shift, y1_scale


def _em_mean_1d(props, edges, sigma, start, max_step=1000, eps2=1e-10, eps1=1e-5):
    pn = props / max(props.sum(), 1e-12)
    mu = start
    for _ in range(max_step):
        a = (edges - mu) / sigma
        G1 = norm.cdf(a); G2 = norm.pdf(a)
        dG1 = G1[1:] - G1[:-1]; dG2 = G2[1:] - G2[:-1]
        m_bin = mu - sigma * dG2 / np.clip(dG1, eps2, None)
        mu_new = float(np.sum(pn * m_bin))
        if abs(mu_new - mu) < eps1:
            mu = mu_new
            break
        mu = mu_new
    return mu


def psid_balance_subsample(X, t, y):
    t_flat = t.reshape(-1)
    tr = (t_flat == 1); ct = (t_flat == 0)
    X_tr = X[tr]; t_tr = t[tr]; y_tr = y[tr]
    X_ct = X[ct]; t_ct = t[ct]; y_ct = y[ct]
    n_control = X_ct.shape[0]; n_keep = min(500, n_control)
    if n_control > n_keep:
        np.random.seed(PSID_BAL_SEED)
        idx = np.random.choice(n_control, n_keep, replace=False)
        X_ct = X_ct[idx]; t_ct = t_ct[idx]; y_ct = y_ct[idx]
    X = np.vstack([X_tr, X_ct]); t = np.concatenate([t_tr, t_ct]); y = np.concatenate([y_tr, y_ct])
    perm = np.random.RandomState(PSID_BAL_SEED).permutation(X.shape[0])
    return X[perm], t[perm], y[perm]


@torch.no_grad()
def _forward_one_arm(model, X_ctx, T_ctx, Y_std_ctx, X_q, t_val, num_features, nbins):
    B, N_ctx, F = X_ctx.shape
    N_q = X_q.shape[1]
    assert F == num_features, (F, num_features)
    t_ctx_col = T_ctx.reshape(B, N_ctx, 1)
    t_q_col   = torch.full((B, N_q, 1), float(t_val), dtype=X_q.dtype, device=X_q.device)
    xt_ctx = torch.cat([t_ctx_col, X_ctx], dim=-1)
    xt_q   = torch.cat([t_q_col,   X_q],   dim=-1)
    x_all  = torch.cat([xt_ctx, xt_q], dim=1)
    x_src = x_all.transpose(0, 1).contiguous()
    y_src = Y_std_ctx.transpose(0, 1).contiguous()
    pred = model(x_src, y_src)
    pred = pred.transpose(0, 1).contiguous()
    return pred[..., -nbins:]


def _compute_std_stats(T_train, Y_train_raw, bin_edges_np, mode):
    """Return (y_ctx_std_np, arm0_stats, arm1_stats, y_min_or_None).
    arm_stats = (shift, scale) — arm0 and arm1 identical for pooled/log/log_winsor,
    differ only for per_arm / log_per_arm.  y_min set only when a log transform
    is applied (needed to un-log at the end)."""
    T = T_train.reshape(-1); Y = Y_train_raw.reshape(-1).astype(np.float64)
    # log-family: transform Y BEFORE computing stats
    y_min = None
    if mode in ('log', 'log_per_arm', 'log_winsor'):
        y_min = float(Y.min())
        Y_work = np.log1p(Y - y_min)
    else:
        Y_work = Y

    if mode in ('per_arm', 'log_per_arm'):
        y0 = Y_work[T < 0.5]; y1 = Y_work[T > 0.5]
        m0, s0 = (float(y0.mean()), float(y0.std() + 1e-6)) if y0.size else (0.0, 1.0)
        m1, s1 = (float(y1.mean()), float(y1.std() + 1e-6)) if y1.size else (0.0, 1.0)
        z = np.where(T > 0.5, (Y_work - m1) / s1, (Y_work - m0) / s0)
        return z.astype(np.float32), (m0, s0), (m1, s1), y_min

    if mode == 'log_winsor':
        lo, hi = np.quantile(Y_work, [0.01, 0.99])
        Yw = np.clip(Y_work, lo, hi)
        m, s = float(Yw.mean()), float(Yw.std() + 1e-6)
        edge_lo, edge_hi = float(bin_edges_np[0]), float(bin_edges_np[-1])
        z = np.clip((Y_work - m) / s, edge_lo, edge_hi)
        return z.astype(np.float32), (m, s), (m, s), y_min

    # pooled / log (pooled on log-Y)
    m, s = float(Y_work.mean()), float(Y_work.std() + 1e-6)
    z = (Y_work - m) / s
    return z.astype(np.float32), (m, s), (m, s), y_min


@torch.no_grad()
def cate_raw_and_em(model, X_train, T_train, Y_train_raw, X_test,
                    num_features, nbins, bin_edges_np):
    y_ctx_std, arm0, arm1, y_min = _compute_std_stats(
        T_train, Y_train_raw, bin_edges_np, STD_MODE)
    X_ctx = torch.from_numpy(X_train.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_ctx = torch.from_numpy(T_train.astype(np.float32)).unsqueeze(0).to(DEVICE)
    Y_ctx = torch.from_numpy(y_ctx_std).unsqueeze(0).to(DEVICE)
    X_q   = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(0).to(DEVICE)
    centers = 0.5 * (bin_edges_np[:-1] + bin_edges_np[1:])

    logits_t0 = _forward_one_arm(model, X_ctx, T_ctx, Y_ctx, X_q, 0.0, num_features, nbins)
    logits_t1 = _forward_one_arm(model, X_ctx, T_ctx, Y_ctx, X_q, 1.0, num_features, nbins)
    p0 = torch.softmax(logits_t0.float(), dim=-1).squeeze(0).cpu().numpy()
    p1 = torch.softmax(logits_t1.float(), dim=-1).squeeze(0).cpu().numpy()

    e_y0_raw = (p0 * centers).sum(axis=-1)
    e_y1_raw = (p1 * centers).sum(axis=-1)

    sigma = float(bin_edges_np[1] - bin_edges_np[0])
    N_q = p0.shape[0]
    e_y0_em = np.empty(N_q, dtype=np.float64)
    e_y1_em = np.empty(N_q, dtype=np.float64)
    for q in range(N_q):
        e_y0_em[q] = _em_mean_1d(p0[q], bin_edges_np, sigma, start=e_y0_raw[q])
        e_y1_em[q] = _em_mean_1d(p1[q], bin_edges_np, sigma, start=e_y1_raw[q])

    def _un(a, arm):
        # de-standardise to (log-)Y space, then expm1 if log was applied
        v = a * arm[1] + arm[0]
        return np.expm1(v) + y_min if y_min is not None else v

    # log-family means the shift no longer cancels — de-standardise each arm
    cate_raw = _un(e_y1_raw, arm1) - _un(e_y0_raw, arm0)
    cate_em  = _un(e_y1_em,  arm1) - _un(e_y0_em,  arm0)
    return cate_raw.astype(np.float32), cate_em.astype(np.float32)


def evaluate(r, ds, model, num_features, nbins, bin_edges_np, apply_psid_balance):
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
    X_tr_p = _pad_features(X_tr_std, num_features)
    X_te_p = _pad_features(X_te_std, num_features)

    cate_raw, cate_em = cate_raw_and_em(model, X_tr_p, T_tr, y_tr, X_te_p,
                                         num_features, nbins, bin_edges_np)

    def _pehe(cate):
        pehe = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
        ate_hat = float(cate.mean())
        err = abs(ate_hat - true_ate) / max(abs(true_ate), 0.1)
        return pehe, err, ate_hat

    p_r, e_r, a_r = _pehe(cate_raw)
    p_e, e_e, a_e = _pehe(cate_em)
    return {
        'dataset': DATASET, 'realization': r, 'true_ate': true_ate,
        'pehe_raw': p_r, 'err_raw': e_r, 'ate_raw': a_r,
        'pehe_em':  p_e, 'err_em':  e_e, 'ate_em':  a_e,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    cfg = ck.get('model_config', {})
    sd  = ck['model_state_dict']
    sd = _strip_prefix(sd, 'model.', drop_no_prefix=True)

    enc_w = sd.get('encoder.weight')
    num_features_plus_t = enc_w.shape[1] if enc_w is not None else 101
    num_features = num_features_plus_t - 1
    ninp    = cfg.get('ninp',    enc_w.shape[0] if enc_w is not None else 384)
    nhid    = cfg.get('nhid',    768)
    nhead   = cfg.get('nhead',   6)
    nlayers = cfg.get('nlayers', 20)
    n_out   = cfg.get('n_out',   10)
    dropout = cfg.get('dropout', 0.0)

    nbins = None
    if isinstance(cfg, dict):
        nbins = cfg.get('nbins') or cfg.get('model', {}).get('nbins')
    if nbins is None:
        head_w = sd.get('head.2.weight')
        if head_w is not None:
            nbins = head_w.shape[0] - n_out
    if nbins is None:
        nbins = 1024

    model = TabDPTLongContextModel(
        dropout=dropout, n_out=n_out, nhead=nhead, nhid=nhid, ninp=ninp,
        nlayers=nlayers, num_features=num_features_plus_t, nbins=nbins,
    ).to(DEVICE)
    bin_edges_np = np.linspace(VMIN, VMAX, nbins + 1).astype(np.float64)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 5 or len(unexpected) > 5:
        raise RuntimeError(
            f'load mismatch — missing: {list(missing)[:5]}, unexpected: {list(unexpected)[:5]}'
        )
    model.eval()

    ds = get_dataset(DATASET)
    apply_psid_balance = (DATASET == 'PSID_bal')
    n = ds.n_tables if not MAX_REAL else min(ds.n_tables, int(MAX_REAL))
    print(f'[bootstrap] {DATASET}  ckpt={CKPT}  ninp={ninp} nhid={nhid} nhead={nhead} '
          f'nlayers={nlayers} nbins={nbins} F={num_features}  n={n}  '
          f'ctx_cap={EVAL_MAX_CONTEXT or "none"}  seed={EVAL_CONTEXT_SEED}  '
          f'STD_MODE={STD_MODE}', flush=True)

    rows = []
    t0 = time.time()
    for r in range(n):
        row = evaluate(r, ds, model, num_features, nbins, bin_edges_np, apply_psid_balance)
        rows.append(row)
        tag = f'{DATASET}_r{r:03d}' if DATASET != 'ACIC' else f'{DATASET}_r{r:02d}'
        np.savez(os.path.join(OUT, f'{tag}.npz'),
                 **{k: np.array(v) for k, v in row.items()})
        print(f'r={r:03d}  raw: pehe={row["pehe_raw"]:6.3f} err={row["err_raw"]:5.3f}  |  '
              f'em: pehe={row["pehe_em"]:6.3f} err={row["err_em"]:5.3f}  '
              f'(true_ate={row["true_ate"]:+6.3f}, {time.time()-t0:.0f}s)', flush=True)

    def _ms(k):
        v = np.array([r[k] for r in rows if np.isfinite(r[k])])
        return (v.mean(), v.std(ddof=1) / np.sqrt(max(len(v), 1)), int(v.size)) if v.size \
            else (float('nan'), float('nan'), 0)

    print(f'\n══ {DATASET} summary  (n={len(rows)}) ══')
    for k in ('pehe_raw', 'err_raw', 'pehe_em', 'err_em'):
        m, s, _ = _ms(k)
        print(f'  {k:10s} = {m:8.3f} ± {s:6.3f}')


if __name__ == '__main__':
    main()
