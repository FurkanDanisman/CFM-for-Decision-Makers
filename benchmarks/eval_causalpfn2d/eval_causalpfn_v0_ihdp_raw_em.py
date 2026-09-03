"""Pure-ICL IHDP eval of a CausalPFN 1D checkpoint — raw AND em CATE.

Copy of eval_causalpfn_v0_ihdp_minimal.py extended so each realization
reports both raw and em mean estimators applied to the same 1D BarDist
marginals. No faiss / no CATEEstimator / no retrieval.

Output schema mirrors eval_cpfn2d_ihdp_em.py so aggregators can share code.

Env: CPFN_V0_LOCAL, OUT, CAUSALPFN
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch
from scipy.stats import norm


CPFN_V0_LOCAL = os.environ.get('CPFN_V0_LOCAL',
                                '/scratch/furkanbd/rpfn_bench_kit/warmstart/causalpfn_v0.pt')
OUT           = os.environ.get('OUT', './results_causalpfn_v0_ihdp_raw_em')
CAUSALPFN     = os.environ['CAUSALPFN']

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, CAUSALPFN)
sys.path.insert(0, CAUSALPFN + '/src')

from benchmarks import IHDPDataset  # noqa: E402
from causalpfn.models.model import TabDPTLongContextModel  # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VMIN = -10.0
VMAX = +10.0


def _strip_prefix(sd, prefix, drop_no_prefix=False):
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        elif not drop_no_prefix:
            out[k] = v
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
    """Same fixed-point recipe as eval_cpfn2d_ihdp_em._em_mean_1d.

    Iterates mu ← Σ_i props[i] · centre_i, where centre_i is the mean of
    a Gaussian truncated to bin i (Bayes-optimal under Gaussian-bin
    residual assumption).
    """
    pn = props / max(props.sum(), 1e-12)
    mu = start
    for _ in range(max_step):
        a = (edges - mu) / sigma
        G1 = norm.cdf(a)
        G2 = norm.pdf(a)
        dG1 = G1[1:] - G1[:-1]
        dG2 = G2[1:] - G2[:-1]
        m_bin = mu - sigma * dG2 / np.clip(dG1, eps2, None)
        mu_new = float(np.sum(pn * m_bin))
        if abs(mu_new - mu) < eps1:
            mu = mu_new
            break
        mu = mu_new
    return mu


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


@torch.no_grad()
def cate_raw_and_em(model, X_train, T_train, Y_train_raw, X_test,
                    num_features, nbins, bin_edges_np):
    y0s, y0sc, y1s, y1sc = _per_arm_shift_scale(T_train, Y_train_raw)
    y_ctx_std = np.where(T_train.reshape(-1) > 0.5,
                          (Y_train_raw - y1s) / y1sc,
                          (Y_train_raw - y0s) / y0sc).astype(np.float32)
    X_ctx = torch.from_numpy(X_train.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_ctx = torch.from_numpy(T_train.astype(np.float32)).unsqueeze(0).to(DEVICE)
    Y_ctx = torch.from_numpy(y_ctx_std).unsqueeze(0).to(DEVICE)
    X_q   = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(0).to(DEVICE)

    bin_centers = 0.5 * (bin_edges_np[:-1] + bin_edges_np[1:])
    centers_t = torch.from_numpy(bin_centers.astype(np.float32)).to(DEVICE)

    logits_t0 = _forward_one_arm(model, X_ctx, T_ctx, Y_ctx, X_q, 0.0, num_features, nbins)
    logits_t1 = _forward_one_arm(model, X_ctx, T_ctx, Y_ctx, X_q, 1.0, num_features, nbins)
    p0 = torch.softmax(logits_t0.float(), dim=-1).squeeze(0).cpu().numpy()  # (N_q, nbins)
    p1 = torch.softmax(logits_t1.float(), dim=-1).squeeze(0).cpu().numpy()

    # Raw mean (in standardised Y).
    e_y0_std_raw = (p0 * bin_centers).sum(axis=-1)
    e_y1_std_raw = (p1 * bin_centers).sum(axis=-1)

    # EM mean per query.
    sigma = float(bin_edges_np[1] - bin_edges_np[0])  # bin width as σ
    N_q = p0.shape[0]
    e_y0_std_em = np.empty(N_q, dtype=np.float32)
    e_y1_std_em = np.empty(N_q, dtype=np.float32)
    for q in range(N_q):
        e_y0_std_em[q] = _em_mean_1d(p0[q], bin_edges_np, sigma, start=e_y0_std_raw[q])
        e_y1_std_em[q] = _em_mean_1d(p1[q], bin_edges_np, sigma, start=e_y1_std_raw[q])

    # Un-standardise per arm.
    def unst(a, sh, sc):  return a * sc + sh
    cate_raw = unst(e_y1_std_raw, y1s, y1sc) - unst(e_y0_std_raw, y0s, y0sc)
    cate_em  = unst(e_y1_std_em,  y1s, y1sc) - unst(e_y0_std_em,  y0s, y0sc)
    return cate_raw.astype(np.float32), cate_em.astype(np.float32)


def evaluate(realization, model, num_features, nbins, bin_edges_np):
    ds = IHDPDataset()
    cate_ds = ds[realization][0]
    X_tr = cate_ds.X_train.astype(np.float32)
    T_tr = cate_ds.t_train.astype(np.float32).reshape(-1)
    y_tr = cate_ds.y_train.astype(np.float32).reshape(-1)
    X_te = cate_ds.X_test.astype(np.float32)
    true_cate = cate_ds.true_cate.astype(np.float32)
    true_ate  = float(true_cate.mean())

    X_tr_std, X_te_std = _standardize_train_test(X_tr, X_te)
    X_tr_p = _pad_features(X_tr_std, num_features)
    X_te_p = _pad_features(X_te_std, num_features)

    cate_raw, cate_em = cate_raw_and_em(model, X_tr_p, T_tr, y_tr, X_te_p,
                                         num_features, nbins, bin_edges_np)

    def _pehe_err(cate):
        pehe = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
        ate_hat = float(cate.mean())
        err = abs(ate_hat - true_ate) / max(abs(true_ate), 0.1)
        return pehe, err, ate_hat

    p_raw, e_raw, ate_raw = _pehe_err(cate_raw)
    p_em,  e_em,  ate_em  = _pehe_err(cate_em)
    return {
        'dataset': 'IHDP', 'realization': realization, 'true_ate': true_ate,
        'pehe_raw': p_raw, 'err_raw': e_raw, 'ate_raw': ate_raw,
        'pehe_em':  p_em,  'err_em':  e_em,  'ate_em':  ate_em,
    }


def main():
    print(f'[bootstrap] device={DEVICE}  ckpt={CPFN_V0_LOCAL}', flush=True)

    ck = torch.load(CPFN_V0_LOCAL, map_location='cpu', weights_only=False)
    cfg = ck.get('model_config', {})
    sd  = ck['model_state_dict']
    sd = _strip_prefix(sd, 'model.', drop_no_prefix=True)

    enc_w = sd.get('encoder.weight')
    num_features_plus_t = enc_w.shape[1] if enc_w is not None else 101
    num_features = num_features_plus_t - 1
    ninp     = cfg.get('ninp',    enc_w.shape[0] if enc_w is not None else 384)
    nhid     = cfg.get('nhid',    768)
    nhead    = cfg.get('nhead',   6)
    nlayers  = cfg.get('nlayers', 20)
    n_out    = cfg.get('n_out',   10)
    dropout  = cfg.get('dropout', 0.0)

    nbins = None
    if isinstance(cfg, dict):
        nbins = cfg.get('nbins') or cfg.get('model', {}).get('nbins')
    if nbins is None:
        head_w = sd.get('head.2.weight')
        if head_w is not None:
            nbins = head_w.shape[0] - n_out
    if nbins is None:
        nbins = 1024
        print('[bootstrap] WARN: nbins defaulted to 1024', flush=True)

    print(f'[bootstrap] ninp={ninp} nhid={nhid} nhead={nhead} nlayers={nlayers} '
          f'nbins={nbins} num_features={num_features} n_out={n_out}', flush=True)

    model = TabDPTLongContextModel(
        dropout=dropout, n_out=n_out, nhead=nhead, nhid=nhid, ninp=ninp,
        nlayers=nlayers, num_features=num_features_plus_t, nbins=nbins,
    ).to(DEVICE)
    bin_edges_np = np.linspace(VMIN, VMAX, nbins + 1).astype(np.float64)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[bootstrap] load: missing={len(missing)} unexpected={len(unexpected)}', flush=True)
    if len(missing) > 5 or len(unexpected) > 5:
        raise RuntimeError(
            f'load mismatch — missing: {list(missing)[:5]}, unexpected: {list(unexpected)[:5]}'
        )
    model.eval()

    os.makedirs(OUT, exist_ok=True)
    rows = []
    t0 = time.time()
    for r in range(100):
        row = evaluate(r, model, num_features, nbins, bin_edges_np)
        np.savez(os.path.join(OUT, f'IHDP_r{r:03d}.npz'),
                 **{k: np.array(v) for k, v in row.items()})
        rows.append(row)
        print(
            f'r={r:03d}  raw: pehe={row["pehe_raw"]:6.3f} err={row["err_raw"]:5.3f}  |  '
            f'em: pehe={row["pehe_em"]:6.3f} err={row["err_em"]:5.3f}  '
            f'(true_ate={row["true_ate"]:+6.3f}, {time.time()-t0:.0f}s)', flush=True,
        )

    def _ms(k):
        v = np.array([r[k] for r in rows])
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

    print(f'\n══ IHDP summary (1D orig pure-ICL, n={len(rows)}) ══')
    for k in ('pehe_raw', 'err_raw', 'pehe_em', 'err_em'):
        m, s = _ms(k)
        print(f'  {k:12s} = {m:8.3f} ± {s:6.3f}')


if __name__ == '__main__':
    main()
