"""IHDP eval with UWYK's target-encoded T (as in dofm_full_conditioning.py).

The ONE concrete difference between UWYK's Table 1 pipeline and our
current eval: UWYK replaces raw {0, 1} T_obs with mean(Y|T) — continuous
values, not binary. This eval applies the same transformation before
passing T_obs to our joint-head model.

Everything else identical to eval_graph2d_ihdp.py:
- Same adjacency convention (0 non-edges, +1 edges, -1 padded)
- Same feature standardization
- Same Y scaling to [-1, 1]
- Same anc / noanc modes
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch


CKPT      = os.environ['CKPT']
OUT       = os.environ.get('OUT', './results_graph2d_ihdp_te')
UWYK      = os.environ['UWYK']
CAUSALPFN = os.environ['CAUSALPFN']

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, UWYK)
sys.path.insert(0, UWYK + '/src')
sys.path.insert(0, CAUSALPFN)
sys.path.insert(0, CAUSALPFN + '/src')

from benchmarks import IHDPDataset  # noqa: E402
from training_graph2d.model_graph_2d import GraphConditioned2DHead  # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_anc_full(F: int, n_real: int) -> np.ndarray:
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    A[0, 1] = 1.0
    for i in range(n_real):
        A[2 + i, 0] = 1.0
        A[2 + i, 1] = 1.0
    for i in range(n_real, F):
        A[2 + i, :] = -1.0
        A[:, 2 + i] = -1.0
        A[2 + i, 2 + i] = -1.0
    return A


def build_anc_none(F: int, n_real: int) -> np.ndarray:
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    for i in range(n_real, F):
        A[2 + i, :] = -1.0
        A[:, 2 + i] = -1.0
        A[2 + i, 2 + i] = -1.0
    return A


def _standardize_train_test(X_tr, X_te):
    mu = X_tr.mean(0, keepdims=True)
    sd = X_tr.std(0, keepdims=True) + 1e-8
    return (X_tr - mu) / sd, (X_te - mu) / sd


def _pad_features(X: np.ndarray, F: int) -> np.ndarray:
    if X.shape[1] == F:
        return X
    if X.shape[1] > F:
        return X[:, :F]
    pad = np.zeros((X.shape[0], F - X.shape[1]), dtype=X.dtype)
    return np.hstack([X, pad])


def _scale_y(y):
    ymin, ymax = float(y.min()), float(y.max())
    yrange = max(ymax - ymin, 1e-9)
    y_scaled = 2.0 * (y - ymin) / yrange - 1.0
    return y_scaled.astype(np.float32), ymin, yrange


@torch.no_grad()
def cate_from_forward(model, X_train, T_train, Y_train_scaled, X_test, adj, J):
    """Joint-head CATE — SAME AS eval_graph2d_ihdp.py."""
    X_obs  = torch.from_numpy(X_train.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_obs  = torch.from_numpy(T_train.astype(np.float32)).reshape(1, -1, 1).to(DEVICE)
    Y_obs  = torch.from_numpy(Y_train_scaled).unsqueeze(0).to(DEVICE)
    X_intv = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(0).to(DEVICE)
    adj_t  = torch.from_numpy(adj).unsqueeze(0).to(DEVICE)

    out = model(X_obs, T_obs, Y_obs, X_intv, adj_t)
    logits = out['predictions'] if isinstance(out, dict) else out
    interior = logits[..., : J * J]
    p = torch.softmax(interior, dim=-1).reshape(1, -1, J, J)
    centres = torch.linspace(-1.0 + 1.0 / J, 1.0 - 1.0 / J, J, device=logits.device)
    p_y0 = p.sum(dim=-1)
    p_y1 = p.sum(dim=-2)
    e_y0 = (p_y0 * centres.view(1, 1, J)).sum(dim=-1)
    e_y1 = (p_y1 * centres.view(1, 1, J)).sum(dim=-1)
    return (e_y1 - e_y0).squeeze(0).cpu().numpy()


def evaluate(realization: int, model, J, F):
    ds = IHDPDataset()
    cate_ds = ds[realization][0]
    X_tr = cate_ds.X_train.astype(np.float32)
    T_tr_binary = cate_ds.t_train.astype(np.float32).reshape(-1)
    y_tr_raw = cate_ds.y_train.astype(np.float32).reshape(-1)
    X_te = cate_ds.X_test.astype(np.float32)
    true_cate = cate_ds.true_cate.astype(np.float32)
    true_ate  = float(true_cate.mean())

    # === THE KEY DIFFERENCE FROM eval_graph2d_ihdp.py ===
    # UWYK target-encodes T: replace {0, 1} with mean(Y | T).
    # See dofm_full_conditioning.py:91-95.
    mean_y_t0 = float(y_tr_raw[T_tr_binary == 0].mean())
    mean_y_t1 = float(y_tr_raw[T_tr_binary == 1].mean())
    T_tr = np.where(T_tr_binary == 0, mean_y_t0, mean_y_t1).astype(np.float32)
    # ====================================================

    n_real = X_tr.shape[1]
    X_tr_std, X_te_std = _standardize_train_test(X_tr, X_te)
    X_tr_p = _pad_features(X_tr_std, F)
    X_te_p = _pad_features(X_te_std, F)
    y_scaled, ymin, yrange = _scale_y(y_tr_raw)
    Y_obs = y_scaled.reshape(-1, 1)

    results = {}
    for mode, adj in (('anc', build_anc_full(F, n_real)),
                       ('noanc', build_anc_none(F, n_real))):
        cate_scaled = cate_from_forward(model, X_tr_p, T_tr, Y_obs, X_te_p, adj, J)
        cate = cate_scaled * yrange / 2.0
        pehe = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
        ate_hat = float(cate.mean())
        err_ate = abs(ate_hat - true_ate) / max(abs(true_ate), 0.1)
        results[f'pehe_graph2d_{mode}'] = pehe
        results[f'err_graph2d_{mode}']  = err_ate
        results[f'ate_graph2d_{mode}']  = ate_hat

    return {
        'dataset': 'IHDP',
        'realization': realization,
        'true_ate': true_ate,
        'mean_y_t0': mean_y_t0,
        'mean_y_t1': mean_y_t1,
        **results,
    }


def main():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    cfg = ck['config']
    print(f'[bootstrap] ckpt={CKPT}')
    print(f'[bootstrap] cfg num_features={cfg["num_features"]}  J={cfg["J"]}  step={ck.get("step")}')
    print(f'[bootstrap] EVAL WITH TARGET-ENCODED T (matches UWYK dofm_full_conditioning.py)')

    model = GraphConditioned2DHead(
        num_features=cfg['num_features'],
        d_model=cfg['d_model'],
        depth=cfg['depth'],
        heads_feat=cfg['heads'],
        heads_samp=cfg['heads'],
        dropout=0.0,
        hidden_mult=cfg['hidden_mult'],
        normalize_features=True,
        J=cfg['J'],
    ).to(DEVICE)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    J = cfg['J']
    F = cfg['num_features']

    os.makedirs(OUT, exist_ok=True)
    all_rows = []
    t0 = time.time()
    for r in range(100):
        row = evaluate(r, model, J, F)
        out_path = os.path.join(OUT, f'r{r:03d}.npz')
        np.savez(out_path, **{k: np.array(v) for k, v in row.items()})
        all_rows.append(row)
        print(
            f'r={r:03d}  '
            f'anc: pehe={row["pehe_graph2d_anc"]:6.3f}  err={row["err_graph2d_anc"]:5.3f}  |  '
            f'noanc: pehe={row["pehe_graph2d_noanc"]:6.3f}  err={row["err_graph2d_noanc"]:5.3f}  '
            f'({time.time()-t0:.0f}s)',
            flush=True,
        )

    def _mean_sem(k):
        v = np.array([r[k] for r in all_rows])
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

    print('\n══ IHDP summary (n={}) ══'.format(len(all_rows)))
    for k in ('pehe_graph2d_anc', 'err_graph2d_anc',
              'pehe_graph2d_noanc', 'err_graph2d_noanc'):
        m, s = _mean_sem(k)
        print(f'  {k:25s} = {m:8.3f} ± {s:6.3f}')


if __name__ == '__main__':
    main()
