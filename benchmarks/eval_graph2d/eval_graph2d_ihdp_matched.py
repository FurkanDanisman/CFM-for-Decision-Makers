"""Match training's anc_matrix convention at eval.

PairedInterventionalDataset produces anc_matrix in {-1, 0, +1}:
  * -1 = confirmed non-ancestor (2 * 0 - 1)
  * +1 = confirmed ancestor    (2 * 1 - 1)
  *  0 = randomly hidden (positions where rand < hide_frac)

hide_frac ~ Uniform(0, 1), so the median training batch has ~50% of
real-block positions hidden as 0.

Our current eval builds a discontinuous input (only +1 on specific edges,
0 everywhere else) that CANNOT come from the training sampler — the only
way to get "0 on non-edges" is hide_frac = 1.0 on non-edges specifically,
but at hide_frac = 1.0 the +1 edges would also be hidden. So the model
sees a pattern it never saw during training and produces a degenerate
output.

This eval:
  1. Builds the TRUE {-1, +1} adjacency (hide_frac = 0, in-distribution).
  2. Also builds partial versions with hide_frac in {0.0, 0.25, 0.5, 0.75}.
     For each, averages predictions over N random hide masks.
  3. Reports PEHE and eps_ATE for each condition.

If ANY hide_frac gives us anc-beats-noanc, we've found an in-distribution
eval protocol without retraining.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch


CKPT      = os.environ['CKPT']
OUT       = os.environ.get('OUT', './results_graph2d_ihdp_matched')
UWYK      = os.environ['UWYK']
CAUSALPFN = os.environ['CAUSALPFN']
N_MASKS   = int(os.environ.get('N_MASKS', 8))

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, UWYK)
sys.path.insert(0, UWYK + '/src')
sys.path.insert(0, CAUSALPFN)
sys.path.insert(0, CAUSALPFN + '/src')

from benchmarks import IHDPDataset  # noqa: E402
from training_graph2d.model_graph_2d import GraphConditioned2DHead  # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_true_anc_signed(F: int, n_real: int) -> np.ndarray:
    """True IHDP adjacency in TRAINING's {-1, +1} convention (hide_frac = 0).

    All real-block non-edges are -1, edges are +1. Padded slots -1.
    This is what PairedInterventionalDataset produces BEFORE hide_mask
    is applied.
    """
    real_size = 2 + n_real
    A = np.full((F + 2, F + 2), -1.0, dtype=np.float32)
    # Set the +1 confirmed edges
    A[0, 1] = 1.0                       # T -> Y
    for i in range(n_real):
        A[2 + i, 0] = 1.0               # feat_i -> T
        A[2 + i, 1] = 1.0               # feat_i -> Y
    return A


def apply_hide_mask(A_signed: np.ndarray, n_real: int,
                     hide_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Apply training's random-hide procedure to a signed anc matrix.

    Sets each real-block position to 0 with probability hide_frac.
    Padded positions untouched (stay -1).
    """
    A = A_signed.copy()
    real_size = 2 + n_real
    hide = rng.random((real_size, real_size)) < hide_frac
    A_real = A[:real_size, :real_size].copy()
    A_real[hide] = 0.0
    A[:real_size, :real_size] = A_real
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


def evaluate_realization(realization: int, model, J, F, hide_fracs, n_masks):
    ds = IHDPDataset()
    cate_ds = ds[realization][0]
    X_tr = cate_ds.X_train.astype(np.float32)
    T_tr = cate_ds.t_train.astype(np.float32).reshape(-1)
    y_tr = cate_ds.y_train.astype(np.float32).reshape(-1)
    X_te = cate_ds.X_test.astype(np.float32)
    true_cate = cate_ds.true_cate.astype(np.float32)
    true_ate = float(true_cate.mean())

    n_real = X_tr.shape[1]
    X_tr_std, X_te_std = _standardize_train_test(X_tr, X_te)
    X_tr_p = _pad_features(X_tr_std, F)
    X_te_p = _pad_features(X_te_std, F)
    y_scaled, ymin, yrange = _scale_y(y_tr)
    Y_obs = y_scaled.reshape(-1, 1)

    A_signed_true = build_true_anc_signed(F, n_real)

    row = {}
    for hf in hide_fracs:
        rng = np.random.default_rng(seed=1234 + realization)
        cates = []
        for _ in range(n_masks):
            if hf == 0.0:
                adj = A_signed_true
            else:
                adj = apply_hide_mask(A_signed_true, n_real, hf, rng)
            cate_scaled = cate_from_forward(model, X_tr_p, T_tr, Y_obs, X_te_p, adj, J)
            cate = cate_scaled * yrange / 2.0
            cates.append(cate)
            if hf == 0.0:  # deterministic — no averaging needed
                break
        cate_mean = np.mean(cates, axis=0)
        pehe = float(np.sqrt(np.mean((cate_mean - true_cate) ** 2)))
        ate_hat = float(cate_mean.mean())
        err_ate = abs(ate_hat - true_ate) / max(abs(true_ate), 0.1)
        key = f'hf_{int(hf*100):03d}'
        row[f'pehe_{key}'] = pehe
        row[f'err_{key}']  = err_ate
        row[f'ate_{key}']  = ate_hat

    return row


def main():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    cfg = ck['config']
    print(f'[bootstrap] ckpt={CKPT}')
    print(f'[bootstrap] cfg num_features={cfg["num_features"]}  J={cfg["J"]}  step={ck.get("step")}')
    print(f'[bootstrap] N_MASKS per hide_frac = {N_MASKS}')

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

    hide_fracs = [0.0, 0.25, 0.5, 0.75, 0.9]
    os.makedirs(OUT, exist_ok=True)
    all_rows = []
    import time
    t0 = time.time()
    for r in range(100):
        row = evaluate_realization(r, model, J, F, hide_fracs, N_MASKS)
        out_path = os.path.join(OUT, f'r{r:03d}.npz')
        np.savez(out_path, **{k: np.array(v) for k, v in row.items()})
        all_rows.append(row)
        if r % 10 == 0 or r == 99:
            print(f'  r={r:03d}  ' + '  '.join(
                f'hf{int(hf*100):02d}:pehe={row[f"pehe_hf_{int(hf*100):03d}"]:.3f}'
                for hf in hide_fracs
            ) + f'  ({time.time()-t0:.0f}s)', flush=True)

    def _mean_sem(k):
        v = np.array([r[k] for r in all_rows])
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

    print('\n══ IHDP summary (n=100) ══')
    print('  hide_frac        pehe (mean ± sem)      eps_ATE (mean ± sem)')
    print('  ──────────────────────────────────────────────────────────────')
    for hf in hide_fracs:
        key = f'hf_{int(hf*100):03d}'
        pehe_m, pehe_s = _mean_sem(f'pehe_{key}')
        err_m, err_s = _mean_sem(f'err_{key}')
        print(f'  {hf:5.2f}         {pehe_m:8.3f} ± {pehe_s:5.3f}         {err_m:6.3f} ± {err_s:5.3f}')


if __name__ == '__main__':
    main()
