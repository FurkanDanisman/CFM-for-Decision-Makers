"""Diagnose whether the graph2d model actually responds to the adjacency
matrix at inference. Load a checkpoint, run 5 IHDP realizations through
FOUR different adjacencies, and print how much the CATE predictions
change across them.

If all four give ~identical predictions, the adjacency path is dead.
If (anc_full) matches (noanc) but both differ from (anc_flipped), the
model reacts to *some* signal but has learned nothing useful.
If (anc_full) differs cleanly from (noanc) in the direction that
improves PEHE, the model IS using the graph correctly and something
else is wrong.

Usage:
    CKPT=... UWYK=... CAUSALPFN=... python diag_adj_sensitivity.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch


CKPT      = os.environ['CKPT']
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


def build_anc_flipped(F: int, n_real: int) -> np.ndarray:
    """Reverse the true edges: Y->T, T->feats, Y->feats. Should be strictly
    WRONG. If model still gives the same output as anc_full, it's not
    reading the direction sensitively."""
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    A[1, 0] = 1.0  # Y -> T
    for i in range(n_real):
        A[0, 2 + i] = 1.0  # T -> feat_i
        A[1, 2 + i] = 1.0  # Y -> feat_i
    for i in range(n_real, F):
        A[2 + i, :] = -1.0
        A[:, 2 + i] = -1.0
        A[2 + i, 2 + i] = -1.0
    return A


def build_anc_all_plus1(F: int, n_real: int) -> np.ndarray:
    """Everything +1 in the real block: 'every node is ancestor of every other'.
    Extreme signal — model should react to this VERY differently than to zeros
    if it's using the adjacency."""
    A = np.ones((F + 2, F + 2), dtype=np.float32)
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


def main():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    cfg = ck['config']
    print(f'[bootstrap] ckpt={CKPT}')
    print(f'[bootstrap] cfg num_features={cfg["num_features"]}  J={cfg["J"]}  step={ck.get("step")}')

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

    ds = IHDPDataset()
    for r in range(5):
        cate_ds = ds[r][0]
        X_tr = cate_ds.X_train.astype(np.float32)
        T_tr = cate_ds.t_train.astype(np.float32).reshape(-1)
        y_tr = cate_ds.y_train.astype(np.float32).reshape(-1)
        X_te = cate_ds.X_test.astype(np.float32)

        n_real = X_tr.shape[1]
        X_tr_std, X_te_std = _standardize_train_test(X_tr, X_te)
        X_tr_p = _pad_features(X_tr_std, F)
        X_te_p = _pad_features(X_te_std, F)
        y_scaled, ymin, yrange = _scale_y(y_tr)
        Y_obs = y_scaled.reshape(-1, 1)

        # Build four adjacencies for this realization
        adjs = {
            'noanc':      build_anc_none(F, n_real),
            'anc_full':   build_anc_full(F, n_real),
            'anc_flipped': build_anc_flipped(F, n_real),
            'anc_all1':   build_anc_all_plus1(F, n_real),
        }

        cates = {}
        for name, adj in adjs.items():
            cate_scaled = cate_from_forward(model, X_tr_p, T_tr, Y_obs, X_te_p, adj, J)
            cate = cate_scaled * yrange / 2.0
            cates[name] = cate

        # Report: how much does each anc mode DIFFER from noanc's predictions?
        base = cates['noanc']
        print(f'\nr={r:03d}  n_real={n_real}  |cate|~{np.abs(base).mean():.3f}')
        for name in ('anc_full', 'anc_flipped', 'anc_all1'):
            diff = cates[name] - base
            print(f'  {name:12s}  mean_diff={diff.mean():+7.4f}  '
                  f'L1_diff={np.abs(diff).mean():7.4f}  '
                  f'L2_diff={np.sqrt((diff**2).mean()):7.4f}  '
                  f'max_|diff|={np.abs(diff).max():.4f}')


if __name__ == '__main__':
    main()
