"""Quick diagnostic on realization 0: show the raw joint p(y0, y1), each
marginal, and the point-mean CATE under both anc modes.

If the two marginals are essentially identical shapes, the model is
undertrained. If they differ but our extraction gives ~0 CATE anyway,
there's an axis or scaling bug in eval_graph2d_ihdp.py.
"""
from __future__ import annotations
import os, sys
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

from benchmarks import IHDPDataset
from training_graph2d.model_graph_2d import GraphConditioned2DHead
from benchmarks.eval_graph2d.eval_graph2d_ihdp import (
    load_model, build_anc_full, build_anc_none, _pad_features, _scale_y,
)


def _forward_p(model, X_obs, T_obs, Y_obs, X_intv, adj, J):
    dev = next(model.parameters()).device
    Xo = torch.from_numpy(X_obs).unsqueeze(0).to(dev).float()
    To = torch.from_numpy(T_obs).reshape(1, -1, 1).to(dev).float()
    Yo = torch.from_numpy(Y_obs).unsqueeze(0).to(dev).float()
    Xi = torch.from_numpy(X_intv).unsqueeze(0).to(dev).float()
    A  = torch.from_numpy(adj).unsqueeze(0).to(dev).float()
    with torch.no_grad():
        out = model(Xo, To, Yo, Xi, A)
    logits = out['predictions'] if isinstance(out, dict) else out
    p = torch.softmax(logits[..., : J * J], dim=-1).reshape(1, -1, J, J)
    return p.cpu().numpy()  # (1, M, J, J)


def main():
    model, cfg = load_model(CKPT)
    J = cfg['J']; F = cfg['num_features']
    step = torch.load(CKPT, map_location='cpu', weights_only=False).get('step')
    print(f'checkpoint step={step}  J={J}  F={F}')

    ds = IHDPDataset()
    cd = ds[0][0]
    n_real = cd.X_train.shape[1]
    X_tr = _pad_features(cd.X_train.astype(np.float32), F)
    X_te = _pad_features(cd.X_test.astype(np.float32),  F)
    T_tr = cd.t_train.astype(np.float32).reshape(-1)
    y_scaled, ymin, yrange = _scale_y(cd.y_train.astype(np.float32).reshape(-1))
    Y_tr = y_scaled.reshape(-1, 1)

    for name, A in (('anc',   build_anc_full(F, n_real)),
                    ('noanc', build_anc_none(F, n_real))):
        p = _forward_p(model, X_tr, T_tr, Y_tr, X_te, A, J)[0]  # (M, J, J)
        centres = -1.0 + 2.0 * (np.arange(J) + 0.5) / J
        p_y0 = p.sum(axis=-1)                    # (M, J)  marg over y0
        p_y1 = p.sum(axis=-2)                    # (M, J)  marg over y1
        e0 = (p_y0 * centres).sum(axis=-1)       # (M,)
        e1 = (p_y1 * centres).sum(axis=-1)       # (M,)
        cate_scaled = e1 - e0
        cate = cate_scaled * yrange / 2.0

        # Concentration diagnostics
        eff_bins_p    = 1.0 / (p.reshape(len(p), -1)**2).sum(axis=-1)  # ~effective #bins with mass
        eff_bins_p_y0 = 1.0 / (p_y0**2).sum(axis=-1)
        eff_bins_p_y1 = 1.0 / (p_y1**2).sum(axis=-1)
        marg_diff = np.abs(p_y0 - p_y1).mean(axis=-1)   # how different y0 and y1 marginals are

        print(f'\n══ {name} ══')
        print(f'  joint concentration: mean effective-#interior-bins = {eff_bins_p.mean():.1f}   (of {J*J})')
        print(f'  y0 marg conc.:       mean effective-#bins           = {eff_bins_p_y0.mean():.2f}   (of {J})')
        print(f'  y1 marg conc.:       mean effective-#bins           = {eff_bins_p_y1.mean():.2f}   (of {J})')
        print(f'  marg y0 vs y1 mean-abs-diff (per query): {marg_diff.mean():.4f}')
        print(f'  E[y0_scaled] over queries: mean={e0.mean():.4f} std={e0.std():.4f}')
        print(f'  E[y1_scaled] over queries: mean={e1.mean():.4f} std={e1.std():.4f}')
        print(f'  CATE_scaled = E[y1]-E[y0]: mean={cate_scaled.mean():.4f} std={cate_scaled.std():.4f}')
        print(f'  CATE (Y-scale):            mean={cate.mean():.4f} std={cate.std():.4f}')

        # Show first 5 queries in detail
        print(f'  first 5 queries: e0={np.round(e0[:5],3)}  e1={np.round(e1[:5],3)}  cate={np.round(cate[:5],2)}')

    print(f'\ntrue CATE mean={cd.true_cate.mean():.2f}   std={cd.true_cate.std():.2f}   first 5: {np.round(cd.true_cate[:5],2)}')


if __name__ == '__main__':
    main()
