"""Evaluate cpfn2d checkpoint on IHDP using RAW MEAN CATE.

CATE per query = E[Y_do1] - E[Y_do0], where E[Y_do_t] is computed by:
  1. Softmaxing the interior K*K logits into a joint p(Y_do0, Y_do1 | x)
  2. Marginalising to p(Y_do_t | x) by summing along the other axis
  3. Weighted sum of bin centres (from the checkpoint's fitted edges)
  4. Un-standardising the standardised expected values back to raw Y units
     using the per-task pooled (y_mean, y_std) computed on Y_context

No MALC. No boundary-region handling. Just raw histogram means, matching how
the joint head was TRAINED (per-task pooled standardisation, edges fitted
from standardised warmup Y).

Env vars:
  CKPT      : path to a step_XXXXX.pt or step_XXXXX_final.pt
  OUT       : output directory for per-realization NPZ files
  CAUSALPFN : path to external/causalpfn (for backbone imports)
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch


CKPT      = os.environ['CKPT']
OUT       = os.environ.get('OUT', './results_cpfn2d_ihdp_raw')
CAUSALPFN = os.environ['CAUSALPFN']

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, CAUSALPFN)
sys.path.insert(0, CAUSALPFN + '/src')

from benchmarks import IHDPDataset  # noqa: E402
from training_causalpfn2d.model_causalpfn_2d import CausalPFN2DHead  # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


@torch.no_grad()
def cate_raw_mean(model, X_train, T_train, Y_train_raw, X_test, edges, J):
    """Return per-query CATE in raw Y units via marginal-mean of the 2D joint."""
    X_ctx = torch.from_numpy(X_train.astype(np.float32)).unsqueeze(0).to(DEVICE)
    t_ctx = torch.from_numpy(T_train.astype(np.float32)).unsqueeze(0).to(DEVICE)
    y_ctx_raw = torch.from_numpy(Y_train_raw.astype(np.float32)).unsqueeze(0).to(DEVICE)
    X_q = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(0).to(DEVICE)

    # Per-task pooled standardisation — matches training's forward().
    y_mean = y_ctx_raw.mean(dim=1, keepdim=True)                # (1, 1)
    y_std  = y_ctx_raw.std(dim=1, keepdim=True).clamp(min=1e-6)  # (1, 1)
    y_ctx_std = (y_ctx_raw - y_mean) / y_std

    # Forward → interior K² joint.
    logits = model._forward_logits(X_ctx, t_ctx, y_ctx_std, X_q)  # (1, N_q, nbins_2d)
    interior = logits[..., : J * J]                                # (1, N_q, J*J)
    p = torch.softmax(interior, dim=-1).reshape(1, -1, J, J)       # (1, N_q, J, J)

    # Marginalise. p[..., j0, j1] = P(Y_do0 in bin j0, Y_do1 in bin j1 | x).
    p_y0 = p.sum(dim=-1)                                           # (1, N_q, J)
    p_y1 = p.sum(dim=-2)                                           # (1, N_q, J)

    # Bin centres from the checkpoint's fitted edges (standardised-space).
    edges_dev = edges.to(DEVICE)
    centres = 0.5 * (edges_dev[:-1] + edges_dev[1:])               # (J,)

    e_y0_std = (p_y0 * centres.view(1, 1, J)).sum(dim=-1)          # (1, N_q)
    e_y1_std = (p_y1 * centres.view(1, 1, J)).sum(dim=-1)

    # Un-standardise back to raw Y units.
    y_mean_1d = y_mean.squeeze(-1)                                 # (1,)
    y_std_1d  = y_std.squeeze(-1)
    e_y0 = e_y0_std * y_std_1d + y_mean_1d
    e_y1 = e_y1_std * y_std_1d + y_mean_1d

    return (e_y1 - e_y0).squeeze(0).cpu().numpy()


def evaluate(realization: int, model, edges, J, F):
    ds = IHDPDataset()
    cate_ds = ds[realization][0]
    X_tr = cate_ds.X_train.astype(np.float32)
    T_tr = cate_ds.t_train.astype(np.float32).reshape(-1)
    y_tr = cate_ds.y_train.astype(np.float32).reshape(-1)
    X_te = cate_ds.X_test.astype(np.float32)
    true_cate = cate_ds.true_cate.astype(np.float32)
    true_ate = float(true_cate.mean())

    X_tr_std, X_te_std = _standardize_train_test(X_tr, X_te)
    X_tr_p = _pad_features(X_tr_std, F)
    X_te_p = _pad_features(X_te_std, F)

    cate = cate_raw_mean(model, X_tr_p, T_tr, y_tr, X_te_p, edges, J)
    pehe = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
    ate_hat = float(cate.mean())
    err_ate = abs(ate_hat - true_ate) / max(abs(true_ate), 1e-9)

    return {
        'dataset': 'IHDP',
        'realization': realization,
        'true_ate': true_ate,
        'pehe_cpfn2d_raw': pehe,
        'err_cpfn2d_raw':  err_ate,
        'ate_cpfn2d_raw':  ate_hat,
    }


def _load_state_dict_safe(model, sd):
    """Handle torch.compile _orig_mod. prefix and shape-mismatch gracefully.

    Uses a GLOBAL replace on '_orig_mod.' — the prefix may be mid-path when
    only a submodule was compiled (e.g. `backbone._orig_mod.encoder.weight`
    when the trainer did `inner.backbone = torch.compile(inner.backbone)`).
    Prior version used k.startswith and silently dropped 148/149 keys.
    """
    if any('_orig_mod.' in k for k in sd):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        print('[eval] stripped _orig_mod. prefix (global) from state_dict keys', flush=True)
    ref = model.state_dict()
    kept, skipped = {}, []
    for k, v in sd.items():
        if k in ref and ref[k].shape != v.shape:
            skipped.append((k, tuple(v.shape), tuple(ref[k].shape)))
            continue
        kept[k] = v
    missing, unexpected = model.load_state_dict(kept, strict=False)
    print(f'[eval] load_state_dict: missing={len(missing)}  unexpected={len(unexpected)}  '
          f'shape-mismatch-skipped={len(skipped)}', flush=True)
    for k, shp_have, shp_want in skipped:
        print(f'  SKIP {k}: ckpt {shp_have} vs model {shp_want}', flush=True)
    # Hard abort: if we're missing most of the backbone, we're about to
    # evaluate a random-init model. That's what silently happened for the
    # entire previous eval sweep (missing=148/149).
    if len(missing) > 20:
        raise RuntimeError(
            f'[eval] ABORT: {len(missing)} missing keys after load — refusing '
            f'to eval a random-init model. First missing: {list(missing)[:8]}'
        )


def main():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    cfg = ck['config']
    edges = ck['edges']
    step = ck.get('step', 'unknown')

    print(f'[bootstrap] ckpt={CKPT}')
    print(f'[bootstrap] step={step}  J={cfg["J"]}  num_features={cfg["num_features"]}')
    print(f'[bootstrap] edges: J+1={edges.numel()} range=[{edges[0].item():.4f}, '
          f'{edges[-1].item():.4f}]  bin_width={((edges[-1]-edges[0])/cfg["J"]).item():.4f}')

    model = CausalPFN2DHead(
        J=cfg['J'],
        num_features=cfg['num_features'],
        ninp=cfg['ninp'],
        nhid=cfg['nhid'],
        nhead=cfg['nhead'],
        nlayers=cfg['nlayers'],
        dropout=cfg.get('dropout', 0.0),
        n_out=cfg.get('n_out', 10),
    ).to(DEVICE)

    _load_state_dict_safe(model, ck['model_state_dict'])
    model.eval()

    J = cfg['J']
    F = cfg['num_features']

    os.makedirs(OUT, exist_ok=True)
    all_rows = []
    t0 = time.time()
    for r in range(100):
        row = evaluate(r, model, edges, J, F)
        out_path = os.path.join(OUT, f'r{r:03d}.npz')
        np.savez(out_path, **{k: np.array(v) for k, v in row.items()})
        all_rows.append(row)
        print(
            f'r={r:03d}  '
            f'pehe={row["pehe_cpfn2d_raw"]:6.3f}  '
            f'err_ate={row["err_cpfn2d_raw"]:5.3f}  '
            f'ate_hat={row["ate_cpfn2d_raw"]:+6.3f}  '
            f'true_ate={row["true_ate"]:+6.3f}  '
            f'({time.time()-t0:.0f}s)',
            flush=True,
        )

    def _mean_sem(k):
        v = np.array([r[k] for r in all_rows])
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

    print(f'\n══ IHDP summary (n={len(all_rows)}, step={step}) ══')
    for k in ('pehe_cpfn2d_raw', 'err_cpfn2d_raw'):
        m, s = _mean_sem(k)
        print(f'  {k:25s} = {m:8.3f} ± {s:6.3f}')


if __name__ == '__main__':
    main()
