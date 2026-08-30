"""Minimal IHDP eval on causalpfn_v0.pt — bypasses CATEEstimator entirely.

Loads the raw state_dict into TabDPTLongContextModel and does pure ICL
inference (all train as context). Does NOT use CausalPFN's retrieval
step (faiss-based nearest-neighbor context selection), because faiss
can't be installed on Alliance Canada.

Reference: CausalPFN's InContextModel does per-arm Y standardization:
  y0_shift/scale from control units, y1_shift/scale from treated units,
  each arm standardised with its own stats.

For each query, run the model twice — once with T=0, once with T=1 —
extract E[Y | T=t] from the 1D softmax head over vmin/vmax=[-10, +10]
with 1024 bins, un-standardise with the arm's own stats, then
  CATE = E[Y | T=1] - E[Y | T=0].

The PEHE we get here should be compared against our cpfn2d which ALSO
does pure ICL — NOT against the paper's PEHE 0.58 (that uses retrieval
which needs faiss).

Env: CPFN_V0_LOCAL, OUT, CAUSALPFN
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch


CPFN_V0_LOCAL = os.environ.get('CPFN_V0_LOCAL',
                                '/scratch/furkanbd/rpfn_bench_kit/warmstart/causalpfn_v0.pt')
OUT           = os.environ.get('OUT', './results_causalpfn_v0_ihdp_minimal')
CAUSALPFN     = os.environ['CAUSALPFN']

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, CAUSALPFN)
sys.path.insert(0, CAUSALPFN + '/src')

from benchmarks import IHDPDataset  # noqa: E402  — CausalPFN's benchmarks pkg
# Import ONLY the model class, not the estimator (which would trigger faiss).
# But causalpfn/__init__.py runs first and imports causal_estimator, so we
# still need the sitecustomize shim to stub faiss. That's OK — we never call
# faiss functions here, we just need the import to succeed.
from causalpfn.models.model import TabDPTLongContextModel  # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CausalPFN's InContextModel defaults — from src/causalpfn/models/icl_model.py
VMIN  = -10.0
VMAX  = +10.0
NBINS = 1024
BIN_EDGES   = torch.linspace(VMIN, VMAX, NBINS + 1)          # (1025,)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])         # (1024,)


def _strip_prefix(sd, prefix, drop_no_prefix=False):
    """Strip `prefix` from any key that has it.

    - drop_no_prefix=False (default): keys without the prefix are kept as-is.
    - drop_no_prefix=True: keys without the prefix are dropped. Used when the
      un-prefixed keys are top-level buffers on the wrapper class (InContextModel)
      that don't exist on our unwrapped TabDPTLongContextModel — those buffers
      are `bin_edges`, `bin_width`, `bin_centers` in causalpfn's InContextModel.

    causalpfn_v0.pt and their epoch_XXXX.pt checkpoints save an
    `InContextModel` state_dict which nests the backbone under `model.`
    and adds 3 top-level buffers. Both need to be handled.
    """
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        elif not drop_no_prefix:
            out[k] = v
        # else: dropped
    return out


def _pad_features(X: np.ndarray, F: int) -> np.ndarray:
    if X.shape[1] == F: return X
    if X.shape[1] > F:  return X[:, :F]
    return np.hstack([X, np.zeros((X.shape[0], F - X.shape[1]), dtype=X.dtype)])


def _standardize_train_test(Xtr, Xte, eps=1e-8):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + eps
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _per_arm_shift_scale(t_context, y_context, eps=1e-6):
    """Match CausalPFN's InContextModel — per-arm mean/std for standardisation."""
    t = t_context.reshape(-1)
    y = y_context.reshape(-1)
    y0 = y[t < 0.5]
    y1 = y[t > 0.5]
    y0_shift, y0_scale = float(y0.mean()), float(y0.std() + eps) if y0.size else (0.0, 1.0)
    y1_shift, y1_scale = float(y1.mean()), float(y1.std() + eps) if y1.size else (0.0, 1.0)
    return y0_shift, y0_scale, y1_shift, y1_scale


@torch.no_grad()
def _forward_one_arm(model, X_ctx, T_ctx, Y_std_ctx, X_q, t_val, num_features):
    """Run TabDPTLongContextModel forward with T_query fixed to t_val at
    every query position. Returns (B, N_q, nbins) logits from the last
    nbins columns of the head output."""
    B, N_ctx, F = X_ctx.shape
    N_q = X_q.shape[1]
    assert F == num_features, (F, num_features)

    # Concatenate (T, X) as the input features. CausalPFN prepends T as
    # column 0 (verified from their trainer).
    t_ctx_col = T_ctx.reshape(B, N_ctx, 1)
    t_q_col   = torch.full((B, N_q, 1), float(t_val), dtype=X_q.dtype, device=X_q.device)
    xt_ctx = torch.cat([t_ctx_col, X_ctx], dim=-1)             # (B, N_ctx, F+1)
    xt_q   = torch.cat([t_q_col,   X_q],   dim=-1)             # (B, N_q,   F+1)
    x_all  = torch.cat([xt_ctx, xt_q], dim=1)                  # (B, N_ctx+N_q, F+1)

    # Backbone expects (S, B, F+1); y_src is context-only.
    x_src = x_all.transpose(0, 1).contiguous()                 # (S, B, F+1)
    y_src = Y_std_ctx.transpose(0, 1).contiguous()             # (N_ctx, B)

    pred = model(x_src, y_src)                                 # (N_q, B, n_out+nbins)
    pred = pred.transpose(0, 1).contiguous()                   # (B, N_q, n_out+nbins)
    return pred[..., -NBINS:]                                  # (B, N_q, nbins)


@torch.no_grad()
def cate_causalpfn_v0(model, X_train, T_train, Y_train_raw, X_test, num_features):
    """CausalPFN pure-ICL CATE per query, in raw Y units."""
    # Per-arm standardisation stats (from raw arrays, then applied per-arm).
    y0s, y0sc, y1s, y1sc = _per_arm_shift_scale(T_train, Y_train_raw)

    # Standardise Y_context per-arm (each row uses its own arm's stats).
    y_ctx_std = np.where(T_train.reshape(-1) > 0.5,
                          (Y_train_raw - y1s) / y1sc,
                          (Y_train_raw - y0s) / y0sc).astype(np.float32)

    X_ctx = torch.from_numpy(X_train.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_ctx = torch.from_numpy(T_train.astype(np.float32)).unsqueeze(0).to(DEVICE)
    Y_ctx = torch.from_numpy(y_ctx_std).unsqueeze(0).to(DEVICE)
    X_q   = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(0).to(DEVICE)

    centers = BIN_CENTERS.to(DEVICE)                              # (nbins,)

    # Two forwards: one per arm.
    logits_t0 = _forward_one_arm(model, X_ctx, T_ctx, Y_ctx, X_q, 0.0, num_features)
    logits_t1 = _forward_one_arm(model, X_ctx, T_ctx, Y_ctx, X_q, 1.0, num_features)

    p0 = torch.softmax(logits_t0.float(), dim=-1)                # (1, N_q, nbins)
    p1 = torch.softmax(logits_t1.float(), dim=-1)
    e_y0_std = (p0 * centers.view(1, 1, -1)).sum(dim=-1).squeeze(0).cpu().numpy()
    e_y1_std = (p1 * centers.view(1, 1, -1)).sum(dim=-1).squeeze(0).cpu().numpy()

    # Un-standardise per arm.
    e_y0 = e_y0_std * y0sc + y0s
    e_y1 = e_y1_std * y1sc + y1s
    return (e_y1 - e_y0).astype(np.float32)


def evaluate(realization, model, num_features):
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

    cate = cate_causalpfn_v0(model, X_tr_p, T_tr, y_tr, X_te_p, num_features)
    pehe = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
    ate_hat = float(cate.mean())
    err_ate = abs(ate_hat - true_ate) / max(abs(true_ate), 1e-9)

    return {'dataset': 'IHDP', 'realization': realization,
            'true_ate': true_ate,
            'pehe_cpfn_v0': pehe, 'err_cpfn_v0': err_ate, 'ate_cpfn_v0': ate_hat}


def main():
    print(f'[bootstrap] device={DEVICE}  ckpt={CPFN_V0_LOCAL}', flush=True)

    ck = torch.load(CPFN_V0_LOCAL, map_location='cpu', weights_only=False)
    cfg = ck.get('model_config', {})
    sd  = ck['model_state_dict']
    # Their trainer saves InContextModel state (which wraps TabDPTLongContextModel
    # under `.model` and adds 3 top-level buffers: bin_edges/width/centers).
    # We only want the TabDPTLongContextModel state → strip `model.` prefix
    # AND drop keys that don't have it.
    sd = _strip_prefix(sd, 'model.', drop_no_prefix=True)
    print(f'[bootstrap] ckpt model_config keys: '
          f'{list(cfg.keys()) if isinstance(cfg, dict) else type(cfg).__name__}', flush=True)
    print(f'[bootstrap] state_dict: {len(sd)} keys after prefix strip + buffer drop',
          flush=True)

    # Instantiate matching architecture. Config hints from state_dict shapes:
    #   ninp=384 (from encoder shapes we probed), nhid=768, nhead=6,
    #   nlayers=20, nbins=1024, num_features from encoder.weight.shape[1]-1.
    # If model_config has explicit values, use those.
    enc_w = sd.get('encoder.weight')
    num_features_plus_t = enc_w.shape[1] if enc_w is not None else 101
    num_features = num_features_plus_t - 1
    ninp     = cfg.get('ninp',    enc_w.shape[0] if enc_w is not None else 384)
    nhid     = cfg.get('nhid',    768)
    nhead    = cfg.get('nhead',   6)
    nlayers  = cfg.get('nlayers', 20)
    n_out    = cfg.get('n_out',   10)
    dropout  = cfg.get('dropout', 0.0)
    print(f'[bootstrap] model: ninp={ninp} nhid={nhid} nhead={nhead} nlayers={nlayers} '
          f'nbins={NBINS} num_features={num_features} n_out={n_out}', flush=True)

    model = TabDPTLongContextModel(
        dropout=dropout, n_out=n_out, nhead=nhead, nhid=nhid, ninp=ninp,
        nlayers=nlayers, num_features=num_features_plus_t, nbins=NBINS,
    ).to(DEVICE)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[bootstrap] load: missing={len(missing)} unexpected={len(unexpected)}', flush=True)
    if len(missing) > 5 or len(unexpected) > 5:
        raise RuntimeError(
            f'load mismatch — first missing: {list(missing)[:5]}, '
            f'first unexpected: {list(unexpected)[:5]}'
        )
    model.eval()

    os.makedirs(OUT, exist_ok=True)
    rows = []
    t0 = time.time()
    for r in range(100):
        row = evaluate(r, model, num_features)
        np.savez(os.path.join(OUT, f'r{r:03d}.npz'),
                 **{k: np.array(v) for k, v in row.items()})
        rows.append(row)
        print(
            f'r={r:03d}  pehe={row["pehe_cpfn_v0"]:6.3f}  '
            f'err_ate={row["err_cpfn_v0"]:5.3f}  '
            f'ate_hat={row["ate_cpfn_v0"]:+6.3f}  true_ate={row["true_ate"]:+6.3f}  '
            f'({time.time()-t0:.0f}s)', flush=True,
        )

    def _ms(k):
        v = np.array([r[k] for r in rows])
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

    print(f'\n══ IHDP summary (causalpfn_v0.pt, pure ICL, n={len(rows)}) ══')
    for k in ('pehe_cpfn_v0', 'err_cpfn_v0'):
        m, s = _ms(k)
        print(f'  {k:15s} = {m:8.3f} ± {s:6.3f}')
    print(f'\n  Note: paper PEHE 0.58 uses retrieval-augmented inference (needs faiss).')
    print(f'  This is PURE ICL — same inference protocol as our cpfn2d evals.')


if __name__ == '__main__':
    main()
