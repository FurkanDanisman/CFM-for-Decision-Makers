"""Evaluate a graph-conditioned 2D-head checkpoint on IHDP.

For each IHDP realization we run inference twice with the same checkpoint:
  * anc   -- true adjacency (T→Y, all real features → T, all real features → Y);
             padded slots masked with -1.
  * noanc -- adjacency zeroed everywhere (padded slots still -1).

Per-realization CATE is the raw mean of the model's 2D joint (interior
softmax over the first J**2 logits, means from bin centres, inverse-
scaled by the observed Y_range). PEHE and epsilon-ATE are computed per
realization and aggregated at the end. Output: one npz per realization
with pehe/err/ate for each of the two adjacency modes.

Usage:
    CKPT=/scratch/.../checkpoints_graph2d/step_10000.pt \
    OUT=/scratch/.../results_graph2d_ihdp \
    UWYK=/scratch/.../external/uwyk \
    CAUSALPFN=/scratch/.../external/causalpfn \
    python -u benchmarks/eval_graph2d/eval_graph2d_ihdp.py
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch


CKPT      = os.environ['CKPT']
OUT       = os.environ.get('OUT', './results_graph2d_ihdp')
UWYK      = os.environ['UWYK']
CAUSALPFN = os.environ['CAUSALPFN']

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, UWYK)
sys.path.insert(0, UWYK + '/src')
sys.path.insert(0, CAUSALPFN)
# CausalPFN's benchmarks/*.py transitively import `from causalpfn.synthetic
# import ...`, so the causalpfn package (which lives at external/causalpfn/src/
# causalpfn/) needs to be importable too. Without this line the eval crashes
# at "ModuleNotFoundError: No module named 'causalpfn'" (job 5035359).
sys.path.insert(0, CAUSALPFN + '/src')

from benchmarks import IHDPDataset  # noqa: E402
from training_graph2d.model_graph_2d import GraphConditioned2DHead  # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _pad_features(X: np.ndarray, F: int) -> np.ndarray:
    if X.shape[1] == F:
        return X
    if X.shape[1] > F:
        return X[:, :F]
    pad = np.zeros((X.shape[0], F - X.shape[1]), dtype=X.dtype)
    return np.hstack([X, pad])


def _standardize_train_test(X_train: np.ndarray, X_test: np.ndarray, eps: float = 1e-8):
    """Z-score X_train and X_test using X_train's mean / std, matching the
    training dataset's _standardize() in PairedInterventionalDataset.py:314.
    The model sees standardised X at every training step, so eval must too."""
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, eps, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd


def _scale_y(y: np.ndarray):
    """Rescale Y to [-1, 1]; return the (ymin, yrange) used so inverse-scaling
    of predicted CATE is available downstream."""
    ymin = float(y.min())
    ymax = float(y.max())
    yrange = max(ymax - ymin, 1e-9)
    y_scaled = 2.0 * (y - ymin) / yrange - 1.0
    return y_scaled.astype(np.float32), ymin, yrange


def build_anc_full(F: int, n_real: int) -> np.ndarray:
    """True-graph adjacency in UWYK's convention (matches
    benchmarks/methods/uwyk.py::build_ancestral_adjacency exactly):
       positions [T, Y, feat_0, ..., feat_{F-1}].
       T -> Y; every real feature -> T; every real feature -> Y.
       Non-edge real positions stay at 0. Padded feature slots masked
       with -1 everywhere.
    """
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    T_idx, Y_idx = 0, 1
    feat_off = 2
    A[T_idx, Y_idx] = 1.0
    for i in range(n_real):
        A[feat_off + i, T_idx] = 1.0
        A[feat_off + i, Y_idx] = 1.0
    for i in range(n_real, F):
        A[feat_off + i, :] = -1.0
        A[:, feat_off + i] = -1.0
        A[feat_off + i, feat_off + i] = -1.0
    return A


def build_anc_none(F: int, n_real: int) -> np.ndarray:
    """No-ancestral-information adjacency: every real edge zeroed; only
    padded slots keep the -1 mask so the model ignores them."""
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    feat_off = 2
    for i in range(n_real, F):
        A[feat_off + i, :] = -1.0
        A[:, feat_off + i] = -1.0
        A[feat_off + i, feat_off + i] = -1.0
    return A


def load_model(ckpt_path: str) -> tuple[GraphConditioned2DHead, dict]:
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ck['config']

    # Attention-sink counts aren't saved to cfg by the trainer. Infer them
    # from the checkpoint's own state_dict: if `sink_rows_x` / `sink_rows_y`
    # (or `sink_cols_x` / `sink_cols_y`) params exist, their first dim IS
    # the sink count. Else 0. This avoids env-var coupling to the sbatch.
    sd = ck['model_state_dict']
    if any('_orig_mod.' in k for k in sd):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        ck['model_state_dict'] = sd

    def _sink_count(prefix):
        for suffix in ('_x', '_y'):
            k = prefix + suffix
            if k in sd and sd[k].dim() >= 1:
                return int(sd[k].shape[0])
        return 0

    n_sample_sink = _sink_count('sink_rows')
    n_feature_sink = _sink_count('sink_cols')

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
        n_sample_attention_sink_rows=n_sample_sink,
        n_feature_attention_sink_cols=n_feature_sink,
    ).to(DEVICE)
    print(f'[load_model] sink_rows={n_sample_sink}  sink_cols={n_feature_sink}  '
          f'(inferred from ckpt state_dict)', flush=True)

    ref = model.state_dict()
    kept = {k: v for k, v in sd.items() if k in ref and ref[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(kept, strict=False)
    print(f'[load_model] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}  '
          f'loaded={len(kept)}/{len(ref)}', flush=True)
    if len(missing) > 5:
        raise RuntimeError(
            f'[load_model] ABORT: {len(missing)} missing keys — refusing to '
            f'eval a partially-loaded model. First missing: {list(missing)[:8]}'
        )
    model.eval()
    return model, cfg


@torch.no_grad()
def cate_from_forward(
    model: GraphConditioned2DHead,
    X_train: np.ndarray, T_train: np.ndarray, Y_train_scaled: np.ndarray,
    X_test: np.ndarray, adj: np.ndarray,
    J: int,
) -> np.ndarray:
    """Run the graph-conditioned 2D-head forward and derive per-query CATE
    on the [-1, 1] scale from the raw interior probabilities."""
    B = 1
    X_obs = torch.from_numpy(X_train.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_obs = torch.from_numpy(T_train.astype(np.float32)).reshape(1, -1, 1).to(DEVICE)
    Y_obs = torch.from_numpy(Y_train_scaled).unsqueeze(0).to(DEVICE)
    X_intv = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(0).to(DEVICE)
    adj_t  = torch.from_numpy(adj).unsqueeze(0).to(DEVICE)

    out = model(X_obs, T_obs, Y_obs, X_intv, adj_t)
    logits = out['predictions'] if isinstance(out, dict) else out
    interior = logits[..., : J * J]
    p = torch.softmax(interior, dim=-1).reshape(B, -1, J, J)

    centres = torch.linspace(
        -1.0 + 1.0 / J, 1.0 - 1.0 / J, J, device=logits.device
    )
    p_y0 = p.sum(dim=-1)
    p_y1 = p.sum(dim=-2)
    e_y0 = (p_y0 * centres.view(1, 1, J)).sum(dim=-1)
    e_y1 = (p_y1 * centres.view(1, 1, J)).sum(dim=-1)
    cate_scaled = (e_y1 - e_y0).squeeze(0).cpu().numpy()
    return cate_scaled


def evaluate(realization: int, model: GraphConditioned2DHead, J: int, F: int):
    ds = IHDPDataset()
    cate_ds = ds[realization][0]
    X_tr_raw = cate_ds.X_train.astype(np.float32)
    T_tr     = cate_ds.t_train.astype(np.float32).reshape(-1)
    y_tr_raw = cate_ds.y_train.astype(np.float32).reshape(-1)
    X_te_raw = cate_ds.X_test.astype(np.float32)
    true_cate = cate_ds.true_cate.astype(np.float32)
    true_ate  = float(true_cate.mean())

    n_real = X_tr_raw.shape[1]
    X_tr_std, X_te_std = _standardize_train_test(X_tr_raw, X_te_raw)
    X_tr = _pad_features(X_tr_std, F)
    X_te = _pad_features(X_te_std, F)
    y_scaled, ymin, yrange = _scale_y(y_tr_raw)
    Y_obs = y_scaled.reshape(-1, 1)

    results = {}
    for mode, adj in (('anc', build_anc_full(F, n_real)),
                       ('noanc', build_anc_none(F, n_real))):
        cate_scaled = cate_from_forward(model, X_tr, T_tr, Y_obs, X_te, adj, J)
        cate = cate_scaled * yrange / 2.0
        pehe = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
        ate_hat = float(cate.mean())
        err_ate = abs(ate_hat - true_ate) / max(abs(true_ate), 1e-9)
        results[f'pehe_graph2d_{mode}'] = pehe
        results[f'err_graph2d_{mode}']  = err_ate
        results[f'ate_graph2d_{mode}']  = ate_hat

    return {
        'dataset': 'IHDP',
        'realization': realization,
        'true_ate': true_ate,
        'n_queries': int(true_cate.size),
        'n_context': int(X_tr_raw.shape[0]),
        **results,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f'[bootstrap] device={DEVICE}  ckpt={CKPT}  out={OUT}', flush=True)
    model, cfg = load_model(CKPT)
    J = cfg['J']
    F = cfg['num_features']
    print(f'[bootstrap] J={J}  F={F}  step={torch.load(CKPT, map_location="cpu", weights_only=False).get("step")}', flush=True)

    n_tables = IHDPDataset().n_tables
    print(f'[bootstrap] IHDP realizations to evaluate: {n_tables}', flush=True)

    all_rows = []
    t0 = time.time()
    for r in range(n_tables):
        row = evaluate(r, model, J, F)
        all_rows.append(row)
        out_path = os.path.join(OUT, f'IHDP_r{r:03d}.npz')
        np.savez(out_path, **{k: np.array(v) for k, v in row.items()})
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
