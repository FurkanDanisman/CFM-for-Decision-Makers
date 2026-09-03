"""UWYK eval on the 6 synthetic case studies — PEHE + ATE err.

Uses the RAW underlying UWYK model (PartialGraphConditionedInterventionalPFN)
via the wrapper's .model attribute — bypasses the sklearn wrapper's
preprocessing which was collapsing predictions to zero on non-standard
inputs. Follows the same forward pattern as eval_graph2d_realcause.py:

  - X padded with 0 to model.num_features
  - Adj matrix (F+2, F+2) with real block + -1 around padded slots
  - Y scaled to [-1, 1] via y_min/y_rng of the training Y
  - Model forward with (X_obs, T_obs, Y_obs, X_intv, adj_t)
  - Concat both do(0) + do(1) queries → one forward pass
  - Extract 1D nbins marginal, compute mean, un-scale, subtract

Env vars: DATASET, OUT, UWYK_SRC, UWYK_CKPT, UWYK_CONFIG (or UWYK_CKPT_DIR),
          ANC_MODE (noanc|anc), MAX_REAL, DOPFN_DATA_ROOT
"""
from __future__ import annotations
import argparse, os, sys, time, importlib
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=os.environ.get('DATASET', 'Observed_Confounder'))
args, _ = parser.parse_known_args()
DATASET       = args.dataset
OUT           = os.environ['OUT']
UWYK_SRC      = os.environ['UWYK_SRC']
UWYK_CKPT_DIR = os.environ.get('UWYK_CKPT_DIR', '')
UWYK_CKPT     = os.environ.get('UWYK_CKPT', '')
UWYK_CONFIG   = os.environ.get('UWYK_CONFIG', '')
ANC_MODE      = os.environ.get('ANC_MODE', 'noanc').lower()
MAX_REAL      = os.environ.get('MAX_REAL', '')
assert ANC_MODE in ('noanc', 'anc'), ANC_MODE

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, os.path.join(REPO_SRC, 'benchmarks'))

from scm_case_study_dataset import SCMCaseStudyDataset  # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── UWYK model load (via wrapper.model — the underlying transformer) ──────
def _load_uwyk():
    saved = {}
    for name in list(sys.modules):
        if name in ('models', 'utils') or name.startswith('models.') or name.startswith('utils.'):
            saved[name] = sys.modules.pop(name)
    sys.path.insert(0, UWYK_SRC)
    pre_mod = importlib.import_module('models.PreprocessingGraphConditionedPFN')
    if UWYK_SRC in sys.path: sys.path.remove(UWYK_SRC)
    for name in list(sys.modules):
        if name in ('models', 'utils') or name.startswith('models.') or name.startswith('utils.'):
            del sys.modules[name]
    sys.modules.update(saved)

    _orig_load = torch.load
    def _patched(*a, **kw):
        kw.setdefault('weights_only', False); return _orig_load(*a, **kw)
    torch.load = _patched
    try:
        if UWYK_CKPT and UWYK_CONFIG:
            ck_p, cfg_p = UWYK_CKPT, UWYK_CONFIG
        else:
            fin_ck  = os.path.join(UWYK_CKPT_DIR, 'final_model_with_bardist.pt')
            fin_cfg = os.path.join(UWYK_CKPT_DIR, 'final_model_with_bardist_config.yaml')
            if os.path.isfile(fin_ck) and os.path.isfile(fin_cfg):
                ck_p, cfg_p = fin_ck, fin_cfg
            else:
                ck_p  = os.path.join(UWYK_CKPT_DIR, 'best_model.pt')
                cfg_p = os.path.join(UWYK_CKPT_DIR, 'best_model_config.yaml')
        assert os.path.isfile(ck_p),  f'UWYK ckpt missing: {ck_p}'
        assert os.path.isfile(cfg_p), f'UWYK config missing: {cfg_p}'
        _dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'[uwyk] loading  ckpt={ck_p}  cfg={cfg_p}  device={_dev}', flush=True)
        wrapper = pre_mod.PreprocessingGraphConditionedPFN(
            config_path=cfg_p, checkpoint_path=ck_p, device=_dev, verbose=False,
            random_state=42, use_clustering=False,
        ).load()
    finally:
        torch.load = _orig_load
    return wrapper


def _pad_features(X, F):
    if X.shape[1] == F: return X
    if X.shape[1] > F:  return X[:, :F]
    return np.hstack([X, np.zeros((X.shape[0], F - X.shape[1]), dtype=X.dtype)])


def _standardize_train_test(Xtr, Xte, eps=1e-8):
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + eps
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _scale_y(y_train):
    """[-1, 1] scaling via y_min/y_max of training Y. Returns scaled y + (y_min, y_rng)."""
    y = np.asarray(y_train, dtype=np.float32).reshape(-1)
    y_min = float(y.min()); y_max = float(y.max())
    y_rng = max(y_max - y_min, 1e-6)
    y_scaled = 2.0 * (y - y_min) / y_rng - 1.0
    return y_scaled.reshape(-1, 1), y_min, y_rng


def _build_anc_none(F, n_real):
    """Real block all 0, padded slots -1 (rows + cols + diagonal)."""
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    feat_off = 2
    for i in range(n_real, F):
        A[feat_off + i, :] = -1.0
        A[:, feat_off + i] = -1.0
        A[feat_off + i, feat_off + i] = -1.0
    return A


def _build_anc_full(F, n_real):
    """+1 for T→Y and each real X→T, X→Y; propagate; -1 around padded slots."""
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    T_idx, Y_idx, feat_off = 0, 1, 2
    A[T_idx, Y_idx] = 1.0
    for i in range(n_real):
        A[feat_off + i, T_idx] = 1.0
        A[feat_off + i, Y_idx] = 1.0
    # Padded slots: -1
    for i in range(n_real, F):
        A[feat_off + i, :] = -1.0
        A[:, feat_off + i] = -1.0
        A[feat_off + i, feat_off + i] = -1.0
    # Propagate ancestor knowledge to fill entailed entries
    try:
        from utils.graph_utils import propagate_ancestor_knowledge
        real_n = 2 + n_real
        real_block = torch.from_numpy(A[:real_n, :real_n].copy())
        real_block = propagate_ancestor_knowledge(real_block, raise_on_inconsistent=False)
        A[:real_n, :real_n] = real_block.numpy().astype(np.float32)
    except Exception:
        pass
    return A


@torch.no_grad()
def _cate_uwyk_raw(model, X_train, T_train, y_train, X_test, anc_mode):
    """Raw forward via wrapper.model — bypass sklearn wrapper preprocessing."""
    underlying = model.model                # PartialGraphConditionedInterventionalPFN
    F = underlying.num_features
    n_real = X_train.shape[1]

    # 1. X: standardise then 0-pad
    X_tr_s, X_te_s = _standardize_train_test(X_train.astype(np.float32),
                                              X_test.astype(np.float32))
    X_tr = _pad_features(X_tr_s, F)
    X_te = _pad_features(X_te_s, F)

    # 2. Y scaled to [-1, 1]
    Y_scaled, y_min, y_rng = _scale_y(y_train)

    # 3. Adj: -1 around padded slots
    adj = _build_anc_none(F, n_real) if anc_mode == 'noanc' else _build_anc_full(F, n_real)

    # 4. Concat both arms into one forward
    M = X_te.shape[0]
    X_intv_np = np.vstack([X_te, X_te]).astype(np.float32)
    T_intv_np = np.concatenate([np.zeros(M, dtype=np.float32),
                                 np.ones(M, dtype=np.float32)])

    X_obs = torch.from_numpy(X_tr).unsqueeze(0).to(DEVICE)                            # (1, N, F)
    T_obs = torch.from_numpy(T_train.astype(np.float32)).reshape(1, -1, 1).to(DEVICE) # (1, N, 1)
    Y_obs = torch.from_numpy(Y_scaled).unsqueeze(0).to(DEVICE)                        # (1, N, 1)
    X_int = torch.from_numpy(X_intv_np).unsqueeze(0).to(DEVICE)                       # (1, 2M, F)
    T_int = torch.from_numpy(T_intv_np).reshape(1, -1, 1).to(DEVICE)                  # (1, 2M, 1)
    adj_t = torch.from_numpy(adj).unsqueeze(0).to(DEVICE)                              # (1, F+2, F+2)

    # Forward — UWYK's underlying takes (X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix)
    out = underlying(X_obs, T_obs, Y_obs, X_int, T_int, adj_t)
    logits = out['predictions'] if isinstance(out, dict) else out
    logits = logits.squeeze(0).float().cpu().numpy()   # (2M, nbins)

    # BarDistribution mean per query
    nbins = logits.shape[-1]
    # Bin centers on [-1, 1] with `nbins` bins
    edges = np.linspace(-1.0, 1.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = np.exp(logits - logits.max(axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True)
    e_scaled = (p * centers).sum(axis=-1)               # (2M,) mean in [-1,1]

    # Un-scale to raw Y
    e_raw = (e_scaled + 1.0) * 0.5 * y_rng + y_min
    e0 = e_raw[:M]; e1 = e_raw[M:]
    return (e1 - e0).astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = SCMCaseStudyDataset(DATASET)
    n = ds.n_tables if not MAX_REAL else min(ds.n_tables, int(MAX_REAL))
    print(f'[bootstrap] UWYK-raw {ANC_MODE}  {DATASET}  n={n}', flush=True)
    model = _load_uwyk()

    rows = []; t0 = time.time()
    for r in range(n):
        cate_ds, _ = ds[r]
        try:
            cate_pred = _cate_uwyk_raw(model,
                                        cate_ds.X_train, cate_ds.t_train, cate_ds.y_train,
                                        cate_ds.X_test, ANC_MODE)
        except Exception as e:
            print(f'r={r:03d}  ERROR: {type(e).__name__}: {e}', flush=True)
            continue
        true_cate = np.asarray(cate_ds.true_cate, dtype=np.float32).reshape(-1)
        pehe = float(np.sqrt(np.mean((cate_pred - true_cate) ** 2)))
        ate_true = float(true_cate.mean()); ate_hat = float(cate_pred.mean())
        err = abs(ate_hat - ate_true) / max(abs(ate_true), 0.1)
        row = {'dataset': DATASET, 'realization': r, 'anc_mode': ANC_MODE,
               'true_ate': ate_true, 'ate_pred': ate_hat,
               'pehe_raw': pehe, 'err_raw': err}
        rows.append(row)
        np.savez(os.path.join(OUT, f'r{r:03d}.npz'), **{k: np.array(v) for k, v in row.items()})
        print(f'r={r:03d}  pehe={pehe:6.3f}  err={err:5.3f}  ate={ate_hat:+5.2f} vs true {ate_true:+5.2f}  '
              f'({time.time()-t0:.0f}s)', flush=True)

    def _ms(k):
        v = np.array([r[k] for r in rows]); return v.mean(), v.std(ddof=1)/np.sqrt(len(v))
    print(f'\n══ {DATASET}  UWYK-raw-{ANC_MODE}  n={len(rows)} ══')
    for k in ('pehe_raw', 'err_raw'):
        m, s = _ms(k); print(f'  {k:10s} = {m:8.3f} ± {s:6.3f}')


if __name__ == '__main__':
    main()
