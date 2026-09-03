"""UWYK eval on the 6 synthetic case studies — PEHE + ATE err.

Calls PartialGraphConditionedInterventionalPFN forward directly (via
wrapper.model) using the SAME preprocessing the sklearn wrapper does:

- X-standardize (mean/std on TRAIN X) then 0-pad to model.num_features
- Y-scale to [-1, 1] via y_min/y_max
- Adj (F+2, F+2): real block has T→Y = +1, X→T = +1, X→Y = +1;
  padded feature slots get -1 on rows/cols/diag
- TWO separate forwards (T=1 arm, T=0 arm) — do NOT concat
- Get scaled mean via wrapper.bar_distribution.mean(out['predictions'])
- CATE = (arm1 − arm0) × y_range / 2   (y_min cancels)

Env: DATASET, OUT, UWYK_SRC, UWYK_CKPT, UWYK_CONFIG (or UWYK_CKPT_DIR),
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


def _build_adj(F, n_real, anc_mode):
    """(F+2, F+2) adjacency. Order: [T=0, Y=1, X_0 ... X_{F-1}].
    noanc: real block all 0, padded slots -1
    anc:   T→Y=+1, X_i→T=+1, X_i→Y=+1 for real; padded slots -1
    """
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    T_idx, Y_idx, feat_off = 0, 1, 2
    if anc_mode == 'anc':
        A[T_idx, Y_idx] = 1.0
        for i in range(n_real):
            A[feat_off + i, T_idx] = 1.0
            A[feat_off + i, Y_idx] = 1.0
    # padded feature slots → -1 rows + cols + diagonal
    for i in range(n_real, F):
        A[feat_off + i, :] = -1.0
        A[:, feat_off + i] = -1.0
        A[feat_off + i, feat_off + i] = -1.0
    return A


def _standardize_and_pad(X_train, X_test, F):
    """Compute (mean, std) on TRAIN X (real cols), pad to F with zeros
    (padded cols have mean=0, std=1 so they stay zero after standardization)."""
    L = X_train.shape[1]
    mean = np.zeros((1, F), dtype=np.float32); mean[0, :L] = X_train.mean(0)
    std  = np.ones ((1, F), dtype=np.float32); std [0, :L] = X_train.std(0) + 1e-8
    Xtr_p = np.zeros((X_train.shape[0], F), dtype=np.float32); Xtr_p[:, :L] = X_train
    Xte_p = np.zeros((X_test .shape[0], F), dtype=np.float32); Xte_p[:, :L] = X_test
    Xtr_s = (Xtr_p - mean) / std
    Xte_s = (Xte_p - mean) / std
    return Xtr_s, Xte_s


@torch.no_grad()
def _cate_uwyk(wrapper, X_train, T_train, y_train, X_test, anc_mode):
    underlying = wrapper.model
    bd = wrapper.bar_distribution
    F = underlying.num_features
    n_real = X_train.shape[1]

    # X: standardize + 0-pad to F
    Xtr_s, Xte_s = _standardize_and_pad(X_train.astype(np.float32),
                                         X_test.astype(np.float32), F)

    # Y: [-1, 1] scaling via TRAIN y_min / y_max
    y_min = float(y_train.min()); y_max = float(y_train.max())
    y_rng = max(y_max - y_min, 1e-8)
    Y_scaled = (2.0 * (y_train.astype(np.float32) - y_min) / y_rng - 1.0)

    # Adj (F+2, F+2)
    adj = _build_adj(F, n_real, anc_mode)

    # Tensors — shapes: X(1,N,F), T(1,N,1), Y(1,N,1), Xq(1,M,F), Tq(1,M,1), adj(1,F+2,F+2)
    X_obs = torch.from_numpy(Xtr_s).unsqueeze(0).to(DEVICE)
    T_obs = torch.from_numpy(T_train.astype(np.float32)).reshape(1, -1, 1).to(DEVICE)
    Y_obs = torch.from_numpy(Y_scaled).reshape(1, -1, 1).to(DEVICE)
    X_intv = torch.from_numpy(Xte_s).unsqueeze(0).to(DEVICE)
    adj_t = torch.from_numpy(adj).unsqueeze(0).to(DEVICE)

    def _arm(t_val):
        T_intv = torch.full((1, X_test.shape[0], 1), float(t_val),
                             dtype=torch.float32, device=DEVICE)
        out = underlying(X_obs, T_obs, Y_obs, X_intv, T_intv, adj_t)
        pred = out['predictions'] if isinstance(out, dict) else out   # (1, M, num_bars+extras)
        # Use BarDistribution mean — same as wrapper does
        mean = bd.mean(pred).squeeze(0).cpu().numpy()                  # (M,) in scaled Y
        return mean

    e0_scaled = _arm(0.0)
    e1_scaled = _arm(1.0)

    # CATE inverse: scaled_diff × y_range / 2  (y_min cancels)
    cate = (e1_scaled - e0_scaled) * (y_rng / 2.0)
    return cate.astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = SCMCaseStudyDataset(DATASET)
    n = ds.n_tables if not MAX_REAL else min(ds.n_tables, int(MAX_REAL))
    print(f'[bootstrap] UWYK {ANC_MODE}  {DATASET}  n={n}', flush=True)
    wrapper = _load_uwyk()

    rows = []; t0 = time.time()
    for r in range(n):
        cate_ds, _ = ds[r]
        try:
            cate_pred = _cate_uwyk(wrapper,
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
    print(f'\n══ {DATASET}  UWYK-{ANC_MODE}  n={len(rows)} ══')
    for k in ('pehe_raw', 'err_raw'):
        m, s = _ms(k); print(f'  {k:10s} = {m:8.3f} ± {s:6.3f}')


if __name__ == '__main__':
    main()
