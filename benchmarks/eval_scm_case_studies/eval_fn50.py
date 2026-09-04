"""fn=50 eval on the 6 synthetic case studies — raw-mean CATE only.

fn=50 uses the InterventionalPFN architecture (NOT CausalPFN2DHead) with a
J×J + 9 + 4 = J²+13 output head. We only need the inner J×J marginals for
raw center-of-mass CATE.

Env vars:
  DATASET       case study name
  OUT           per-realization NPZ dir
  CKPT          fn=50 checkpoint path
  DOPFN_DATA_ROOT
  DOPFN_ROOT
  MAX_REAL      optional cap
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=os.environ.get('DATASET', 'Observed_Confounder'))
args, _ = parser.parse_known_args()
DATASET  = args.dataset
OUT      = os.environ['OUT']
CKPT     = os.environ['CKPT']
MAX_REAL = os.environ.get('MAX_REAL', '')

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, os.path.join(REPO_SRC, 'benchmarks'))

from scm_case_study_dataset import SCMCaseStudyDataset  # noqa: E402
from models.InterventionalPFN import InterventionalPFN   # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _rescale_and_pad(X, F):
    """Match ours_densities: normalize per-feature to [-1,1] then pad to F cols."""
    X = np.asarray(X, dtype=np.float32)
    if X.shape[1] < F:
        X = np.hstack([X, np.zeros((X.shape[0], F - X.shape[1]), dtype=np.float32)])
    elif X.shape[1] > F:
        X = X[:, :F]
    return X


def _load_fn50(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ck['config']; J = cfg['J']
    edges_np = ck['edges'].cpu().numpy()
    model = InterventionalPFN(
        num_features=cfg['num_features'], d_model=cfg['d_model'], depth=cfg['depth'],
        heads_feat=cfg['heads'], heads_samp=cfg['heads'], dropout=0.0,
        output_dim=J*J + 9 + 4, hidden_mult=cfg['hidden_mult'],
        normalize_features=True, normalize_treatment=False,
        use_treatment_in_query=False, use_checkpoint=False,
    ).to(DEVICE).eval()
    model.load_state_dict(ck['model_state_dict'])
    return model, J, edges_np, cfg['num_features']


DENSITY_DUMP = os.environ.get('DENSITY_DUMP', '0') == '1'


@torch.no_grad()
def _cate_fn50(model, J, edges_np, num_features, X_train, T_train, y_train, X_test):
    """One forward pass, extract J×J inner marginals, compute raw center-of-mass CATE.
    Returns (cate_pred, dens_dict) — dens_dict populated only when DENSITY_DUMP=1.
    """
    y_min = float(y_train.min())
    y_rng = max(float(y_train.max() - y_train.min()), 1e-6)
    Y_scaled = ((y_train.astype(np.float32) - y_min) / y_rng * 2.0 - 1.0)

    Xc = _rescale_and_pad(X_train, num_features)
    Xq = _rescale_and_pad(X_test,  num_features)

    Xc_t = torch.from_numpy(Xc).unsqueeze(0).to(DEVICE)                        # (1, N, F)
    Tc_t = torch.from_numpy(T_train.astype(np.float32).reshape(-1, 1)).unsqueeze(0).to(DEVICE)
    Yc_t = torch.from_numpy(Y_scaled.reshape(-1, 1)).unsqueeze(0).to(DEVICE)   # (1, N, 1)
    Xq_t = torch.from_numpy(Xq).unsqueeze(0).to(DEVICE)                        # (1, M, F)

    out = model(Xc_t, Tc_t, Yc_t, Xq_t)
    # InterventionalPFN.forward returns Dict[str, Tensor] with key 'predictions'
    logits = out['predictions'] if isinstance(out, dict) else out
    logits = logits.squeeze(0).float().cpu().numpy()   # (M, J²+9+4)

    interior = logits[..., : J * J]
    p = np.exp(interior - interior.max(axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True)
    p_mat = p.reshape(-1, J, J)
    p_y0 = p_mat.sum(axis=-1); p_y0 /= p_y0.sum(axis=-1, keepdims=True)
    p_y1 = p_mat.sum(axis=-2); p_y1 /= p_y1.sum(axis=-1, keepdims=True)

    centers = 0.5 * (edges_np[:-1] + edges_np[1:])
    e0_scaled = (p_y0 * centers).sum(axis=-1)
    e1_scaled = (p_y1 * centers).sum(axis=-1)

    # Un-scale: scaled Y is (Y - y_min) / y_rng * 2 - 1 → raw = (scaled + 1) * y_rng/2 + y_min
    # CATE is a difference so y_min shift cancels; multiply diff by y_rng/2.
    cate_pred = (e1_scaled - e0_scaled) * (y_rng / 2.0)
    dens = None
    if DENSITY_DUMP:
        # y_shift/y_scale s.t.  y_raw = y_scaled * y_scale + y_shift
        y_scale = y_rng / 2.0
        y_shift = y_min + y_scale
        dens = dict(
            edges=edges_np.astype(np.float32),                    # (J+1,) scaled Y edges [-1,+1]
            p_y0_scaled=p_y0.astype(np.float32),                  # (M, J)
            p_y1_scaled=p_y1.astype(np.float32),                  # (M, J)
            p_joint_scaled=p_mat.astype(np.float32),              # (M, J, J)
            y_shift=np.float32(y_shift),
            y_scale=np.float32(y_scale),
        )
    return cate_pred.astype(np.float32), dens


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f'[bootstrap] fn=50  {DATASET}  ckpt={CKPT}', flush=True)
    model, J, edges_np, num_features = _load_fn50(CKPT)
    ds = SCMCaseStudyDataset(DATASET)
    n = ds.n_tables if not MAX_REAL else min(ds.n_tables, int(MAX_REAL))
    print(f'[bootstrap] J={J}  num_features={num_features}  n={n}', flush=True)

    rows = []; t0 = time.time()
    for r in range(n):
        cate_ds, _ = ds[r]
        cate_pred, dens = _cate_fn50(model, J, edges_np, num_features,
                                cate_ds.X_train, cate_ds.t_train, cate_ds.y_train,
                                cate_ds.X_test)
        true_cate = np.asarray(cate_ds.true_cate, dtype=np.float32).reshape(-1)
        pehe = float(np.sqrt(np.mean((cate_pred - true_cate) ** 2)))
        ate_true = float(true_cate.mean()); ate_hat = float(cate_pred.mean())
        err = abs(ate_hat - ate_true) / max(abs(ate_true), 0.1)
        row = {'dataset': DATASET, 'realization': r,
               'true_ate': ate_true, 'ate_pred': ate_hat,
               'pehe_raw': pehe, 'err_raw': err}
        rows.append(row)
        shard = {k: np.array(v) for k, v in row.items()}
        if dens is not None: shard.update(dens)
        np.savez(os.path.join(OUT, f'r{r:03d}.npz'), **shard)
        print(f'r={r:03d}  pehe={pehe:6.3f}  err={err:5.3f}  ate={ate_hat:+5.2f} vs true {ate_true:+5.2f}  '
              f'({time.time()-t0:.0f}s)', flush=True)

    def _ms(k):
        v = np.array([r[k] for r in rows]); return v.mean(), v.std(ddof=1)/np.sqrt(len(v))
    print(f'\n══ {DATASET}  fn=50  n={len(rows)} ══')
    for k in ('pehe_raw', 'err_raw'):
        m, s = _ms(k); print(f'  {k:10s} = {m:8.3f} ± {s:6.3f}')


if __name__ == '__main__':
    main()
