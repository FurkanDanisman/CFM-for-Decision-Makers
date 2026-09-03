"""UWYK eval on the 6 synthetic case studies — PEHE + ATE err.

Delegates the actual per-arm density computation to
`benchmarks/l2_ihdp/methods_densities.py::uwyk_noanc_densities` (or
`uwyk_anc_densities` when ANC_MODE=anc). That code path is battle-tested
on the IHDP L2 pipeline. We just wrap it in the case-study loader and
turn its density output into a point CATE per query.

Env vars:
  DATASET          case study name (Observed_Confounder / ...)
  OUT              per-realization NPZ dir
  UWYK_SRC         path to UWYK repo src (has models/ + utils/)
  UWYK_CKPT        explicit .pt path (preferred)
  UWYK_CONFIG      explicit .yaml path (preferred)
  UWYK_CKPT_DIR    fallback dir when UWYK_CKPT/UWYK_CONFIG unset
  ANC_MODE         noanc | anc  (default noanc)
  MAX_REAL         optional cap
  DOPFN_DATA_ROOT  where the prior_sampling pkls live
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
sys.path.insert(0, os.path.join(REPO_SRC, 'benchmarks'))     # scm_case_study_dataset
sys.path.insert(0, os.path.join(REPO_SRC, 'benchmarks', 'l2_ihdp'))  # methods_densities

from scm_case_study_dataset import SCMCaseStudyDataset  # noqa: E402


# ── UWYK model bootstrap (isolated from local models/utils collisions) ──
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
        assert os.path.isfile(cfg_p), f'UWYK config missing: {cfg_p} (set UWYK_CONFIG)'
        print(f'[uwyk] loading  ckpt={ck_p}  cfg={cfg_p}', flush=True)
        m = pre_mod.PreprocessingGraphConditionedPFN(
            config_path=cfg_p, checkpoint_path=ck_p, device='cpu', verbose=False,
            random_state=42, use_clustering=False,
        ).load()
    finally:
        torch.load = _orig_load
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = SCMCaseStudyDataset(DATASET)
    n = ds.n_tables if not MAX_REAL else min(ds.n_tables, int(MAX_REAL))
    print(f'[bootstrap] UWYK {ANC_MODE}  {DATASET}  n={n}', flush=True)
    uwyk_model = _load_uwyk()
    num_features = uwyk_model.model.num_features
    density_fn_name = 'uwyk_noanc_densities' if ANC_MODE == 'noanc' else 'uwyk_anc_densities'

    # Import methods_densities lazily (avoids IHDP/dopfn dependency until needed)
    from methods_densities import uwyk_noanc_densities, uwyk_anc_densities
    density_fn = uwyk_noanc_densities if ANC_MODE == 'noanc' else uwyk_anc_densities

    rows = []; t0 = time.time()
    for r in range(n):
        cate_ds, _ = ds[r]
        # Compute y_min/y_rng from THIS realization's training Y
        y_train = np.asarray(cate_ds.y_train, dtype=np.float32).reshape(-1)
        y_min = float(y_train.min())
        y_rng = max(float(y_train.max() - y_train.min()), 1e-6)

        # methods_densities.uwyk_*_densities returns a dict with per-query densities +
        # 'cate_raw_scaled' (mean-based CATE in scaled Y).
        try:
            d = density_fn(cate_ds, uwyk_model, num_features,
                            y_min=y_min, y_rng=y_rng, n_context=None)
        except Exception as e:
            print(f'r={r:03d}  ERROR: {type(e).__name__}: {e}', flush=True)
            continue

        # Recover CATE per query. cate_raw_scaled is in scaled Y ([-1, 1]);
        # multiply by y_rng/2 to get raw Y units.
        cate_scaled = d.get('cate_raw_scaled')
        if cate_scaled is None:
            # Fall back to inferring from p_y0/p_y1 marginals
            p_y0 = d['p_y0']; p_y1 = d['p_y1']
            from methods_densities import Y_CENTERS
            e0 = (p_y0 * Y_CENTERS).sum(axis=-1)
            e1 = (p_y1 * Y_CENTERS).sum(axis=-1)
            cate_scaled = e1 - e0
        cate_pred = np.asarray(cate_scaled, dtype=np.float32).reshape(-1) * (y_rng / 2.0)
        true_cate = np.asarray(cate_ds.true_cate, dtype=np.float32).reshape(-1)

        pehe = float(np.sqrt(np.mean((cate_pred - true_cate) ** 2)))
        ate_true = float(true_cate.mean()); ate_hat = float(cate_pred.mean())
        err = abs(ate_hat - ate_true) / max(abs(ate_true), 1e-9)
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
