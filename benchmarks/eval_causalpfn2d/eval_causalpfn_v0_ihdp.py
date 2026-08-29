"""Reference eval: run vdblm/causalpfn's OWN pretrained causalpfn_v0.pt
through our IHDP loop.

The point is NOT to develop anything — it's a validation that our IHDP
pipeline (dataset loading, standardisation, per-realization CATE + PEHE
+ err_ATE calculation) produces sensible numbers on a model known to
work. If PEHE ≈ 0.58 (their published number), our eval is correct and
any joint-head gap is training-side. If PEHE is much worse, we have an
eval bug and our joint-head "ceiling" story is unreliable.

Uses their own CATEEstimator API, which:
  1. Downloads causalpfn_v0.pt from HuggingFace (~75MB, cached locally)
  2. Fits by storing context data (zero-shot ICL, no gradient updates)
  3. estimate_cate() runs their 1D single-arm HL-Gauss head twice per
     query — once with T=0, once with T=1 — and returns E[Y1]-E[Y0]

Env vars:
  OUT       : output directory for per-realization NPZ files
  CAUSALPFN : path to external/causalpfn (for imports + shims)
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch


OUT       = os.environ.get('OUT', './results_causalpfn_v0_ihdp')
CAUSALPFN = os.environ['CAUSALPFN']
# Optional local checkpoint path — bypasses HF download entirely so this
# eval can run on compute nodes without internet. Pre-download once on
# the login node into $DEPLOY_ROOT/warmstart/causalpfn_v0.pt.
CPFN_V0_LOCAL = os.environ.get('CPFN_V0_LOCAL', '')

# Same sys.path setup as the working raw/em evals — IHDPDataset comes
# from CausalPFN's benchmarks package (their external/causalpfn/benchmarks/
# module — R-PFN's benchmarks/__init__.py doesn't export IHDPDataset).
# The sitecustomize.py shim on PYTHONPATH stubs faiss so CausalPFN's
# import chain (benchmarks/__init__ -> polynomial -> base -> causalpfn ->
# causal_estimator -> faiss) doesn't die.
REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, CAUSALPFN)
sys.path.insert(0, CAUSALPFN + '/src')

from benchmarks import IHDPDataset  # noqa: E402  — CausalPFN's benchmarks
from causalpfn import CATEEstimator  # noqa: E402


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate(realization: int, estimator: CATEEstimator):
    ds = IHDPDataset()
    cate_ds = ds[realization][0]
    X_tr = cate_ds.X_train.astype(np.float32)
    T_tr = cate_ds.t_train.astype(np.float32).reshape(-1)
    y_tr = cate_ds.y_train.astype(np.float32).reshape(-1)
    X_te = cate_ds.X_test.astype(np.float32)
    true_cate = cate_ds.true_cate.astype(np.float32)
    true_ate = float(true_cate.mean())

    # CausalPFN's own fit/predict — zero-shot ICL, no gradient updates.
    estimator.fit(X_tr, T_tr, y_tr)
    cate_hat = np.asarray(estimator.estimate_cate(X_te)).reshape(-1).astype(np.float32)

    pehe = float(np.sqrt(np.mean((cate_hat - true_cate) ** 2)))
    ate_hat = float(cate_hat.mean())
    err_ate = abs(ate_hat - true_ate) / max(abs(true_ate), 1e-9)

    return {
        'dataset':      'IHDP',
        'realization':  realization,
        'true_ate':     true_ate,
        'pehe_cpfn_v0': pehe,
        'err_cpfn_v0':  err_ate,
        'ate_cpfn_v0':  ate_hat,
    }


def main():
    print(f'[bootstrap] device={DEVICE}  out={OUT}', flush=True)
    # Prefer local ckpt (no compute-node internet needed). Falls back to
    # CATEEstimator's HF auto-download if CPFN_V0_LOCAL isn't set.
    if CPFN_V0_LOCAL and os.path.isfile(CPFN_V0_LOCAL):
        print(f'[bootstrap] instantiating CATEEstimator from local ckpt: '
              f'{CPFN_V0_LOCAL}', flush=True)
        estimator = CATEEstimator(device=DEVICE, model_path=CPFN_V0_LOCAL, verbose=False)
    else:
        print(f'[bootstrap] instantiating CATEEstimator (auto-downloads '
              f'vdblm/causalpfn/causalpfn_v0.pt on first run — may fail on '
              f'compute nodes without internet)…', flush=True)
        estimator = CATEEstimator(device=DEVICE, verbose=False)
    print(f'[bootstrap] estimator ready', flush=True)

    os.makedirs(OUT, exist_ok=True)
    all_rows = []
    t0 = time.time()
    for r in range(100):
        row = evaluate(r, estimator)
        np.savez(os.path.join(OUT, f'r{r:03d}.npz'),
                 **{k: np.array(v) for k, v in row.items()})
        all_rows.append(row)
        print(
            f'r={r:03d}  '
            f'pehe={row["pehe_cpfn_v0"]:6.3f}  '
            f'err_ate={row["err_cpfn_v0"]:5.3f}  '
            f'ate_hat={row["ate_cpfn_v0"]:+6.3f}  '
            f'true_ate={row["true_ate"]:+6.3f}  '
            f'({time.time()-t0:.0f}s)',
            flush=True,
        )

    def _ms(k):
        v = np.array([r[k] for r in all_rows])
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

    print(f'\n══ IHDP reference summary (n={len(all_rows)}, model=causalpfn_v0.pt) ══')
    for k in ('pehe_cpfn_v0', 'err_cpfn_v0'):
        m, s = _ms(k)
        print(f'  {k:15s} = {m:8.3f} ± {s:6.3f}')
    print(f'\n  Paper target: PEHE ≈ 0.58, err_ATE ≈ 0.03')


if __name__ == '__main__':
    main()
