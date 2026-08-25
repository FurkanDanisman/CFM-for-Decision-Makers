"""Diagnostic: which context-handling change reproduces paper's PSID PEHE?

Observation:
  - n_train < 1000  (IHDP=672, PSIDbal=685) → single forward pass → MATCHES paper
  - n_train > 1000  (PSID=2675, CPS=14559) → clustering kicks in → DIVERGES

The clustering path is where the fix must go. This probe runs realization 0
of PSID under four modes and prints the PEHE for each:

  A. baseline                — use_clustering=True (default), no subsample
                                → what we get today (~22k, wrong)
  B. subsample n=1000, strat  — pick 1000 rows stratified by T, use_clustering=False
                                → mirrors PSIDbal but keeps unbalanced ratio
  C. subsample n=1000, random — pick 1000 rows uniformly, use_clustering=False
                                → weaker than B; tests whether stratification matters
  D. clustering + pinned seed — use_clustering=True, random_state=42
                                → tests whether the divergence is KMeans nondeterminism

Whichever lands near paper's 13096 is the fix — we bake that into run_one.py.
"""
import os
import sys
import numpy as np

UWYK      = os.environ['UWYK']
CKPT      = os.environ['CKPT']
CAUSALPFN = os.environ['CAUSALPFN']

sys.path.insert(0, UWYK)
sys.path.insert(0, UWYK + '/RealCauseEval')
sys.path.insert(0, CAUSALPFN)
shim = os.path.join(os.path.dirname(__file__), 'shims')
if os.path.isdir(shim):
    sys.path.insert(0, shim)

from benchmarks import RealCauseLalondePSIDDataset  # noqa
from src.models.PreprocessingGraphConditionedPFN import (  # noqa
    PreprocessingGraphConditionedPFN,
)


def _pad(X, F=50):
    if X.shape[1] < F:
        return np.hstack([X, np.zeros((len(X), F - X.shape[1]), dtype=X.dtype)])
    return X[:, :F]


def _pehe(model, X_train, t_train, y_train, X_test, true_cate, y_range,
          n_test_probe=500):
    n = min(n_test_probe, len(X_test))
    F = X_train.shape[1]
    T1 = np.ones((n, 1), dtype=np.float32)
    T0 = np.zeros_like(T1)
    adj = np.zeros((F + 2, F + 2), dtype=np.float32)
    y1 = model.predict(
        X_obs=X_train, T_obs=t_train.reshape(-1, 1), Y_obs=y_train.reshape(-1, 1),
        X_intv=X_test[:n], T_intv=T1, adjacency_matrix=adj, prediction_type='mean',
    )
    y0 = model.predict(
        X_obs=X_train, T_obs=t_train.reshape(-1, 1), Y_obs=y_train.reshape(-1, 1),
        X_intv=X_test[:n], T_intv=T0, adjacency_matrix=adj, prediction_type='mean',
    )
    cate = (np.asarray(y1).flatten() - np.asarray(y0).flatten()) * y_range / 2.0
    pehe = float(np.sqrt(np.mean((cate - true_cate[:n]) ** 2)))
    return pehe, cate


def stratified_subsample(X, t, y, n_keep, rng):
    """Sample `n_keep` rows, preserving the treated-fraction of the input."""
    t_flat = t.flatten()
    n_treat_total = int((t_flat == 1).sum())
    n_ctrl_total  = int((t_flat == 0).sum())
    p_treat = n_treat_total / len(t_flat)
    n_treat_keep = int(round(n_keep * p_treat))
    n_ctrl_keep  = n_keep - n_treat_keep
    n_treat_keep = min(n_treat_keep, n_treat_total)
    n_ctrl_keep  = min(n_ctrl_keep,  n_ctrl_total)

    idx_treat = np.where(t_flat == 1)[0]
    idx_ctrl  = np.where(t_flat == 0)[0]
    keep_treat = rng.choice(idx_treat, size=n_treat_keep, replace=False)
    keep_ctrl  = rng.choice(idx_ctrl,  size=n_ctrl_keep,  replace=False)
    keep = np.concatenate([keep_treat, keep_ctrl])
    rng.shuffle(keep)
    return X[keep], t[keep], y[keep], n_treat_keep, n_ctrl_keep


def random_subsample(X, t, y, n_keep, rng):
    idx = rng.choice(len(X), size=n_keep, replace=False)
    return X[idx], t[idx], y[idx]


def build_model(use_clustering, random_state):
    m = PreprocessingGraphConditionedPFN(
        config_path=os.path.join(CKPT, 'best_model_config.yaml'),
        checkpoint_path=os.path.join(CKPT, 'best_model.pt'),
        use_clustering=use_clustering,
        random_state=random_state,
        verbose=False,
    )
    m.load()
    return m


def main():
    ds = RealCauseLalondePSIDDataset()
    cate_ds = ds[0][0]
    X_train = _pad(cate_ds.X_train)
    t_train = cate_ds.t_train.astype(np.float32)
    y_train = cate_ds.y_train.astype(np.float32)
    X_test  = _pad(cate_ds.X_test)
    true    = cate_ds.true_cate
    y_range = float(cate_ds.y_train.max() - cate_ds.y_train.min())

    n_treated = int((t_train == 1).sum())
    print(f'\nPSID r=0: n_train={len(X_train)}  n_treated={n_treated}  '
          f'p_treated={n_treated/len(X_train):.3f}')
    print(f'paper target PEHE ≈ 13096 (noanc) / 12975 (anc, adj matters but adj=0 here)')
    print('─' * 72)

    print('\nA. baseline (use_clustering=True, random_state=None)')
    m = build_model(use_clustering=True, random_state=None)
    pehe, _ = _pehe(m, X_train, t_train, y_train, X_test, true, y_range)
    print(f'   PEHE = {pehe:.2f}')

    print('\nB. subsample to 1000 stratified by T + use_clustering=False')
    for seed in (0, 42, 12345):
        rng = np.random.default_rng(seed)
        Xs, ts, ys, nT, nC = stratified_subsample(X_train, t_train, y_train, 1000, rng)
        m = build_model(use_clustering=False, random_state=None)
        pehe, _ = _pehe(m, Xs, ts, ys, X_test, true, y_range)
        print(f'   subsample seed={seed:>5}  (nT={nT}, nC={nC})  PEHE = {pehe:.2f}')

    print('\nC. subsample to 1000 random + use_clustering=False')
    for seed in (0, 42, 12345):
        rng = np.random.default_rng(seed)
        Xs, ts, ys = random_subsample(X_train, t_train, y_train, 1000, rng)
        m = build_model(use_clustering=False, random_state=None)
        pehe, _ = _pehe(m, Xs, ts, ys, X_test, true, y_range)
        print(f'   subsample seed={seed:>5}  PEHE = {pehe:.2f}')

    print('\nD. use_clustering=True + pinned random_state=42')
    m = build_model(use_clustering=True, random_state=42)
    pehe, _ = _pehe(m, X_train, t_train, y_train, X_test, true, y_range)
    print(f'   PEHE = {pehe:.2f}')

    print('\nInterpretation:')
    print('  B near 13k → fix is subsample-1000-stratified, no clustering.')
    print('  C near 13k → random subsample also works; stratification not required.')
    print('  D near 13k → the entire divergence is KMeans nondeterminism.')
    print('  A ≈ 22k, others also ≈ 22k → clustering itself is broken; deeper dig needed.')


if __name__ == '__main__':
    main()
