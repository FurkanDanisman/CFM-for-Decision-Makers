"""Diagnostic: is UWYK's PSID-unbalanced PEHE dominated by the KMeans seed?

Background:
  - PSIDbal (~685 rows) sits below the 1000-clustering threshold → single
    forward pass → matches paper.
  - PSID unbal (2675 rows) triggers clustering into 3 chunks of ~900. Only
    ~185 treated total, so per-chunk treated count is small and highly seed-
    dependent. Small clusters with too-few treated can't estimate CATE →
    predictions collapse.

This probe runs `PreprocessingGraphConditionedPFN` on realization 0 of PSID
(unbalanced) with several `random_state` values and prints the resulting
PEHE + prediction mean/std.

Interpretation:
  - Wide spread (12k-25k)  → KMeans seed dominates. Paper's 13k is one draw
                             from a wide distribution. Fix by matching their
                             seed OR by rebalancing before clustering.
  - All ~22k               → seed doesn't matter; something else is wrong
                             (e.g. our adjacency semantics differ).
  - All ~13k               → we accidentally reproduce paper — Table 1/3
                             wrappers have an ADDITIONAL bug on top.

Also runs CPS realization 0 as a control (should give ~13k regardless).
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

from benchmarks import RealCauseLalondeCPSDataset, RealCauseLalondePSIDDataset  # noqa
from src.models.PreprocessingGraphConditionedPFN import (  # noqa
    PreprocessingGraphConditionedPFN,
)


def probe(ds_name, ds, seeds):
    print(f'\n══ {ds_name} ══')
    cate_ds = ds[0][0]
    X_train = cate_ds.X_train
    t_train = cate_ds.t_train.reshape(-1, 1)
    y_train = cate_ds.y_train.reshape(-1, 1)
    X_test  = cate_ds.X_test
    true    = cate_ds.true_cate
    n_treated = int((t_train == 1).sum())
    print(f'  n_train={len(X_train)}  n_test={len(X_test)}  '
          f'n_features={X_train.shape[1]}  n_treated={n_treated}  '
          f'p_treated={n_treated/len(X_train):.3f}')

    F = 50
    if X_train.shape[1] < F:
        X_train = np.hstack([X_train, np.zeros((len(X_train), F - X_train.shape[1]))])
        X_test  = np.hstack([X_test,  np.zeros((len(X_test),  F - X_test.shape[1]))])
    else:
        X_train = X_train[:, :F]
        X_test  = X_test[:,  :F]

    y_range = float(y_train.max() - y_train.min())
    n_test_probe = min(500, len(X_test))
    T1 = np.ones((n_test_probe, 1), dtype=np.float32)
    T0 = np.zeros_like(T1)
    adj = np.zeros((F + 2, F + 2), dtype=np.float32)

    for seed in seeds:
        m = PreprocessingGraphConditionedPFN(
            config_path=os.path.join(CKPT, 'best_model_config.yaml'),
            checkpoint_path=os.path.join(CKPT, 'best_model.pt'),
            random_state=seed,
            verbose=False,
        )
        m.load()
        y1 = m.predict(
            X_obs=X_train, T_obs=t_train, Y_obs=y_train,
            X_intv=X_test[:n_test_probe], T_intv=T1,
            adjacency_matrix=adj, prediction_type='mean',
        )
        y0 = m.predict(
            X_obs=X_train, T_obs=t_train, Y_obs=y_train,
            X_intv=X_test[:n_test_probe], T_intv=T0,
            adjacency_matrix=adj, prediction_type='mean',
        )
        cate = (np.asarray(y1).flatten() - np.asarray(y0).flatten()) * y_range / 2.0
        pehe = float(np.sqrt(np.mean((cate - true[:n_test_probe]) ** 2)))
        print(f'  seed={seed:>7}   PEHE={pehe:>10.2f}   '
              f'CATE mean={cate.mean():>10.1f}  std={cate.std():>10.1f}')


def main():
    SEEDS = (0, 7, 42, 100, 12345, 999999)
    # PSID is the main event
    probe('PSID (unbalanced)', RealCauseLalondePSIDDataset(), SEEDS)
    # CPS as a control — should give ~13k regardless of seed
    probe('CPS (control — should be seed-stable ~13k)',
          RealCauseLalondeCPSDataset(), SEEDS[:3])


if __name__ == '__main__':
    main()
