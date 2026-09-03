"""Diagnostic: load ArikReuter's reproduce-realcause-results 1D-head checkpoint
via their PreprocessingGraphConditionedPFN wrapper, run on ACIC (10 realizations)
with both anc and noanc adjacency matrices we build here.

Purpose: isolate whether the anc-degradation observed on our 2D-head checkpoint
is a property of the *evaluation* (adjacency construction, mode inputs) or a
property of our *trained model*.

If the numbers here match their published ACIC table
    Ancestral Info.:    PEHE = 2.79
    No Ancestral Info.: PEHE = 3.47
then our adjacency construction + eval flow is correct, and the anc-degradation
we observe on our own checkpoint is training-side.

Env vars:
    UWYK_REPRO  = path to a clone of Graphs4CausalFoundationModels
                    (reproduce-realcause-results branch, git-lfs pulled)
    CAUSALPFN   = path to the pinned CausalPFN checkout referenced in
                    their REPRODUCE_REALCAUSE_RESULTS.md (a3de754)
    CKPT        = path to their best_model.pt
    CONFIG      = path to their best_model_config.yaml
    OUT         = per-realization NPZ dir  (default ./results_diag_reproduce_acic)
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch


UWYK_REPRO = os.environ['UWYK_REPRO']
CAUSALPFN  = os.environ['CAUSALPFN']
CKPT       = os.environ['CKPT']
CONFIG     = os.environ['CONFIG']
OUT        = os.environ.get('OUT', './results_diag_reproduce_acic')

# Path setup so their src.models.* imports resolve.
sys.path.insert(0, UWYK_REPRO)
sys.path.insert(0, os.path.join(UWYK_REPRO, 'RealCauseEval'))
sys.path.insert(0, CAUSALPFN)
sys.path.insert(0, os.path.join(CAUSALPFN, 'src'))

from src.models.PreprocessingGraphConditionedPFN import PreprocessingGraphConditionedPFN  # noqa: E402
from benchmarks import ACIC2016Dataset  # noqa: E402  (CausalPFN's benchmarks pkg)


# ── Our adjacency-matrix builders (verbatim from eval_graph2d_realcause.py) ──
def build_anc_full(F, n_real):
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    T_idx, Y_idx, feat_off = 0, 1, 2
    A[T_idx, Y_idx] = 1.0
    for i in range(n_real):
        A[feat_off + i, T_idx] = 1.0
        A[feat_off + i, Y_idx] = 1.0
    for i in range(n_real, F):
        A[feat_off + i, :] = -1.0
        A[:, feat_off + i] = -1.0
        A[feat_off + i, feat_off + i] = -1.0
    return A


def build_anc_none(F, n_real):
    A = np.zeros((F + 2, F + 2), dtype=np.float32)
    feat_off = 2
    for i in range(n_real, F):
        A[feat_off + i, :] = -1.0
        A[:, feat_off + i] = -1.0
        A[feat_off + i, feat_off + i] = -1.0
    return A


# ── The pipeline (mirrors dofm_no_clustering.py::create_dofm_no_clustering_pipeline
# byte-for-byte, but takes the adjacency as an explicit argument). ──────────
def run_cate(model, cate_dataset, adjacency_matrix):
    X_train = cate_dataset.X_train
    t_train_orig = cate_dataset.t_train.reshape(-1, 1) if cate_dataset.t_train.ndim == 1 else cate_dataset.t_train
    y_train_orig = cate_dataset.y_train.reshape(-1, 1) if cate_dataset.y_train.ndim == 1 else cate_dataset.y_train
    X_test = cate_dataset.X_test
    y_train = y_train_orig

    n_test = X_test.shape[0]

    # Target encoding for treatment: replace T with mean(Y|T). This is what
    # dofm_no_clustering.py does — needed because their model was trained
    # with T going through TPreprocessor.
    t_flat = t_train_orig.flatten()
    y_flat = y_train.flatten()
    mean_y_t0 = float(y_flat[t_flat == 0].mean())
    mean_y_t1 = float(y_flat[t_flat == 1].mean())
    t_train = np.where(t_train_orig == 0, mean_y_t0, mean_y_t1).astype(np.float32)
    t_intv_0_encoded = mean_y_t0
    t_intv_1_encoded = mean_y_t1

    # Fit preprocessing on this realization's training set (their pipeline
    # refits per realization since IHDP/ACIC/CPS/PSID stats vary).
    model.fit(X_train, t_train, y_train)

    # Two forwards per query — one with each intervention value.
    T_intv_1 = np.full((n_test, 1), t_intv_1_encoded, dtype=np.float32)
    y_pred_1 = model.predict(
        X_obs=X_train, T_obs=t_train, Y_obs=y_train,
        X_intv=X_test, T_intv=T_intv_1,
        adjacency_matrix=adjacency_matrix,
        prediction_type='mean', inverse_transform=True,
    )

    T_intv_0 = np.full((n_test, 1), t_intv_0_encoded, dtype=np.float32)
    y_pred_0 = model.predict(
        X_obs=X_train, T_obs=t_train, Y_obs=y_train,
        X_intv=X_test, T_intv=T_intv_0,
        adjacency_matrix=adjacency_matrix,
        prediction_type='mean', inverse_transform=True,
    )

    return (y_pred_1 - y_pred_0).flatten()


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f'[bootstrap] ckpt={CKPT}\n            config={CONFIG}\n            out={OUT}',
          flush=True)

    # Load their model via their wrapper. use_clustering=False matches
    # dofm_no_clustering.py.
    model = PreprocessingGraphConditionedPFN(
        config_path=CONFIG,
        checkpoint_path=CKPT,
        verbose=False,
        use_clustering=False,
    )
    model.load()

    ds = ACIC2016Dataset()
    F = model.model.num_features
    print(f'[bootstrap] ACIC realizations={ds.n_tables}  model.num_features={F}',
          flush=True)

    rows = []
    t0 = time.time()
    for r in range(ds.n_tables):
        cate_ds = ds[r][0]
        true_cate = np.asarray(cate_ds.true_cate, dtype=np.float32)
        true_ate = float(true_cate.mean())

        n_real = min(int(cate_ds.X_train.shape[1]), F)
        adj_anc   = build_anc_full(F, n_real)
        adj_noanc = build_anc_none(F, n_real)

        cate_anc   = run_cate(model, cate_ds, adj_anc)
        cate_noanc = run_cate(model, cate_ds, adj_noanc)

        def _pehe(cate):
            pehe = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
            ate  = float(cate.mean())
            err  = abs(ate - true_ate) / max(abs(true_ate), 0.1)
            return pehe, err, ate

        pehe_anc,   err_anc,   ate_anc   = _pehe(cate_anc)
        pehe_noanc, err_noanc, ate_noanc = _pehe(cate_noanc)

        row = {
            'realization': r, 'true_ate': true_ate,
            'pehe_anc':   pehe_anc,   'err_anc':   err_anc,   'ate_anc':   ate_anc,
            'pehe_noanc': pehe_noanc, 'err_noanc': err_noanc, 'ate_noanc': ate_noanc,
        }
        rows.append(row)
        np.savez(os.path.join(OUT, f'ACIC_r{r:02d}.npz'),
                 **{k: np.array(v) for k, v in row.items()})
        print(
            f'r={r:02d}  '
            f'anc: pehe={pehe_anc:6.3f} err={err_anc:5.3f}  |  '
            f'noanc: pehe={pehe_noanc:6.3f} err={err_noanc:5.3f}  '
            f'({time.time()-t0:.0f}s)',
            flush=True,
        )

    def ms(k):
        v = np.array([r[k] for r in rows if np.isfinite(r[k])])
        if v.size < 2: return float('nan'), float('nan')
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

    print(f'\n══ ACIC summary (n={len(rows)}, their published anc=2.79, noanc=3.47) ══')
    for k in ('pehe_anc', 'err_anc', 'pehe_noanc', 'err_noanc'):
        m, s = ms(k)
        print(f'  {k:15s} = {m:8.3f} ± {s:6.3f}')


if __name__ == '__main__':
    main()
