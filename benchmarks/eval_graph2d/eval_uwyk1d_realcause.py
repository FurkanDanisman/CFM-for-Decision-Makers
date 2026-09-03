"""CONTROL: run UWYK's own 1D checkpoint through OUR eval harness.

Purpose. Our graph2d eval shows anc ≈ noanc (IHDP) or anc worse (ACIC/CPS/PSID).
Two explanations are confounded:
    (a) the harness — PAM construction, preprocessing, context handling
    (b) our training — the 2D head / weight-decay-collapsed graph params

This script removes the confound by holding (a) fixed and swapping only the
model. Everything below — get_dataset, _standardize_train_test, _scale_y,
_pad_features, build_adjacency_matrix (incl. FIX_DIAG), psid_balance_subsample,
EVAL_MAX_CONTEXT, the PEHE / err_ATE computation and the npz layout — is
imported from eval_graph2d_realcause. The ONLY difference is that the CATE
comes from UWYK's 1D bar-distribution model run twice (T=1, T=0) instead of
from our 2D joint head run once.

Read the result as:
    anc beats noanc here  → harness is fine, look at our training
    anc ties/loses here   → the harness (PAM encoding / preprocessing) is
                            suppressing the graph signal for everyone

T_ENCODING (env, default 'binary'):
    'binary'  feed T ∈ {0,1} and query T_intv ∈ {0,1} — what OUR harness does
    'target'  feed T ← mean(Y|T) and query the two encoded values — what UWYK's
              own dofm_full_conditioning.py does
Run both: the difference tells you how much of UWYK's reported gap is carried
by the treatment encoding rather than by the graph.

ANC_MODE (env, read by the imported harness): selects which adjacency variants
are evaluated, via the shared eval_graph2d_realcause.build_mode_list. Set it to
the same value your graph2d run used — e.g. ANC_MODE=v6a_only emits
`pehe_raw_v6a` / `pehe_raw_noanc`, the identical keys and identical adjacency
matrices, so the two npz trees drop straight into one table.

Usage (GPU node):
    CKPT=<uwyk best_model.pt>  CONFIG=<uwyk best_model_config.yaml> \
    UWYK=$PWD/g4cfm  CAUSALPFN=/path/to/CausalPFN \
    DATASET=IHDP  OUT=./results_uwyk1d/IHDP \
    EVAL_MAX_CONTEXT=1000  FIX_DIAG=1  T_ENCODING=binary  ANC_MODE=v6a_only \
    python -u benchmarks/eval_graph2d/eval_uwyk1d_realcause.py

Or as a Slurm array over all 5 datasets × both encodings:
    sbatch benchmarks/cluster/submit_eval_uwyk1d_realcause.sbatch
"""
from __future__ import annotations
import importlib.util
import os
import sys
import time

import numpy as np

# The harness module reads env at import time, so everything must be set first.
os.environ.setdefault('CKPT', '')          # unused here; satisfies its env read
CONFIG      = os.environ['CONFIG']
CKPT        = os.environ['CKPT']
T_ENCODING  = os.environ.get('T_ENCODING', 'binary')
assert T_ENCODING in ('binary', 'target'), T_ENCODING
QUERY_CHUNK = int(os.environ.get('QUERY_CHUNK', '512'))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HARNESS_PATH = os.path.join(os.path.dirname(__file__), 'eval_graph2d_realcause.py')
spec = importlib.util.spec_from_file_location('_eval_graph2d_realcause_harness',
                                              HARNESS_PATH)
assert spec is not None and spec.loader is not None
H = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = H
spec.loader.exec_module(H)

sys.path.insert(0, os.environ['UWYK'])
sys.path.insert(0, os.environ['UWYK'] + '/src')
from models.GraphConditionedInterventionalPFN_sklearn import (  # noqa: E402
    GraphConditionedInterventionalPFNSklearn,
)

DATASET = H.DATASET
OUT     = os.environ.get('OUT', f'./results_uwyk1d_{DATASET}')


def load_uwyk():
    w = GraphConditionedInterventionalPFNSklearn(
        config_path=CONFIG, checkpoint_path=CKPT, verbose=True)
    w.load()
    F = w.model.num_features
    print(f'[load_uwyk] num_features={F}  mode={w.graph_conditioning_mode}  '
          f'bar_dist={w.use_bar_distribution}', flush=True)
    return w, F


def cate_from_uwyk(w, X_tr, T_tr, Y_tr, X_te, adj, t0_val, t1_val):
    """Two forward passes (do(t1), do(t0)); returns CATE on the scaled y axis."""
    preds = {}
    for tag, tval in (('1', t1_val), ('0', t0_val)):
        out = []
        for s in range(0, X_te.shape[0], QUERY_CHUNK):
            Xq = X_te[s:s + QUERY_CHUNK]
            Tq = np.full((Xq.shape[0], 1), tval, dtype=np.float32)
            out.append(np.asarray(w.predict(
                X_obs=X_tr, T_obs=T_tr, Y_obs=Y_tr,
                X_intv=Xq, T_intv=Tq,
                adjacency_matrix=adj,
                prediction_type='mean',
            )).reshape(-1))
        preds[tag] = np.concatenate(out)
    return (preds['1'] - preds['0']).astype(np.float32)


def evaluate(realization, ds, w, F, apply_psid_balance):
    cate_ds  = ds[realization][0]
    X_tr_raw = np.asarray(cate_ds.X_train, dtype=np.float32)
    T_tr     = np.asarray(cate_ds.t_train, dtype=np.float32).reshape(-1)
    y_tr_raw = np.asarray(cate_ds.y_train, dtype=np.float32).reshape(-1)
    X_te_raw = np.asarray(cate_ds.X_test,  dtype=np.float32)
    true_cate = np.asarray(cate_ds.true_cate, dtype=np.float32)
    true_ate  = float(true_cate.mean())

    if apply_psid_balance:
        X_tr_raw, T_tr, y_tr_raw = H.psid_balance_subsample(X_tr_raw, T_tr, y_tr_raw)

    if H.EVAL_MAX_CONTEXT:
        cap = int(H.EVAL_MAX_CONTEXT)
        if X_tr_raw.shape[0] > cap:
            rng = np.random.default_rng(H.EVAL_CONTEXT_SEED + realization)
            idx = rng.choice(X_tr_raw.shape[0], cap, replace=False)
            X_tr_raw, T_tr, y_tr_raw = X_tr_raw[idx], T_tr[idx], y_tr_raw[idx]

    n_real = min(X_tr_raw.shape[1], F)
    X_tr_std, X_te_std = H._standardize_train_test(X_tr_raw, X_te_raw)
    X_tr = H._pad_features(X_tr_std, F)
    X_te = H._pad_features(X_te_std, F)
    y_scaled, ymin, yrange = H._scale_y(y_tr_raw)
    Y_obs = y_scaled.reshape(-1, 1)

    # Treatment encoding — the one deliberate knob.
    if T_ENCODING == 'target':
        m0 = float(y_scaled[T_tr == 0].mean())
        m1 = float(y_scaled[T_tr == 1].mean())
        T_feed = np.where(T_tr == 0, m0, m1).astype(np.float32).reshape(-1, 1)
        t0_val, t1_val = m0, m1
    else:
        T_feed = T_tr.astype(np.float32).reshape(-1, 1)
        t0_val, t1_val = 0.0, 1.0

    results = {}
    # Same ANC_MODE dispatch the graph2d eval uses, so a v6a_only run here
    # produces the identical adjacency matrices and npz keys as the joint run.
    for mode, adj in H.build_mode_list(F, n_real):
        cate_scaled = cate_from_uwyk(w, X_tr, T_feed, Y_obs, X_te, adj, t0_val, t1_val)
        cate = cate_scaled * yrange / 2.0
        results[f'pehe_raw_{mode}'] = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
        results[f'ate_raw_{mode}']  = float(cate.mean())
        results[f'err_raw_{mode}']  = abs(float(cate.mean()) - true_ate) / max(abs(true_ate), 1e-9)
        # keep the npz schema identical to the graph2d eval
        results[f'pehe_em_{mode}'] = results[f'pehe_raw_{mode}']
        results[f'ate_em_{mode}']  = results[f'ate_raw_{mode}']
        results[f'err_em_{mode}']  = results[f'err_raw_{mode}']

    return {'dataset': DATASET, 'realization': realization, 'true_ate': true_ate,
            'n_queries': int(true_cate.size), 'n_context': int(X_tr_raw.shape[0]),
            **results}


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f'[bootstrap] CONTROL uwyk-1d-in-our-harness  dataset={DATASET}  '
          f'T_ENCODING={T_ENCODING}  ANC_MODE={H.ANC_MODE}  '
          f'EVAL_MAX_CONTEXT={H.EVAL_MAX_CONTEXT or "(none)"}', flush=True)

    ds = H.get_dataset(DATASET)
    w, F = load_uwyk()
    apply_psid_balance = (DATASET == 'PSID_bal')

    rows = []
    tags = []  # adjacency tags actually emitted, in build_mode_list order
    t0 = time.time()
    cap = int(os.environ.get('MAX_REAL', ds.n_tables))
    for r in range(min(ds.n_tables, cap)):
        row = evaluate(r, ds, w, F, apply_psid_balance)
        rows.append(row)
        if not tags:
            tags = [k[len('pehe_raw_'):] for k in row if k.startswith('pehe_raw_')]
        np.savez(os.path.join(OUT, f'{DATASET}_r{r:03d}.npz'),
                 **{k: np.array(v) for k, v in row.items()})
        cells = '  |  '.join(
            f'{t}: pehe={row[f"pehe_raw_{t}"]:9.3f} err={row[f"err_raw_{t}"]:5.3f}' for t in tags)
        print(f'r={r:03d}  {cells}  ({time.time()-t0:.0f}s)', flush=True)

    def ms(k):
        v = np.array([r[k] for r in rows if np.isfinite(r[k])])
        return (v.mean(), v.std(ddof=1) / np.sqrt(len(v))) if v.size > 1 else (float('nan'),) * 2

    print(f'\n══ {DATASET} CONTROL summary (n={len(rows)}) ══')
    for k in [f'pehe_raw_{t}' for t in tags] + [f'err_raw_{t}' for t in tags]:
        m, s = ms(k)
        print(f'  {k:18s} = {m:10.3f} ± {s:8.3f}')
    # Paired anc-vs-noanc delta, whichever positive-adjacency tag was requested.
    pos = [t for t in tags if t != 'noanc']
    if 'noanc' in tags and len(pos) == 1:
        t = pos[0]
        d = np.array([r[f'pehe_raw_{t}'] - r['pehe_raw_noanc'] for r in rows])
        print(f'  paired ΔPEHE({t}-noanc) = {d.mean():+10.4f} ± {d.std(ddof=1)/np.sqrt(len(d)):.4f}'
              f'   {t} wins {int((d < 0).sum())}/{len(d)}')


if __name__ == '__main__':
    main()
