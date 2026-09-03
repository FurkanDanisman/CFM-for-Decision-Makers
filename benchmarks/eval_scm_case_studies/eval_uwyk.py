"""UWYK eval on the 6 synthetic case studies — PEHE + ATE err.

VERBATIM copy of UWYK's own dofm_full_conditioning.py pipeline
(reproduced in benchmarks/uwyk_direct_repro.py). Key details we were
missing before:

- T is TARGET-ENCODED: t_train = where(T==0, mean_y_t0, mean_y_t1).
  NOT 0/1. Same for T_intv (query treatment is the encoded value).
- Two separate wrapper.predict(prediction_type='mean') calls (T=1 arm,
  T=0 arm). Wrapper handles X standardization + padding + Y scaling
  internally.
- Adjacency mode: 'all_unknown' (= noanc) or 'full_graph' (= anc).

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

# Map our anc_mode → UWYK's graph_mode
GRAPH_MODE = 'all_unknown' if ANC_MODE == 'noanc' else 'full_graph'

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, os.path.join(REPO_SRC, 'benchmarks'))

from scm_case_study_dataset import SCMCaseStudyDataset  # noqa: E402


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


def _pad_negatives(A, feat_off, n_real, model_n_features):
    """Mark padded feature slots as -1 across their rows/cols/diagonal."""
    for i in range(n_real, model_n_features):
        idx = feat_off + i
        A[idx, :] = -1.0
        A[:, idx] = -1.0
        A[idx, idx] = -1.0
    return A


def build_adjacency_matrix_for_case(case_study, model_n_features, n_real, graph_mode):
    """Per-case-study true-DAG adjacency (matching DoPFN paper Fig 2).

    all_unknown: real block all 0, padded slots -1 (regardless of case).

    full_graph:  the actual DAG of each case study. Nodes are
    [T=0, Y=1, X_0, X_1, ...]; entries in {-1, 0, +1} where +1 = row is
    ancestor of col.  For case studies where X_i is DOWNSTREAM of T
    (mediators, front-door), we set A[T, X_i]=+1 instead of A[X_i, T]=+1.
    """
    A = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
    T, Y, off = 0, 1, 2

    if graph_mode == 'all_unknown':
        return _pad_negatives(A, off, n_real, model_n_features)

    if graph_mode != 'full_graph':
        raise ValueError(graph_mode)

    if case_study in ('Observed_Confounder', 'Backdoor_Criterion'):
        # X → T, X → Y, T → Y  (all X are confounders of the T→Y edge)
        A[T, Y] = 1.0
        for i in range(n_real):
            A[off + i, T] = 1.0
            A[off + i, Y] = 1.0

    elif case_study == 'Observed_Mediator':
        # T → X → Y  (X is mediator; T → Y also holds transitively)
        A[T, Y] = 1.0
        for i in range(n_real):
            A[T, off + i] = 1.0        # T ancestor of X
            A[off + i, Y] = 1.0        # X ancestor of Y

    elif case_study == 'Observed_Mediator_and_Confounder':
        # X_0 = confounder (X_0 → T, X_0 → Y)
        # X_1..X_{n-1} = mediators (T → X_i → Y)
        # T → Y direct
        A[T, Y] = 1.0
        if n_real >= 1:
            A[off + 0, T] = 1.0        # confounder → T
            A[off + 0, Y] = 1.0        # confounder → Y
        for i in range(1, n_real):
            A[T, off + i] = 1.0        # T → mediator
            A[off + i, Y] = 1.0        # mediator → Y

    elif case_study == 'Unobserved_Confounder':
        # Unobserved U → T, U → Y. Observed X carries NO known ancestor info.
        # Only T → Y is (weakly) known.
        A[T, Y] = 1.0

    elif case_study == 'Frontdoor_Criterion':
        # T → X → Y (X = mediator on the front-door path)
        # Unobserved U → T, U → Y (T→Y confounded).
        # Observed X is the front-door variable.
        A[T, Y] = 1.0
        for i in range(n_real):
            A[T, off + i] = 1.0        # T → X (frontdoor)
            A[off + i, Y] = 1.0        # X → Y

    else:
        raise ValueError(f'unknown case_study: {case_study}')

    A = _pad_negatives(A, off, n_real, model_n_features)

    # Propagate ancestor knowledge on the real block (fills entailed entries)
    try:
        from utils.graph_utils import propagate_ancestor_knowledge
        real_n = 2 + n_real
        real_block = torch.from_numpy(A[:real_n, :real_n].copy())
        real_block = propagate_ancestor_knowledge(real_block, raise_on_inconsistent=False)
        A[:real_n, :real_n] = real_block.numpy().astype(np.float32)
    except Exception:
        pass
    return A


# Backward-compat alias — UWYK's original name
def build_adjacency_matrix(model_n_features, n_real_features, graph_mode):
    """Old name (case-agnostic). Kept for callers that don't have case_study."""
    return build_adjacency_matrix_for_case(
        DATASET, model_n_features, n_real_features, graph_mode)


def _cate_uwyk_paper_pipeline(model, cate_dataset, graph_mode):
    """Call UWYK's own predict_cate() method — verbatim what the wrapper
    ships. It uses raw binary T_intv (1.0 / 0.0), inverse_transform=False
    per arm, then applies _inverse_transform_cate to the difference.
    """
    X_train = np.asarray(cate_dataset.X_train, dtype=np.float32)
    t_train = np.asarray(cate_dataset.t_train, dtype=np.float32).reshape(-1, 1)
    y_train = np.asarray(cate_dataset.y_train, dtype=np.float32).reshape(-1, 1)
    X_test  = np.asarray(cate_dataset.X_test,  dtype=np.float32)

    n_features_orig = X_train.shape[1]
    model_n_features = model.model.num_features
    n_real_features = min(n_features_orig, model_n_features)

    # Fit wrapper's preprocessing state on RAW binary T (0/1)
    model.fit(X_train, t_train, y_train)

    adjacency_matrix = build_adjacency_matrix_for_case(
        DATASET, model_n_features, n_real_features, graph_mode)

    # Wrapper's own predict_cate — handles the arm-difference math correctly
    cate_pred = model.predict_cate(
        X_obs=X_train, T_obs=t_train, Y_obs=y_train,
        X_intv=X_test,
        adjacency_matrix=adjacency_matrix,
        prediction_type='mean',
    )
    return np.asarray(cate_pred, dtype=np.float32).reshape(-1)


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = SCMCaseStudyDataset(DATASET)
    n = ds.n_tables if not MAX_REAL else min(ds.n_tables, int(MAX_REAL))
    print(f'[bootstrap] UWYK-paper-pipeline  graph_mode={GRAPH_MODE}  '
          f'{DATASET}  n={n}', flush=True)
    wrapper = _load_uwyk()

    rows = []; t0 = time.time()
    for r in range(n):
        cate_ds, _ = ds[r]
        try:
            cate_pred = _cate_uwyk_paper_pipeline(wrapper, cate_ds, GRAPH_MODE)
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
