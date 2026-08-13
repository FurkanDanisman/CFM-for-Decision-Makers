"""Direct reproduction test: runs UWYK's dofm_full_conditioning.py logic verbatim.

Purpose: isolate whether the ~71% PSID (unbalanced) and 15% CPS Anc gap comes
from our wrapper code (benchmarks/methods/uwyk.py) or from something outside
it (deployed wrapper file, environment, etc.).

This script does NOT touch our uwyk.py or run_one.py. It imports UWYK's
PreprocessingGraphConditionedPFN directly and calls .predict() the way
UWYK's own dofm_full_conditioning.py does.

Usage on cluster:
    cd $DEPLOY_ROOT/external/dopfn
    python $DEPLOY_ROOT/R-PFN/benchmarks/uwyk_direct_repro.py \
        --dataset PSID --realization 0 --graph_mode full_graph
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np


def build_adjacency_matrix(model_n_features, n_real_features, graph_mode="full_graph"):
    """VERBATIM copy of dofm_full_conditioning.py::build_adjacency_matrix."""
    adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
    T_idx = 0
    Y_idx = 1
    feature_offset = 2

    if graph_mode == "all_unknown":
        pass  # nothing to set
    elif graph_mode == "full_graph":
        adjacency_matrix[T_idx, Y_idx] = 1.0
        for i in range(n_real_features):
            adjacency_matrix[feature_offset + i, T_idx] = 1.0
            adjacency_matrix[feature_offset + i, Y_idx] = 1.0
    else:
        raise ValueError(graph_mode)

    for i in range(n_real_features, model_n_features):
        feat_idx = feature_offset + i
        adjacency_matrix[feat_idx, :] = -1.0
        adjacency_matrix[:, feat_idx] = -1.0
        adjacency_matrix[feat_idx, feat_idx] = -1.0

    return adjacency_matrix


def paper_pipeline(model, cate_dataset, graph_mode):
    """VERBATIM copy of dofm_full_conditioning.py::dofm_full_conditioning_pipeline."""
    X_train = np.asarray(cate_dataset.X_train, dtype=np.float32)
    t_train_orig = np.asarray(cate_dataset.t_train, dtype=np.float32)
    t_train_orig = t_train_orig.reshape(-1, 1) if t_train_orig.ndim == 1 else t_train_orig
    y_train_orig = np.asarray(cate_dataset.y_train, dtype=np.float32)
    y_train_orig = y_train_orig.reshape(-1, 1) if y_train_orig.ndim == 1 else y_train_orig
    X_test = np.asarray(cate_dataset.X_test, dtype=np.float32)
    y_train = y_train_orig

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_features = X_train.shape[1]
    print(f"[Dataset] Train samples: {n_train}, Test samples: {n_test}, Features: {n_features}")

    t_flat = t_train_orig.flatten()
    y_flat = y_train.flatten()
    mean_y_t0 = y_flat[t_flat == 0].mean()
    mean_y_t1 = y_flat[t_flat == 1].mean()
    t_train = np.where(t_train_orig == 0, mean_y_t0, mean_y_t1).astype(np.float32)

    t_intv_0_encoded = mean_y_t0
    t_intv_1_encoded = mean_y_t1
    print(f"[Target Encoding] T=0 -> {mean_y_t0:.4f}, T=1 -> {mean_y_t1:.4f}")

    n_features_orig = X_train.shape[1]
    model_n_features = model.model.num_features

    model.fit(X_train, t_train, y_train)

    n_real_features = min(n_features_orig, model_n_features)
    adjacency_matrix = build_adjacency_matrix(model_n_features, n_real_features, graph_mode)

    T_intv_1 = np.full((n_test, 1), t_intv_1_encoded, dtype=np.float32)
    y_pred_1 = model.predict(
        X_obs=X_train, T_obs=t_train, Y_obs=y_train,
        X_intv=X_test, T_intv=T_intv_1,
        adjacency_matrix=adjacency_matrix,
        prediction_type="mean", inverse_transform=True,
    )

    T_intv_0 = np.full((n_test, 1), t_intv_0_encoded, dtype=np.float32)
    y_pred_0 = model.predict(
        X_obs=X_train, T_obs=t_train, Y_obs=y_train,
        X_intv=X_test, T_intv=T_intv_0,
        adjacency_matrix=adjacency_matrix,
        prediction_type="mean", inverse_transform=True,
    )

    cate_pred = y_pred_1 - y_pred_0
    return np.asarray(cate_pred).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['CPS', 'PSID', 'ACIC', 'IHDP'])
    ap.add_argument('--realization', type=int, default=0)
    ap.add_argument('--graph_mode', choices=['full_graph', 'all_unknown'], default='full_graph')
    ap.add_argument('--random_state', type=int, default=None,
                    help='Pass to PreprocessingGraphConditionedPFN. None = paper default (uses global RNG).')
    ap.add_argument('--deploy_root', default=os.environ.get('DEPLOY_ROOT', ''))
    args = ap.parse_args()

    if not args.deploy_root:
        raise SystemExit('Set DEPLOY_ROOT env var or pass --deploy_root')

    sys.path.insert(0, args.deploy_root + '/external/uwyk')
    sys.path.insert(0, args.deploy_root + '/external/causalpfn')

    from src.models.PreprocessingGraphConditionedPFN import PreprocessingGraphConditionedPFN
    from benchmarks import (IHDPDataset, ACIC2016Dataset,
                             RealCauseLalondeCPSDataset, RealCauseLalondePSIDDataset)
    DS_MAP = {'IHDP': IHDPDataset, 'ACIC': ACIC2016Dataset,
              'CPS': RealCauseLalondeCPSDataset, 'PSID': RealCauseLalondePSIDDataset}

    CKPT_DIR = (args.deploy_root +
                '/external/uwyk/experiments/checkpoints/full_conditioned_model/'
                'final_earlytest_full_conditioning_16773252.0')

    print(f"[direct-repro] loading UWYK model (random_state={args.random_state})")
    mdl_kw = dict(config_path=CKPT_DIR + '/best_model_config.yaml',
                   checkpoint_path=CKPT_DIR + '/best_model.pt',
                   device='cpu', verbose=False)
    if args.random_state is not None:
        mdl_kw['random_state'] = args.random_state
    mdl = PreprocessingGraphConditionedPFN(**mdl_kw).load()

    print(f"[direct-repro] loading {args.dataset} realization {args.realization}")
    cd, ad = DS_MAP[args.dataset]()[args.realization]

    print(f"[direct-repro] running paper's pipeline verbatim, graph_mode={args.graph_mode}")
    cate = paper_pipeline(mdl, cd, graph_mode=args.graph_mode)

    true_cate = np.asarray(cd.true_cate, dtype=np.float32).reshape(-1)
    pehe = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
    ate_pred = float(cate.mean())
    ate_true = float(true_cate.mean())
    ate_err = abs(ate_pred - ate_true) / max(abs(ate_true), 1e-9)

    print()
    print(f'=== DIRECT REPRO ({args.dataset} r{args.realization}, {args.graph_mode}) ===')
    print(f'  PEHE:     {pehe:.2f}')
    print(f'  ATE(pred):{ate_pred:+.4f}   ATE(true):{ate_true:+.4f}   eps_ATE:{ate_err:.4f}')
    print()
    print('Paper Table 3 (n=100 mean ± SEM):')
    if args.dataset == 'PSID':
        print('  Anc:   12975 ± 24     NoAnc: 13096 ± 26')
    elif args.dataset == 'CPS':
        print('  Anc:   11213 ± 60     NoAnc: 12800 ± 55')
    elif args.dataset == 'ACIC':
        print('  Anc:    2.79 ± 0.45   NoAnc:  3.47 ± 0.47')
    elif args.dataset == 'IHDP':
        print('  Anc:    5.49 ± 0.78   NoAnc:  6.28 ± 0.79')


if __name__ == '__main__':
    main()
