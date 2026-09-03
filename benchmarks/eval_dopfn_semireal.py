"""Evaluate OURS on Do-PFN's semi-real known-graph datasets (sales, law_race).

Uses DoPFN's own load_dataset() and generate_valid_split() to get the same
train/test splits as inference_example.py. Runs ours_pipeline with any
combination of --backbone dopfn_bb, --skip-malc, --y-log-transform.

Because MALC/EM values require the joint diag-integration and Y-preprocessing
un-scaling would need per-quantity Jacobians, only raw-mean is meaningful when
--skip-malc or Y-transform is on (matches how other Ours pipelines behave in
that mode). Prints raw-mean PEHE + eps_ATE side by side with Do-PFN's own
reported numbers (for eyeballing).

Also runs Do-PFN's own DoPFNRegressor.predict_cate for a paired reference on
the same split — that's the exact baseline the paper reports.

Usage
-----
    python R-PFN/benchmarks/eval_dopfn_semireal.py \\
        --dataset sales \\
        --repo /scratch/.../R-PFN \\
        --dopfn /scratch/.../external/dopfn \\
        --causalpfn /scratch/.../external/causalpfn \\
        --checkpoint /scratch/.../checkpoints_dopfn_backbone_j10/step_200000.pt \\
        --backbone dopfn_bb --skip-malc --y-log-transform \\
        --outdir /scratch/.../results_dopfn_semireal
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import types

import numpy as np
import torch


def _to_np(a):
    if isinstance(a, torch.Tensor): return a.numpy()
    return np.asarray(a)


class _CateDS:
    """Duck-typed CATE dataset object matching what ours_pipeline expects."""
    def __init__(self, X_train, t_train, y_train, X_test, true_cate):
        self.X_train = torch.from_numpy(X_train.astype(np.float32))
        self.t_train = torch.from_numpy(t_train.astype(np.float32))
        self.y_train = torch.from_numpy(y_train.astype(np.float32))
        self.X_test  = torch.from_numpy(X_test.astype(np.float32))
        self.true_cate = torch.from_numpy(true_cate.astype(np.float32))


def _load_dopfn_dataset(dopfn_root, ds_name):
    """Import DoPFN's datasets module from its repo root and load one dataset."""
    _cwd = os.getcwd()
    if dopfn_root not in sys.path:
        sys.path.insert(0, dopfn_root)
    try:
        os.chdir(dopfn_root)
        from datasets import load_dataset as _dopfn_load
        ds = _dopfn_load(ds_name=ds_name)
    finally:
        os.chdir(_cwd)
    return ds


def _split_to_cate(train_ds, test_ds):
    """DoPFN's TabularDataset has .x with T as column 0 and .y as outcome.
    .cate on test_ds gives per-unit true CATE."""
    xt = _to_np(train_ds.x); yt = _to_np(train_ds.y)
    xe = _to_np(test_ds.x)
    true_cate = _to_np(test_ds.cate).reshape(-1)
    X_train = xt[:, 1:].astype(np.float32)
    t_train = xt[:, 0].astype(np.float32)
    y_train = yt.astype(np.float32)
    X_test  = xe[:, 1:].astype(np.float32)
    return X_train, t_train, y_train, X_test, true_cate


def _load_our_model(args):
    """Twin of benchmarks/run_one.py::_load_our_model but stripped."""
    sys.path.insert(0, args.repo); sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['config']; J = cfg['J']
    edges_np = ckpt['edges'].cpu().numpy()
    bin_width = float(edges_np[1] - edges_np[0])
    centers = 0.5 * (edges_np[:-1] + edges_np[1:])
    if args.backbone == 'dopfn_bb':
        sys.path.insert(0, os.path.join(args.repo, 'training_dopfn_base'))
        from dopfn_backbone_head import DoPFNBackboneWith2DHead
        m = DoPFNBackboneWith2DHead(dopfn_root=args.dopfn, K=J).eval()
        m.load_state_dict(ckpt['model_state_dict'])
        NUM_FEATURES = -1
    else:
        from models.InterventionalPFN import InterventionalPFN
        NUM_FEATURES = cfg['num_features']
        m = InterventionalPFN(
            num_features=NUM_FEATURES, d_model=cfg['d_model'], depth=cfg['depth'],
            heads_feat=cfg['heads'], heads_samp=cfg['heads'], dropout=0.0,
            output_dim=J*J + 9 + 4, hidden_mult=cfg['hidden_mult'],
            normalize_features=True, normalize_treatment=False,
            use_treatment_in_query=False, use_checkpoint=False,
        ).eval()
        m.load_state_dict(ckpt['model_state_dict'])
    ot_dir = os.path.join(args.repo, 'MALC', 'Optimal_Transport')
    if ot_dir not in sys.path: sys.path.insert(0, ot_dir)
    from ot_barycenter import wasserstein_barycenter_1d
    return m, edges_np, J, bin_width, centers, NUM_FEATURES, wasserstein_barycenter_1d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['sales', 'law_race'])
    ap.add_argument('--split-number', type=int, default=1)
    ap.add_argument('--n-splits', type=int, default=5)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--dopfn', required=True)
    ap.add_argument('--causalpfn', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--backbone', choices=['ipfn', 'dopfn_bb'], default='dopfn_bb')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--malc-B', type=int, default=100)
    ap.add_argument('--malc-max-K', type=int, default=3)
    ap.add_argument('--n-eval', type=int, default=200)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--skip-malc', action='store_true')
    ap.add_argument('--y-log-transform', action='store_true')
    ap.add_argument('--y-power-transform', action='store_true')
    ap.add_argument('--y-winsorize', type=float, default=0.0)
    ap.add_argument('--ours-max-n-train', type=int, default=1000)
    ap.add_argument('--ours-query-chunk',  type=int, default=20)
    ap.add_argument('--also-dopfn', action='store_true',
                    help='Also run Do-PFN\'s own DoPFNRegressor for reference.')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f'[dopfn-semireal] loading dataset={args.dataset}', flush=True)
    ds = _load_dopfn_dataset(args.dopfn, args.dataset)
    train_ds, test_ds = ds.generate_valid_split(
        split_number=args.split_number, n_splits=args.n_splits)
    X_train, t_train, y_train, X_test, true_cate = _split_to_cate(train_ds, test_ds)
    print(f'  N_train={X_train.shape[0]}  d={X_train.shape[1]}  '
          f'N_test={X_test.shape[0]}  y_range=[{y_train.min():.3f}, {y_train.max():.3f}]',
          flush=True)

    # Load OURS
    print(f'[dopfn-semireal] loading OURS ({args.backbone})', flush=True)
    (our_model, edges_np, J, bin_width, centers, NUM_FEATURES,
     wasserstein_barycenter_1d) = _load_our_model(args)

    # Run pipeline
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks'))
    from methods.ours import ours_pipeline

    cd = _CateDS(X_train, t_train, y_train, X_test, true_cate)
    t0 = time.time()
    print(f'[dopfn-semireal] OURS inference', flush=True)
    ours = ours_pipeline(cd, our_model, edges_np, J, bin_width, NUM_FEATURES,
                          centers, args, wasserstein_barycenter_1d)
    dt = time.time() - t0

    true_ate = float(np.mean(true_cate))
    ours_cate = np.asarray(ours['ours_mean'], dtype=float)
    ours_ate = float(np.mean(ours_cate))
    pehe = float(np.sqrt(np.mean((ours_cate - true_cate) ** 2)))
    eps_ate = float(abs(ours_ate - true_ate) / max(abs(true_ate), 0.1))

    out = dict(
        dataset=args.dataset,
        split_number=args.split_number,
        n_splits=args.n_splits,
        backbone=args.backbone,
        checkpoint=args.checkpoint,
        y_log_transform=args.y_log_transform,
        y_power_transform=args.y_power_transform,
        skip_malc=args.skip_malc,
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        d=int(X_train.shape[1]),
        true_ate=true_ate,
        ours_ate=ours_ate,
        pehe_ours_mean=pehe,
        err_ours_mean=eps_ate,
        runtime_s=dt,
    )
    print()
    print(f'=== OURS ({args.backbone}) on {args.dataset} split {args.split_number}/{args.n_splits} ===')
    print(f'  PEHE(raw-mean):    {pehe:.4f}')
    print(f'  eps_ATE(raw-mean): {eps_ate:.4f}')
    print(f'  true_ATE:          {true_ate:+.4f}   ours_ATE: {ours_ate:+.4f}')
    print(f'  runtime:           {dt:.1f}s')

    # Save OURS shard FIRST so a downstream Do-PFN error doesn't waste OURS work.
    shard = os.path.join(args.outdir,
                          f'{args.dataset}_split{args.split_number}_of{args.n_splits}'
                          f'{"_logy" if args.y_log_transform else ""}'
                          f'{"_pt"   if args.y_power_transform else ""}.npz')
    np.savez(shard, **{k: np.asarray(v) for k, v in out.items()})
    print(f'\n[save] {shard}', flush=True)

    if args.also_dopfn:
        print()
        print('[dopfn-semireal] running Do-PFN reference (DoPFNRegressor.predict_cate)', flush=True)

        def _t(x):
            if isinstance(x, torch.Tensor):
                return x.float()
            return torch.as_tensor(np.asarray(x), dtype=torch.float32)

        _cwd = os.getcwd()
        try:
            os.chdir(args.dopfn)
            sys.path.insert(0, args.dopfn)
            from scripts.transformer_prediction_interface.base import DoPFNRegressor
            reg = DoPFNRegressor()
            reg.fit(_t(train_ds.x), _t(train_ds.y))
            dp_cate = np.asarray(reg.predict_cate(_t(test_ds.x))).reshape(-1)
        except Exception as e:
            print(f'[warn] Do-PFN reference failed: {type(e).__name__}: {e}')
            dp_cate = None
        finally:
            os.chdir(_cwd)

        if dp_cate is not None:
            dp_ate = float(np.mean(dp_cate))
            dp_pehe = float(np.sqrt(np.mean((dp_cate - true_cate) ** 2)))
            dp_eps  = float(abs(dp_ate - true_ate) / max(abs(true_ate), 0.1))
            out['pehe_dopfn'] = dp_pehe
            out['err_dopfn']  = dp_eps
            out['dopfn_ate']  = dp_ate
            # Re-save with DoPFN fields appended
            np.savez(shard, **{k: np.asarray(v) for k, v in out.items()})
            print(f'=== Do-PFN reference on {args.dataset} split {args.split_number} ===')
            print(f'  PEHE:              {dp_pehe:.4f}')
            print(f'  eps_ATE:           {dp_eps:.4f}')
            print(f'  true_ATE:          {true_ate:+.4f}   dopfn_ATE: {dp_ate:+.4f}')


if __name__ == '__main__':
    main()
