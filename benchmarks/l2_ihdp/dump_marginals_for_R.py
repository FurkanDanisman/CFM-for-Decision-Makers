"""Dump one IHDP realization's raw J-bin marginals and true Y_CENTERS
densities to a JSON file so R MALC can be evaluated on the SAME input
our Python pipeline sees.

Runs the standard ours_densities pipeline on one realization, then extracts:
  - per-query raw p_mat marginals (J bins each, for both y0 and y1)
  - per-query true densities on Y_CENTERS (100 bins each, for both y0 and y1)
  - the J bin edges + Y_CENTERS grid so R can reproduce the geometry
  - Python's OWN MALC output on Y_CENTERS (so we can compare exact L2 numbers)

Usage:
    python dump_marginals_for_R.py \\
        --repo $DEPLOY_ROOT/R-PFN \\
        --dopfn $DEPLOY_ROOT/external/dopfn \\
        --causalpfn $DEPLOY_ROOT/external/CausalPFN \\
        --checkpoint-dopfn-bb $DEPLOY_ROOT/checkpoints_dopfn_backbone_realj10/step_150000.pt \\
        --realization 0 \\
        --out ihdp_marginals_r0.json

Output JSON is small (~few hundred KB) and safe to git commit.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--dopfn', required=True)
    ap.add_argument('--causalpfn', required=True,
                     help='Path to the CausalPFN repo (has benchmarks.IHDPDataset).')
    ap.add_argument('--checkpoint-dopfn-bb', required=True)
    ap.add_argument('--realization', type=int, default=0)
    ap.add_argument('--n-context', type=int, default=100)
    ap.add_argument('--n-queries-max', type=int, default=15,
                     help='Cap number of queries dumped to keep JSON small.')
    ap.add_argument('--y-scaling', default='min_max', choices=['min_max', 'std'])
    ap.add_argument('--malc-B', type=int, default=60)
    ap.add_argument('--malc-max-K', type=int, default=3)
    ap.add_argument('--n-eval', type=int, default=200)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'l2_ihdp'))
    sys.path.insert(0, os.path.join(args.repo, 'training_dopfn_base'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC', 'Optimal_Transport'))
    sys.path.insert(0, args.repo)
    sys.path.insert(0, args.causalpfn)

    import torch
    from true_ihdp import load_ihdp_truth, true_marginals_per_query, Y_CENTERS
    from methods_densities import ours_densities
    from dopfn_backbone_head import DoPFNBackboneWith2DHead
    from losses.BarDistribution2D import fit_malc_inner, unpack_pred
    from malc_2d import dmalc_2d
    from benchmarks import IHDPDataset

    cd, _ = IHDPDataset()[args.realization]
    y_train_full = np.asarray(cd.y_train.detach().cpu()
                              if hasattr(cd.y_train, 'detach') else cd.y_train)
    truth = load_ihdp_truth(args.realization, args.causalpfn, y_train_full)

    # ── Load model + get raw p_mats ────────────────────────────────
    print(f'[load] {args.checkpoint_dopfn_bb}', flush=True)
    ckpt = torch.load(args.checkpoint_dopfn_bb, map_location='cpu', weights_only=False)
    cfg = ckpt['config']; J = cfg['J']
    edges_np = ckpt['edges'].cpu().numpy()
    bin_width = float(edges_np[1] - edges_np[0])
    model = DoPFNBackboneWith2DHead(dopfn_root=args.dopfn, K=J).eval()
    model.load_state_dict(ckpt['model_state_dict'])

    # Run full ours_densities to get Python's MALC output for comparison
    print(f'[ours] running ours_densities ...', flush=True)
    d = ours_densities(
        cd, model, edges_np, J, bin_width, -1,
        y_min=truth.y_min, y_rng=truth.y_rng,
        malc_B=args.malc_B, malc_max_K=args.malc_max_K, n_eval=args.n_eval,
        n_context=args.n_context,
        fit_malc_inner=fit_malc_inner, dmalc_2d=dmalc_2d,
        y_scaling=args.y_scaling,
    )

    # Redo the model forward JUST to grab raw p_mats (ours_densities already
    # did this internally but didn't return them — simplest is to redo).
    from methods_densities import _np, _rescale_and_pad
    X_ctx_full = _np(cd.X_train); t_ctx_full = _np(cd.t_train); y_ctx_full = _np(cd.y_train)
    N = args.n_context if args.n_context else X_ctx_full.shape[0]
    X_ctx = X_ctx_full[:N].astype(np.float32)
    T_ctx = t_ctx_full[:N].astype(np.float32).reshape(-1, 1)
    Y_ctx = y_ctx_full[:N].astype(np.float32).reshape(-1, 1)
    if args.y_scaling == 'std':
        _mu = float(y_ctx_full.astype(np.float64).mean())
        _sig = float(y_ctx_full.astype(np.float64).std()) if y_ctx_full.size > 1 else 1.0
        _y_scale = max(_sig / 0.3, 1e-8)
        Y_ctx = ((Y_ctx - _mu) / _y_scale).astype(np.float32)
    else:
        Y_ctx = ((Y_ctx - truth.y_min) / truth.y_rng * 2.0 - 1.0).astype(np.float32)
    X_ctx = _rescale_and_pad(X_ctx, -1)
    X_test_p = _rescale_and_pad(_np(cd.X_test).astype(np.float32), -1)
    with torch.no_grad():
        pred = model(torch.from_numpy(X_ctx).unsqueeze(0),
                     torch.from_numpy(T_ctx).unsqueeze(0),
                     torch.from_numpy(Y_ctx).unsqueeze(0),
                     torch.from_numpy(X_test_p).unsqueeze(0))['predictions'][0]
    n_test = X_test_p.shape[0]
    p_mats = np.zeros((n_test, J, J), dtype=np.float32)
    for q in range(n_test):
        p_mat, *_ = unpack_pred(pred[q], J, bin_width)
        p_mats[q] = p_mat.detach().cpu().numpy()

    # True densities on Y_CENTERS
    p_y0_true, p_y1_true = true_marginals_per_query(truth)

    # Cap n_queries for JSON size
    n_dump = min(args.n_queries_max, n_test)
    print(f'[dump] {n_dump} queries → {args.out}', flush=True)

    blob = {
        'realization': args.realization,
        'J': int(J),
        'n_queries': int(n_dump),
        'y_scaling': args.y_scaling,
        'edges_scaled': edges_np.tolist(),        # (J+1,)  bin edges in scaled Y
        'Y_CENTERS': Y_CENTERS.tolist(),          # (100,)  common comparison grid (scaled)
        'y_min': float(truth.y_min),
        'y_rng': float(truth.y_rng),
        'queries': [],
    }
    for q in range(n_dump):
        blob['queries'].append({
            'q': q,
            'p_marg_y0_raw': p_mats[q].sum(axis=1).tolist(),    # (J,)
            'p_marg_y1_raw': p_mats[q].sum(axis=0).tolist(),    # (J,)
            'p_y0_true': p_y0_true[q].tolist(),                 # (100,) density on Y_CENTERS
            'p_y1_true': p_y1_true[q].tolist(),
            'py_malc_p_y0': d['p_y0'][q].tolist(),              # our Python MALC output
            'py_malc_p_y1': d['p_y1'][q].tolist(),
        })
    with open(args.out, 'w') as f:
        json.dump(blob, f)
    sz = os.path.getsize(args.out) / 1024
    print(f'[done] {args.out}  ({sz:.1f} KB)', flush=True)


if __name__ == '__main__':
    main()
