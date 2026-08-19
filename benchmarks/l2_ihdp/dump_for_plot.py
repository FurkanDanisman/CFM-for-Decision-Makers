"""Unified dump for R-vs-Python MALC plots. Handles ihdp, acic, and
synthetic polynomial-SCM cases. For one realization/seed + one query,
dumps to JSON:
  - raw J-bin marginals for y0 and y1 (input to MALC)
  - true densities on Y_CENTERS (comparison grid)
  - Python MALC output (default: MALC-1D via CVXPY + log-linear eval)
  - Do-PFN's density (from its own bar distribution + criterion.borders)

Default MALC hyperparams: B=100, max_K=1 (single log-concave, no mixture)
per user 2026-08-19.

Usage:
    # IHDP r=14
    python dump_for_plot.py --case ihdp --realization 14 --query 0 \\
        --repo $DEPLOY_ROOT/R-PFN --dopfn $DEPLOY_ROOT/external/dopfn \\
        --causalpfn $DEPLOY_ROOT/external/causalpfn \\
        --checkpoint-dopfn-bb $DEPLOY_ROOT/checkpoints_dopfn_backbone_realj10/step_150000.pt \\
        --out ihdp_r14_q0.json

    # ACIC r=6
    python dump_for_plot.py --case acic --realization 6 --query 0 [...same args...] \\
        --out acic_r6_q0.json

    # Synthetic polynomial SCM d=6 seed=6
    python dump_for_plot.py --case synthetic --d 6 --N 200 --seed 6 --query 0 [...] \\
        --out synthetic_d6_seed6_q0.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np


def _setup_paths(repo, causalpfn=None):
    sys.path.insert(0, os.path.join(repo, 'benchmarks', 'l2_ihdp'))
    sys.path.insert(0, os.path.join(repo, 'benchmarks', 'l2_acic'))
    sys.path.insert(0, os.path.join(repo, 'benchmarks', 'empirical_tests'))
    sys.path.insert(0, os.path.join(repo, 'benchmarks'))
    sys.path.insert(0, os.path.join(repo, 'training_dopfn_base'))
    sys.path.insert(0, os.path.join(repo, 'MALC'))
    sys.path.insert(0, os.path.join(repo, 'MALC', 'Optimal_Transport'))
    sys.path.insert(0, repo)
    if causalpfn: sys.path.insert(0, causalpfn)


def _load_bb_model(args):
    import torch
    from dopfn_backbone_head import DoPFNBackboneWith2DHead
    ckpt = torch.load(args.checkpoint_dopfn_bb, map_location='cpu', weights_only=False)
    cfg = ckpt['config']; J = cfg['J']
    edges_np = ckpt['edges'].cpu().numpy()
    bin_width = float(edges_np[1] - edges_np[0])
    model = DoPFNBackboneWith2DHead(dopfn_root=args.dopfn, K=J).eval()
    model.load_state_dict(ckpt['model_state_dict'])
    return model, edges_np, J, bin_width


def _get_p_mats(model, cd, edges_np, J, bin_width, y_min, y_rng, n_context, y_scaling):
    """Redo model forward to extract raw J×J p_mats per query."""
    import torch
    from losses.BarDistribution2D import unpack_pred
    from methods_densities import _np, _rescale_and_pad
    X_ctx_full = _np(cd.X_train); t_ctx_full = _np(cd.t_train); y_ctx_full = _np(cd.y_train)
    N = n_context if n_context else X_ctx_full.shape[0]
    X_ctx = X_ctx_full[:N].astype(np.float32)
    T_ctx = t_ctx_full[:N].astype(np.float32).reshape(-1, 1)
    Y_ctx = y_ctx_full[:N].astype(np.float32).reshape(-1, 1)
    if y_scaling == 'std':
        _mu = float(y_ctx_full.astype(np.float64).mean())
        _sig = float(y_ctx_full.astype(np.float64).std()) if y_ctx_full.size > 1 else 1.0
        _y_scale = max(_sig / 0.3, 1e-8)
        Y_ctx = ((Y_ctx - _mu) / _y_scale).astype(np.float32)
    else:
        Y_ctx = ((Y_ctx - y_min) / y_rng * 2.0 - 1.0).astype(np.float32)
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
    return p_mats


def _load_ihdp_case(args):
    from true_ihdp import load_ihdp_truth, true_marginals_per_query, true_cate_per_query, Y_CENTERS, TAU_CENTERS
    from benchmarks import IHDPDataset
    cd, _ = IHDPDataset()[args.realization]
    y_train_full = np.asarray(cd.y_train.detach().cpu()
                              if hasattr(cd.y_train, 'detach') else cd.y_train)
    truth = load_ihdp_truth(args.realization, args.causalpfn, y_train_full)
    p_y0_true, p_y1_true = true_marginals_per_query(truth)
    p_tau_true = true_cate_per_query(truth)
    return cd, truth.y_min, truth.y_rng, p_y0_true, p_y1_true, p_tau_true, Y_CENTERS, TAU_CENTERS


def _load_acic_case(args):
    """ACIC per-realization truth + context data. Mirrors benchmarks/l2_acic/eval_realization.py."""
    # ACIC's true_acic.py lives in l2_acic/ (not l2_ihdp/) — its Y_CENTERS may
    # differ from IHDP's; source from the same place.
    from true_acic import (load_acic_truth, true_marginals_per_query,
                            true_cate_per_query, Y_CENTERS, TAU_CENTERS)
    # ACIC via causalpfn.benchmarks
    from benchmarks import ACIC2016Dataset
    # Reuse the datasets-module shim from l2_acic/eval_realization.py so
    # dopfn's benchmark imports resolve.
    from eval_realization import _install_dopfn_datasets_shim   # noqa: F401
    try:
        _install_dopfn_datasets_shim(args.dopfn)
    except Exception as e:
        print(f'[warn] dopfn datasets shim failed: {type(e).__name__}: {e}', flush=True)
    cd, _ = ACIC2016Dataset()[args.realization]
    y_train_full = np.asarray(cd.y_train.detach().cpu()
                              if hasattr(cd.y_train, 'detach') else cd.y_train)
    truth = load_acic_truth(args.realization, y_train_full,
                             cache_dir=(args.acic_cache_dir or None))
    p_y0_true, p_y1_true = true_marginals_per_query(truth)
    p_tau_true = true_cate_per_query(truth)
    return cd, truth.y_min, truth.y_rng, p_y0_true, p_y1_true, p_tau_true, Y_CENTERS, TAU_CENTERS


def _load_synthetic_case(args):
    from fig2_pehe_l2 import (make_polynomial_scm, truth_marginals, truth_cate,
                                Y_GRID, TAU_GRID)
    cd = make_polynomial_scm(seed=args.seed, n_context=args.N, n_test=args.n_test,
                              rho_eff=min(args.rho, 0.99),
                              x_dim=args.d, degree=args.degree,
                              sigma_eps=args.sigma_eps)
    p_y0_true = truth_marginals(cd)[0]
    p_y1_true = truth_marginals(cd)[1]
    p_tau_true = truth_cate(cd)
    y_ctx = cd.y_train.numpy()
    y_min = float(y_ctx.min()); y_max = float(y_ctx.max())
    y_rng = max(y_max - y_min, 1e-6)
    return cd, y_min, y_rng, p_y0_true, p_y1_true, p_tau_true, Y_GRID, TAU_GRID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True, choices=['ihdp', 'acic', 'synthetic'])
    ap.add_argument('--repo', required=True)
    ap.add_argument('--dopfn', required=True)
    ap.add_argument('--causalpfn', default='',
                     help='Only needed for ihdp/acic cases.')
    ap.add_argument('--checkpoint-dopfn-bb', required=True)
    ap.add_argument('--realization', type=int, default=0,
                     help='For ihdp/acic cases.')
    ap.add_argument('--acic-cache-dir', default='',
                     help='Optional cache dir for ACIC truth (empty = use default).')
    ap.add_argument('--query', type=int, default=0,
                     help='Which test query to dump for plotting.')
    ap.add_argument('--n-context', type=int, default=100)
    ap.add_argument('--y-scaling', default='min_max', choices=['min_max', 'std'])
    # Synthetic-only
    ap.add_argument('--d', type=int, default=6)
    ap.add_argument('--N', type=int, default=200)
    ap.add_argument('--n-test', type=int, default=25)
    ap.add_argument('--seed', type=int, default=6)
    ap.add_argument('--rho', type=float, default=0.0)
    ap.add_argument('--sigma-eps', type=float, default=1.0)
    ap.add_argument('--degree', type=int, default=3)
    # MALC hyperparams (user 2026-08-19: B=100, K=1)
    ap.add_argument('--malc-B', type=int, default=100)
    ap.add_argument('--malc-max-K', type=int, default=1)
    ap.add_argument('--n-eval', type=int, default=200)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    _setup_paths(args.repo, args.causalpfn)

    import torch
    from losses.BarDistribution2D import fit_malc_inner
    from malc_2d import dmalc_2d
    from methods_densities import ours_densities, dopfn_densities
    from dopfn_helpers import load_dopfn

    print(f'[case] {args.case}', flush=True)
    if args.case == 'ihdp':
        cd, y_min, y_rng, p_y0_true, p_y1_true, p_tau_true, Y_CENTERS, TAU_CENTERS = _load_ihdp_case(args)
    elif args.case == 'acic':
        cd, y_min, y_rng, p_y0_true, p_y1_true, p_tau_true, Y_CENTERS, TAU_CENTERS = _load_acic_case(args)
    else:
        cd, y_min, y_rng, p_y0_true, p_y1_true, p_tau_true, Y_CENTERS, TAU_CENTERS = _load_synthetic_case(args)

    # Load DoPFN-bb model + get p_mats
    model, edges_np, J, bin_width = _load_bb_model(args)
    print(f'[bb] J={J}  running ours_densities (B={args.malc_B}, max_K={args.malc_max_K}) ...', flush=True)
    d_bb = ours_densities(
        cd, model, edges_np, J, bin_width, -1,
        y_min=y_min, y_rng=y_rng,
        malc_B=args.malc_B, malc_max_K=args.malc_max_K, n_eval=args.n_eval,
        n_context=args.n_context,
        fit_malc_inner=fit_malc_inner, dmalc_2d=dmalc_2d,
        y_scaling=args.y_scaling,
    )
    p_mats = _get_p_mats(model, cd, edges_np, J, bin_width, y_min, y_rng,
                          args.n_context, args.y_scaling)

    # Do-PFN densities
    DoPFNRegressor = load_dopfn(args)
    print(f'[dopfn] running dopfn_densities ...', flush=True)
    d_dopfn = dopfn_densities(cd, DoPFNRegressor, y_min=y_min, y_rng=y_rng,
                                dopfn_root=args.dopfn, n_context=args.n_context)

    q = args.query
    n_test = p_mats.shape[0]
    if q >= n_test:
        print(f'[warn] q={q} out of range (n_test={n_test}), clamping to 0'); q = 0

    print(f'[dump] q={q}  n_test={n_test}  → {args.out}', flush=True)
    blob = {
        'case': args.case,
        'realization': int(args.realization),
        'seed': int(args.seed) if args.case == 'synthetic' else None,
        'query': int(q),
        'J': int(J),
        'y_scaling': args.y_scaling,
        'malc_B': int(args.malc_B),
        'malc_max_K': int(args.malc_max_K),
        'edges_scaled': edges_np.tolist(),
        'Y_CENTERS': np.asarray(Y_CENTERS).tolist(),
        'TAU_CENTERS': np.asarray(TAU_CENTERS).tolist(),
        'y_min': float(y_min),
        'y_rng': float(y_rng),
        # Chosen query
        'p_marg_y0_raw': p_mats[q].sum(axis=1).tolist(),
        'p_marg_y1_raw': p_mats[q].sum(axis=0).tolist(),
        # Ground truth
        'p_y0_true': np.asarray(p_y0_true[q]).tolist(),
        'p_y1_true': np.asarray(p_y1_true[q]).tolist(),
        'p_tau_true': np.asarray(p_tau_true[q]).tolist(),
        # Python-side MALC output (DoPFN-bb)
        'py_bb_p_y0': d_bb['p_y0'][q].tolist(),
        'py_bb_p_y1': d_bb['p_y1'][q].tolist(),
        'py_bb_p_tau': d_bb['p_tau'][q].tolist(),
        # Do-PFN output
        'dopfn_p_y0': d_dopfn['p_y0'][q].tolist(),
        'dopfn_p_y1': d_dopfn['p_y1'][q].tolist(),
        'dopfn_p_tau': d_dopfn['p_tau'][q].tolist(),
    }
    with open(args.out, 'w') as f:
        json.dump(blob, f)
    sz = os.path.getsize(args.out) / 1024
    print(f'[done] {args.out}  ({sz:.1f} KB)', flush=True)


if __name__ == '__main__':
    main()
