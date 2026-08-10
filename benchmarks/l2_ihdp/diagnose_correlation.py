"""Measure ρ predicted by fn=50 and fn=10 joint densities on IHDP.

For each query, compute Pearson correlation from the raw p_mat (bin
probabilities on the K×K grid). Truth ρ = 0 (Hill 2011 uses independent
noise per arm). Report aggregate stats.

Two forms of ρ per query:
  raw    : ρ from the raw K×K p_mat (transformer output, no MALC)
  smooth : ρ from the MALC-smoothed continuous density on the fine grid

Also breaks queries into groups (easy vs hard) based on the true CATE
value: if fn=10 mispredicts correlation specifically on hard queries,
the aggregate splits will show it.

Usage
-----
    python R-PFN/benchmarks/l2_ihdp/diagnose_correlation.py \\
        --repo /scratch/furkanbd/rpfn_bench_kit/R-PFN \\
        --causalpfn /scratch/furkanbd/rpfn_bench_kit/external/causalpfn \\
        --dopfn /scratch/furkanbd/rpfn_bench_kit/external/dopfn \\
        --checkpoint50 /scratch/furkanbd/rpfn_bench_kit/R-PFN/checkpoints/step_50000_final.pt \\
        --checkpoint10 /scratch/furkanbd/rpfn_bench_kit/R-PFN/checkpoints_dopfn/step_50000_final.pt \\
        --realization 0 --malc-B 60 --n-eval 200
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import types

import numpy as np
import torch


def _corr_from_2d(p, y0_grid, y1_grid, dy0, dy1):
    """Pearson ρ from a 2D density `p[i, j]` where i=y1_index, j=y0_index."""
    # marginals
    m_y0 = p.sum(axis=0) * dy1                    # shape (len(y0_grid),)
    m_y1 = p.sum(axis=1) * dy0                    # shape (len(y1_grid),)
    E_y0 = float((y0_grid * m_y0).sum() * dy0)
    E_y1 = float((y1_grid * m_y1).sum() * dy1)
    E_y0y1 = float((y1_grid[:, None] * y0_grid[None, :] * p).sum() * dy0 * dy1)
    Var_y0 = max(float(((y0_grid - E_y0) ** 2 * m_y0).sum() * dy0), 1e-16)
    Var_y1 = max(float(((y1_grid - E_y1) ** 2 * m_y1).sum() * dy1), 1e-16)
    Cov = E_y0y1 - E_y0 * E_y1
    return Cov / (np.sqrt(Var_y0) * np.sqrt(Var_y1))


def _install_dopfn_shim(dopfn_dir):
    if 'datasets' in sys.modules:
        return
    sys.path.insert(0, dopfn_dir)
    ds_mod = types.ModuleType('datasets')
    with open(os.path.join(dopfn_dir, 'datasets/__init__.py')) as fp:
        src = fp.read().split('def load_semi_real')[0]
    exec(src, ds_mod.__dict__)
    sys.modules['datasets'] = ds_mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo',           required=True)
    ap.add_argument('--causalpfn',      required=True)
    ap.add_argument('--dopfn',          required=True)
    ap.add_argument('--checkpoint50',   required=True)
    ap.add_argument('--checkpoint10',   required=True)
    ap.add_argument('--realization',    type=int, default=0)
    ap.add_argument('--malc-B',         type=int, default=60)
    ap.add_argument('--n-eval',         type=int, default=200)
    ap.add_argument('--malc-max-K',     type=int, default=1)
    args = ap.parse_args()

    _install_dopfn_shim(args.dopfn)
    sys.path.insert(0, args.causalpfn)
    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC', 'Optimal_Transport'))

    from benchmarks import IHDPDataset
    from models.InterventionalPFN import InterventionalPFN
    from losses.BarDistribution2D import unpack_pred, fit_malc_inner
    from malc_2d import dmalc_2d

    cd, _ = IHDPDataset()[args.realization]
    y_train_full = cd.y_train.numpy() if hasattr(cd.y_train, 'numpy') else np.asarray(cd.y_train)
    X_train_full = cd.X_train.numpy() if hasattr(cd.X_train, 'numpy') else np.asarray(cd.X_train)
    t_train_full = cd.t_train.numpy() if hasattr(cd.t_train, 'numpy') else np.asarray(cd.t_train)
    X_test = cd.X_test.numpy() if hasattr(cd.X_test, 'numpy') else np.asarray(cd.X_test)
    true_cate = cd.true_cate.numpy() if hasattr(cd.true_cate, 'numpy') else np.asarray(cd.true_cate)
    true_cate = true_cate.reshape(-1)

    y_min = float(y_train_full.min()); y_max = float(y_train_full.max())
    y_rng = max(y_max - y_min, 1e-6)
    true_cate_scaled = true_cate * (2.0 / y_rng)

    # Split queries by |true tau|: easy = smallest half, hard = largest half
    order = np.argsort(np.abs(true_cate_scaled))
    easy_idx = order[:len(order) // 2].tolist()
    hard_idx = order[len(order) // 2:].tolist()

    for label, ckpt_path in [('fn=50', args.checkpoint50), ('fn=10', args.checkpoint10)]:
        print(f'\n{"=" * 72}\n[{label}]  {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        cfg = ckpt['config']; J = cfg['J']; num_features = cfg['num_features']
        edges_np = ckpt['edges'].cpu().numpy()
        bin_width = float(edges_np[1] - edges_np[0])
        bin_centres = 0.5 * (edges_np[:-1] + edges_np[1:])
        print(f'  cfg: J={J}  num_features={num_features}  edges=[{edges_np[0]:.2f},'
              f' {edges_np[-1]:.2f}]  bin_width={bin_width:.4f}')

        model = InterventionalPFN(
            num_features=num_features, d_model=cfg['d_model'], depth=cfg['depth'],
            heads_feat=cfg['heads'], heads_samp=cfg['heads'], dropout=0.0,
            output_dim=J*J + 9 + 4, hidden_mult=cfg['hidden_mult'],
            normalize_features=True, normalize_treatment=False,
            use_treatment_in_query=False, use_checkpoint=False,
        ).eval()
        model.load_state_dict(ckpt['model_state_dict'])

        N = X_train_full.shape[0]
        X_context = X_train_full[:N].astype(np.float32)
        T_context = t_train_full[:N].astype(np.float32).reshape(-1, 1)
        Y_context = y_train_full[:N].astype(np.float32).reshape(-1, 1)
        Y_context = ((Y_context - y_min) / y_rng * 2.0 - 1.0).astype(np.float32)

        d = X_context.shape[1]
        if d < num_features:
            pad = np.full((X_context.shape[0], num_features - d), np.nan, dtype=np.float32)
            X_context = np.concatenate([X_context, pad], axis=1)
            Xt_pad = np.full((X_test.shape[0], num_features - d), np.nan, dtype=np.float32)
            X_test_p = np.concatenate([X_test.astype(np.float32), Xt_pad], axis=1)
        else:
            X_context = X_context[:, :num_features]
            X_test_p  = X_test.astype(np.float32)[:, :num_features]

        Xc = torch.from_numpy(X_context).unsqueeze(0)
        Tc = torch.from_numpy(T_context).unsqueeze(0)
        Yc = torch.from_numpy(Y_context).unsqueeze(0)
        Xq = torch.from_numpy(X_test_p).unsqueeze(0)
        with torch.no_grad():
            pred = model(Xc, Tc, Yc, Xq)['predictions'][0]

        xs = np.linspace(edges_np[0], edges_np[-1], args.n_eval)
        ys = np.linspace(edges_np[0], edges_np[-1], args.n_eval)
        XX, YY = np.meshgrid(xs, ys, indexing='xy')
        eval_pts = np.column_stack([XX.ravel(), YY.ravel()])
        dxs = float(xs[1] - xs[0]); dys = float(ys[1] - ys[0])

        rho_raw_all = []
        rho_smooth_all = []
        n_test = X_test_p.shape[0]
        for q in range(n_test):
            p_mat, *_ = unpack_pred(pred[q], J, bin_width)
            p_mat_np = p_mat.detach().cpu().numpy()
            # Normalise so ∑ p_ij bin_width^2 = 1 (per-bin density)
            p_dens = p_mat_np / (p_mat_np.sum() * bin_width * bin_width)
            # p_mat_np convention: p_mat[y0_idx, y1_idx]  (see plot_ihdp_n10.py:
            # imshow(p_mats.T) with row=y1). For _corr_from_2d we need p[y1, y0]
            # so transpose.
            p_dens_yx = p_dens.T
            rho_raw = _corr_from_2d(p_dens_yx, bin_centres, bin_centres,
                                    bin_width, bin_width)

            seed = int(hashlib.md5(f'q{q}_{label}'.encode()).hexdigest()[:8], 16) % 10**8
            fit = fit_malc_inner(p_mat_np.T, edges_np, edges_np,
                                 B_fit=args.malc_B, B_select=args.malc_B,
                                 max_K=args.malc_max_K, seed=seed, parallel=False)
            density_2d = dmalc_2d(fit, eval_pts).reshape(args.n_eval, args.n_eval)
            rho_smooth = _corr_from_2d(density_2d, xs, ys, dxs, dys)

            rho_raw_all.append(rho_raw)
            rho_smooth_all.append(rho_smooth)

        rho_raw_all = np.array(rho_raw_all)
        rho_smooth_all = np.array(rho_smooth_all)

        def _pr(name, arr):
            print(f'  {name:26s} n={arr.size:>3d}   mean={arr.mean():+.3f}   '
                  f'median={np.median(arr):+.3f}   std={arr.std(ddof=1):.3f}   '
                  f'min={arr.min():+.3f}  max={arr.max():+.3f}')

        print(f'  --- ρ from raw p_mat (transformer output, no MALC) ---')
        _pr('all queries',   rho_raw_all)
        _pr('easy (small τ)', rho_raw_all[easy_idx])
        _pr('hard (large τ)', rho_raw_all[hard_idx])
        print(f'  --- ρ from MALC-smoothed density (K={args.malc_max_K}) ---')
        _pr('all queries',   rho_smooth_all)
        _pr('easy (small τ)', rho_smooth_all[easy_idx])
        _pr('hard (large τ)', rho_smooth_all[hard_idx])

    print('\nTruth ρ on IHDP = 0 exactly (Hill 2011 uses independent noise per arm).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
