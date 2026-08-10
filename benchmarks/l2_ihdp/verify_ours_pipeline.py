"""End-to-end verification for fn=50 vs fn=10 CATE pipeline.

Loads a single IHDP realization, runs both checkpoints, and prints:
  1. Checkpoint config diff (edges range, bin_width, num_features, J)
  2. Raw p_mat statistics per query (max value, entropy)
  3. Raw p_mat marginal moments vs MALC-smoothed marginal moments
  4. CATE via three methods per checkpoint:
       (a) joint-diagonal integration (current pipeline)
       (b) marginals-under-independence convolution (Do-PFN's recipe)
       (c) analytical truth (Gaussian)
  5. L2 to truth for each method.

Reveals whether:
  - fn=10's p_mat is truly spiky (pipeline correct, model over-confident)
  - fn=10's joint-diagonal is buggy relative to marginals-conv
  - Any difference in edges/bin_width between the two checkpoints

Usage
-----
    python R-PFN/benchmarks/l2_ihdp/verify_ours_pipeline.py \\
        --repo /scratch/furkanbd/rpfn_bench_kit/R-PFN \\
        --causalpfn /scratch/furkanbd/rpfn_bench_kit/external/causalpfn \\
        --dopfn /scratch/furkanbd/rpfn_bench_kit/external/dopfn \\
        --checkpoint50 /scratch/furkanbd/rpfn_bench_kit/R-PFN/checkpoints/step_50000_final.pt \\
        --checkpoint10 /scratch/furkanbd/rpfn_bench_kit/R-PFN/checkpoints_dopfn/step_50000_final.pt \\
        --realization 0 --queries 4,2,18,66,64,42 --malc-B 60 --n-eval 200
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import types

import numpy as np
import torch


TAU_EDGES = np.linspace(-3.0, 3.0, 601)
TAU_CENTERS = 0.5 * (TAU_EDGES[:-1] + TAU_EDGES[1:])
TAU_BIN = float(TAU_CENTERS[1] - TAU_CENTERS[0])
Y_EDGES = np.linspace(-1.5, 1.5, 101)
Y_CENTERS = 0.5 * (Y_EDGES[:-1] + Y_EDGES[1:])
Y_BIN = float(Y_CENTERS[1] - Y_CENTERS[0])


def _trap(f, x):
    fn = getattr(np, 'trapezoid', np.trapz)
    return float(fn(f, x))


def _l2(f, g, grid):
    return float(np.sqrt(_trap((f - g) ** 2, grid)))


def _gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2.0 * np.pi))


def _naive_p_tau(p_y0, p_y1):
    """CATE under independence from marginals on Y_CENTERS -> density on TAU_CENTERS."""
    n = p_y0.shape[0]
    density = np.correlate(p_y1, p_y0, mode='full')
    tau_native = np.arange(-(n - 1), n) * Y_BIN
    out = np.interp(TAU_CENTERS, tau_native, density, left=0.0, right=0.0)
    s = out.sum() * TAU_BIN
    if s > 0:
        out /= s
    return out


def _resample(src_grid, src_density, dst_grid):
    out = np.interp(dst_grid, src_grid, src_density, left=0.0, right=0.0)
    dx = float(dst_grid[1] - dst_grid[0])
    total = float(out.sum() * dx)
    if total > 0:
        out /= total
    return out


def _cate_from_2d_diag(density_2d, xs, ys, tau_grid, off_by_one=False):
    """Diagonal integration. Returns p_tau on tau_grid."""
    dxs = float(xs[1] - xs[0])
    dys = float(ys[1] - ys[0])
    p_tau = np.zeros(len(tau_grid))
    for k, t in enumerate(tau_grid):
        y1_target = xs + t
        valid = (y1_target >= ys[0]) & (y1_target <= ys[-1])
        if not np.any(valid):
            continue
        ss = np.searchsorted(xs, xs[valid])
        col = np.clip(ss - (1 if off_by_one else 0), 0, len(xs) - 1)
        row_f = (y1_target[valid] - ys[0]) / dys
        row_lo = np.clip(np.floor(row_f).astype(int), 0, len(ys) - 2)
        row_hi = row_lo + 1
        w_hi = row_f - row_lo
        w_lo = 1.0 - w_hi
        f_diag = w_lo * density_2d[row_lo, col] + w_hi * density_2d[row_hi, col]
        p_tau[k] = f_diag.sum() * dxs
    s = p_tau.sum() * (tau_grid[1] - tau_grid[0])
    if s > 0:
        p_tau /= s
    return p_tau


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo',           required=True)
    ap.add_argument('--causalpfn',      required=True)
    ap.add_argument('--dopfn',          required=True)
    ap.add_argument('--checkpoint50',   required=True)
    ap.add_argument('--checkpoint10',   required=True)
    ap.add_argument('--realization',    type=int, default=0)
    ap.add_argument('--queries',        default='4,2,18,66,64,42',
                    help='comma-separated test indices to inspect')
    ap.add_argument('--malc-B',         type=int, default=60)
    ap.add_argument('--n-eval',         type=int, default=200)
    args = ap.parse_args()

    query_ids = [int(q) for q in args.queries.split(',')]
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

    # True densities via CausalPFN NPZ (same recipe as true_ihdp.py)
    ihdp_dir = os.path.join(args.causalpfn, 'benchmarks', 'IHDP')
    if not os.path.isdir(ihdp_dir):
        ihdp_dir = os.path.join(args.causalpfn, 'IHDP')
    tr = np.load(os.path.join(ihdp_dir, 'ihdp_npci_1-100.train.npz'))
    te = np.load(os.path.join(ihdp_dir, 'ihdp_npci_1-100.test.npz'))
    mu0 = te['mu0'][..., args.realization].astype(np.float32).reshape(-1)
    mu1 = te['mu1'][..., args.realization].astype(np.float32).reshape(-1)
    yf_tr = tr['yf'][..., args.realization].astype(np.float32).reshape(-1)
    t_tr  = tr['t' ][..., args.realization].astype(np.float32).reshape(-1)
    mu0_tr = tr['mu0'][..., args.realization].astype(np.float32).reshape(-1)
    mu1_tr = tr['mu1'][..., args.realization].astype(np.float32).reshape(-1)
    sig_orig = float(np.std(yf_tr - np.where(t_tr > 0.5, mu1_tr, mu0_tr), ddof=1))
    y_min = float(y_train_full.min()); y_max = float(y_train_full.max())
    y_rng = max(y_max - y_min, 1e-6)
    mu0_s = (mu0 - y_min) / y_rng * 2.0 - 1.0
    mu1_s = (mu1 - y_min) / y_rng * 2.0 - 1.0
    sig_s = sig_orig * (2.0 / y_rng)

    for label, ckpt_path in [('fn=50', args.checkpoint50), ('fn=10', args.checkpoint10)]:
        print(f'\n{"=" * 76}\n[{label}]  {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        cfg = ckpt['config']; J = cfg['J']; num_features = cfg['num_features']
        edges_np = ckpt['edges'].cpu().numpy()
        bin_width = float(edges_np[1] - edges_np[0])
        print(f'  cfg: J={J}  num_features={num_features}  d_model={cfg["d_model"]}  '
              f'depth={cfg["depth"]}  heads={cfg["heads"]}  hidden_mult={cfg["hidden_mult"]}')
        print(f'  edges: range=[{edges_np[0]:.4f}, {edges_np[-1]:.4f}]  '
              f'n_edges={len(edges_np)}  bin_width={bin_width:.6f}')

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

        for q in query_ids:
            p_mat, *_ = unpack_pred(pred[q], J, bin_width)
            p_mat_np = p_mat.detach().cpu().numpy()
            p_max = float(p_mat_np.max())
            p_arg = np.unravel_index(int(np.argmax(p_mat_np)), p_mat_np.shape)
            entropy = -float((p_mat_np * np.log(p_mat_np + 1e-12)).sum())
            top1_mass = p_max
            # top-3 mass concentration
            flat = np.sort(p_mat_np.reshape(-1))[::-1]
            top3_mass = float(flat[:3].sum())
            print(f'\n  query {q}   true tau ~ N({mu1_s[q] - mu0_s[q]:+.3f}, '
                  f'{np.sqrt(2) * sig_s:.3f}²)')
            print(f'    raw p_mat: max={p_max:.4f} at idx={p_arg}  '
                  f'entropy={entropy:.3f}  top3_mass={top3_mass:.3f}')

            # Fit MALC on transposed p_mat (per sibling convention)
            seed = int(hashlib.md5(f'q{q}_{label}'.encode()).hexdigest()[:8], 16) % 10**8
            for max_K in (1, 3):
                fit = fit_malc_inner(p_mat_np.T, edges_np, edges_np,
                                     B_fit=args.malc_B, B_select=args.malc_B,
                                     max_K=max_K, seed=seed, parallel=False)
                density_2d = dmalc_2d(fit, eval_pts).reshape(args.n_eval, args.n_eval)
                # Marginals from smoothed joint
                m_y0_fine = density_2d.sum(axis=0) * dys
                m_y1_fine = density_2d.sum(axis=1) * dxs
                p_y0_100 = _resample(xs, m_y0_fine, Y_CENTERS)
                p_y1_100 = _resample(ys, m_y1_fine, Y_CENTERS)

                # CATE via joint-diagonal (both correct and buggy variants)
                p_tau_diag = _cate_from_2d_diag(density_2d, xs, ys, TAU_CENTERS, off_by_one=False)
                p_tau_diag_bug = _cate_from_2d_diag(density_2d, xs, ys, TAU_CENTERS, off_by_one=True)
                # CATE via independence conv of the marginals we just derived
                p_tau_marg = _naive_p_tau(p_y0_100, p_y1_100)

                # Truth
                p_y0_true = _gauss(Y_CENTERS, mu0_s[q], sig_s)
                p_y1_true = _gauss(Y_CENTERS, mu1_s[q], sig_s)
                p_tau_true = _gauss(TAU_CENTERS, mu1_s[q] - mu0_s[q], np.sqrt(2) * sig_s)

                l2_y0 = _l2(p_y0_100, p_y0_true, Y_CENTERS)
                l2_y1 = _l2(p_y1_100, p_y1_true, Y_CENTERS)
                l2_diag     = _l2(p_tau_diag,     p_tau_true, TAU_CENTERS)
                l2_diag_bug = _l2(p_tau_diag_bug, p_tau_true, TAU_CENTERS)
                l2_marg     = _l2(p_tau_marg,     p_tau_true, TAU_CENTERS)
                print(f'    MALC K={max_K}:  L2(y0)={l2_y0:.3f}  L2(y1)={l2_y1:.3f}  '
                      f'L2(tau|diag_fixed)={l2_diag:.3f}  '
                      f'L2(tau|diag_off1)={l2_diag_bug:.3f}  '
                      f'L2(tau|marg_conv)={l2_marg:.3f}')

    return 0


def _install_dopfn_shim(dopfn_dir):
    if 'datasets' in sys.modules:
        return
    sys.path.insert(0, dopfn_dir)
    ds_mod = types.ModuleType('datasets')
    with open(os.path.join(dopfn_dir, 'datasets/__init__.py')) as fp:
        src = fp.read().split('def load_semi_real')[0]
    exec(src, ds_mod.__dict__)
    sys.modules['datasets'] = ds_mod


if __name__ == '__main__':
    sys.exit(main())
