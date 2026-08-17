"""Dump raw p_mat[i,j] for every test query at one SCM. Prints per-query
Marg-KLfwd for Y0 and Y1 so we can see if Ours' marginal collapse is
sporadic or systematic. Also draws an ASCII heatmap of p_mat for query 0.

Usage:
    python inspect_ours_pmat.py --repo $DEPLOY_ROOT/R-PFN \
        --checkpoint $DEPLOY_ROOT/R-PFN/checkpoints/step_50000_final.pt \
        --N-context 200 --N-test 10 --rho 0.8 --seed 40000
"""
import argparse, os, sys, numpy as np, torch

here  = os.path.dirname(os.path.abspath(__file__))
bench = os.path.dirname(here)
if bench not in sys.path: sys.path.insert(0, bench)
if here  not in sys.path: sys.path.insert(0, here)
from methods import dopfn as _  # noqa: F401 — sklearn shim
from fig2_pehe_l2 import (
    make_polynomial_scm, ours_forward, load_ours_ipfn,
    truth_marginals, Y_GRID, Y_DX, _to_common_grid, _discrete_to_density,
    kl_1d,
)


def _sparkbar(vec, width=None):
    bars = ' ▁▂▃▄▅▆▇█'
    v = np.asarray(vec, dtype=np.float64)
    peak = v.max() if v.max() > 0 else 1.0
    if width is not None and len(v) > width:
        step = len(v) // width
        v = np.array([v[i:i+step].max() for i in range(0, len(v), step)])[:width]
    idxs = np.clip((v / peak * (len(bars) - 1)).astype(int), 0, len(bars) - 1)
    return ''.join(bars[i] for i in idxs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--N-context', type=int, default=200)
    ap.add_argument('--N-test',    type=int, default=10)
    ap.add_argument('--rho',       type=float, default=0.8)
    ap.add_argument('--seed',      type=int, default=40000)
    ap.add_argument('--causalpfn', default='')
    args = ap.parse_args()

    if args.causalpfn:
        sys.path.insert(0, args.causalpfn)

    class _A: pass
    a = _A(); a.repo = args.repo
    model, edges, J, bw, NF, _ = load_ours_ipfn(a, args.checkpoint)
    print(f'[load] J={J}  NF={NF}  bw={bw:.4f}  edges=[{edges[0]:+.2f}, {edges[-1]:+.2f}]')

    cd = make_polynomial_scm(args.seed, args.N_context, args.N_test, args.rho)
    p_mat_all, centers = ours_forward(model, edges, J, bw, NF, cd)
    print(f'[fwd]  p_mat shape={p_mat_all.shape}  centers=[{centers[0]:+.2f}, {centers[-1]:+.2f}]')

    p0_true, p1_true = truth_marginals(cd)  # (n_test, Y_BINS) on Y_GRID
    n_test = cd.X_test.shape[0]

    # Per-query Marg-KLfwd on both arms
    print(f'\nq   μ0     μ1     Y_do(0) KLfwd   Y_do(1) KLfwd   p0_peak/truth   p1_peak/truth')
    print('-' * 100)
    for q in range(n_test):
        p0_raw = p_mat_all[q].sum(axis=1)   # marginal over Y0
        p1_raw = p_mat_all[q].sum(axis=0)   # marginal over Y1
        d0 = _discrete_to_density(p0_raw, centers)
        d1 = _discrete_to_density(p1_raw, centers)
        p0_common = _to_common_grid(d0, centers, Y_GRID)
        p1_common = _to_common_grid(d1, centers, Y_GRID)
        kl0 = kl_1d(p0_true[q], p0_common, Y_DX)
        kl1 = kl_1d(p1_true[q], p1_common, Y_DX)
        peak0_ratio = p0_common.max() / max(p0_true[q].max(), 1e-12)
        peak1_ratio = p1_common.max() / max(p1_true[q].max(), 1e-12)
        print(f'{q:2d}  {cd._mu0_test[q]:+5.2f}  {cd._mu1_test[q]:+5.2f}  '
              f'{kl0:14.3f}  {kl1:14.3f}       {peak0_ratio:6.2f}          {peak1_ratio:6.2f}')

    # ── p_mat heatmap for q=0 ──
    print(f'\n── p_mat[i,j] for q=0  (rows=Y0 bins, cols=Y1 bins,  J={J}) ──')
    pm = p_mat_all[0]
    print(f'   sum={pm.sum():.4f}  peak={pm.max():.4f}  peak@(i,j)=({np.unravel_index(pm.argmax(), pm.shape)})')
    # Show every-few-rows-and-cols compact heatmap
    step = max(J // 30, 1)
    idxs = list(range(0, J, step))
    print(f'   col centers (Y1): {" ".join(f"{centers[j]:+.1f}" for j in idxs)}')
    print(f'   row centers (Y0)')
    for i in idxs:
        print(f'   {centers[i]:+5.2f}   {_sparkbar(pm[i, idxs])}   sum={pm[i,:].sum():.3f}')

    # Row/col-sum sparklines
    print(f'\n   p0[i] = Σ_j p_mat[i,j] (Y_do(0) marginal, raw):')
    print(f'   {_sparkbar(pm.sum(axis=1))}')
    print(f'   p1[j] = Σ_i p_mat[i,j] (Y_do(1) marginal, raw):')
    print(f'   {_sparkbar(pm.sum(axis=0))}')


if __name__ == '__main__':
    main()
