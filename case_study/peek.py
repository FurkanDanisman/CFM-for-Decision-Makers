"""Show the head of a case-study dataset: covariates, treatment, outcomes, truth.

    python case_study/peek.py <npz>
    python case_study/peek.py --root case_study/d_variation/shift+2/d3 \
        --case Observed_Confounder --n 1000 --r 0 --rows 8
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def show(path, rows=10):
    z = np.load(path, allow_pickle=True)
    names = [str(x) for x in z['feature_names']]
    X, T, Y = z['X'], z['T'], z['Y']
    cate, mu0, mu1 = z['cate'], z['mu_0'], z['mu_1']
    edges = [tuple(map(str, e)) for e in z['graph_edges']]
    hidden = sorted({n for e in edges for n in e} - set(names) - {'T', 'Y'})

    print(f"{path}")
    print(f"  case={str(z['case_study'])}  N={int(z['n_context'])}  d={X.shape[1]}"
          f"  seed={int(z['seed'])}")
    print(f"  exo_std={float(z['exo_std']):.4f}  noise_std={float(z['noise_std']):.4f}"
          f"  cate_shift={float(z['cate_shift']):+.2f}")
    print(f"  X columns : {names}")
    print(f"  hidden    : {hidden if hidden else '(none)'}")
    print(f"  edges     : {edges}")

    k = min(rows, X.shape[0])
    w = 9
    print(f"\n  {'row':>4s} " + "".join(f"{n:>{w}s}" for n in names)
          + f"{'T':>{w}s}{'Y':>{w}s}{'mu_0':>{w}s}{'mu_1':>{w}s}{'CATE':>{w}s}")
    print("  " + "-" * (5 + w * (len(names) + 5)))
    for i in range(k):
        print(f"  {i:>4d} " + "".join(f"{X[i, j]:>{w}.4f}" for j in range(X.shape[1]))
              + f"{T[i]:>{w}.0f}{Y[i]:>{w}.4f}{mu0[i]:>{w}.4f}{mu1[i]:>{w}.4f}"
              f"{cate[i]:>{w}.4f}")

    print(f"\n  over all {X.shape[0]} rows:")
    print(f"    T mean (treated fraction) = {T.mean():.4f}")
    print(f"    Y     mean={Y.mean():+.4f}  sd={Y.std():.4f}")
    print(f"    CATE  mean={cate.mean():+.4f}  sd={cate.std():.4f}"
          f"  min={cate.min():+.4f}  max={cate.max():+.4f}")
    print(f"    ATE (= mean CATE)         = {cate.mean():+.4f}")
    # the identity the density eval depends on
    gap = float(np.max(np.abs((mu1 - mu0) - cate)))
    print(f"    max |mu_1 - mu_0 - CATE|  = {gap:.2e}  "
          + ("(tau is deterministic)" if gap < 1e-5 else "*** MISMATCH ***"))
    for j, n in enumerate(names):
        print(f"    {n:<6s} mean={X[:, j].mean():+.4f}  sd={X[:, j].std():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz', nargs='?')
    ap.add_argument('--root', help='e.g. case_study/d_variation/shift+2/d3')
    ap.add_argument('--case', default='Observed_Confounder')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--r', type=int, default=0)
    ap.add_argument('--rows', type=int, default=10)
    a = ap.parse_args()
    if a.npz:
        path = a.npz
    else:
        if not a.root:
            ap.error('give an npz path or --root')
        path = os.path.join(a.root, a.case, f'N{a.n}', f'{a.case}_{a.r}.npz')
        if not os.path.isfile(path):
            hits = glob.glob(os.path.join(a.root, a.case, f'N{a.n}', '*.npz'))
            ap.error(f'{path} not found ({len(hits)} realizations in that cell)')
    show(path, a.rows)


if __name__ == '__main__':
    main()
