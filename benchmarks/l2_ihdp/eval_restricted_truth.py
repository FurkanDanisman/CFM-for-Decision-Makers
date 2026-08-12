"""Recompute IHDP L2 for existing shards against restricted-features truth.

For each realisation r, builds truth conditioned on the first `d_restrict`
covariates via k-NN mixture-of-Gaussians (see true_ihdp_restricted.py),
then reads each method's predicted densities from its shard and computes
L2 against this new truth.

Do-PFN's predictions in the existing shards were computed with all 25 IHDP
features. Its L2 against a 10-feature truth is therefore not a fair test of
"Do-PFN restricted to 10 features"; it is a test of "Do-PFN's 25-feature
prediction compared against the 10-feature marginal". Interpret accordingly.
To do the fully-fair comparison, Do-PFN would also need to be re-run with
first-10-features input; that requires a fresh sweep (not covered here).

Usage
-----
    python R-PFN/benchmarks/l2_ihdp/eval_restricted_truth.py \\
        --shards-glob "/scratch/furkanbd/rpfn_bench_kit/l2_ihdp/out.r*.npz" \\
        --override-glob "ours_fn10=/scratch/furkanbd/rpfn_bench_kit/l2_ihdp/out_maxK1.r*.npz" \\
        --causalpfn /scratch/furkanbd/rpfn_bench_kit/external/causalpfn \\
        --dopfn /scratch/furkanbd/rpfn_bench_kit/external/dopfn \\
        --d-restrict 10 --K 20
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import types

import numpy as np


METHODS = ['ours_fn50', 'ours_fn10', 'dopfn', 'uwyk_noanc']
LABELS = {
    'ours_fn50':  'Ours(fn=50)',
    'ours_fn10':  'Ours(fn=10)',
    'dopfn':      'Do-PFN',
    'uwyk_noanc': 'UWYK-NoAnc',
}


def _install_dopfn_shim(dopfn_dir):
    if 'datasets' in sys.modules:
        return
    sys.path.insert(0, dopfn_dir)
    ds_mod = types.ModuleType('datasets')
    with open(os.path.join(dopfn_dir, 'datasets/__init__.py')) as fp:
        src = fp.read().split('def load_semi_real')[0]
    exec(src, ds_mod.__dict__)
    sys.modules['datasets'] = ds_mod


def _r_from_path(path: str) -> int:
    m = re.search(r'r(\d{3})\.npz$', path)
    if m is None:
        raise ValueError(f'cannot extract realization from {path}')
    return int(m.group(1))


def _trap(f, x):
    fn = getattr(np, 'trapezoid', np.trapz)
    return float(fn(f, x))


def _l2(f, g, grid):
    return float(np.sqrt(_trap((f - g) ** 2, grid)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards-glob', required=True,
                    help='default source of per-realization shards')
    ap.add_argument('--override-glob', action='append', default=[],
                    metavar='METHOD=GLOB',
                    help='per-method shard glob override (e.g. ours_fn10=out_maxK1.r*.npz)')
    ap.add_argument('--causalpfn', required=True)
    ap.add_argument('--dopfn', required=True)
    ap.add_argument('--d-restrict', type=int, default=10)
    ap.add_argument('--K', type=int, default=20)
    ap.add_argument('--repo', default='')
    args = ap.parse_args()

    if not args.repo:
        args.repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))

    _install_dopfn_shim(args.dopfn)
    sys.path.insert(0, args.causalpfn)
    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.join(args.repo, 'MALC', 'Optimal_Transport'))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from benchmarks import IHDPDataset
    from ot_barycenter import wasserstein_barycenter_1d
    from true_ihdp_restricted import (
        TAU_CENTERS, TAU_BIN, Y_CENTERS,
        load_ihdp_restricted_truth,
        restricted_marginals_per_query,
        restricted_cate_per_query,
        restricted_ate_barycenter,
        induced_correlation,
    )

    # Per-method shard resolution
    overrides = {}
    for entry in args.override_glob:
        if '=' not in entry:
            print(f'[fatal] bad --override-glob {entry!r}'); return 2
        m, g = entry.split('=', 1)
        overrides[m.strip()] = g.strip()

    per_method_shards = {m: sorted(glob.glob(overrides.get(m, args.shards_glob)))
                         for m in METHODS}
    # Index shards by realisation for cross-method alignment
    per_method_shard_by_r = {
        m: {_r_from_path(p): p for p in per_method_shards[m]} for m in METHODS}

    # union of realisations that appear in any method's shards
    all_rs = sorted({r for m in METHODS for r in per_method_shard_by_r[m].keys()})
    if not all_rs:
        print(f'[fatal] no shards match {args.shards_glob}'); return 2
    print(f'[load] scanning {len(all_rs)} realisations '
          f'({len(per_method_shard_by_r["ours_fn50"])} default, per-method may differ)')

    l2 = {m: {k: [] for k in ('p_y0', 'p_y1', 'p_tau')} for m in METHODS}
    l2_ate = {m: [] for m in METHODS}
    induced_rho_all = []

    for r in all_rs:
        cd, _ = IHDPDataset()[r]
        X_train_full = _np(cd.X_train)
        X_test_full  = _np(cd.X_test)
        y_train_full = _np(cd.y_train)

        truth = load_ihdp_restricted_truth(
            r, args.causalpfn, X_train_full, X_test_full, y_train_full,
            d_restrict=args.d_restrict, K=args.K)
        p_y0_true, p_y1_true = restricted_marginals_per_query(truth)
        p_tau_true = restricted_cate_per_query(truth)
        p_ate_true = restricted_ate_barycenter(p_tau_true, wasserstein_barycenter_1d)
        induced_rho_all.append(induced_correlation(truth))

        for m in METHODS:
            if r not in per_method_shard_by_r[m]:
                continue
            path = per_method_shard_by_r[m][r]
            with np.load(path) as f:
                if f'{m}__p_y0' not in f.files:
                    continue
                p_y0 = np.asarray(f[f'{m}__p_y0'])
                p_y1 = np.asarray(f[f'{m}__p_y1'])
                p_tau = np.asarray(f[f'{m}__p_tau'])

            n_q = p_y0_true.shape[0]
            for q in range(n_q):
                l2[m]['p_y0'].append(_l2(p_y0[q], p_y0_true[q], Y_CENTERS))
                l2[m]['p_y1'].append(_l2(p_y1[q], p_y1_true[q], Y_CENTERS))
                l2[m]['p_tau'].append(_l2(p_tau[q], p_tau_true[q], TAU_CENTERS))
            p_ate = wasserstein_barycenter_1d(p_tau, TAU_CENTERS)
            s = p_ate.sum() * TAU_BIN
            if s > 0: p_ate /= s
            l2_ate[m].append(_l2(p_ate, p_ate_true, TAU_CENTERS))

    induced_rho_all = np.concatenate(induced_rho_all)

    print()
    print(f'IHDP with truth restricted to first {args.d_restrict} features '
          f'(K = {args.K} nearest neighbours in standardised space)')
    print()
    print(f'{"density":10s} {"method":22s} {"n":>6s} {"mean":>8s} {"median":>8s} {"std":>8s}')
    print('-' * 72)
    for dk in ('p_y0', 'p_y1', 'p_tau'):
        for m in METHODS:
            v = np.array(l2[m][dk])
            if v.size == 0:
                continue
            print(f'{dk:10s} {LABELS[m]:22s} {v.size:>6d} '
                  f'{v.mean():>8.4f} {np.median(v):>8.4f} {v.std(ddof=1):>8.4f}')
    for m in METHODS:
        v = np.array(l2_ate[m])
        if v.size == 0:
            continue
        print(f'{"p_ate":10s} {LABELS[m]:22s} {v.size:>6d} '
              f'{v.mean():>8.4f} {np.median(v):>8.4f} {v.std(ddof=1):>8.4f}')

    print()
    print(f'induced rho of restricted-truth joint (target for fn=10 to match): '
          f'mean={induced_rho_all.mean():+.3f}  '
          f'median={np.median(induced_rho_all):+.3f}  '
          f'std={induced_rho_all.std(ddof=1):.3f}')
    print('(compare to fn=10 predicted rho ≈ +0.94; fn=50 predicted rho ≈ +0.05)')
    return 0


def _np(a):
    import torch
    if isinstance(a, torch.Tensor):
        return a.numpy()
    return np.asarray(a)


if __name__ == '__main__':
    sys.exit(main())
