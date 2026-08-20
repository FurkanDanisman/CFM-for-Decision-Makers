"""Compare truth-density statistics between IHDP and the d=6 polynomial SCM.

For each dataset, samples truth (μ0, μ1, σ) per query, converts to scaled Y,
and reports the distributions of:
  - raw μ0, μ1, σ
  - scaled μ0, μ1 (should live in [-1, 1] for the model to see them)
  - scaled σ (relative to J=10 bin_w = 0.2)
  - fraction of Gaussian mass inside [-1, 1] after scaling
  - implied y_rng, y_min

Run on cluster:
    python benchmarks/empirical_tests/compare_truth_stats.py \\
      --repo $DEPLOY_ROOT/R-PFN --dopfn $DEPLOY_ROOT/external/dopfn \\
      --causalpfn $DEPLOY_ROOT/external/causalpfn \\
      --n-ihdp 10 --n-d6-seeds 10
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--dopfn', required=True)
    ap.add_argument('--causalpfn', required=True)
    ap.add_argument('--n-ihdp', type=int, default=10, help='IHDP realizations to sample')
    ap.add_argument('--n-d6-seeds', type=int, default=10, help='d=6 seeds to sample')
    ap.add_argument('--d6-N', type=int, default=200)
    ap.add_argument('--d6-n-test', type=int, default=25)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'l2_ihdp'))
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'empirical_tests'))
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks'))
    sys.path.insert(0, args.repo)
    sys.path.insert(0, args.causalpfn)

    from scipy.stats import norm

    def summarize(name, values, unit=''):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            print(f'  {name:22s}  (empty)'); return
        print(f'  {name:22s}  n={arr.size:5d}  '
              f'mean={arr.mean():+.4f}  std={arr.std():.4f}  '
              f'min={arr.min():+.4f}  q25={np.quantile(arr,0.25):+.4f}  '
              f'med={np.median(arr):+.4f}  q75={np.quantile(arr,0.75):+.4f}  '
              f'max={arr.max():+.4f}{"  "+unit if unit else ""}')

    # ---- IHDP ----
    from true_ihdp import load_ihdp_truth
    from benchmarks import IHDPDataset

    print('=' * 90)
    print(f'IHDP (first {args.n_ihdp} realizations)')
    print('=' * 90)
    ihdp = dict(mu0_raw=[], mu1_raw=[], sigma_raw=[],
                 mu0_sc=[], mu1_sc=[], sigma_sc=[],
                 mass_in_pm1=[], y_min=[], y_rng=[])
    for r in range(args.n_ihdp):
        cd, _ = IHDPDataset()[r]
        y_train = np.asarray(cd.y_train.detach().cpu()
                              if hasattr(cd.y_train, 'detach') else cd.y_train).reshape(-1)
        truth = load_ihdp_truth(r, args.causalpfn, y_train)
        mu0_sc = np.asarray(truth.mu0_test_scaled).reshape(-1)
        mu1_sc = np.asarray(truth.mu1_test_scaled).reshape(-1)
        sigma_sc = float(truth.sigma_scaled)
        y_min = float(truth.y_min); y_rng = float(truth.y_rng)
        # Convert back to raw
        mu0_raw = (mu0_sc + 1) * y_rng / 2 + y_min
        mu1_raw = (mu1_sc + 1) * y_rng / 2 + y_min
        sigma_raw = sigma_sc * y_rng / 2
        ihdp['mu0_raw'].extend(mu0_raw); ihdp['mu1_raw'].extend(mu1_raw)
        ihdp['sigma_raw'].append(sigma_raw)
        ihdp['mu0_sc'].extend(mu0_sc); ihdp['mu1_sc'].extend(mu1_sc)
        ihdp['sigma_sc'].append(sigma_sc)
        # Mass in [-1, 1] under N(mu_sc, sigma_sc)
        for m in list(mu0_sc) + list(mu1_sc):
            ihdp['mass_in_pm1'].append(norm.cdf(1, m, sigma_sc) - norm.cdf(-1, m, sigma_sc))
        ihdp['y_min'].append(y_min); ihdp['y_rng'].append(y_rng)

    summarize('raw μ0', ihdp['mu0_raw'])
    summarize('raw μ1', ihdp['mu1_raw'])
    summarize('raw σ', ihdp['sigma_raw'])
    summarize('scaled μ0', ihdp['mu0_sc'])
    summarize('scaled μ1', ihdp['mu1_sc'])
    summarize('scaled σ', ihdp['sigma_sc'], unit='(bin_w=0.2)')
    summarize('mass in [-1,1] scaled', ihdp['mass_in_pm1'])
    summarize('y_min (per realization)', ihdp['y_min'])
    summarize('y_rng (per realization)', ihdp['y_rng'])

    # ---- d=6 ----
    from fig2_pehe_l2 import make_polynomial_scm

    print('\n' + '=' * 90)
    print(f'd=6 polynomial SCM (first {args.n_d6_seeds} seeds, N={args.d6_N}, n_test={args.d6_n_test})')
    print('=' * 90)
    d6 = dict(mu0_raw=[], mu1_raw=[], sigma_raw=[],
              mu0_sc=[], mu1_sc=[], sigma_sc=[],
              mass_in_pm1=[], y_min=[], y_rng=[])
    for seed in range(args.n_d6_seeds):
        cd = make_polynomial_scm(seed=seed, n_context=args.d6_N, n_test=args.d6_n_test,
                                  rho_eff=0.0, x_dim=6, degree=3, sigma_eps=1.0)
        y_ctx = cd.y_train.numpy() if hasattr(cd.y_train, 'numpy') else np.asarray(cd.y_train)
        y_min = float(y_ctx.min()); y_rng = max(float(y_ctx.max()) - y_min, 1e-6)
        mu0_raw = np.asarray(cd._mu0_test).reshape(-1)
        mu1_raw = np.asarray(cd._mu1_test).reshape(-1)
        sigma_raw = float(cd._sigma_eps)
        mu0_sc = (mu0_raw - y_min) / y_rng * 2 - 1
        mu1_sc = (mu1_raw - y_min) / y_rng * 2 - 1
        sigma_sc = sigma_raw * 2 / y_rng
        d6['mu0_raw'].extend(mu0_raw); d6['mu1_raw'].extend(mu1_raw)
        d6['sigma_raw'].append(sigma_raw)
        d6['mu0_sc'].extend(mu0_sc); d6['mu1_sc'].extend(mu1_sc)
        d6['sigma_sc'].append(sigma_sc)
        for m in list(mu0_sc) + list(mu1_sc):
            d6['mass_in_pm1'].append(norm.cdf(1, m, sigma_sc) - norm.cdf(-1, m, sigma_sc))
        d6['y_min'].append(y_min); d6['y_rng'].append(y_rng)

    summarize('raw μ0', d6['mu0_raw'])
    summarize('raw μ1', d6['mu1_raw'])
    summarize('raw σ', d6['sigma_raw'])
    summarize('scaled μ0', d6['mu0_sc'])
    summarize('scaled μ1', d6['mu1_sc'])
    summarize('scaled σ', d6['sigma_sc'], unit='(bin_w=0.2)')
    summarize('mass in [-1,1] scaled', d6['mass_in_pm1'])
    summarize('y_min (per seed)', d6['y_min'])
    summarize('y_rng (per seed)', d6['y_rng'])

    # ---- Comparison lines ----
    print('\n' + '=' * 90)
    print('Comparison — how similar are the truth distributions?')
    print('=' * 90)
    def _cmp(name, a, b, unit=''):
        a = np.asarray(a); b = np.asarray(b)
        print(f'  {name:22s}  IHDP mean={np.mean(a):+.4f} std={np.std(a):.4f}  '
              f'|  d=6 mean={np.mean(b):+.4f} std={np.std(b):.4f}{"  "+unit if unit else ""}')
    _cmp('scaled μ0',    ihdp['mu0_sc'], d6['mu0_sc'])
    _cmp('scaled μ1',    ihdp['mu1_sc'], d6['mu1_sc'])
    _cmp('scaled σ',     ihdp['sigma_sc'], d6['sigma_sc'], '(units of bin_w=0.2)')
    _cmp('mass in [-1,1]', ihdp['mass_in_pm1'], d6['mass_in_pm1'])
    _cmp('σ/bin_w',      np.asarray(ihdp['sigma_sc'])/0.2, np.asarray(d6['sigma_sc'])/0.2,
          '(bells per bin — smaller = more concentrated)')
    _cmp('raw σ',        ihdp['sigma_raw'], d6['sigma_raw'])
    _cmp('raw y_rng',    ihdp['y_rng'], d6['y_rng'])

    print('\nInterpretation:')
    print('  - Similar scaled μ ranges + similar σ/bin_w means the L2 comparison is')
    print('    "apples-to-apples" — truth shapes look alike per query.')
    print('  - If d=6 has WAY tighter σ_scaled (e.g. 10x smaller), each query is a')
    print('    razor-thin spike inside one J=10 bin, and L2 numbers will be much')
    print('    larger for any method that fails to concentrate mass tightly.')
    print('  - "mass in [-1,1]" < 1 means the Gaussian truth spills outside the')
    print('    evaluation support (partial truncation) — non-trivial re-normalisation.')


if __name__ == '__main__':
    main()
