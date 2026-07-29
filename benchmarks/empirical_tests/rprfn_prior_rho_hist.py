"""Step 1: What ρ range does R-PFN's own SCM prior naturally produce?

Sample K SCMs from scm_prior.sample_as_cate_dataset, for each SCM compute
the empirical correlation ρ = Corr(Y_do0_raw, Y_do1_raw) from the paired
test outputs. Report the histogram + summary stats.

Decision rule (per user):
  - If ρ span is wide (say [0.0, 0.95] with reasonable density): use
    Option A (natural variation from R-PFN prior).
  - If ρ concentrates in a narrow high band ([0.9, 1.0]): need Option B
    (inject exogenous noise correlation as a knob).
"""
from __future__ import annotations
import argparse, os, sys, traceback
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K',           type=int, default=200)
    ap.add_argument('--n-context',   type=int, default=200)
    ap.add_argument('--n-test',      type=int, default=200,
                    help='per-SCM test units used to estimate ρ')
    ap.add_argument('--out',         default='/Users/furkandanisman/R-PFN/paper/theory/figures/rprfn_prior_rho_hist.png')
    args = ap.parse_args()

    # Path priority: R-PFN root FIRST (so `training.data.*` resolves to
    # R-PFN's paired sampler helpers, not UWYK's training module which
    # shares the `training` package name), THEN UWYK's src (so
    # `priors.causal_prior.*` resolves to UWYK's SCM primitives).
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))   # …/R-PFN
    sweep_dir = os.path.join(repo_root, 'benchmarks', 'context_sweep')
    uwyk_src = os.environ.get(
        'UWYK_SRC',
        '/Users/furkandanisman/.claude/jobs/7758df90/tmp/uwyk_upstream/src',
    )
    for p in (uwyk_src, sweep_dir, repo_root):
        if os.path.isdir(p) and p in sys.path: sys.path.remove(p)
    # Insert LAST first so it ends up FIRST after the sequence of inserts.
    for p in (uwyk_src, sweep_dir, repo_root):
        if os.path.isdir(p): sys.path.insert(0, p)
    from scm_prior import sample_as_cate_dataset

    rhos = []; skipped = 0
    for k in range(args.K):
        try:
            _cd, s = sample_as_cate_dataset(scm_seed=k, n_context=args.n_context,
                                              n_test=args.n_test)
            y0 = np.asarray(s['Y_do0_raw']).reshape(-1)
            y1 = np.asarray(s['Y_do1_raw']).reshape(-1)
            if len(y0) < 5 or y0.std() < 1e-9 or y1.std() < 1e-9:
                skipped += 1; continue
            rho = float(np.corrcoef(y0, y1)[0, 1])
            if np.isfinite(rho):
                rhos.append(rho)
            else:
                skipped += 1
        except Exception:
            skipped += 1
            traceback.print_exc()
        if (k + 1) % 20 == 0:
            print(f'[progress] {k+1}/{args.K}  n_valid={len(rhos)}  skipped={skipped}',
                  flush=True)

    rhos = np.array(rhos)
    print(f'\n[summary] {len(rhos)} valid SCMs, {skipped} skipped')
    print(f'  min      = {rhos.min():+.3f}')
    print(f'  10th pct = {np.percentile(rhos, 10):+.3f}')
    print(f'  50th pct = {np.percentile(rhos, 50):+.3f}')
    print(f'  90th pct = {np.percentile(rhos, 90):+.3f}')
    print(f'  max      = {rhos.max():+.3f}')
    print(f'  mean±std = {rhos.mean():+.3f} ± {rhos.std():.3f}')

    # Coverage check
    bins = np.linspace(-1, 1, 21)  # 20 bins of width 0.1
    hist, _ = np.histogram(rhos, bins=bins)
    print('\n[coverage] SCMs per 0.1-wide ρ bin:')
    for i, edge in enumerate(bins[:-1]):
        marker = '█' * int(np.round(20 * hist[i] / max(hist.max(), 1)))
        print(f'  [{edge:+.1f}, {edge+0.1:+.1f})  n={hist[i]:>4d}  {marker}')

    # Plot
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.hist(rhos, bins=bins, color='#2E4A6F', alpha=0.7, edgecolor='white')
    ax.set_xlabel(r'Empirical $\rho = \mathrm{Corr}(Y_{do0}, Y_{do1})$')
    ax.set_ylabel(f'Number of SCMs (of {len(rhos)})')
    ax.set_title(f"R-PFN prior — natural ρ distribution across {len(rhos)} SCMs\n"
                  f"(n_context={args.n_context}, n_test={args.n_test})")
    ax.axvline(rhos.mean(), color='red', ls='--', lw=1.5,
                label=f'mean = {rhos.mean():+.3f}')
    ax.axvline(0, color='k', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlim(-1, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'\n[save] {args.out}')

    # Decision
    span_p10_p90 = float(np.percentile(rhos, 90) - np.percentile(rhos, 10))
    print(f'\n[decision] 10th-to-90th percentile span = {span_p10_p90:.3f}')
    if rhos.min() < 0.2 and rhos.max() > 0.7 and span_p10_p90 > 0.3:
        print('  → Option A viable (natural variation covers a useful ρ range)')
    else:
        print('  → Option B needed (natural range too narrow; inject a ρ knob)')


if __name__ == '__main__':
    main()
