"""Diagnose why malc_1d_cvxpy collapses a log-concave Gaussian to a spike.

Construct a synthetic Gaussian p_marg on J=100 bins (the same J the fn=50
checkpoint uses), which IS log-concave, so the true log-concave MLE must
equal the empirical (p_fit == p_marg). Run the CVXPY problem with SCS,
ECOS, and CLARABEL separately, print:

  - final relative L1  |p_fit - p_marg| / p_marg summed
  - peak height ratio  max(p_fit) / max(p_marg)   (spike ⇒ >> 1)
  - shape of returned log_p values before and after subtract-max
  - solver status + iterations

If SCS returns a wrong solution while ECOS/CLARABEL return the empirical,
that pinpoints the collapse to SCS's first-order convergence on this LP.

Reference behavior — R's `logcondiscr::logConDiscrMLE` (used by the R
implementation in /Users/furkandanisman/DensOLog_VS/R/malc.R) uses an
active-set method that's tight on such problems, so R "never" produces
this spike.

Usage:
    python inspect_malc_1d_solver.py            # default J=100, σ=1 bin
    python inspect_malc_1d_solver.py --J 50 --sigma 3
"""
import argparse, sys, os
import numpy as np


def make_p_marg(J, sigma_bins, seed=0):
    """Broad Gaussian pmf on J bins; μ near center, σ in bin units."""
    rng = np.random.default_rng(seed)
    xs = np.arange(J) - (J - 1) / 2.0
    p = np.exp(-0.5 * (xs / sigma_bins) ** 2)
    p /= p.sum()
    return p


def try_solver(prob, log_p, p_param, p_marg, solver):
    import cvxpy as cp
    p_param.value = p_marg
    try:
        prob.solve(solver=solver, verbose=False)
    except Exception as e:
        return {'solver': solver, 'ok': False, 'err': f'{type(e).__name__}: {e}'}
    if log_p.value is None:
        return {'solver': solver, 'ok': False, 'err': 'log_p.value is None',
                'status': prob.status}
    lp = np.asarray(log_p.value, dtype=np.float64)
    # Pre-subtract-max view
    truth_log = np.log(p_marg + 1e-300)
    truth_log = truth_log - truth_log.max()
    lp_norm = lp - lp.max()
    # p_fit via softmax
    p_fit = np.exp(lp_norm)
    p_fit /= max(p_fit.sum(), 1e-300)
    # metrics
    l1 = float(np.abs(p_fit - p_marg).sum())
    peak_ratio = p_fit.max() / max(p_marg.max(), 1e-300)
    lp_max_gap = float(lp.max() - np.median(lp))
    truth_max_gap = float(truth_log.max() - np.median(truth_log))
    return {
        'solver': solver, 'ok': True, 'status': prob.status,
        'l1': l1, 'peak_ratio': peak_ratio,
        'lp_top_gap':    lp_max_gap,      # solver's log_p max - median
        'truth_top_gap': truth_max_gap,   # log(p_marg) max - median (correct answer)
        'lp': lp_norm, 'p_fit': p_fit,
    }


def _sparkbar(v, width=None):
    bars = ' ▁▂▃▄▅▆▇█'
    v = np.asarray(v, dtype=np.float64)
    peak = v.max() if v.max() > 0 else 1.0
    if width and len(v) > width:
        step = len(v) // width
        v = np.array([v[i:i+step].max() for i in range(0, len(v), step)])[:width]
    idxs = np.clip((v / peak * (len(bars) - 1)).astype(int), 0, len(bars) - 1)
    return ''.join(bars[i] for i in idxs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--J',     type=int,   default=100)
    ap.add_argument('--sigma', type=float, default=10.0,
                    help='Std dev of Gaussian in bin units (10 = broad).')
    ap.add_argument('--solvers', default='SCS,ECOS,CLARABEL',
                    help='Comma-separated CVXPY solvers to try.')
    args = ap.parse_args()

    import cvxpy as cp
    print(f'CVXPY version: {cp.__version__}')
    print(f'Available solvers: {cp.installed_solvers()}\n')

    p_marg = make_p_marg(args.J, args.sigma)
    print(f'Input p_marg:  J={args.J}  σ={args.sigma} bins  '
          f'peak={p_marg.max():.4f}  peak_bin={p_marg.argmax()}')
    print(f'    {_sparkbar(p_marg, width=80)}')
    print(f'    (true MLE = p_marg exactly, since p_marg is already log-concave)\n')

    # Build the CVXPY problem once
    p_param = cp.Parameter(args.J, nonneg=True)
    log_p = cp.Variable(args.J)
    prob = cp.Problem(
        cp.Maximize(p_param @ log_p),
        [cp.log_sum_exp(log_p) <= 0, cp.diff(log_p, 2) <= 0],
    )

    for slv in [s.strip() for s in args.solvers.split(',') if s.strip()]:
        print(f'── Solver: {slv} ─────────────────────────────────────────')
        r = try_solver(prob, log_p, p_param, p_marg, slv)
        if not r['ok']:
            print(f'    FAILED: {r.get("err","?")}  status={r.get("status","?")}\n')
            continue
        print(f'    status={r["status"]}   L1(p_fit − p_marg)={r["l1"]:.4f}   '
              f'peak_ratio={r["peak_ratio"]:.2f}')
        print(f'    solver log_p top-gap = {r["lp_top_gap"]:+.2f}   '
              f'(truth = {r["truth_top_gap"]:+.2f})')
        print(f'    p_fit (norm):  {_sparkbar(r["p_fit"], width=80)}')
        print(f'    residual*10:   {_sparkbar(np.abs(r["p_fit"]-p_marg)*10, width=80)}')
        # If peak_ratio > ~2, we have the spike bug.
        if r['peak_ratio'] > 2.0:
            print(f'    *** SPIKE COLLAPSE — p_fit peak is {r["peak_ratio"]:.1f}× '
                  f'the truth. This solver is broken for this problem. ***')
        elif r['l1'] < 0.05:
            print(f'    OK — solver reproduces the empirical MLE (as it should '
                  f'for a log-concave input).')
        else:
            print(f'    WARN — noticeable deviation but no spike.')
        print()


if __name__ == '__main__':
    main()
