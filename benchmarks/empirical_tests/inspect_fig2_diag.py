"""Text-mode diagnostic for a Fig 2 diag shard (out.rhoN.diag.npz).

Prints, for the first query at ρ=rho:
  - the truth marginal's [5%, 95%] support in raw y
  - each method's marginal support (where density > 0.5% of peak)
  - the fraction of each method's mass outside the truth's [5%, 95%] window
    (missing tail mass — the KL_fwd blow-up signature)
  - peak height ratio  method/truth  (over-smoothing signature)

Usage:
    python inspect_fig2_diag.py $DEPLOY_ROOT/fig2_pehe_l2_smoke/out.rho4.diag.npz
"""
import os, sys, numpy as np

METHODS = ('uwyk_noanc', 'uwyk_anc', 'dopfn', 'ours_fn50')
LABEL   = {'uwyk_noanc':'UWYK-NoAnc','uwyk_anc':'UWYK-FullAnc',
           'dopfn':'Do-PFN','ours_fn50':'Ours(fn=50)'}


def _quantile_support(p, grid, q_lo=0.05, q_hi=0.95):
    dx = grid[1] - grid[0]
    cdf = np.cumsum(p) * dx
    cdf /= max(cdf[-1], 1e-12)
    lo = float(np.interp(q_lo, cdf, grid))
    hi = float(np.interp(q_hi, cdf, grid))
    return lo, hi


def _peak_support(p, grid, frac_of_peak=0.005):
    """Range of grid where p >= frac_of_peak * max(p)."""
    peak = p.max()
    if peak <= 0:
        return None, None
    mask = p >= frac_of_peak * peak
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None, None
    return float(grid[idx[0]]), float(grid[idx[-1]])


def _mass_outside(p, grid, lo, hi):
    dx = grid[1] - grid[0]
    m = ((grid < lo) | (grid > hi)).astype(float)
    return float((p * m).sum() * dx)


def _sparkline(p, width=60):
    """ASCII bar chart across the grid (min=0, max=peak)."""
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    step = max(n // width, 1)
    tr = np.array([p[i:i+step].max() for i in range(0, n, step)])[:width]
    peak = tr.max() if tr.max() > 0 else 1.0
    bars = ' ▁▂▃▄▅▆▇█'
    idxs = np.clip((tr / peak * (len(bars) - 1)).astype(int), 0, len(bars) - 1)
    return ''.join(bars[i] for i in idxs)


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: inspect_fig2_diag.py <path/to/out.rhoN.diag.npz>')
    p = os.path.expandvars(sys.argv[1])
    with np.load(p) as f:
        d = {k: f[k] for k in f.files}
    rho = float(d['rho'])
    # Grids match fig2_pehe_l2.py:
    Y_GRID = np.linspace(-8.0, 8.0, 501)
    TAU_GRID = np.linspace(-10.0, 10.0, 501)

    print(f'\nshard: {p}\nρ = {rho}    (only q=0 is stored in diag)\n')

    for arm_key, label in [('p_y0_q0', 'Y_do(0)'), ('p_y1_q0', 'Y_do(1)')]:
        true_key = f'true_{arm_key}'
        p_true = d[true_key]
        lo_t, hi_t = _quantile_support(p_true, Y_GRID)
        peak_t = p_true.max()

        print(f'── {label}  |  truth support (5–95%): [{lo_t:+.2f}, {hi_t:+.2f}]   '
              f'peak={peak_t:.3f}')
        print(f'   truth :  {_sparkline(p_true)}')
        for m in METHODS:
            key = f'{m}_{arm_key}'
            if key not in d:
                continue
            p_est = d[key]
            lo, hi = _peak_support(p_est, Y_GRID)
            miss = _mass_outside(p_est, Y_GRID, lo_t, hi_t)
            peak = p_est.max()
            print(f'   {LABEL[m]:<14s}  supp≈[{lo:+.2f}, {hi:+.2f}]   '
                  f'miss={miss*100:5.2f}%   peak={peak:.3f}   '
                  f'peak_ratio={peak/max(peak_t,1e-12):.2f}')
            print(f'                    {_sparkline(p_est)}')
        print()

    # CATE panel (q=0)
    print(f'── τ = Y1 − Y0  (q=0)')
    p_true = d['true_p_tau_q0']
    lo_t, hi_t = _quantile_support(p_true, TAU_GRID)
    peak_t = p_true.max()
    print(f'   truth  supp[5-95%]=[{lo_t:+.2f}, {hi_t:+.2f}]   peak={peak_t:.3f}')
    print(f'   truth :  {_sparkline(p_true)}')
    for m in METHODS:
        key = f'{m}_p_tau_q0'
        if key not in d:
            continue
        p_est = d[key]
        lo, hi = _peak_support(p_est, TAU_GRID)
        miss = _mass_outside(p_est, TAU_GRID, lo_t, hi_t)
        peak = p_est.max()
        print(f'   {LABEL[m]:<14s}  supp≈[{lo:+.2f}, {hi:+.2f}]   '
              f'miss={miss*100:5.2f}%   peak_ratio={peak/max(peak_t,1e-12):.2f}')
        print(f'                    {_sparkline(p_est)}')


if __name__ == '__main__':
    main()
