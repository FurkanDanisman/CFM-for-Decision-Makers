"""Interval + distributional metrics for a predictive tau density.

Consumes exactly what the raw (non-MALC) path in `density_common.py` already
produces: a density evaluated on `TAU_CENTERS`. Adds four scores:

    coverage   fraction of queries whose 1-alpha interval contains true tau
    length     width of that interval
    winkler    length + (2/alpha) * distance outside the interval  (proper)
    crps       \\int (F(z) - 1{z >= tau*})^2 dz                      (proper)

WHY THESE FOUR. Coverage alone is maximised by an infinite interval; length
alone by an empty one. Winkler combines them and is proper, so neither trick
wins. CRPS scores the whole distribution rather than one level -- it reduces to
|tau_hat - tau*| for a point mass, so it is directly comparable to the CATE MAE
`density_common.point_metrics` reports, and the gap between them is the price
of the density's width. CRPS also stays finite where NLL diverges, which
matters for the coarse J=10 joint bins.

UNITS. The tau grid is in the SCALED outcome units the models work in. Pass
`y_scale` (the same one `point_metrics` takes) and length / winkler / crps come
back in original outcome units; all three are homogeneous of degree 1 in tau,
so this is an exact rescale, not an approximation. Coverage is dimensionless.

GRID TRUNCATION. TAU_CENTERS spans [-3, 3] only. A wide density gets its tails
cut, which biases length and winkler DOWN and crps DOWN -- i.e. it flatters a
diffuse model, the exact failure this eval exists to catch. Every function here
reports `censored` / `mass`; aggregate them and gate on them rather than
averaging blind. See `summarize`.
"""
from __future__ import annotations

import numpy as np

_TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

# ALPHAS, not nominal coverages: alpha=0.05 <-> a 95% interval. Every function
# here takes alpha; `nominal` in the output is 1-alpha.
DEFAULT_LEVELS = (0.50, 0.20, 0.10, 0.05)


# ── CDF ──────────────────────────────────────────────────────────────────────
def predictive_cdf(p, grid):
    """Cumulative trapezoid of a density on `grid`. Not renormalised: the
    terminal value IS the mass diagnostic and must stay visible."""
    p = np.asarray(p, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    if p.shape != grid.shape:
        raise ValueError(f'density {p.shape} and grid {grid.shape} must match')
    if np.any(p < 0):
        raise ValueError('density has negative values')
    step = np.diff(grid)
    return np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * step)])


# ── intervals ────────────────────────────────────────────────────────────────
def interval_equal_tailed(p, grid, alpha, cdf=None):
    """Central 1-alpha interval: [F^-1(alpha/2), F^-1(1-alpha/2)].

    Returns (lo, hi, censored). `censored` is True when a requested quantile
    lies outside the grid's mass range, i.e. the endpoint was clamped to a grid
    edge and the true interval is WIDER than reported.
    """
    grid = np.asarray(grid, dtype=np.float64)
    F = predictive_cdf(p, grid) if cdf is None else np.asarray(cdf, np.float64)
    total = F[-1]
    if not np.isfinite(total) or total <= 0:
        raise ValueError('density integrates to a non-positive value')
    q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0
    # Quantiles are taken of the NORMALISED cdf; truncation shows up in
    # `censored` and in the mass diagnostic, never as a silent renormalisation
    # of the score.
    # Truncation test must use the UNNORMALISED mass: after dividing by
    # `total` the endpoints are 0 and 1 by construction and no truncation is
    # ever visible. Missing mass exceeding one tail means the true quantile
    # lies off-grid and the reported interval is NARROWER than the real one.
    Fn = F / total
    censored = bool((1.0 - total) > alpha / 2.0)
    lo = float(np.interp(q_lo, Fn, grid))
    hi = float(np.interp(q_hi, Fn, grid))
    return lo, hi, censored


def interval_hpd(p, grid, alpha, n_bisect=200):
    """Highest-posterior-density region at level 1-alpha.

    Returns (lo, hi, censored, disjoint, measure, level). `lo`/`hi` are the
    region's HULL; `measure` is its true total width. For a UNIMODAL density
    the two agree. For a multimodal one the region is disjoint and the hull
    spans the gap, so the hull can be LONGER than the equal-tailed interval
    while the region itself is shorter -- `measure` is the honest sharpness
    number and `level` lets a caller test membership exactly
    (`interp(y, grid, p) >= level`) instead of using the hull.
    """
    p = np.asarray(p, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    total = _TRAPZ(p, grid)
    if not np.isfinite(total) or total <= 0:
        raise ValueError('density integrates to a non-positive value')
    target = (1.0 - alpha) * total

    lo_c, hi_c = 0.0, float(p.max())
    for _ in range(n_bisect):                       # bisect the density level
        c = 0.5 * (lo_c + hi_c)
        if _TRAPZ(np.where(p >= c, p, 0.0), grid) >= target:
            lo_c = c
        else:
            hi_c = c
    keep = p >= lo_c
    if not keep.any():
        raise RuntimeError('empty HPD region')
    idx = np.flatnonzero(keep)
    disjoint = bool(np.any(np.diff(idx) > 1))
    lo, hi = float(grid[idx[0]]), float(grid[idx[-1]])
    censored = bool(keep[0] or keep[-1] or (1.0 - total) > alpha)
    measure = float(_TRAPZ(keep.astype(np.float64), grid))
    return lo, hi, censored, disjoint, measure, float(lo_c)


# ── scores ───────────────────────────────────────────────────────────────────
def covered(lo, hi, y_true):
    return bool(lo <= y_true <= hi)


def length(lo, hi):
    return float(hi - lo)


def winkler(lo, hi, y_true, alpha):
    """Winkler / interval score. Lower is better. PROPER: widening to buy
    coverage costs `length`, narrowing costs the (2/alpha) miss penalty, so
    neither degenerate strategy wins."""
    w = float(hi - lo)
    if y_true < lo:
        w += (2.0 / alpha) * (lo - y_true)
    elif y_true > hi:
        w += (2.0 / alpha) * (y_true - hi)
    return w


def crps(p, grid, y_true, cdf=None):
    """\\int (F(z) - 1{z >= y})^2 dz. Lower is better, units of tau.

    Reduces to |y_hat - y| for a point mass, so it is on the same scale as the
    CATE MAE. Finite even where the density is 0 at the truth (unlike NLL).

    The integrand JUMPS at y, so a plain trapezoid over the whole grid is only
    O(dz) accurate -- at dz=5e-4 that is a ~2e-4 bias, the same order as the
    differences between models we want to resolve. We therefore split the
    integral at y and insert it as an exact node: \\int F^2 below, \\int (F-1)^2
    above. Both pieces are smooth, so this is O(dz^2).
    """
    grid = np.asarray(grid, dtype=np.float64)
    F = predictive_cdf(p, grid) if cdf is None else np.asarray(cdf, np.float64)
    total = F[-1]
    if not np.isfinite(total) or total <= 0:
        raise ValueError('density integrates to a non-positive value')
    Fn = F / total
    y = float(y_true)

    if y <= grid[0]:
        return float(_TRAPZ((Fn - 1.0) ** 2, grid))
    if y >= grid[-1]:
        return float(_TRAPZ(Fn ** 2, grid))

    k = int(np.searchsorted(grid, y))
    Fy = float(np.interp(y, grid, Fn))
    g_lo = np.concatenate([grid[:k], [y]])
    F_lo = np.concatenate([Fn[:k], [Fy]])
    g_hi = np.concatenate([[y], grid[k:]])
    F_hi = np.concatenate([[Fy], Fn[k:]])
    return float(_TRAPZ(F_lo ** 2, g_lo) + _TRAPZ((F_hi - 1.0) ** 2, g_hi))


# ── per-query driver ─────────────────────────────────────────────────────────
def query_metrics(p, grid, y_true, levels=DEFAULT_LEVELS, y_scale=1.0,
                  method='equal-tailed'):
    """All four scores for ONE query density against ONE true tau.

    `y_true` must be in the SAME (scaled) units as `grid`; length / winkler /
    crps are returned multiplied by `y_scale`, i.e. in original outcome units.
    """
    grid = np.asarray(grid, dtype=np.float64)
    y_true = float(y_true)
    s = float(y_scale)
    F = predictive_cdf(p, grid)
    out = {'mass': float(F[-1]), 'crps': crps(p, grid, y_true, cdf=F) * s,
           'levels': {}}
    for a in levels:
        if method == 'equal-tailed':
            lo, hi, cens = interval_equal_tailed(p, grid, a, cdf=F)
            disj = False
        elif method == 'hpd':
            lo, hi, cens, disj, measure, lvl = interval_hpd(p, grid, a)
        else:
            raise ValueError("method must be 'equal-tailed' or 'hpd'")
        if method == 'hpd':
            # exact region membership, not hull containment
            is_cov = bool(float(np.interp(y_true, grid, np.asarray(p))) >= lvl)
            width = measure
        else:
            is_cov = covered(lo, hi, y_true)
            width = length(lo, hi)
        out['levels'][a] = {
            'nominal': 1.0 - a,
            'covered': is_cov,
            'length': width * s,
            'winkler': winkler(lo, hi, y_true, a) * s,
            'lo': lo * s, 'hi': hi * s,
            'censored': cens, 'disjoint': disj,
        }
    return out


def summarize(per_query, levels=DEFAULT_LEVELS, mass_tol=0.01):
    """Aggregate `query_metrics` dicts into the reportable table.

    Coverage is a plain percentage, as requested. `censored_frac` and
    `mass_fail_frac` are reported ALONGSIDE, never folded in: a run with a
    large censored fraction has understated length and winkler, and the
    comparison is not trustworthy at that level.
    """
    if not per_query:
        raise ValueError('no per-query results to summarize')
    n = len(per_query)
    mass = np.array([q['mass'] for q in per_query], dtype=np.float64)
    out = {
        'n_queries': n,
        'crps_mean': float(np.mean([q['crps'] for q in per_query])),
        'crps_median': float(np.median([q['crps'] for q in per_query])),
        'mass_mean': float(mass.mean()),
        'mass_min': float(mass.min()),
        'mass_fail_frac': float(np.mean(np.abs(mass - 1.0) > mass_tol)),
        'levels': {},
    }
    for a in levels:
        cov = np.array([q['levels'][a]['covered'] for q in per_query], float)
        ln = np.array([q['levels'][a]['length'] for q in per_query], float)
        wk = np.array([q['levels'][a]['winkler'] for q in per_query], float)
        cn = np.array([q['levels'][a]['censored'] for q in per_query], float)
        dj = np.array([q['levels'][a]['disjoint'] for q in per_query], float)
        out['levels'][a] = {
            'nominal_pct': 100.0 * (1.0 - a),
            'coverage_pct': 100.0 * float(cov.mean()),
            # binomial se of the coverage estimate, in percentage points
            'coverage_se_pct': 100.0 * float(np.sqrt(cov.mean() *
                                                     (1 - cov.mean()) / n)),
            'length_mean': float(ln.mean()),
            'length_median': float(np.median(ln)),
            'winkler_mean': float(wk.mean()),
            'censored_frac': float(cn.mean()),
            'disjoint_frac': float(dj.mean()),
        }
    return out


def format_table(summary, title=''):
    """One-model text block: coverage / length / winkler per level, then CRPS."""
    L = []
    if title:
        L.append(title)
    L.append(f"  n_queries={summary['n_queries']}  "
             f"mass[mean={summary['mass_mean']:.5f} min={summary['mass_min']:.5f} "
             f"fail={100*summary['mass_fail_frac']:.1f}%]")
    L.append(f"  {'nominal':>8s} {'coverage':>12s} {'length':>10s} "
             f"{'winkler':>10s} {'censored':>9s}")
    for a in sorted(summary['levels'], reverse=True):
        d = summary['levels'][a]
        L.append(f"  {d['nominal_pct']:7.0f}% "
                 f"{d['coverage_pct']:8.1f}±{d['coverage_se_pct']:<3.1f}% "
                 f"{d['length_mean']:10.4f} {d['winkler_mean']:10.4f} "
                 f"{100*d['censored_frac']:8.1f}%")
    L.append(f"  CRPS mean={summary['crps_mean']:.5f}  "
             f"median={summary['crps_median']:.5f}")
    return '\n'.join(L)
