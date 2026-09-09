"""Tier-C density eval, MALC arm: p(tau | x) with region 0 smoothed.

The MALC counterpart of eval_density_tauC.py. Same three methods, same truth,
same metrics, same output schema -- summarize_density_tauC.py reads this
directory unchanged. The ONLY difference is inside region 0, the inner x inner
block, where the head's raw histogram is replaced by a K=1 log-concave MLE
(see density_common's MALC section for what that estimator does and why K=1).
Every one of the 8 non-interior regions goes through the identical code path
with identical quadrature, so a raw-vs-MALC difference is attributable to
region 0 and nothing else.

Runs entirely off the prediction dumps eval_density_tauC.py writes with
SAVE_PREDICTIONS=1. CPU only -- no checkpoint, no GPU, no harness import, no
dataset loader. Parallel over QUERIES, which at ~1 s each keeps the pool
balanced better than parallelising over realizations.

FALLBACK. The log-concave MLE has compact support, so its interior term at
tau* can be exactly zero. The TOTAL density is still positive there, because
the 8 tail regions have infinite support -- which is the trap: the NLL comes
back finite but is pure tail, ~1e-3 of what it should be, scoring ~6.7 nats
where the same realization's other queries score 0.38. So the trigger is the
INTERIOR term, not the total, and it coincides exactly with tau* falling
outside malc_tau_hull_{lo,hi}. Measured rate at B=100: 0.8% (joint), 2.7%
(uwyk_native), 0.8% (uwyk_matched); ~0 at B=1000. Those queries fall back to
the RAW density and are counted, not silently dropped -- two of them out of 75
would move a realization mean by ~0.17 nats, half the headline effect, and a
silent drop would quietly change which queries each method is scored on.

Usage (CPU node):
    DUMPS=./results_density_tauC/5312884/IHDP/predictions \
    OUT=./results_density_tauC_malc/IHDP \
    MALC_B=100 N_WORKERS=32 \
    python -u benchmarks/eval_graph2d/eval_density_tauC_malc.py
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import sys
import time

# MALC's conic solver and numpy are both threaded; with one process per query
# they oversubscribe badly. BLAS reads these AT IMPORT, so this must stay
# ABOVE `import numpy` -- set below it they are silently already ignored.
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from density_common import (                                        # noqa: E402
    Joint2D, UWYK1D, independent_f2d, joint_tau_density, uwyk_tau_density,
    truth_tau_density, l2_distance, kl, mass, point_metrics, TAU_CENTERS,
    fit_malc_interior, malc_tau_density, malc_interior_tau,
    malc_hull_tau_range, malc_inner_mean,
    malc_synthetic_points, MALC_B_DEFAULT, MALC_N_Y0,
)

DUMPS = os.environ['DUMPS']
OUT = os.environ.get('OUT', './results_density_tauC_malc')
MALC_B = int(os.environ.get('MALC_B', str(MALC_B_DEFAULT)))
MALC_N_Y0_ENV = int(os.environ.get('MALC_N_Y0', str(MALC_N_Y0)))
N_Y0 = int(os.environ.get('N_Y0', '4096'))          # tails, matches the raw run
N_WORKERS = int(os.environ.get(
    'N_WORKERS', os.environ.get('SLURM_CPUS_PER_TASK', '8')))
REAL_START = int(os.environ.get('REAL_START', '0'))
REAL_END = os.environ.get('REAL_END')
SAVE_FALLBACK_POINTS = os.environ.get('SAVE_FALLBACK_POINTS', '1') == '1'

METHODS = ('uwyk_native', 'uwyk_matched', 'joint')
_G: dict = {}


# ---------------------------------------------------------------------------
def score(p_est, p_true, tau_star_density, tau_grid=TAU_CENTERS):
    """Byte-identical to eval_density_tauC.score -- same metrics, same order."""
    return dict(
        nll=float(-np.log(tau_star_density)),
        l2=l2_distance(p_true, p_est, tau_grid),
        kl_fwd=kl(p_true, p_est, tau_grid),
        kl_rev=kl(p_est, p_true, tau_grid),
        mass=mass(p_est, tau_grid),
    )


def uwyk_arm_mean(f: UWYK1D, inner_mean: float) -> float:
    """UWYK1D.mean() with the bar term replaced by a MALC interior mean.

    Mirrors density_common.UWYK1D.mean exactly: interior mass x interior mean,
    plus the two half-normal tail means. Only the interior mean changes.
    """
    h = math.sqrt(2.0 / math.pi)
    return float(float(np.exp(f.log_pBars).sum()) * float(inner_mean)
                 + math.exp(f.log_pL) * (f.edges[0] - h * f.sL)
                 + math.exp(f.log_pR) * (f.edges[-1] + h * f.sR))


def build_query(q):
    """The three (p_mat, edges, w0, tails, raw) bundles for one query.

    This is the ONLY place the three methods differ. Everything downstream is
    shared, which is what makes the comparison a comparison.
    """
    d = _G['d']
    jt = Joint2D.from_pred(d['joint_logits'][q], _G['J'], _G['e2'])
    f0 = UWYK1D.from_pred(d['uwyk_pred0'][q], _G['be'], _G['bw'],
                          _G['sL'], _G['sR'])
    f1 = UWYK1D.from_pred(d['uwyk_pred1'][q], _G['be'], _G['bw'],
                          _G['sL'], _G['sR'])
    f0m, f1m = f0.rebin(_G['e2']), f1.rebin(_G['e2'])

    def uwyk_bundle(a, b):
        pa, pb = np.exp(a.log_pBars), np.exp(b.log_pBars)
        return dict(
            p_mat=np.outer(pa, pb),                  # sums to w0 by construction
            edges=np.asarray(a.edges, dtype=np.float64),
            w0=float(pa.sum() * pb.sum()),
            tail_f2d=independent_f2d(a, b),
            pad=8.0 * max(a.max_scale, b.max_scale),
            align_bins=len(a.widths),
            raw=lambda t, a=a, b=b: uwyk_tau_density(a, b, t, n_y0=N_Y0),
            raw_mean=lambda a=a, b=b: b.mean() - a.mean(),
            malc_mean=lambda inner, a=a, b=b: (uwyk_arm_mean(b, inner[1])
                                               - uwyk_arm_mean(a, inner[0])),
        )

    return {
        'joint': dict(
            p_mat=jt.p_mat, edges=np.asarray(_G['e2'], dtype=np.float64),
            w0=float(jt.w[0]), tail_f2d=jt.density,
            pad=8.0 * jt.max_scale, align_bins=jt.p_mat.shape[0],
            raw=lambda t: joint_tau_density(jt, t, n_y0=N_Y0),
            raw_mean=lambda: (lambda m: m[1] - m[0])(jt.mean()),
            malc_mean=lambda inner: (lambda m: m[1] - m[0])(jt.mean(inner=inner)),
        ),
        'uwyk_native': uwyk_bundle(f0, f1),
        'uwyk_matched': uwyk_bundle(f0m, f1m),
    }


def run_query(q):
    """One query, all three methods. Returns plain dicts (picklable)."""
    d = _G['d']
    tau_star = float(d['tau_star_scaled'][q])
    t_star = np.array([tau_star])
    p_true = truth_tau_density(float(d['mu0_scaled'][q]),
                               float(d['mu1_scaled'][q]),
                               float(d['sigma_scaled']), TAU_CENTERS)
    bundles = build_query(q)
    out = {}
    for name in METHODS:
        b = bundles[name]
        lo, hi = float(b['edges'][0]), float(b['edges'][-1])
        # Seed on (realization, query, method): reproducible, and independent
        # across methods so a bad draw cannot favour one of them.
        seed = (_G['r'] * 1_000_003 + q * 1009
                + METHODS.index(name) * 31) % (2 ** 31 - 1)
        fit = fit_malc_interior(b['p_mat'], b['edges'], B=MALC_B, seed=seed)

        rec = dict(fallback=0, reason='', tau_hull_lo=float('nan'),
                   tau_hull_hi=float('nan'), p_star=float('nan'),
                   interior_star=float('nan'), points=None)
        p_grid = d_star = cate = None

        if fit is None:
            rec['fallback'], rec['reason'] = 1, 'fit_failed'
        else:
            rec['tau_hull_lo'], rec['tau_hull_hi'] = malc_hull_tau_range(fit)
            kw = dict(w0=b['w0'], lo=lo, hi=hi, pad=b['pad'],
                      n_y0_malc=MALC_N_Y0_ENV, n_y0_tail=N_Y0,
                      align_bins=b['align_bins'])
            # THE INTERIOR TERM ALONE decides the fallback, not the total.
            # Adding the 8 tail regions makes the total strictly positive
            # everywhere, so `d_star <= 0` almost never fires -- what actually
            # happens when tau* misses the hull is that region 0 contributes
            # NOTHING and the density comes back as pure tail: measured
            # 1.2e-3 against a median of 0.68 on the same realization, i.e.
            # 6.7 nats where the other queries score 0.38. Finite, so nothing
            # raises, and two such queries out of 75 shift the realization
            # mean by ~0.17 nats -- half the headline effect. Trigger on the
            # interior instead, which is exactly the hull tau-range.
            interior_star = float(malc_interior_tau(
                fit, lo, hi, t_star, b['w0'], n_y0=MALC_N_Y0_ENV)[0])
            rec['interior_star'] = interior_star
            d_star = float(malc_tau_density(fit, b['tail_f2d'], t_star, **kw)[0])
            rec['p_star'] = d_star
            inner = malc_inner_mean(fit, lo, hi)
            if inner is None:
                rec['fallback'], rec['reason'] = 1, 'empty_fit'
            elif interior_star <= 0.0:
                rec['fallback'], rec['reason'] = 1, 'tau_outside_hull'
            elif not np.isfinite(d_star) or d_star <= 0.0:
                rec['fallback'], rec['reason'] = 1, 'zero_density'
            if rec['fallback']:
                if SAVE_FALLBACK_POINTS:
                    rec['points'] = malc_synthetic_points(fit)
                d_star = None
            else:
                p_grid = malc_tau_density(fit, b['tail_f2d'], TAU_CENTERS, **kw)
                cate = float(b['malc_mean'](inner))

        if p_grid is None:                      # raw fallback for this query
            p_grid = b['raw'](TAU_CENTERS)
            d_star = float(b['raw'](t_star)[0])
            cate = float(b['raw_mean']())

        out[name] = dict(score=score(p_grid, p_true, d_star), cate=cate, **rec)
    return q, out


def _init(payload):
    _G.update(payload)


# ---------------------------------------------------------------------------
def evaluate(path, pool_cls):
    with np.load(path, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    r = int(d['realization'])
    payload = dict(d=d, r=r, J=int(d['J']), e2=d['edges2d'],
                   be=d['bar_edges'], bw=d['bar_widths'],
                   sL=float(d['base_sL']), sR=float(d['base_sR']))
    n_q = len(d['tau_star_scaled'])

    if N_WORKERS > 1:
        with pool_cls(processes=min(N_WORKERS, n_q), initializer=_init,
                      initargs=(payload,)) as pool:
            results = pool.map(run_query, range(n_q), chunksize=1)
    else:
        _init(payload)
        results = [run_query(q) for q in range(n_q)]
    results.sort(key=lambda t: t[0])

    true_cate = np.asarray(d['true_cate'], dtype=np.float64).reshape(-1)
    y_scale = float(d['y_scale'])
    row = {'dataset': str(d['dataset']), 'realization': r, 'n_queries': n_q,
           'n_context': int(d['n_context']), 'anc_tag': str(d['anc_tag']),
           'truth_noise_source': str(d['truth_noise_source']),
           'sigma_raw': float(d['sigma_raw']),
           'sigma_scaled': float(d['sigma_scaled']),
           'sigma_residual_raw': float(d['sigma_residual_raw']),
           'sigma_residual_scaled': float(d['sigma_residual_scaled']),
           'y_scale': y_scale, 'y_shift': float(d['y_shift']),
           'y_scaling': str(d['y_scaling']), 'std_target': str(d['std_target']),
           'true_cate': true_cate,
           'malc_B': MALC_B, 'malc_K': 1, 'malc_n_y0': MALC_N_Y0_ENV,
           'n_y0': N_Y0, 'source_dump': str(path),
           'frac_tau_outside_grid': float(
               np.mean(np.abs(d['tau_star_scaled']) > 3.0))}

    for name in METHODS:
        recs = [res[name] for _, res in results]
        for k in recs[0]['score']:
            row[f'{k}_{name}'] = float(np.mean([x['score'][k] for x in recs]))
        means = [x['cate'] for x in recs]
        for metric, value in point_metrics(means, true_cate, y_scale).items():
            row[f'{metric}_{name}'] = value
        row[f'cate_pred_{name}'] = np.asarray(means) * y_scale
        # Per-query MALC diagnostics: everything needed to plot where the
        # synthetic samples landed relative to tau*.
        row[f'malc_tau_hull_lo_{name}'] = np.array(
            [x['tau_hull_lo'] for x in recs])
        row[f'malc_tau_hull_hi_{name}'] = np.array(
            [x['tau_hull_hi'] for x in recs])
        row[f'malc_p_star_{name}'] = np.array([x['p_star'] for x in recs])
        row[f'malc_interior_star_{name}'] = np.array(
            [x['interior_star'] for x in recs])
        row[f'malc_fallback_{name}'] = np.array(
            [x['fallback'] for x in recs], dtype=np.int8)
        row[f'malc_fallback_reason_{name}'] = np.array(
            [x['reason'] for x in recs])
        row[f'n_fallback_{name}'] = int(sum(x['fallback'] for x in recs))
        row[f'tau_star_scaled'] = np.asarray(d['tau_star_scaled'])
        if SAVE_FALLBACK_POINTS:
            fq = [q for q, x in enumerate(recs) if x['points'] is not None]
            pts = [recs[q]['points'] for q in fq]
            row[f'malc_fallback_query_{name}'] = np.array(fq, dtype=np.int32)
            row[f'malc_fallback_points_{name}'] = (
                np.concatenate(pts, axis=0) if pts else np.empty((0, 2)))
    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    paths = sorted(p for p in os.listdir(DUMPS) if p.endswith('.npz'))
    if not paths:
        raise SystemExit(f'[tauC-malc] no dumps in {DUMPS}')
    dataset = paths[0].rsplit('_r', 1)[0]
    lo, hi = max(0, REAL_START), (len(paths) if REAL_END is None
                                  else min(len(paths), int(REAL_END)))
    print(f'[tauC-malc] dataset={dataset} dumps={DUMPS} '
          f'B={MALC_B} K=1 malc_n_y0={MALC_N_Y0_ENV} tail_n_y0={N_Y0} '
          f'workers={N_WORKERS}  realizations [{lo}, {hi}) of {len(paths)}',
          flush=True)

    # 'fork' shares the dump arrays copy-on-write; 'spawn' would repickle the
    # whole npz to every worker for every realization.
    pool_cls = mp.get_context('fork').Pool

    t0 = time.time()
    for path in paths[lo:hi]:
        row = evaluate(os.path.join(DUMPS, path), pool_cls)
        r = int(row['realization'])
        np.savez(os.path.join(OUT, f'{dataset}_r{r:03d}.npz'),
                 **{k: np.array(v) for k, v in row.items()})
        print(f'r={r:03d}  ' + '  |  '.join(
            f'{m}: nll={row[f"nll_{m}"]:7.3f} l2={row[f"l2_{m}"]:6.3f} '
            f'klrev={row[f"kl_rev_{m}"]:7.4f} pehe={row[f"pehe_{m}"]:7.3f} '
            f'fb={row[f"n_fallback_{m}"]:d}' for m in METHODS)
            + f'   ({time.time()-t0:.0f}s)', flush=True)
    print(f'[tauC-malc] done in {time.time()-t0:.0f}s -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
