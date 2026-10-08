"""Table 21 (tab:density-nll) recomputed from OUR RealCause density dumps, CPU only.

All three NLLs are reported in RAW outcome units (the loader's original y units):

    L_{Y0+Y1} = -(1/Q) sum_q [log f0(mu0_q) + log f1(mu1_q)]
    L_tau     = -(1/Q) sum_q  log f_tau,q(tau*_q),   tau*_q = Y1_q - Y0_q (realized)
    L_ATE     = -log f_ATE(theta*),                 theta* = mean_q (mu1_q - mu0_q)

per realization, then mean +- SE over realizations (IHDP r000-r099, ACIC r000-r009).

No density is reimplemented here. Every model goes through the collaborator's
objects in density_common.py and the exact routines eval_density_tauC.py /
eval_density_marginals.py / eval_density_ate.py call:

    1D arms   UWYK1D / DoPFN1D / CausalPFN1D  .density           -> f0, f1
              uwyk_tau_density / dopfn_tau_density / causalpfn_tau_density
                                         (independence convolution) -> f_tau
    2D joint  Joint2D.from_pred(full J*J+9+4 head output) [.affine]
              joint_marginals                                     -> f0, f1
              joint_tau_density (exact interior diagonal + tail-region quadrature)
    ATE       eval_density_ate.tau_densities on TAU_CENTERS -> W2 barycenter
              (ot_barycenter.wasserstein_barycenter_1d) -> eval_density_ate._normalise
              -> np.interp at theta*  (eval_density_ate.score's nll line)

WORKING AXIS (numerical device only, not a reported scale). The collaborator's
grid routines live on a fixed tau grid TAU_CENTERS = [-3, 3] (step 5e-4), so the
densities are first mapped affinely onto a per-realization working axis
w = (y - shift_w) / scale_w shared by all models, evaluated there, and converted
back to raw units with the Jacobian in `work_to_raw_nll`. Point NLLs (Y0+Y1, tau)
are exactly invariant to the working axis; L_ATE is invariant up to the grid
discretization / [-3, 3] truncation of the barycenter (the self-test checks this).
`working_axis` is the one place to change it. NB the collaborator's own tables
used the harness min-max axis of the (on ACIC, 1000-row-subsampled) training
context and reported scaled-axis NLLs; these raw-unit numbers differ from those
by k*log(y_scale) per realization.

Usage:
    python benchmarks/eval_graph2d/nll_from_dumps.py --dataset IHDP \
        --out-csv nll_IHDP.csv --workers 16
"""
from __future__ import annotations

import argparse
import glob
import math
import multiprocessing as mp
import os
import re
import sys
import time
from dataclasses import replace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from density_common import (                                        # noqa: E402
    Joint2D, UWYK1D, DoPFN1D, CausalPFN1D, TAU_CENTERS, joint_marginals,
    joint_tau_density, uwyk_tau_density, dopfn_tau_density, causalpfn_tau_density,
)
from density_truth import load_density_truth                       # noqa: E402
from eval_density_ate import (                                     # noqa: E402
    tau_densities, _normalise, wasserstein_barycenter_1d,
)

N_Y0 = 4096                      # eval_density_tauC.N_Y0 default (tail quadrature)
N_REAL = {'IHDP': 100, 'ACIC': 10}

# (row label, dump format, root template key, adjacency mode) in Table 21 row order.
MODELS = (
    ('Do-PFN',         'dopfn1d', 'dopfn1d', None),
    ('Do-PFN 2D',      'dopfn2d', 'dopfn2d', None),
    ('UWYK No-Anc',    'uwyk1d',  'uwyk1d',  'noanc'),
    ('UWYK No-Anc 2D', 'uwyk2d',  'uwyk2d',  'noanc'),
    ('UWYK Anc',       'uwyk1d',  'uwyk1d',  'v3a'),
    ('UWYK Anc 2D',    'uwyk2d',  'uwyk2d',  'v3a'),
    ('CausalPFN-C',    'cpfn1d',  'cpfn1d',  None),
    ('CausalPFN-C 2D', 'cpfn2d',  'cpfn2d',  None),
)
DEFAULT_ROOTS = {
    'dopfn1d': '{SCRATCH}/rc_dens_uni/dopfn_native/{ds}',
    'dopfn2d': '{SCRATCH}/dumps_all/dopfn_repro_joint2d_262k_t1/rc/dopfn_native/{ds}',
    'uwyk1d':  '{SCRATCH}/rc_dens_uni/uwyk1d/{ds}',
    'uwyk2d':  '{SCRATCH}/rc_nll2d/graph2d/{ds}',
    'cpfn1d':  '{SCRATCH}/rc_dens_uni/cpfn1d/{ds}',
    'cpfn2d':  '{SCRATCH}/rc_nll2d/cpfn2d_pooled/{ds}',
}


# ---------------------------------------------------------------------------
# Axis: the one place the working scale is defined, and its Jacobian
# ---------------------------------------------------------------------------
def working_axis(mu0_raw, mu1_raw, sigma_raw):
    """(shift_w, scale_w) with w = (y_raw - shift_w) / scale_w.

    Purely numerical: chosen so every true tau (|mu1-mu0| + noise) sits well
    inside TAU_CENTERS = [-3, 3]. Span = [min mu - 4 sigma, max mu + 4 sigma],
    mapped to [-1, 1], so any tau inside that span is within +-2 working units.
    Final numbers are in raw units regardless (see work_to_raw_nll).
    """
    mu = np.concatenate([np.ravel(mu0_raw), np.ravel(mu1_raw)])
    lo, hi = float(mu.min()) - 4 * sigma_raw, float(mu.max()) + 4 * sigma_raw
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


def work_to_raw_nll(nll_work, scale_w, n_dims):
    """f_raw(y) = f_work(w) / scale_w per dimension  =>  NLL_raw = NLL_work + k log scale_w.

    n_dims = 2 for the Y0+Y1 sum (two 1D densities), 1 for tau and ATE.
    """
    return np.asarray(nll_work, dtype=np.float64) + n_dims * math.log(scale_w)


def _log(p):
    with np.errstate(divide='ignore'):
        return np.log(np.asarray(p, dtype=np.float64))


def _uwyk_affine(f: UWYK1D, factor, offset) -> UWYK1D:
    """UWYK1D on y_new = factor * y_old + offset (bars, widths, tail scales)."""
    return replace(f, edges=f.edges * factor + offset, widths=f.widths * factor,
                   sL=f.sL * factor, sR=f.sR * factor)


# ---------------------------------------------------------------------------
# Dump readers -> per-query density objects on the working axis
# ---------------------------------------------------------------------------
def _realization_files(root):
    """{r: path} for every dump under root (rNNN / IHDP_rNNN / ACIC_rNN)."""
    out = {}
    for p in glob.glob(os.path.join(root, '*.npz')):
        # dump names only: rNNN.npz (Do-PFN) or <DATASET>_rNNN.npz; other files share the folder
        # (e.g. malc_ci_B100_K1_E2001_r000.npz) and must not be read as dumps
        m = re.fullmatch(r'(?:IHDP_|ACIC_)?r(\d+)\.npz', os.path.basename(p))
        if not m:
            continue
        r = int(m.group(1))
        if r in out:
            raise SystemExit(f'two dumps for r={r} under {root}: {out[r]} and {p}')
        out[r] = p
    return out


def _dump_affine(z):
    """Dump axis -> raw: raw = d * y_scale + y_shift."""
    return float(z['y_shift']), float(z['y_scale'])


def build_objects(fmt, mode, z, shift_w, scale_w):
    """-> (family, kind, per_query) exactly as eval_density_marginals.build_objects
    shapes it, every object on the working axis."""
    if fmt == 'dopfn1d':
        # Writer: eval_scm_case_studies/eval_native_dopfn.py:739-773. edges are
        # criterion.borders in raw units (y_shift=0, y_scale=1), p = softmax of
        # predict_full logits; first/last bins are DoPFN's half-normal end bins.
        # DoPFN1D derives tail scales from the raw borders, as density_dopfn.py does.
        b = np.asarray(z['edges'], np.float64)
        ds_, dsc = _dump_affine(z)
        b = b * dsc + ds_
        mk = lambda p: DoPFN1D.from_pred(_log(p), b, y_shift=shift_w, y_scale=scale_w)
        p0, p1 = z['p_y0_scaled'], z['p_y1_scaled']
        return 'dopfn', '1d', [(mk(p0[q]), mk(p1[q])) for q in range(len(p0))]

    if fmt == 'dopfn2d':
        # Writer: eval_native_dopfn.py:_predict_joint2d (:520-633). Head output on
        # the TRAINING grid; raw y = train y * data_std + data_mean. Unpack with the
        # training-grid bin width (BarDistribution2D._unpack), then map affinely,
        # exactly as density_dopfn.DoPFNModelSet does for repro_joint.
        lg = np.asarray(z['logits_2d'], np.float64)
        ge = np.asarray(z['grid_edges_train'], np.float64)
        dstd, dmean = float(z['data_std']), float(z['data_mean'])
        J = len(ge) - 1
        if lg.shape[1] != J * J + 13:
            raise ValueError(f'dopfn2d logits width {lg.shape[1]} != J^2+13 for J={J}')
        if 'edges' in z.files and not np.allclose(
                np.asarray(z['edges'], np.float64), ge * dstd + dmean, rtol=1e-4, atol=1e-4):
            raise ValueError('dopfn2d: stored raw edges != grid_edges_train*data_std+data_mean')
        fac, off = dstd / scale_w, (dmean - shift_w) / scale_w
        return 'dopfn', 'joint', [Joint2D.from_pred(lg[q], J, ge).affine(fac, off)
                                  for q in range(len(lg))]

    if fmt == 'uwyk1d':
        # Writer: eval_uwyk1d_realcause.py:243-275. p = softmax over [pL, bars, pR];
        # sL/sR raw per arm; scale = base * (softplus(raw) + floor). Dump axis = harness minmax.
        e = np.asarray(z['edges'], np.float64)
        w = np.diff(e)
        bL, bR, fl = float(z['base_s_left']), float(z['base_s_right']), float(z['scale_floor'])
        ds_, dsc = _dump_affine(z)
        fac, off = dsc / scale_w, (ds_ - shift_w) / scale_w
        p = {t: np.asarray(z[f'p_y{t}_scaled_{mode}'], np.float64) for t in (0, 1)}
        sl = {t: np.asarray(z[f'sL_raw_{t}_{mode}'], np.float64) for t in (0, 1)}
        sr = {t: np.asarray(z[f'sR_raw_{t}_{mode}'], np.float64) for t in (0, 1)}
        if p[0].shape[1] != len(w) + 2:
            raise ValueError(f'uwyk1d: {p[0].shape[1]} probs for {len(w)} bars (expected K+2)')

        def mk(t, q):
            pred = np.concatenate([_log(p[t][q]), [sl[t][q], sr[t][q]]])
            return _uwyk_affine(UWYK1D.from_pred(pred, e, w, bL, bR, scale_floor=fl), fac, off)
        return 'uwyk', '1d', [(mk(0, q), mk(1, q)) for q in range(len(p[0]))]

    if fmt in ('uwyk2d', 'cpfn2d'):
        # uwyk2d writer: eval_graph2d_realcause.py:1150-1195 (edges = linspace(-1,1,J+1)
        # on the harness minmax axis). cpfn2d writer: eval_cpfn2d_realcause.py:494-509
        # (edges = checkpoint training grid on the pooled-standardized axis). Both:
        # head trained with bin_width = (edges[-1]-edges[0])/J of exactly these edges.
        key = f'logits_2d_{mode}' if fmt == 'uwyk2d' else 'logits_2d'
        lg = np.asarray(z[key], np.float64)
        e = np.asarray(z['edges'], np.float64)
        J = len(e) - 1
        if lg.shape[1] != J * J + 13:
            raise ValueError(f'{fmt}: logits width {lg.shape[1]} != J^2+13 for J={J}')
        ds_, dsc = _dump_affine(z)
        fac, off = dsc / scale_w, (ds_ - shift_w) / scale_w
        fam = 'uwyk' if fmt == 'uwyk2d' else 'causalpfn'
        return fam, 'joint', [Joint2D.from_pred(lg[q], J, e).affine(fac, off)
                              for q in range(len(lg))]

    if fmt == 'cpfn1d':
        # Writer: eval_causalpfn2d/eval_causalpfn_v0_realcause.py:302-331. Finite
        # histogram probs over the native bin edges, pooled axis raw = d*y_scale+y_shift.
        ds_, dsc = _dump_affine(z)
        e = (np.asarray(z['edges'], np.float64) * dsc + ds_ - shift_w) / scale_w
        CausalPFN1D.from_pred(np.zeros(len(e) - 1), e)          # validates the grid
        p0, p1 = (np.asarray(z[k], np.float64) for k in ('p_y0_scaled', 'p_y1_scaled'))
        # probabilities, not logits: build directly (from_pred rejects log(0) = -inf)
        mk = lambda p: CausalPFN1D(p / p.sum(), e)
        return 'causalpfn', '1d', [(mk(p0[q]), mk(p1[q])) for q in range(len(p0))]
    raise ValueError(fmt)


# ---------------------------------------------------------------------------
# Scoring one (model, realization)
# ---------------------------------------------------------------------------
TAU_1D = {'uwyk': uwyk_tau_density, 'dopfn': dopfn_tau_density,
          'causalpfn': causalpfn_tau_density}


def _tau_at(family, kind, obj, t):
    t = np.array([t])
    if kind == 'joint':
        return float(joint_tau_density(obj, t, n_y0=N_Y0)[0])
    f0, f1 = obj
    if family == 'uwyk':
        return float(uwyk_tau_density(f0, f1, t, n_y0=N_Y0)[0])
    return float(TAU_1D[family](f0, f1, t)[0])


def score_objects(family, kind, per_query, truth_w, scale_w):
    """Raw-unit NLLs from objects on the working axis. truth_w: dict of working-axis
    mu0, mu1, tau_star, theta. Returns (row dict, per-query arrays)."""
    mu0, mu1, ts = truth_w['mu0'], truth_w['mu1'], truth_w['tau_star']
    Q = len(per_query)
    d0, d1, dt = np.empty(Q), np.empty(Q), np.empty(Q)
    for q, obj in enumerate(per_query):
        y = np.array([mu0[q], mu1[q]])
        if kind == 'joint':
            p0, p1 = joint_marginals(obj, y)
        else:
            p0, p1 = obj[0].density(y), obj[1].density(y)
        d0[q], d1[q] = p0[0], p1[1]
        dt[q] = _tau_at(family, kind, obj, ts[q])
    with np.errstate(divide='ignore', invalid='ignore'):
        nll_y = work_to_raw_nll(-(np.log(d0) + np.log(d1)), scale_w, 2)
        nll_t = work_to_raw_nll(-np.log(dt), scale_w, 1)

    # ATE: eval_density_ate.run_realization's construction, verbatim operators.
    pt = tau_densities(family, kind, per_query, N_Y0)
    tau_mass = np.trapezoid(pt, TAU_CENTERS, axis=1)
    p_ate = _normalise(wasserstein_barycenter_1d(pt, TAU_CENTERS))
    at = float(np.interp(truth_w['theta'], TAU_CENTERS, p_ate))
    nll_a = float('inf') if at <= 0 else float(work_to_raw_nll(-math.log(at), scale_w, 1))

    bad_y, bad_t = ~np.isfinite(nll_y), ~np.isfinite(nll_t)
    row = dict(L_y0y1=float(np.mean(nll_y)), L_tau=float(np.mean(nll_t)), L_ate=nll_a, Q=Q,
               n_nonfinite_y0y1=int(bad_y.sum()), n_nonfinite_tau=int(bad_t.sum()),
               nonfinite_ate=int(not np.isfinite(nll_a)),
               tau_grid_mass_min=float(tau_mass.min()))
    return row, dict(nll_y=nll_y, nll_t=nll_t)


def _work(task):
    label, fmt, mode, path, r, truth_raw, ds = task
    t0 = time.time()
    z = np.load(path, allow_pickle=True)
    # Query order: dump's true_cate_per_query vs truth mu1-mu0 (as eval_density_tauC.py:322-339).
    tc = np.asarray(z['true_cate_per_query'], np.float64).reshape(-1)
    cate = truth_raw['mu1'] - truth_raw['mu0']
    if tc.shape != cate.shape:
        raise RuntimeError(f'{label} r={r}: dump has {tc.size} queries, truth {cate.size} ({path})')
    gap = float(np.abs(tc - cate).max())
    if gap > 1e-3 * max(np.abs(cate).max(), 1e-9):
        raise RuntimeError(f'{label} r={r}: true_cate_per_query misaligned with truth '
                           f'(max |diff| {gap:.3e}) in {path}')
    shift_w, scale_w = truth_raw['axis']
    tw = {k: (truth_raw[k] - shift_w) / scale_w for k in ('mu0', 'mu1')}
    tw['tau_star'] = truth_raw['tau_star'] / scale_w
    tw['theta'] = float(cate.mean()) / scale_w
    family, kind, pq = build_objects(fmt, mode, z, shift_w, scale_w)
    if len(pq) != cate.size:
        raise RuntimeError(f'{label} r={r}: {len(pq)} density rows vs {cate.size} queries')
    row, _ = score_objects(family, kind, pq, tw, scale_w)
    row.update(model=label, realization=r, dataset=ds, working_shift=shift_w,
               working_scale=scale_w, seconds=round(time.time() - t0, 1), dump=path)
    return row


# ---------------------------------------------------------------------------
def load_truth_raw(ds, r, causalpfn_dir, acic_cache):
    t = load_density_truth(ds, r, y_shift=0.0, y_scale=1.0,
                           causalpfn_dir=causalpfn_dir, acic_cache_dir=acic_cache)
    out = dict(mu0=t.mu0_scaled, mu1=t.mu1_scaled, tau_star=t.tau_star_scaled,
               sigma=t.sigma_raw)
    out['axis'] = working_axis(out['mu0'], out['mu1'], t.sigma_raw)
    return out


def summarize(df, labels):
    import pandas as pd
    lines = ['| Model | n | L_{Y0+Y1} | L_tau | L_ATE | non-finite (y / tau / ate) |',
             '|---|---|---|---|---|---|']
    for lab in labels:
        d = df[df.model == lab]
        if d.empty:
            continue
        cells = []
        for k in ('L_y0y1', 'L_tau', 'L_ate'):
            v = d[k].to_numpy(float)
            if not np.isfinite(v).all():
                cells.append(f'inf ({int((~np.isfinite(v)).sum())}/{len(v)} real. non-finite)')
                continue
            se = v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float('nan')
            cells.append(f'{v.mean():.4f} ± {se:.4f}')
        nf = (f'{int(d.n_nonfinite_y0y1.sum())} / {int(d.n_nonfinite_tau.sum())} / '
              f'{int(d.nonfinite_ate.sum())}')
        lines.append(f'| {lab} | {len(d)} | ' + ' | '.join(cells) + f' | {nf} |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', required=True, choices=('IHDP', 'ACIC'))
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--out-md', default='', help='append the summary table here')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--models', default='', help='comma-separated row labels (default all 8)')
    ap.add_argument('--realizations', default='',
                    help='e.g. 0-4; default and REQUIRED for the paper: the full set')
    ap.add_argument('--causalpfn', default=os.environ.get('CAUSALPFN', ''),
                    help='CausalPFN checkout (IHDP npz truth); default $CAUSALPFN')
    ap.add_argument('--acic-cache', default=os.environ.get(
        'ACIC_CACHE_DIR', os.path.join(_REPO, 'data', 'acic_cache')))
    for key, tmpl in DEFAULT_ROOTS.items():
        ap.add_argument(f'--root-{key}', default=tmpl,
                        help=f'default {tmpl} ({{SCRATCH}} from env, {{ds}} = dataset)')
    a = ap.parse_args()
    ds = a.dataset
    scratch = os.environ.get('SCRATCH', '')

    expected = set(range(N_REAL[ds]))
    if a.realizations:
        lo, _, hi = a.realizations.partition('-')
        expected = set(range(int(lo), int(hi or lo) + 1))
    wanted = [m for m in MODELS if not a.models or m[0] in a.models.split(',')]
    if not wanted:
        raise SystemExit(f'no model matches --models {a.models!r}')

    files = {}
    for label, fmt, key, mode in wanted:
        root = getattr(a, f'root_{key}').format(SCRATCH=scratch, ds=ds)
        found = _realization_files(root)
        missing = sorted(expected - set(found))
        if missing:
            raise SystemExit(f'{label}: {len(missing)} realizations missing under {root}: '
                             f'{missing[:10]}')
        if not a.realizations and set(found) != expected:
            raise SystemExit(f'{label}: unexpected realizations under {root}: '
                             f'{sorted(set(found) - expected)[:10]}')
        files[label] = (fmt, mode, found)

    truths = {r: load_truth_raw(ds, r, a.causalpfn, a.acic_cache) for r in sorted(expected)}
    tasks = [(label, fmt, mode, found[r], r, truths[r], ds)
             for label, (fmt, mode, found) in files.items() for r in sorted(expected)]
    tasks.sort(key=lambda t: t[1].endswith('2d'), reverse=True)   # slow joints first
    print(f'[nll] {ds}: {len(wanted)} models x {len(expected)} realizations = '
          f'{len(tasks)} tasks, {a.workers} workers', flush=True)

    rows, t0 = [], time.time()

    def _consume(it):
        for i, row in enumerate(it, 1):
            rows.append(row)
            print(f'[{i}/{len(tasks)}] {row["model"]:15s} r{row["realization"]:03d} '
                  f'Ly={row["L_y0y1"]:.4f} Lt={row["L_tau"]:.4f} La={row["L_ate"]:.4f} '
                  f'({row["seconds"]}s, {time.time() - t0:.0f}s total)', flush=True)
    if a.workers > 1:
        with mp.Pool(a.workers) as pool:
            _consume(pool.imap_unordered(_work, tasks))
    else:
        _consume(map(_work, tasks))

    import pandas as pd
    order = {m[0]: i for i, m in enumerate(MODELS)}
    df = pd.DataFrame(rows)
    df['_o'] = df.model.map(order)
    df = df.sort_values(['_o', 'realization']).drop(columns='_o')
    cols = ['model', 'realization', 'L_y0y1', 'L_tau', 'L_ate', 'Q']
    df = df[cols + [c for c in df.columns if c not in cols]]
    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
    df.to_csv(a.out_csv, index=False)
    table = summarize(df, [m[0] for m in wanted])
    head = (f'### {ds}: NLL in raw outcome units, mean ± SE over '
            f'{len(expected)} realizations')
    print('\n' + head + '\n' + table, flush=True)
    if a.out_md:
        with open(a.out_md, 'a') as fh:
            fh.write(head + '\n\n' + table + '\n\n')


if __name__ == '__main__':
    main()
