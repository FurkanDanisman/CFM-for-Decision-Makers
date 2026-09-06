"""Tier-C density eval: p(tau | x) for UWYK-1D vs Joint-2D on IHDP / ACIC.

v1 = RAW path only (no MALC). See density_eval_pipeline.md for the full plan;
the MALC arm reuses everything here with one substitution inside region 0.

What this produces, per realization, per method:
    nll     -log p(tau*)         at the OBSERVED tau* (proper scoring rule)
    l2      ||p_true - p_est||_2 on TAU_CENTERS
    kl_fwd  KL(truth || est)     ~ NLL up to the truth's entropy
    kl_rev  KL(est || truth)     the one carrying independent information
    mass    int p_est dtau       diagnostic: how much sits on the tau grid

Methods (rows):
    uwyk_native   UWYK's K=1000 bars, convolved under independence
    uwyk_matched  UWYK re-binned to the 2D head's J bins, then convolved
                  -- the resolution-matched control
    joint         the 2D head's own joint, diagonal-integrated

`uwyk_*` rows are 'UWYK (x) indep': UWYK emits no joint, the independence
assumption is ours. Never label them plain 'UWYK'.

Both models are fed byte-identical context, preprocessing, adjacency and y
axis by importing eval_graph2d_realcause as the harness. Full densities with
tails on both sides -- see density_common for why truncate-and-renormalise is
not an option here.

Usage (GPU node):
    CKPT=<graph2d ckpt>  UWYK_CKPT=<uwyk best_model.pt>  UWYK_CFG=<yaml>
    UWYK=$PWD/g4cfm  CAUSALPFN=/path/to/CausalPFN
    DATASET=IHDP  ANC_TAG=v6a  OUT=./results_density_tauC/IHDP
    python -u benchmarks/eval_graph2d/eval_density_tauC.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))

# The harness reads env at import time, so everything must already be set.
os.environ.setdefault('ANC_MODE', 'v6a_only')
_spec = importlib.util.spec_from_file_location(
    '_tauC_harness', os.path.join(_HERE, 'eval_graph2d_realcause.py'))
H = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = H
_spec.loader.exec_module(H)

sys.path.insert(0, os.environ['UWYK'])
sys.path.insert(0, os.environ['UWYK'] + '/src')
sys.path.insert(0, _HERE)
from models.GraphConditionedInterventionalPFN_sklearn import (      # noqa: E402
    GraphConditionedInterventionalPFNSklearn,
)
from density_common import (                                        # noqa: E402
    Joint2D, UWYK1D, joint_tau_density, uwyk_tau_density,
    truth_tau_density, l2_distance, kl, mass, TAU_CENTERS,
)

DATASET = H.DATASET
ANC_TAG = os.environ.get('ANC_TAG', 'v6a')
OUT = os.environ.get('OUT', f'./results_density_tauC/{DATASET}')
UWYK_CKPT = os.environ['UWYK_CKPT']
UWYK_CFG = os.environ['UWYK_CFG']
QUERY_CHUNK = int(os.environ.get('QUERY_CHUNK', '512'))
# Tail-quadrature resolution. 2048 measured at ~0.64 s/query with int p(tau)
# within 3e-4 of 1; 1024 is 2.3x faster but drifts to 3e-3. Only the 8 smooth
# tail regions use quadrature -- the interior is closed-form.
N_Y0 = int(os.environ.get('N_Y0', '2048'))
# Realization slice, so a Slurm array can split the work. Both models stay in
# ONE process per slice: splitting by model instead would only give 2x and
# would rest on both processes deriving byte-identical preprocessing.
REAL_START = int(os.environ.get('REAL_START', '0'))
REAL_END = os.environ.get('REAL_END')            # exclusive; None = all
ACIC_CACHE = (os.environ.get('ACIC_CACHE_DIR')
              or os.environ.get('ACIC_CACHE')
              or os.path.join(_REPO, 'data', 'acic_cache'))


# ---------------------------------------------------------------------------
# Observed potential outcomes (tau* targets). CausalPFN's loaders keep only
# `true_cate`, so both datasets are read from their source files directly --
# the same thing true_ihdp.py / true_acic.py already do for mu0/mu1.
# ---------------------------------------------------------------------------
def observed_y0_y1(dataset: str, r: int):
    if dataset == 'IHDP':
        d = np.load(os.path.join(os.environ['CAUSALPFN'], 'benchmarks', 'IHDP',
                                 'ihdp_npci_1-100.test.npz'))
        t = d['t'][..., r].reshape(-1)
        yf = d['yf'][..., r].reshape(-1)
        ycf = d['ycf'][..., r].reshape(-1)
        # IHDPDataset builds X_test straight from the npz with no permutation,
        # so npz row order == cd.X_test row order.
        return (np.where(t == 0, yf, ycf).astype(np.float64),
                np.where(t == 1, yf, ycf).astype(np.float64))
    if dataset == 'ACIC':
        sys.path.insert(0, os.path.join(_REPO, 'benchmarks'))
        from l2_acic.true_acic import _load_zy_frame          # noqa
        sim = _load_zy_frame(r, cache_dir=ACIC_CACHE)
        sim.columns = ['z', 'y0', 'y1', 'mu0', 'mu1']
        n = len(sim)
        # Must match true_acic.py::load_acic_truth byte-for-byte, or y0/y1 and
        # mu0/mu1 refer to different units.
        perm = np.random.default_rng(42 + r).permutation(n)
        test_idx = perm[int(n * (1 - 0.1)):]
        return (sim['y0'].values[test_idx].astype(np.float64),
                sim['y1'].values[test_idx].astype(np.float64))
    raise ValueError(f'Tier C v1 covers IHDP and ACIC only, got {dataset}')


def load_truth(dataset: str, r: int, y_train_ctx: np.ndarray):
    """mu0, mu1, sigma on the scaled axis.

    y_train_ctx MUST be the post-subsample training y the harness handed to
    _scale_y. IHDP's 672 training rows sit under EVAL_MAX_CONTEXT=1000 so the
    cap is a no-op there, but ACIC's ~4.3k rows are subsampled -- passing the
    full training y instead would put truth and model on different axes.
    """
    sys.path.insert(0, os.path.join(_REPO, 'benchmarks'))
    if dataset == 'IHDP':
        from l2_ihdp.true_ihdp import load_ihdp_truth
        t = load_ihdp_truth(r, os.environ['CAUSALPFN'], y_train_ctx)
    else:
        from l2_acic.true_acic import load_acic_truth
        t = load_acic_truth(r, y_train_ctx, cache_dir=ACIC_CACHE)
    return t.mu0_test_scaled, t.mu1_test_scaled, float(t.sigma_scaled)


# ---------------------------------------------------------------------------
# UWYK raw head output
# ---------------------------------------------------------------------------
def capture_raw_preds(w, X_obs, T_obs, Y_obs, X_intv, T_intv, adj, n_test):
    """The (n_test, K+4) raw BarDistribution parameters.

    Ported off PreprocessingGraphConditionedPFN (methods_densities.py:659),
    which passes inverse_transform=False -- an argument the BASE wrapper's
    predict does not accept. Same monkey-patch on bar_distribution.mode to
    grab the tensor before postprocessing; the mode value is discarded.
    """
    cap = {'raw': None}
    orig = w.bar_distribution.mode

    def _grab(raw_preds):
        cap['raw'] = raw_preds.detach().cpu()
        return orig(raw_preds)

    w.bar_distribution.mode = _grab
    try:
        w.predict(X_obs=X_obs, T_obs=T_obs, Y_obs=Y_obs,
                  X_intv=X_intv, T_intv=T_intv,
                  adjacency_matrix=adj, prediction_type='mode')
    finally:
        w.bar_distribution.mode = orig
    if cap['raw'] is None:
        raise RuntimeError('bar_distribution.mode was never called -- the '
                           'wrapper took a non-BarDistribution path')
    return cap['raw'].squeeze(0).double().numpy()[:n_test]


def uwyk_preds_chunked(w, X_tr, T_tr, Y_tr, X_te, t_val, adj):
    out = []
    for s in range(0, X_te.shape[0], QUERY_CHUNK):
        Xq = X_te[s:s + QUERY_CHUNK]
        Tq = np.full((Xq.shape[0], 1), t_val, dtype=np.float32)
        out.append(capture_raw_preds(w, X_tr, T_tr, Y_tr, Xq, Tq, adj,
                                     Xq.shape[0]))
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
def score(p_est, p_true, tau_star_density, tau_grid=TAU_CENTERS):
    """One method, one query. NLL is taken at the observed tau*, not read off
    the grid, so it never inherits the grid's interpolation error."""
    return dict(
        nll=float(-np.log(tau_star_density)),
        l2=l2_distance(p_true, p_est, tau_grid),
        kl_fwd=kl(p_true, p_est, tau_grid),
        kl_rev=kl(p_est, p_true, tau_grid),
        mass=mass(p_est, tau_grid),
    )


def evaluate(r, ds, model2d, J, edges2d, uwyk, F):
    cate = ds[r][0]
    X_tr_raw = np.asarray(cate.X_train, dtype=np.float32)
    T_tr = np.asarray(cate.t_train, dtype=np.float32).reshape(-1)
    y_tr_raw = np.asarray(cate.y_train, dtype=np.float32).reshape(-1)
    X_te_raw = np.asarray(cate.X_test, dtype=np.float32)

    if H.EVAL_MAX_CONTEXT:
        cap = int(H.EVAL_MAX_CONTEXT)
        if X_tr_raw.shape[0] > cap:
            rng = np.random.default_rng(H.EVAL_CONTEXT_SEED + r)
            idx = rng.choice(X_tr_raw.shape[0], cap, replace=False)
            X_tr_raw, T_tr, y_tr_raw = X_tr_raw[idx], T_tr[idx], y_tr_raw[idx]

    n_real = min(X_tr_raw.shape[1], F)
    X_tr_std, X_te_std = H._standardize_train_test(X_tr_raw, X_te_raw)
    X_tr = H._pad_features(X_tr_std, F)
    X_te = H._pad_features(X_te_std, F)
    y_scaled, ymin, yrange = H._scale_y(y_tr_raw)
    Y_obs = y_scaled.reshape(-1, 1)
    T_feed = T_tr.astype(np.float32).reshape(-1, 1)      # binary: matched to 2D

    adj = dict(H.build_mode_list(F, n_real))[ANC_TAG]

    # -- targets and truth, on the axis _scale_y just defined -------------
    y0_raw, y1_raw = observed_y0_y1(DATASET, r)
    to_scaled = lambda y: (2.0 * (y - ymin) / yrange - 1.0)
    tau_star = to_scaled(y1_raw) - to_scaled(y0_raw)     # offsets cancel
    mu0, mu1, sigma = load_truth(DATASET, r, y_tr_raw)

    # ALIGNMENT GUARD. observed_y0_y1 and load_truth both index the source
    # files directly and each reconstructs the test split on its own -- IHDP by
    # assuming npz row order survives into cd.X_test, ACIC by replaying the
    # rng. If either is off, rows silently refer to different units and every
    # number downstream is quietly wrong. cd.true_cate is mu1-mu0 in raw units,
    # so it pins the ordering against the truth loader.
    cate_from_truth = (mu1 - mu0) * (yrange / 2.0)
    true_cate = np.asarray(cate.true_cate, dtype=np.float64).reshape(-1)
    if cate_from_truth.shape != true_cate.shape:
        raise RuntimeError(
            f'r={r}: truth has {cate_from_truth.shape} rows, dataset has '
            f'{true_cate.shape} -- test splits disagree')
    misalign = float(np.abs(cate_from_truth - true_cate).max())
    if misalign > 1e-3 * max(np.abs(true_cate).max(), 1e-9):
        raise RuntimeError(
            f'r={r}: truth mu1-mu0 does not match cd.true_cate (max abs diff '
            f'{misalign:.3e}) -- test-row ordering is misaligned between '
            f'{DATASET}\'s loader and the truth/potential-outcome readers')
    if len(y0_raw) != true_cate.size:
        raise RuntimeError(
            f'r={r}: {len(y0_raw)} potential-outcome rows vs {true_cate.size} '
            f'test queries')

    # -- Joint-2D: one forward pass ---------------------------------------
    _, _, logits, _ = H.marginals_from_forward(
        model2d, X_tr, T_feed.reshape(-1), Y_obs, X_te, adj, J)

    # -- UWYK-1D: two forward passes --------------------------------------
    pred0 = uwyk_preds_chunked(uwyk, X_tr, T_feed, Y_obs, X_te, 0.0, adj)
    pred1 = uwyk_preds_chunked(uwyk, X_tr, T_feed, Y_obs, X_te, 1.0, adj)
    bd = uwyk.bar_distribution
    bar_edges = bd.edges.detach().cpu().double().numpy()
    bar_widths = bd.widths.detach().cpu().double().numpy()
    base_sL = float(bd.base_s_left)
    base_sR = float(bd.base_s_right)

    n_q = X_te.shape[0]
    rows = {m: [] for m in ('uwyk_native', 'uwyk_matched', 'joint')}
    for q in range(n_q):
        p_true = truth_tau_density(mu0[q], mu1[q], sigma, TAU_CENTERS)
        t_star = np.array([tau_star[q]])
        d_true = float(truth_tau_density(mu0[q], mu1[q], sigma, t_star)[0])

        f0 = UWYK1D.from_pred(pred0[q], bar_edges, bar_widths, base_sL, base_sR)
        f1 = UWYK1D.from_pred(pred1[q], bar_edges, bar_widths, base_sL, base_sR)
        jt = Joint2D.from_pred(logits[q], J, edges2d)

        for name, fn in (
            ('uwyk_native', lambda: (
                uwyk_tau_density(f0, f1, TAU_CENTERS, n_y0=N_Y0),
                uwyk_tau_density(f0, f1, t_star, n_y0=N_Y0)[0])),
            ('uwyk_matched', lambda: (
                uwyk_tau_density(f0.rebin(edges2d), f1.rebin(edges2d),
                                 TAU_CENTERS, n_y0=N_Y0),
                uwyk_tau_density(f0.rebin(edges2d), f1.rebin(edges2d),
                                 t_star, n_y0=N_Y0)[0])),
            ('joint', lambda: (
                joint_tau_density(jt, TAU_CENTERS, n_y0=N_Y0),
                joint_tau_density(jt, t_star, n_y0=N_Y0)[0])),
        ):
            p_grid, d_star = fn()
            rows[name].append(score(p_grid, p_true, d_star))

    out = {'dataset': DATASET, 'realization': r, 'n_queries': n_q,
           'n_context': int(X_tr_raw.shape[0]), 'anc_tag': ANC_TAG,
           'sigma_scaled': sigma,
           'frac_tau_outside_grid': float(np.mean(np.abs(tau_star) > 3.0))}
    for name, rr in rows.items():
        for k in rr[0]:
            out[f'{k}_{name}'] = float(np.mean([x[k] for x in rr]))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f'[tauC] dataset={DATASET} anc={ANC_TAG} '
          f'ctx={H.EVAL_MAX_CONTEXT or "(full)"} n_y0={N_Y0}', flush=True)

    ds = H.get_dataset(DATASET)
    model2d, cfg = H.load_model(os.environ['CKPT'])
    J = int(cfg['J'])
    edges2d = torch.load(os.environ['CKPT'], map_location='cpu',
                         weights_only=False)['edges'].double().numpy()
    assert len(edges2d) == J + 1, f'edges {len(edges2d)} vs J+1 {J+1}'

    uwyk = GraphConditionedInterventionalPFNSklearn(
        config_path=UWYK_CFG, checkpoint_path=UWYK_CKPT, verbose=True)
    uwyk.load()
    F = uwyk.model.num_features
    assert F == cfg['num_features'], (
        f'feature caps differ: uwyk {F} vs joint {cfg["num_features"]}')
    print(f'[tauC] J={J} bw={edges2d[1]-edges2d[0]:.4f}   '
          f'UWYK K={len(uwyk.bar_distribution.centers)} '
          f'bw={float(uwyk.bar_distribution.widths[0]):.5f}', flush=True)

    lo = max(0, REAL_START)
    hi = ds.n_tables if REAL_END is None else min(ds.n_tables, int(REAL_END))
    print(f'[tauC] realizations [{lo}, {hi}) of {ds.n_tables}', flush=True)
    t0 = time.time()
    for r in range(lo, hi):
        row = evaluate(r, ds, model2d, J, edges2d, uwyk, F)
        np.savez(os.path.join(OUT, f'{DATASET}_r{r:03d}.npz'),
                 **{k: np.array(v) for k, v in row.items()})
        print(f'r={r:03d}  ' + '  |  '.join(
            f'{m}: nll={row[f"nll_{m}"]:7.3f} l2={row[f"l2_{m}"]:6.3f} '
            f'klrev={row[f"kl_rev_{m}"]:7.4f}'
            for m in ('uwyk_native', 'uwyk_matched', 'joint'))
            + f'   ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
