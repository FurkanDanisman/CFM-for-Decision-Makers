"""Tier-C density eval: p(tau | x) for UWYK and/or DoPFN on IHDP / ACIC.

v1 = RAW path only (no MALC). See density_eval_pipeline.md for the full plan;
the MALC arm reuses everything here with one substitution inside region 0.

What this produces, per realization, per method:
    nll     -log p(tau*)         at the OBSERVED tau* (proper scoring rule)
    l2      ||p_true - p_est||_2 on TAU_CENTERS
    kl_fwd  KL(truth || est)     ~ NLL up to the truth's entropy
    kl_rev  KL(est || truth)     the one carrying independent information
    mass    int p_est dtau       diagnostic: how much sits on the tau grid
    pehe / cate_l1 / ate_abs_err  point errors of the full-density mean,
                                 in original outcome units

Also scores the joint's interior-only mean (joint_inner). All point estimates
come from the SAME logits used for density scoring, with no extra forwards.
SAVE_PREDICTIONS=1 (default) writes logits, axes and truth under OUT/predictions
so subsequent numerical checks can run on CPU without either checkpoint.
Truth uses the documented generator noise sigma=1 in raw outcome units.
Full-training factual residual scales are saved as diagnostics only. Both
minmax and std truth scaling follow the model's actual context transform.

Methods (rows):
    uwyk_native   UWYK's K=1000 bars, convolved under independence
    uwyk_matched  UWYK re-binned to the 2D head's J bins, then convolved
                  -- the resolution-matched control
    joint         the 2D head's own joint, diagonal-integrated
    dopfn_native  DoPFNRegressor.predict_full, convolved under independence
    dopfn_joint   DoPFN backbone with the trained 2D head, diagonal-integrated

MODEL_FAMILY=uwyk (default), dopfn, or all selects model pairs. DoPFN needs
DOPFN_ROOT (upstream checkout with artifacts); DOPFN_JOINT_CKPT defaults to
Required_checkpoints/dopfn_bb_j10_step_150000.pt. DOPFN_QUERY_CHUNK defaults
to 20. DoPFN-only runs do not load UWYK checkpoints. See README.md here.

`uwyk_*` rows are 'UWYK (x) indep': UWYK emits no joint, the independence
assumption is ours. Never label them plain 'UWYK'.

The UWYK models share context, feature preprocessing and the y axis by importing
eval_graph2d_realcause as the harness. DoPFN shares the selected context and
scoring axis, retains all covariates, and uses its native preprocessing for
the 1D model. UWYK's wrapper additionally propagates
the supplied adjacency; prediction dumps record both matrices. Full densities with
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
# COPY of benchmarks/eval_graph2d/eval_density_tauC.py, relocated under
# case_study/. The originals there are READ-ONLY for this folder and are never
# modified. Only two files were copied (this one + density_common.py); the
# remaining siblings this runner needs -- eval_graph2d_realcause.py (the
# harness) and density_truth.py -- are still imported from the original
# directory via _ORIG. _HERE is inserted AFTER _ORIG so it takes precedence on
# sys.path: `density_common` resolves to the LOCAL copy, everything else falls
# through to the originals.
#
# RE-SYNCING: this file is a snapshot. If the original changes, re-copy it and
# re-apply the three hunks below (_ORIG definition, the harness path, and the
# sys.path order). `diff` against the original should show ONLY those three.
_ORIG = os.path.join(_REPO, 'benchmarks', 'eval_graph2d')
if not os.path.isdir(_ORIG):
    raise RuntimeError(f'original eval_graph2d dir not found: {_ORIG}')
MODEL_FAMILY = os.environ.get('MODEL_FAMILY', 'uwyk').lower()
if MODEL_FAMILY not in ('uwyk', 'dopfn', 'all'):
    raise ValueError('MODEL_FAMILY must be uwyk, dopfn, or all')
USE_UWYK = MODEL_FAMILY in ('uwyk', 'all')
USE_DOPFN = MODEL_FAMILY in ('dopfn', 'all')
# The shared harness imports graph definitions even for a DoPFN-only run.
# These defaults locate source code; no UWYK checkpoint is loaded in that mode.
os.environ.setdefault('UWYK', os.path.join(_REPO, 'g4cfm'))
os.environ.setdefault('CKPT', os.path.join(_REPO, 'Required_checkpoints', 'graph2d_step_50000.pt'))

# The harness reads env at import time, so everything must already be set.
_spec = importlib.util.spec_from_file_location(
    '_tauC_harness', os.path.join(_ORIG, 'eval_graph2d_realcause.py'))
H = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = H
_spec.loader.exec_module(H)

sys.path.insert(0, os.environ['UWYK'])
sys.path.insert(0, os.environ['UWYK'] + '/src')
sys.path.insert(0, _ORIG)    # density_truth.py and other siblings
sys.path.insert(0, _HERE)    # LOCAL density_common.py wins over the original
from models.GraphConditionedInterventionalPFN_sklearn import (      # noqa: E402
    GraphConditionedInterventionalPFNSklearn,
)
from density_common import (                                        # noqa: E402
    Joint2D, UWYK1D, joint_tau_density, uwyk_tau_density, dopfn_tau_density,
    truth_tau_density, l2_distance, kl, mass, point_metrics, TAU_CENTERS,
)
from density_truth import harness_y_affine, load_density_truth        # noqa: E402

DATASET = H.DATASET
# This eval conditions on exactly ONE adjacency, so ANC_TAG is the only knob.
# The harness's ANC_MODE picks a whole family for its sweep (v6a_only ->
# v6a+noanc, full -> anc+noanc, ...) and a tag maps to the same builder in
# every family that emits it, so the family carries no information we need --
# we just find one that contains the tag. ANC_MODE is deliberately NOT read:
# it was a second name for the same choice and could silently disagree.
# Tags don't depend on F/n_real, so resolve on a dummy (4, 2) now and fail
# here instead of 20 min in, after the checkpoints and the dataset have loaded.
ANC_TAG = os.environ.get('ANC_TAG', 'v6a') if USE_UWYK else 'none'
_FAMILIES = ['full', 'v6a_only', 'v6b_only', 'v4a_only', 'v5a_only',
             'v5b_only', 'v3b_only', 'ty_only', 'ty_antisym', 'all_variants',
             'v3_family', 'v3_v6_extended', 'focus4', 'three_edge_all',
             'all_combos']
if os.environ.get('ANC_VARIANT'):   # SCM case studies: the tag IS the variant
    _FAMILIES.append('case_variant')
ANC_FAMILY = (next(
    (f for f in _FAMILIES if ANC_TAG in dict(H.build_mode_list(4, 2, f))), None)
    if USE_UWYK else None)
if USE_UWYK and ANC_FAMILY is None:
    _common = sorted({t for f in ('full', 'all_variants', 'v3_v6_extended',
                                  'focus4', 'ty_only', 'ty_antisym')
                      for t, _ in H.build_mode_list(4, 2, f)})
    raise SystemExit(
        f'[tauC] ANC_TAG={ANC_TAG!r} names no adjacency the harness builds. '
        f'Named tags: {_common}, plus the PNB/PNBO codes of three_edge_all '
        f'and all_combos. Usual choices: v6a (no +1 edges, only the -1s '
        f'implied by unconfoundedness) | anc (build_anc_full: T->Y, X_i->T, '
        f'X_i->Y asserted +1) | noanc (all-zero).')
if USE_UWYK and os.environ.get('ANC_MODE'):
    print(f'[tauC] note: ANC_MODE={os.environ["ANC_MODE"]!r} is ignored here; '
          f'ANC_TAG={ANC_TAG!r} alone selects the adjacency '
          f'(resolved via {ANC_FAMILY!r}).', flush=True)
OUT = os.environ.get('OUT', f'./results_density_tauC/{DATASET}')
UWYK_CKPT = os.environ['UWYK_CKPT'] if USE_UWYK else ''
UWYK_CFG = os.environ['UWYK_CFG'] if USE_UWYK else ''
DOPFN_ROOT = os.environ.get('DOPFN_ROOT', os.environ.get('DOPFN', ''))
DOPFN_JOINT_CKPT = os.environ.get('DOPFN_JOINT_CKPT', os.path.join(
    _REPO, 'Required_checkpoints', 'dopfn_bb_j10_step_150000.pt'))
DOPFN_QUERY_CHUNK = int(os.environ.get('DOPFN_QUERY_CHUNK', '20'))
if USE_DOPFN and not DOPFN_ROOT:
    raise ValueError('Set DOPFN_ROOT to the DoPFN checkout containing scripts/ and artifacts/')
METHODS = (('uwyk_native', 'uwyk_matched', 'joint') if USE_UWYK else ()) + (
    ('dopfn_native', 'dopfn_joint') if USE_DOPFN else ())
QUERY_CHUNK = int(os.environ.get('QUERY_CHUNK', '512'))
SAVE_PREDICTIONS = os.environ.get('SAVE_PREDICTIONS', '1') == '1'
# y0-quadrature resolution for the 8 TAIL regions only; the interior is
# closed-form and free at any tau resolution. Measured on the 12001-point
# knot-aligned tau grid with midpoint quadrature:
#   n_y0=1024  0.14 s/query  int p(tau)=0.99922  KL err 8.1e-4
#   n_y0=4096  0.72 s/query  int p(tau)=0.99990  KL err 1.0e-4
# 4096 costs about what the OLD 600-point grid did and is 1e-4 accurate, which
# is ~0.03% of the 0.38-0.70 nat effects being measured.
N_Y0 = int(os.environ.get('N_Y0', '4096'))
# Realization slice, so a Slurm array can split the work. Both models stay in
# ONE process per slice: splitting by model instead would only give 2x and
# would rest on both processes deriving byte-identical preprocessing.
REAL_START = int(os.environ.get('REAL_START', '0'))
REAL_END = os.environ.get('REAL_END')            # exclusive; None = all
ACIC_CACHE = (os.environ.get('ACIC_CACHE_DIR')
              or os.environ.get('ACIC_CACHE')
              or os.path.join(_REPO, 'data', 'acic_cache'))


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


def evaluate(r, ds, model2d, J, edges2d, uwyk, F, dopfn=None):
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

    X_tr_std, X_te_std = H._standardize_train_test(X_tr_raw, X_te_raw)
    if uwyk is not None:
        n_real = min(X_tr_raw.shape[1], F)
        X_tr = H._pad_features(X_tr_std, F)
        X_te = H._pad_features(X_te_std, F)
    y_scaled, y_offset, y_span = H._scale_y(y_tr_raw)
    y_shift, y_scale = harness_y_affine(y_offset, y_span, H.Y_SCALING)
    Y_obs = y_scaled.reshape(-1, 1)
    T_feed = T_tr.astype(np.float32).reshape(-1, 1)      # binary: matched to 2D

    adj = (dict(H.build_mode_list(F, n_real, ANC_FAMILY))[ANC_TAG]
           if uwyk is not None else None)

    # -- targets and truth, on the axis _scale_y just defined -------------
    truth = load_density_truth(DATASET, r, y_shift=y_shift, y_scale=y_scale,
                               causalpfn_dir=H.CAUSALPFN,
                               acic_cache_dir=ACIC_CACHE)
    mu0, mu1, sigma = truth.mu0_scaled, truth.mu1_scaled, truth.sigma_scaled
    tau_star = truth.tau_star_scaled
    truth_metadata = truth.noise_metadata()

    # The truth reader loads means and paired outcomes in a single test order.
    # Check that order against the model dataset's raw-unit true_cate.
    cate_from_truth = (mu1 - mu0) * y_scale
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
    if len(tau_star) != true_cate.size:
        raise RuntimeError(
            f'r={r}: {len(tau_star)} potential-outcome rows vs {true_cate.size} '
            f'test queries')

    prediction_fields = {}
    if uwyk is not None:
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

        adj_uwyk = uwyk._preprocess_adjacency_matrix(
            torch.from_numpy(adj).unsqueeze(0).to(uwyk.device)).cpu().numpy()[0]
        prediction_fields.update(
            joint_logits=logits, uwyk_pred0=pred0, uwyk_pred1=pred1,
            J=J, edges2d=edges2d, bar_edges=bar_edges, bar_widths=bar_widths,
            base_sL=base_sL, base_sR=base_sR,
            adj_joint=adj, adj_uwyk=adj_uwyk,
            ckpt=H.CKPT, uwyk_ckpt=UWYK_CKPT, uwyk_cfg=UWYK_CFG)
    if dopfn is not None:
        dopfn_arms, dopfn_logits, dopfn_dump = dopfn.predict(
            X_tr_raw, T_tr, y_tr_raw, X_te_raw,
            X_tr_std, y_scaled, X_te_std, y_shift=y_shift, y_scale=y_scale)
        prediction_fields.update(dopfn_dump)

    if SAVE_PREDICTIONS:
        prediction_dir = os.path.join(OUT, 'predictions')
        os.makedirs(prediction_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(prediction_dir, f'{DATASET}_r{r:03d}.npz'),
            dataset=DATASET, realization=r, anc_tag=ANC_TAG,
            **prediction_fields, model_family=MODEL_FAMILY,
            y_scale=y_scale, y_shift=y_shift,
            true_cate=true_cate, mu0_scaled=mu0, mu1_scaled=mu1,
            tau_star_scaled=tau_star, **truth_metadata,
            n_context=X_tr_raw.shape[0], context_seed=H.EVAL_CONTEXT_SEED + r,
            y_scaling=H.Y_SCALING, std_target=H.STD_TARGET,
            x_clip_quantile=H.X_CLIP_QUANTILE,
            bias_edge_scale=H.BIAS_EDGE_SCALE, t_intv_override=H.T_INTV_OVERRIDE,
            n_y0=N_Y0, tau_grid=TAU_CENTERS,
        )

    n_q = X_te_raw.shape[0]
    methods = (('uwyk_native', 'uwyk_matched', 'joint') if uwyk is not None else ()) + (
        ('dopfn_native', 'dopfn_joint') if dopfn is not None else ())
    rows = {m: [] for m in methods}
    inner_methods = tuple(m + '_inner' for m in ('joint', 'dopfn_joint') if m in methods)
    cate_means = {m: [] for m in (*rows, *inner_methods)}
    grid_means = {m: [] for m in rows}
    for q in range(n_q):
        p_true = truth_tau_density(mu0[q], mu1[q], sigma, TAU_CENTERS)
        t_star = np.array([tau_star[q]])

        scorers = []
        if uwyk is not None:
            f0 = UWYK1D.from_pred(pred0[q], bar_edges, bar_widths, base_sL, base_sR)
            f1 = UWYK1D.from_pred(pred1[q], bar_edges, bar_widths, base_sL, base_sR)
            jt = Joint2D.from_pred(logits[q], J, edges2d)
            f0_matched, f1_matched = f0.rebin(edges2d), f1.rebin(edges2d)
            m0, m1 = jt.mean()
            inner0, inner1 = jt.inner_mean()
            cate_means['joint'].append(m1 - m0)
            cate_means['joint_inner'].append(inner1 - inner0)
            cate_means['uwyk_native'].append(f1.mean() - f0.mean())
            cate_means['uwyk_matched'].append(f1_matched.mean() - f0_matched.mean())

            scorers.extend((
                ('uwyk_native', lambda: (
                    uwyk_tau_density(f0, f1, TAU_CENTERS, n_y0=N_Y0),
                    uwyk_tau_density(f0, f1, t_star, n_y0=N_Y0)[0])),
                ('uwyk_matched', lambda: (
                    uwyk_tau_density(f0_matched, f1_matched,
                                     TAU_CENTERS, n_y0=N_Y0),
                    uwyk_tau_density(f0_matched, f1_matched,
                                     t_star, n_y0=N_Y0)[0])),
                ('joint', lambda: (
                    joint_tau_density(jt, TAU_CENTERS, n_y0=N_Y0),
                    joint_tau_density(jt, t_star, n_y0=N_Y0)[0])),
            ))
        if dopfn is not None:
            df0, df1 = dopfn_arms[0][q], dopfn_arms[1][q]
            djt = Joint2D.from_pred(dopfn_logits[q], dopfn.J, dopfn.edges)
            m0, m1 = djt.mean()
            i0, i1 = djt.inner_mean()
            cate_means['dopfn_native'].append(df1.mean() - df0.mean())
            cate_means['dopfn_joint'].append(m1 - m0)
            cate_means['dopfn_joint_inner'].append(i1 - i0)
            scorers.extend((
                ('dopfn_native', lambda: (
                    dopfn_tau_density(df0, df1, TAU_CENTERS),
                    dopfn_tau_density(df0, df1, t_star)[0])),
                ('dopfn_joint', lambda: (
                    joint_tau_density(djt, TAU_CENTERS, n_y0=N_Y0),
                    joint_tau_density(djt, t_star, n_y0=N_Y0)[0])),
            ))
        for name, fn in scorers:
            p_grid, d_star = fn()
            rows[name].append(score(p_grid, p_true, d_star))
            grid_means[name].append(mass(TAU_CENTERS * p_grid, TAU_CENTERS))

    out = {'dataset': DATASET, 'realization': r, 'n_queries': n_q,
           'n_context': int(X_tr_raw.shape[0]), 'anc_tag': ANC_TAG,
           'model_family': MODEL_FAMILY, 'methods': np.asarray(methods),
           **truth_metadata,
           'y_scale': y_scale, 'y_shift': y_shift,
           'y_scaling': H.Y_SCALING, 'std_target': H.STD_TARGET,
           'true_cate': true_cate,
           'frac_tau_outside_grid': float(np.mean(np.abs(tau_star) > 3.0))}
    for name, rr in rows.items():
        for k in rr[0]:
            out[f'{k}_{name}'] = float(np.mean([x[k] for x in rr]))
    for name, means in cate_means.items():
        for metric, value in point_metrics(means, true_cate, y_scale).items():
            out[f'{metric}_{name}'] = value
        out[f'cate_pred_{name}'] = np.asarray(means) * y_scale
        if name in grid_means:
            # Unnormalised integral over the finite tau grid, diagnostic only.
            # The reported PEHE uses exact full-density means above.
            out[f'grid_mean_max_abs_diff_{name}'] = float(np.max(np.abs(
                np.asarray(grid_means[name]) - means)) * y_scale)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    graph_desc = (f'UWYK_graph={ANC_TAG}' if USE_UWYK else 'UWYK_graph=n/a')
    if USE_DOPFN:
        graph_desc += ' DoPFN_graph=none'
    print(f'[tauC] dataset={DATASET} family={MODEL_FAMILY} {graph_desc} '
          f'ctx={H.EVAL_MAX_CONTEXT or "(full)"} n_y0={N_Y0}', flush=True)

    ds = H.get_dataset(DATASET)
    model2d = uwyk = J = edges2d = F = None
    if USE_UWYK:
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

    dopfn = None
    if USE_DOPFN:
        from density_dopfn import DoPFNDensityModels
        dopfn = DoPFNDensityModels(DOPFN_ROOT, DOPFN_JOINT_CKPT,
                                   H.DEVICE, DOPFN_QUERY_CHUNK)
        print(f'[tauC] DoPFN joint J={dopfn.J} ckpt={dopfn.checkpoint}', flush=True)

    lo = max(0, REAL_START)
    hi = ds.n_tables if REAL_END is None else min(ds.n_tables, int(REAL_END))
    print(f'[tauC] realizations [{lo}, {hi}) of {ds.n_tables}', flush=True)
    t0 = time.time()
    for r in range(lo, hi):
        row = evaluate(r, ds, model2d, J, edges2d, uwyk, F, dopfn)
        np.savez(os.path.join(OUT, f'{DATASET}_r{r:03d}.npz'),
                 **{k: np.array(v) for k, v in row.items()})
        print(f'r={r:03d}  ' + '  |  '.join(
            f'{m}: nll={row[f"nll_{m}"]:7.3f} l2={row[f"l2_{m}"]:6.3f} '
            f'klrev={row[f"kl_rev_{m}"]:7.4f} pehe={row[f"pehe_{m}"]:7.3f}'
            for m in METHODS)
            + f'  sigma_raw={row["sigma_raw"]:.3f}'
              f' residual_sigma_raw={row["sigma_residual_raw"]:.3f}'
              f'   ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
