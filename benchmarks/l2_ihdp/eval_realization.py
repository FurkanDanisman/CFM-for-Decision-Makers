"""Per-realization L2 evaluation on IHDP.

For one IHDP realization r, computes:
  1. Analytical true densities per query (marginals, CATE) and the true
     population ATE via the 2-Wasserstein barycenter of per-query CATEs.
  2. Predicted densities from each of the 4 methods:
       ours_fn50, ours_fn10, uwyk_noanc, dopfn
     on the common Y_CENTERS / TAU_CENTERS grids.
  3. Per-query L2 distances between predicted and true for p_y0, p_y1, p_tau.
  4. Per-realization L2 for the ATE density (predicted-vs-true barycenter).

Output: one .npz shard per realization at $OUT.r{r:03d}.npz.

Environment
-----------
  REPO             R-PFN repo root                   (auto-detected)
  CAUSALPFN        path to CausalPFN source repo
  DOPFN            path to Do-PFN source repo
  UWYK_SRC         path to Graphs4CausalFoundationModels/src
  UWYK_CKPT_DIR    UWYK No-Ancestral checkpoint dir
  CHECKPOINT50     Ours(fn=50) .pt file
  CHECKPOINT10     Ours(fn=10) .pt file
  OUT              output NPZ path prefix   (default: $REPO/l2_ihdp/out)
  REALIZATION      int, 0..99                 (default: 0)
  N_CONTEXT        max training-context size  (default: full training set)
  MALC_B           MALC bootstrap size        (default: 60)
  MALC_MAX_K       MALC mixture order cap     (default: 3)
  N_EVAL           MALC evaluation grid size  (default: 200)
  UWYK_N_SAMPLES   samples per query per arm  (default: 1024)
  METHODS          comma list, subset of {ours_fn50, ours_fn10, uwyk_noanc, dopfn}
                     default: all four
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
import types

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--realization', type=int,
                     default=int(os.environ.get('REALIZATION', 0)))
    ap.add_argument('--out', default=os.environ.get('OUT', 'l2_ihdp/out'))
    ap.add_argument('--n-context', type=int,
                     default=int(os.environ.get('N_CONTEXT', 0)))         # 0 = full
    ap.add_argument('--malc-B', type=int,
                     default=int(os.environ.get('MALC_B', 100)))
    ap.add_argument('--malc-max-K', type=int,
                     default=int(os.environ.get('MALC_MAX_K', 1)))
    ap.add_argument('--n-eval', type=int,
                     default=int(os.environ.get('N_EVAL', 200)))
    ap.add_argument('--uwyk-n-samples', type=int,
                     default=int(os.environ.get('UWYK_N_SAMPLES', 1024)))
    ap.add_argument('--methods', default=os.environ.get(
        'METHODS', 'ours_fn50,ours_fn10,uwyk_noanc,dopfn'))
    ap.add_argument('--restrict-features', type=int,
                     default=int(os.environ.get('RESTRICT_FEATURES', 0)),
                     help='If > 0, truncate IHDP covariates to the first N '
                          'columns for ALL methods, and compute truth on the '
                          'same restricted feature set (k-NN mixture).')
    ap.add_argument('--knn-K', type=int,
                     default=int(os.environ.get('KNN_K', 20)),
                     help='k-NN size for the restricted truth mixture')
    ap.add_argument('--repo', default=os.environ.get('REPO', ''))
    ap.add_argument('--causalpfn', default=os.environ.get('CAUSALPFN', ''))
    ap.add_argument('--dopfn', default=os.environ.get('DOPFN', ''))
    ap.add_argument('--uwyk-src', default=os.environ.get('UWYK_SRC', ''))
    ap.add_argument('--uwyk-ckpt-dir', default=os.environ.get('UWYK_CKPT_DIR', ''))
    ap.add_argument('--checkpoint50', default=os.environ.get('CHECKPOINT50', ''))
    ap.add_argument('--checkpoint10', default=os.environ.get('CHECKPOINT10', ''))
    ap.add_argument('--checkpoint-dopfn-bb',
                     default=os.environ.get('CHECKPOINT_DOPFN_BB', ''),
                     help='Ours model built on DoPFN backbone (PerFeatureTransformer)')
    ap.add_argument('--y-scaling', default=os.environ.get('Y_SCALING', 'min_max'),
                     choices=['min_max', 'std'],
                     help='Y-scaling scheme for ours_densities (only). min_max = legacy '
                          'per-task (y_min, y_rng); std = (y - mean)/(std/std_target) as '
                          'in eval_dopfn_bb_raw.py.')
    ap.add_argument('--std-target', type=float,
                     default=float(os.environ.get('STD_TARGET', 0.3)),
                     help='Target scaled sigma for --y-scaling std.')
    ap.add_argument('--marginals-from-2d', action='store_true',
                     default=(os.environ.get('MARGINALS_FROM_2D', '0') == '1'),
                     help='Take p(Y_do0), p(Y_do1) by marginalising the 2D-MALC '
                          'joint (same fit CATE uses) instead of running 1D MALC '
                          'on the raw marginals. Applies to all "ours_*" methods.')
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    assert all(m in {'ours_fn50', 'ours_fn10', 'uwyk_noanc', 'uwyk_anc', 'dopfn', 'ours_dopfn_bb'}
                for m in methods), f'bad methods: {methods}'
    if args.restrict_features > 0:
        allowed = {'dopfn', 'ours_fn10', 'ours_dopfn_bb'}
        dropped = [m for m in methods if m not in allowed]
        methods = [m for m in methods if m in allowed]
        if dropped:
            print(f'[restrict] --restrict-features > 0: dropped methods '
                  f'{dropped} (only {sorted(allowed)} tested in restricted mode)',
                  flush=True)

    if not args.repo:
        args.repo = _guess_repo()

    _require_paths(args, methods)

    # Path plumbing — mirror the sibling plot_ihdp_n10*.py convention.
    # Our own package name (`benchmarks`) collides with CausalPFN's `benchmarks`,
    # so we insert `l2_ihdp/` directly and import our modules unqualified.
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC', 'Optimal_Transport'))
    sys.path.insert(0, _here)

    from l2 import l2_distance
    from true_ihdp import (
        TAU_CENTERS, Y_CENTERS, load_ihdp_truth,
        true_ate_barycenter, true_cate_per_query, true_marginals_per_query,
    )
    from true_ihdp_restricted import (
        load_ihdp_restricted_truth,
        restricted_marginals_per_query, restricted_cate_per_query,
        restricted_ate_barycenter, induced_correlation,
    )
    from ot_barycenter import wasserstein_barycenter_1d
    from methods_densities import (
        dopfn_densities, ours_densities, uwyk_noanc_densities, uwyk_anc_densities,
    )

    # ── Load IHDP realization ───────────────────────────────────────────
    print(f'[start] realization {args.realization}  methods={methods}', flush=True)
    _install_dopfn_datasets_shim(args.dopfn)
    sys.path.insert(0, args.causalpfn)
    from benchmarks import IHDPDataset

    cd, _ = IHDPDataset()[args.realization]
    y_train_full = _np(cd.y_train)
    n_ctx = args.n_context if args.n_context > 0 else y_train_full.shape[0]

    # Optional feature restriction — truncate IHDP to the first N covariates
    # for ALL methods, and build truth on the same restricted feature set.
    if args.restrict_features > 0:
        X_train_arr = _np(cd.X_train).astype(np.float32)
        X_test_arr  = _np(cd.X_test).astype(np.float32)
        d_r = min(args.restrict_features, X_train_arr.shape[1])
        cd = _truncate_covariates(cd, d_r)
        print(f'[restrict] using first {d_r} of {X_train_arr.shape[1]} '
              f'covariates for all methods and truth', flush=True)

    # ── Truth ───────────────────────────────────────────────────────────
    if args.restrict_features > 0:
        truth = load_ihdp_restricted_truth(
            args.realization, args.causalpfn,
            X_train_arr, X_test_arr, y_train_full,
            d_restrict=args.restrict_features, K=args.knn_K,
        )
        n_queries = truth.mu0_neighbours_scaled.shape[0]
        induced = induced_correlation(truth)
        print(f'[truth] restricted r={args.realization}  n_queries={n_queries}  '
              f'sigma_scaled={truth.sigma_scaled:.4f}  d_r={truth.d_restrict}  '
              f'K={truth.K}  induced_rho: mean={induced.mean():+.3f} '
              f'median={np.median(induced):+.3f}', flush=True)
        p_y0_true, p_y1_true = restricted_marginals_per_query(truth)
        p_tau_true = restricted_cate_per_query(truth)
        p_ate_true = restricted_ate_barycenter(p_tau_true, wasserstein_barycenter_1d)
    else:
        truth = load_ihdp_truth(args.realization, args.causalpfn, y_train_full)
        n_queries = truth.mu0_test_scaled.shape[0]
        print(f'[truth] r={args.realization}  n_queries={n_queries}  '
              f'sigma_scaled={truth.sigma_scaled:.4f}', flush=True)
        p_y0_true, p_y1_true = true_marginals_per_query(truth)
        p_tau_true = true_cate_per_query(truth)
        p_ate_true = true_ate_barycenter(p_tau_true, wasserstein_barycenter_1d)

    # ── Method densities ────────────────────────────────────────────────
    # Order matters: UWYK's constructor loads its own `utils` and `models`
    # modules and pollutes sys.path/modules in ways that break subsequent
    # imports of Do-PFN (which has its own `utils` package with different
    # symbols). Run UWYK last so downstream imports never encounter its
    # cruft.
    RUN_ORDER = ['ours_fn50', 'ours_fn10', 'ours_dopfn_bb', 'dopfn', 'uwyk_noanc', 'uwyk_anc']
    method_out: dict[str, dict[str, np.ndarray]] = {}
    for m in RUN_ORDER:
        if m not in methods:
            continue
        if m == 'ours_fn50':
            method_out[m] = _run_ours(cd, args.checkpoint50, truth, args, n_ctx)
        elif m == 'ours_fn10':
            method_out[m] = _run_ours(cd, args.checkpoint10, truth, args, n_ctx)
        elif m == 'ours_dopfn_bb':
            method_out[m] = _run_ours_dopfn_bb(
                cd, args.checkpoint_dopfn_bb, truth, args, n_ctx)
        elif m == 'dopfn':
            method_out[m] = _run_dopfn(cd, truth, args, n_ctx)
        elif m == 'uwyk_noanc':
            method_out[m] = _run_uwyk_noanc(cd, truth, args, n_ctx)
        elif m == 'uwyk_anc':
            method_out[m] = _run_uwyk_anc(cd, truth, args, n_ctx)

    # ── Independence-assumption variant for Ours-DoPFN-bb ──────────────
    # Same p_y0/p_y1 marginals from the 2D joint, but re-derive p(τ) under
    # the independence assumption (naive convolution of marginals). Reveals
    # how much of Ours-DoPFN-bb's CATE-density fidelity comes from the joint
    # vs the marginal quality alone. Zero extra compute — reuses marginals.
    if 'ours_dopfn_bb' in method_out:
        from methods_densities import naive_p_tau_from_marginals
        _d = method_out['ours_dopfn_bb']
        _p_y0, _p_y1 = _d['p_y0'], _d['p_y1']
        _n = _p_y0.shape[0]
        method_out['ours_dopfn_bb_indep'] = dict(
            p_y0=_p_y0, p_y1=_p_y1,
            p_tau=np.stack([naive_p_tau_from_marginals(_p_y0[q], _p_y1[q])
                              for q in range(_n)]),
            # Point-estimate CATEs: use raw-mean (identical joint marginal means)
            # since independence doesn't change E[Y1] - E[Y0].
            cate_raw_scaled=_d.get('cate_raw_scaled'),
        )
        # Raw-marginal diagnostic: same p_tau as the MALC-smoothed main run,
        # but p_y0 / p_y1 come from the raw p_mat marginals (no MALC, no
        # log-linear interp). Lets us see whether MALC actually helps y0/y1
        # or if it's over-smoothing at J=10.
        if 'p_y0_raw' in _d and 'p_y1_raw' in _d:
            method_out['ours_dopfn_bb_rawmarg'] = dict(
                p_y0=_d['p_y0_raw'], p_y1=_d['p_y1_raw'],
                p_tau=_d['p_tau'],
                cate_raw_scaled=_d.get('cate_raw_scaled'),
            )
        # OLD-MALC diagnostic: MALC-1D-smoothed marginal but with linear-in-
        # prob interp (resample_onto) instead of log-linear evaluate.
        # Reproduces pre-log-linear-fix behaviour. Same p_tau as main.
        if 'p_y0_malc_old' in _d and 'p_y1_malc_old' in _d:
            method_out['ours_dopfn_bb_old'] = dict(
                p_y0=_d['p_y0_malc_old'], p_y1=_d['p_y1_malc_old'],
                p_tau=_d['p_tau'],
                cate_raw_scaled=_d.get('cate_raw_scaled'),
            )
        # MEAN-ADJUSTED MALC diagnostic: R's get_fhatn recipe (Beta jitter
        # to encode empirical-mean shift + refit log-concave). Same p_tau
        # as main; only y0/y1 marginals differ.
        if 'p_y0_malc_madj' in _d and 'p_y1_malc_madj' in _d:
            method_out['ours_dopfn_bb_madj'] = dict(
                p_y0=_d['p_y0_malc_madj'], p_y1=_d['p_y1_malc_madj'],
                p_tau=_d['p_tau'],
                cate_raw_scaled=_d.get('cate_raw_scaled'),
            )

    # True per-query CATE and true ATE in RAW Y units (Table-3 convention).
    # Densities live in scaled Y ([-1, 1]); convert back via factor y_rng/2.
    true_cate_raw = _np(cd.true_cate).reshape(-1)                      # (n_test,)
    y_rng_over_2 = truth.y_rng / 2.0
    true_ate_raw = float(true_cate_raw.mean())

    # ── Per-realization density diagnostic PNG (methods vs truth) ─────
    # Saved BEFORE metric computation so it lands even if a downstream
    # numeric step raises. One 4-panel PNG per realization at
    # {args.out}.density_diag.r{r:03d}.png.
    try:
        from density_plots import save_density_diag_png
        save_density_diag_png(
            out_path=f'{args.out}.density_diag.r{args.realization:03d}.png',
            r=args.realization,
            method_out=method_out,
            p_y0_true=p_y0_true, p_y1_true=p_y1_true,
            p_tau_true=p_tau_true, p_ate_true=p_ate_true,
            Y_CENTERS=Y_CENTERS, TAU_CENTERS=TAU_CENTERS,
            wb_fn=wasserstein_barycenter_1d,
            q_show=0,
        )
        print(f'[density-diag] saved r={args.realization}', flush=True)
    except Exception as e:
        print(f'[warn] density diag plot failed: {type(e).__name__}: {e}', flush=True)

    # ── L2 distances + point-estimate metrics (PEHE, eps_ATE) ──────────
    out: dict[str, np.ndarray] = dict(
        r=np.int32(args.realization),
        n_queries=np.int32(n_queries),
        p_y0_true=p_y0_true, p_y1_true=p_y1_true,
        p_tau_true=p_tau_true, p_ate_true=p_ate_true,
        true_cate_raw=true_cate_raw.astype(np.float32),
        true_ate_raw=np.float32(true_ate_raw),
        y_rng=np.float32(truth.y_rng),
    )
    for name, d in method_out.items():
        p_ate = wasserstein_barycenter_1d(d['p_tau'], TAU_CENTERS)
        s = p_ate.sum() * float(TAU_CENTERS[1] - TAU_CENTERS[0])
        if s > 0: p_ate = p_ate / s

        # ── Point estimates — TWO variants for CATE / PEHE ────────────
        # (a) MALC-mean: E[τ] under MALC-smoothed p(τ)
        # (b) raw-mean:  E[Y1] - E[Y0] from raw p_mat marginals (Ours-only)
        # The eps_ATE numbers reported below are the OT-mean version (which,
        # by the barycenter-mean identity, equals mean-of-per-query-MALC-means).
        tau_bin = float(TAU_CENTERS[1] - TAU_CENTERS[0])
        cate_hat_scaled = (TAU_CENTERS[None, :] * d['p_tau']).sum(axis=1) * tau_bin
        cate_hat_raw = cate_hat_scaled * y_rng_over_2                    # MALC-mean
        # ATE via OT barycenter's mean (== mean-of-per-query-MALC-means)
        ate_hat_scaled = float((TAU_CENTERS * p_ate).sum() * tau_bin)
        ate_hat_raw = ate_hat_scaled * y_rng_over_2

        # Table-3 metrics — MALC-mean flavour (existing)
        pehe = float(np.sqrt(np.mean((cate_hat_raw - true_cate_raw) ** 2)))
        eps_ate = float(abs(ate_hat_raw - true_ate_raw)
                        / max(abs(true_ate_raw), 1e-9))

        # Raw-mean flavour (Ours-methods only — dopfn/uwyk_noanc don't use MALC
        # so their raw and MALC means coincide with `cate_hat_raw` above).
        pehe_raw = None
        eps_ate_raw = None
        if 'cate_raw_scaled' in d:
            cate_raw_raw = d['cate_raw_scaled'] * y_rng_over_2           # raw-mean
            ate_raw_raw = float(cate_raw_raw.mean())
            pehe_raw = float(np.sqrt(np.mean((cate_raw_raw - true_cate_raw) ** 2)))
            eps_ate_raw = float(abs(ate_raw_raw - true_ate_raw)
                                / max(abs(true_ate_raw), 1e-9))

        # EM-mean flavours (Ours-methods only). Two variants: K=1 forced,
        # and whatever K MALC selected via BIC (mixture, pi-weighted).
        # NaN entries → per-query MALC fit failed for that variant.
        def _malc_em_pehe(cate_em_scaled):
            arr = np.asarray(cate_em_scaled)
            cate_em_raw = arr * y_rng_over_2
            valid = np.isfinite(cate_em_raw)
            if not np.any(valid):
                return None, None
            ate_em = float(cate_em_raw[valid].mean())
            pehe = float(np.sqrt(np.mean(
                (cate_em_raw[valid] - true_cate_raw[valid]) ** 2)))
            eps  = float(abs(ate_em - true_ate_raw) / max(abs(true_ate_raw), 1e-9))
            return pehe, eps
        pehe_em_mix = eps_em_mix = None
        pehe_em_k1  = eps_em_k1  = None
        if 'cate_em_mix_scaled' in d:
            pehe_em_mix, eps_em_mix = _malc_em_pehe(d['cate_em_mix_scaled'])
        if 'cate_em_k1_scaled' in d:
            pehe_em_k1,  eps_em_k1  = _malc_em_pehe(d['cate_em_k1_scaled'])

        # Density L2 (existing)
        l2_y0 = np.array([l2_distance(d['p_y0'][q], p_y0_true[q], Y_CENTERS)
                          for q in range(n_queries)])
        l2_y1 = np.array([l2_distance(d['p_y1'][q], p_y1_true[q], Y_CENTERS)
                          for q in range(n_queries)])
        l2_tau = np.array([l2_distance(d['p_tau'][q], p_tau_true[q], TAU_CENTERS)
                           for q in range(n_queries)])
        l2_ate = l2_distance(p_ate, p_ate_true, TAU_CENTERS)

        # KL divergence, both directions (per user request 2026-08-16).
        # KL_forward = KL(truth || est)   — small when est has mass where truth has mass
        # KL_reverse = KL(est || truth)   — small when est doesn't put mass where truth is zero
        # Numerical safety: floor densities at eps before log; measures the L^1-integral
        # on the common grid so scale is comparable to L2 above.
        Y_DX   = float(Y_CENTERS[1] - Y_CENTERS[0])
        TAU_DX = float(TAU_CENTERS[1] - TAU_CENTERS[0])
        def _kl(p, q, dx, eps=1e-12):
            p = np.asarray(p, dtype=np.float64) + eps
            q = np.asarray(q, dtype=np.float64) + eps
            return float(np.sum(p * np.log(p / q)) * dx)
        kl_y0_fwd = np.array([_kl(p_y0_true[q], d['p_y0'][q], Y_DX)   for q in range(n_queries)])
        kl_y1_fwd = np.array([_kl(p_y1_true[q], d['p_y1'][q], Y_DX)   for q in range(n_queries)])
        kl_tau_fwd = np.array([_kl(p_tau_true[q], d['p_tau'][q], TAU_DX) for q in range(n_queries)])
        kl_ate_fwd = _kl(p_ate_true, p_ate, TAU_DX)
        kl_y0_rev = np.array([_kl(d['p_y0'][q], p_y0_true[q], Y_DX)   for q in range(n_queries)])
        kl_y1_rev = np.array([_kl(d['p_y1'][q], p_y1_true[q], Y_DX)   for q in range(n_queries)])
        kl_tau_rev = np.array([_kl(d['p_tau'][q], p_tau_true[q], TAU_DX) for q in range(n_queries)])
        kl_ate_rev = _kl(p_ate, p_ate_true, TAU_DX)

        out[f'{name}__p_y0']  = d['p_y0']
        out[f'{name}__p_y1']  = d['p_y1']
        out[f'{name}__p_tau'] = d['p_tau']
        out[f'{name}__p_ate'] = p_ate
        # Exact per-bin probabilities on J=10 edges (only Ours-DoPFN-bb with
        # marginals_from_2d=True populates these — recipe-strict form for the
        # bb_2dmarg row in l2_per_bin_prob.py).
        if 'p_y0_bins_j10' in d and 'p_y1_bins_j10' in d:
            out[f'{name}__p_y0_bins_j10'] = d['p_y0_bins_j10'].astype(np.float32)
            out[f'{name}__p_y1_bins_j10'] = d['p_y1_bins_j10'].astype(np.float32)
        out[f'{name}__l2_y0']  = l2_y0.astype(np.float32)
        out[f'{name}__l2_y1']  = l2_y1.astype(np.float32)
        out[f'{name}__l2_tau'] = l2_tau.astype(np.float32)
        out[f'{name}__l2_ate'] = np.float32(l2_ate)
        out[f'{name}__kl_y0_fwd']  = kl_y0_fwd.astype(np.float32)
        out[f'{name}__kl_y1_fwd']  = kl_y1_fwd.astype(np.float32)
        out[f'{name}__kl_tau_fwd'] = kl_tau_fwd.astype(np.float32)
        out[f'{name}__kl_ate_fwd'] = np.float32(kl_ate_fwd)
        out[f'{name}__kl_y0_rev']  = kl_y0_rev.astype(np.float32)
        out[f'{name}__kl_y1_rev']  = kl_y1_rev.astype(np.float32)
        out[f'{name}__kl_tau_rev'] = kl_tau_rev.astype(np.float32)
        out[f'{name}__kl_ate_rev'] = np.float32(kl_ate_rev)
        out[f'{name}__cate_hat_raw'] = cate_hat_raw.astype(np.float32)
        out[f'{name}__ate_hat_raw']  = np.float32(ate_hat_raw)
        out[f'{name}__pehe']         = np.float32(pehe)
        out[f'{name}__eps_ate']      = np.float32(eps_ate)
        if pehe_raw is not None:
            out[f'{name}__pehe_raw']    = np.float32(pehe_raw)
            out[f'{name}__eps_ate_raw'] = np.float32(eps_ate_raw)
        if pehe_em_mix is not None:
            out[f'{name}__pehe_em_mix']    = np.float32(pehe_em_mix)
            out[f'{name}__eps_ate_em_mix'] = np.float32(eps_em_mix)
        if pehe_em_k1 is not None:
            out[f'{name}__pehe_em_k1']     = np.float32(pehe_em_k1)
            out[f'{name}__eps_ate_em_k1']  = np.float32(eps_em_k1)
        print(f'[l2 ] {name:14s}  y0={l2_y0.mean():.4f}  y1={l2_y1.mean():.4f}  '
              f'tau={l2_tau.mean():.4f}  ate={l2_ate:.4f}', flush=True)
        print(f'[KL_fwd] {name:14s}  y0={kl_y0_fwd.mean():.4f}  y1={kl_y1_fwd.mean():.4f}  '
              f'tau={kl_tau_fwd.mean():.4f}  ate={kl_ate_fwd:.4f}', flush=True)
        print(f'[KL_rev] {name:14s}  y0={kl_y0_rev.mean():.4f}  y1={kl_y1_rev.mean():.4f}  '
              f'tau={kl_tau_rev.mean():.4f}  ate={kl_ate_rev:.4f}', flush=True)
        if pehe_raw is not None:
            print(f'[pnt] {name:14s}  PEHE(MALC)={pehe:.4f}  '
                  f'PEHE(raw)={pehe_raw:.4f}    '
                  f'eps_ATE(MALC)={eps_ate:.4f}  eps_ATE(raw)={eps_ate_raw:.4f}  '
                  f'(true_ATE={true_ate_raw:+.3f})', flush=True)
        else:
            print(f'[pnt] {name:14s}  PEHE={pehe:.4f}  '
                  f'eps_ATE={eps_ate:.4f}  '
                  f'(true_ATE={true_ate_raw:+.3f}, '
                  f'pred_ATE={ate_hat_raw:+.3f})', flush=True)

    # ── Save shard ──────────────────────────────────────────────────────
    shard = f'{args.out}.r{args.realization:03d}.npz'
    os.makedirs(os.path.dirname(shard) or '.', exist_ok=True)
    np.savez_compressed(shard, **out)
    print(f'[save] {shard}', flush=True)
    return 0


# ── Method drivers ──────────────────────────────────────────────────────
def _run_ours(cd, ckpt_path, truth, args, n_ctx):
    from models.InterventionalPFN import InterventionalPFN
    from losses.BarDistribution2D import fit_malc_inner
    from malc_2d import dmalc_2d
    from methods_densities import ours_densities

    print(f'[ours] loading {ckpt_path}', flush=True)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']; J = cfg['J']
    edges_np = ckpt['edges'].cpu().numpy()
    bin_width = float(edges_np[1] - edges_np[0])
    num_features = cfg['num_features']
    model = InterventionalPFN(
        num_features=num_features, d_model=cfg['d_model'], depth=cfg['depth'],
        heads_feat=cfg['heads'], heads_samp=cfg['heads'], dropout=0.0,
        output_dim=J*J + 9 + 4, hidden_mult=cfg['hidden_mult'],
        normalize_features=True, normalize_treatment=False,
        use_treatment_in_query=False, use_checkpoint=False,
    ).eval()
    model.load_state_dict(ckpt['model_state_dict'])

    t0 = time.time()
    d = ours_densities(
        cd, model, edges_np, J, bin_width, num_features,
        y_min=truth.y_min, y_rng=truth.y_rng,
        malc_B=args.malc_B, malc_max_K=args.malc_max_K, n_eval=args.n_eval,
        n_context=n_ctx,
        fit_malc_inner=fit_malc_inner, dmalc_2d=dmalc_2d,
        marginals_from_2d=args.marginals_from_2d,
    )
    print(f'[ours] done in {time.time() - t0:.1f}s   '
          f'(marginals_from_2d={args.marginals_from_2d})', flush=True)
    return d


def _run_ours_dopfn_bb(cd, ckpt_path, truth, args, n_ctx):
    """Load the DoPFN-backbone R-PFN checkpoint and run the same MALC+diag pipeline.

    Uses the DoPFNBackboneWith2DHead wrapper (PerFeatureTransformer + 2D head)
    trained from scratch. Forward interface is identical to InterventionalPFN
    (X_context, T_context, Y_context, X_query -> predictions), so the MALC and
    diagonal-integration downstream in ours_densities() works unchanged.

    Because the DoPFN backbone uses per-feature attention (no num_features
    cap), we pass -1 for num_features so ours_densities skips the pad/truncate
    logic and feeds the raw IHDP X unchanged.
    """
    sys.path.insert(0, os.path.join(args.repo, 'training_dopfn_base'))
    from dopfn_backbone_head import DoPFNBackboneWith2DHead
    from losses.BarDistribution2D import fit_malc_inner
    from malc_2d import dmalc_2d
    from methods_densities import ours_densities

    print(f'[ours-dopfn-bb] loading {ckpt_path}', flush=True)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']; J = cfg['J']
    edges_np = ckpt['edges'].cpu().numpy()
    bin_width = float(edges_np[1] - edges_np[0])

    # Build the wrapper. dopfn_root must point at the DoPFN repo — the
    # wrapper only uses it to load the *architecture* (pickle + config); the
    # actual trained weights come from `ckpt['model_state_dict']`.
    model = DoPFNBackboneWith2DHead(dopfn_root=args.dopfn, K=J).eval()
    model.load_state_dict(ckpt['model_state_dict'])

    # No num_features cap for this backbone — pass -1 so ours_densities
    # doesn't try to pad/truncate.
    num_features_dummy = -1

    t0 = time.time()
    d = ours_densities(
        cd, model, edges_np, J, bin_width, num_features_dummy,
        y_min=truth.y_min, y_rng=truth.y_rng,
        malc_B=args.malc_B, malc_max_K=args.malc_max_K, n_eval=args.n_eval,
        n_context=n_ctx,
        fit_malc_inner=fit_malc_inner, dmalc_2d=dmalc_2d,
        y_scaling=args.y_scaling,
        std_target=args.std_target,
        marginals_from_2d=args.marginals_from_2d,
    )
    print(f'[ours-dopfn-bb] done in {time.time() - t0:.1f}s   '
          f'(y_scaling={args.y_scaling}, marginals_from_2d={args.marginals_from_2d})',
          flush=True)
    return d


def _run_uwyk_noanc(cd, truth, args, n_ctx):
    from methods_densities import uwyk_noanc_densities

    # UWYK's imports collide with local models/utils — isolate as
    # plot_ihdp_n10_uwyk_noanc.py does.
    _saved = {}
    for name in list(sys.modules):
        if (name == 'models' or name.startswith('models.') or
                name == 'utils' or name.startswith('utils.')):
            _saved[name] = sys.modules.pop(name)
    sys.path.insert(0, args.uwyk_src)
    pre_mod = importlib.import_module('models.PreprocessingGraphConditionedPFN')
    sys.path.remove(args.uwyk_src)
    for name in list(sys.modules):
        if (name == 'models' or name.startswith('models.') or
                name == 'utils' or name.startswith('utils.')):
            del sys.modules[name]
    sys.modules.update(_saved)

    print('[uwyk] loading checkpoint', flush=True)
    _orig_load = torch.load
    def _p_load(*a, **kw):
        kw.setdefault('weights_only', False); return _orig_load(*a, **kw)
    torch.load = _p_load
    _final_ck  = os.path.join(args.uwyk_ckpt_dir, 'final_model_with_bardist.pt')
    _final_cfg = os.path.join(args.uwyk_ckpt_dir, 'final_model_with_bardist_config.yaml')
    if os.path.isfile(_final_ck) and os.path.isfile(_final_cfg):
        _cfg_p, _ck_p = _final_cfg, _final_ck
    else:
        _cfg_p = os.path.join(args.uwyk_ckpt_dir, 'best_model_config.yaml')
        _ck_p  = os.path.join(args.uwyk_ckpt_dir, 'best_model.pt')
        print(f'[warn] UWYK final_model_with_bardist.pt not found; falling back to best_model.pt', flush=True)
    uwyk_model = pre_mod.PreprocessingGraphConditionedPFN(
        config_path=_cfg_p, checkpoint_path=_ck_p, device='cpu', verbose=False,
        random_state=42,
        # Wrapper's clustered + prediction_type='sample' path has a
        # (num_samples, num_bars) vs (n_test, num_bars) shape confusion in
        # _predict_single_cluster that we cannot patch without a rewrite.
        # Disable clustering for the L2 densities call: wrapper subsamples
        # training to max_n_train (1000) instead. Documented limitation.
        use_clustering=False,
    ).load()
    torch.load = _orig_load
    num_features = uwyk_model.model.num_features

    t0 = time.time()
    d = uwyk_noanc_densities(
        cd, uwyk_model, num_features,
        y_min=truth.y_min, y_rng=truth.y_rng,
        n_context=n_ctx, n_samples=args.uwyk_n_samples,
    )
    print(f'[uwyk] done in {time.time() - t0:.1f}s', flush=True)
    return d


def _run_uwyk_anc(cd, truth, args, n_ctx):
    """UWYK Full-Ancestral — same checkpoint as no-anc but full-graph adjacency."""
    from methods_densities import uwyk_anc_densities

    _saved = {}
    for name in list(sys.modules):
        if (name == 'models' or name.startswith('models.') or
                name == 'utils' or name.startswith('utils.')):
            _saved[name] = sys.modules.pop(name)
    sys.path.insert(0, args.uwyk_src)
    pre_mod = importlib.import_module('models.PreprocessingGraphConditionedPFN')
    sys.path.remove(args.uwyk_src)
    for name in list(sys.modules):
        if (name == 'models' or name.startswith('models.') or
                name == 'utils' or name.startswith('utils.')):
            del sys.modules[name]
    sys.modules.update(_saved)

    print('[uwyk-anc] loading checkpoint', flush=True)
    _orig_load = torch.load
    def _p_load(*a, **kw):
        kw.setdefault('weights_only', False); return _orig_load(*a, **kw)
    torch.load = _p_load
    _final_ck  = os.path.join(args.uwyk_ckpt_dir, 'final_model_with_bardist.pt')
    _final_cfg = os.path.join(args.uwyk_ckpt_dir, 'final_model_with_bardist_config.yaml')
    if os.path.isfile(_final_ck) and os.path.isfile(_final_cfg):
        _cfg_p, _ck_p = _final_cfg, _final_ck
    else:
        _cfg_p = os.path.join(args.uwyk_ckpt_dir, 'best_model_config.yaml')
        _ck_p  = os.path.join(args.uwyk_ckpt_dir, 'best_model.pt')
        print(f'[warn] UWYK final_model_with_bardist.pt not found; falling back to best_model.pt', flush=True)
    uwyk_model = pre_mod.PreprocessingGraphConditionedPFN(
        config_path=_cfg_p, checkpoint_path=_ck_p, device='cpu', verbose=False,
        random_state=42,
        # Wrapper's clustered + prediction_type='sample' path has a
        # (num_samples, num_bars) vs (n_test, num_bars) shape confusion in
        # _predict_single_cluster that we cannot patch without a rewrite.
        # Disable clustering for the L2 densities call: wrapper subsamples
        # training to max_n_train (1000) instead. Documented limitation.
        use_clustering=False,
    ).load()
    torch.load = _orig_load
    num_features = uwyk_model.model.num_features

    t0 = time.time()
    d = uwyk_anc_densities(
        cd, uwyk_model, num_features,
        y_min=truth.y_min, y_rng=truth.y_rng,
        n_context=n_ctx, n_samples=args.uwyk_n_samples,
    )
    print(f'[uwyk-anc] done in {time.time() - t0:.1f}s', flush=True)
    return d


def _run_dopfn(cd, truth, args, n_ctx):
    # dopfn.py installs an sklearn check_array shim on import — pull it in
    # under a temporary sys.path since we can't `from benchmarks.methods...`
    # (the `benchmarks` package name is taken by CausalPFN).
    _bench_methods = os.path.join(args.repo, 'benchmarks', 'methods')
    if _bench_methods not in sys.path:
        sys.path.insert(0, _bench_methods)
    import dopfn as _dopfn_shim  # noqa: F401  — imports install the shim
    from scripts.transformer_prediction_interface.base import DoPFNRegressor
    # DoPFN's base module binds its own `check_array` at import; re-run the
    # shim now that the module is loaded so our patch reaches it.
    _dopfn_shim._repatch_dopfn_check_array()
    from methods_densities import dopfn_densities

    # Pull J=10 scaled-Y edges from the DoPFN-bb checkpoint (same edges
    # the aggregator uses) so dopfn_densities can emit recipe-strict
    # per-J=10-bin probabilities via CDF-interp of the native bar dist.
    _bb_ckpt = torch.load(args.checkpoint_dopfn_bb, map_location='cpu', weights_only=False)
    _edges_j10 = _bb_ckpt['edges'].cpu().numpy()

    t0 = time.time()
    d = dopfn_densities(
        cd, DoPFNRegressor,
        y_min=truth.y_min, y_rng=truth.y_rng,
        dopfn_root=args.dopfn,
        n_context=n_ctx,
        edges_j10_scaled=_edges_j10,
    )
    print(f'[dopfn] done in {time.time() - t0:.1f}s', flush=True)
    return d


# ── Path plumbing ───────────────────────────────────────────────────────
def _truncate_covariates(cd, d_r: int):
    """Return a shim CATE_Dataset with X_train, X_test truncated to first d_r cols."""
    import torch

    def _trim(A):
        if isinstance(A, torch.Tensor):
            return A[:, :d_r].contiguous()
        return np.asarray(A)[:, :d_r]

    class _Shim:
        pass
    shim = _Shim()
    for attr in dir(cd):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(cd, attr)
        except Exception:
            continue
        if attr in ('X_train', 'X_test'):
            setattr(shim, attr, _trim(val))
        else:
            setattr(shim, attr, val)
    return shim


def _install_dopfn_datasets_shim(dopfn_dir: str) -> None:
    """CausalPFN's IHDPDataset imports a bare `datasets` module — shim it in."""
    if 'datasets' in sys.modules:
        return
    sys.path.insert(0, dopfn_dir)
    ds_mod = types.ModuleType('datasets')
    with open(os.path.join(dopfn_dir, 'datasets/__init__.py')) as fp:
        src = fp.read().split('def load_semi_real')[0]
    exec(src, ds_mod.__dict__)
    sys.modules['datasets'] = ds_mod


def _require_paths(args, methods) -> None:
    need_ours = any(m.startswith('ours_') for m in methods)
    need_uwyk = 'uwyk_noanc' in methods or 'uwyk_anc' in methods
    need_dopfn = 'dopfn' in methods

    for label, path, cond in [
        ('CAUSALPFN', args.causalpfn, True),
        ('DOPFN',     args.dopfn,     True),
        ('CHECKPOINT50', args.checkpoint50, 'ours_fn50' in methods),
        ('CHECKPOINT10', args.checkpoint10, 'ours_fn10' in methods),
        ('UWYK_SRC',      args.uwyk_src,      need_uwyk),
        ('UWYK_CKPT_DIR', args.uwyk_ckpt_dir, need_uwyk),
    ]:
        if not cond:
            continue
        exists = os.path.isdir(path) or os.path.isfile(path)
        if not exists:
            print(f'[fatal] {label} not found at {path!r}')
            sys.exit(2)


def _guess_repo() -> str:
    here = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def _np(a):
    if isinstance(a, torch.Tensor): return a.numpy()
    return np.asarray(a)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc(); sys.exit(1)
