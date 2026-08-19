"""Per-bin-probability L2 aggregator — the "clean" recipe agreed 2026-08-19.

For MARGINALS (y0, y1) — J=10 bins on scaled Y:
  edges_Y = [-1.0, -0.8, ..., 1.0]
  Truth (Gaussian per query): t_j = Φ((e_{j+1} - μ)/σ) - Φ((e_j - μ)/σ)
  Do-PFN-bb raw: use raw softmax p_j on the same J=10 bins directly.
  Do-PFN-bb MALC: use MALC discrete p_j directly (Python malc_1d_cvxpy is discrete
    on the same J=10 bins).
  Do-PFN: aggregate its native ~100-bar softmax into J=10 buckets by summing
    all native bins whose center falls inside each J=10 bin.

For CATE (τ = Y1 - Y0) — 20 τ bins width 0.2 on [-2, 2]:
  edges_τ = [-2.0, -1.8, ..., 2.0]
  Truth Gaussian: μ_τ = μ_1 - μ_0, σ_τ = sqrt(σ_0² + σ_1²) (independence).
  Do-PFN-bb raw joint (J=10 × J=10 = 100 bins): each joint bin (j0, j1) contributes
    its probability to the τ bin containing 0.2·(j1 - j0).
  Do-PFN-bb MALC (2D fit on joint): integrate ĥ(y0, y1) over each τ-diagonal strip
    using the fine (xs, ys) grid where the MALC density was evaluated.
  Do-PFN (marginals only, assumes Y0 ⊥ Y1): convolve p_y0_agg and p_y1_agg
    to get per-τ-offset probabilities, aligned with τ bins.

For ATE — same τ bins:
  Take the Wasserstein-barycentric ATE density (as stored in shards), integrate
  over each τ bin. Truth ATE = barycenter of per-query truth τ densities.

L2 formula (marginals AND τ AND ATE):
  L2 = sqrt( Σ_bin (p_method[bin] - t[bin])² / bin_w )

Usage:
  python l2_per_bin_prob.py \\
      --shards-glob "$DEPLOY_ROOT/ihdp_l2_dopfnbb_j10_s150k_B100K1_all5.r*.npz" \\
      --dopfn-shards-glob "$DEPLOY_ROOT/ihdp_l2_dopfnbb_j10_s150k/out.r[0-9][0-9][0-9].npz" \\
      --repo $DEPLOY_ROOT/R-PFN --causalpfn $DEPLOY_ROOT/external/causalpfn \\
      --checkpoint-dopfn-bb $DEPLOY_ROOT/checkpoints_dopfn_backbone_realj10/step_150000.pt \\
      --dataset ihdp
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards-glob', required=True,
                     help='B=100 BB main sweep (1D-MALC marginals, 2D-MALC τ). '
                          'Row label: "Do-PFN-bb MALC (2D-τ) 1D-marg [B=100]".')
    ap.add_argument('--bb-b500-shards-glob', default='',
                     help='B=500 BB main sweep — same as --shards-glob but B=500.')
    ap.add_argument('--bb-2dmarg-shards-glob', default='',
                     help='B=100 BB sweep with marginals_from_2d=True. Row label: '
                          '"Do-PFN-bb MALC (2D-τ) 2D-marg [B=100]".')
    ap.add_argument('--bb-2dmarg-b500-shards-glob', default='',
                     help='B=500 BB sweep with marginals_from_2d=True.')
    ap.add_argument('--bb-b1000-shards-glob', default='',
                     help='B=1000 BB main sweep (1D-MALC marginals).')
    ap.add_argument('--bb-2dmarg-b1000-shards-glob', default='',
                     help='B=1000 BB sweep with marginals_from_2d=True.')
    ap.add_argument('--dopfn-shards-glob', default='',
                     help='Do-PFN shards (any sweep that has dopfn__* keys — B does not apply).')
    ap.add_argument('--fn50-shards-glob', default='',
                     help='fn=50 @ B=100 with 1D-MALC marginals (marginals_from_2d=0).')
    ap.add_argument('--fn50-b500-shards-glob', default='',
                     help='fn=50 @ B=500 with 1D-MALC marginals.')
    ap.add_argument('--fn50-2dmarg-shards-glob', default='',
                     help='fn=50 @ B=100 with marginals_from_2d=True.')
    ap.add_argument('--fn50-2dmarg-b500-shards-glob', default='',
                     help='fn=50 @ B=500 with marginals_from_2d=True.')
    ap.add_argument('--fn50-b1000-shards-glob', default='',
                     help='fn=50 @ B=1000 with 1D-MALC marginals.')
    ap.add_argument('--fn50-2dmarg-b1000-shards-glob', default='',
                     help='fn=50 @ B=1000 with marginals_from_2d=True.')
    ap.add_argument('--uwyk-shards-glob', default='',
                     help='UWYK sweep — reads uwyk_noanc__* and uwyk_anc__* keys.')
    ap.add_argument('--repo', required=True)
    ap.add_argument('--causalpfn', required=True)
    ap.add_argument('--dopfn', default='', help='DoPFN repo path (ACIC only)')
    ap.add_argument('--checkpoint-dopfn-bb', required=True,
                     help='For J=10 bin edges (edges_np from checkpoint).')
    ap.add_argument('--dataset', choices=['ihdp', 'acic'], default='ihdp')
    ap.add_argument('--acic-cache-dir', default='')
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'l2_ihdp'))
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'l2_acic'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC', 'Optimal_Transport'))
    sys.path.insert(0, args.repo)
    sys.path.insert(0, args.causalpfn)

    import torch
    from scipy.stats import norm
    from ot_barycenter import wasserstein_barycenter_1d
    if args.dataset == 'ihdp':
        from true_ihdp import load_ihdp_truth
        from benchmarks import IHDPDataset
    else:
        from true_acic import load_acic_truth
        from benchmarks import ACIC2016Dataset

    # ── J=10 grid (for BB / Do-PFN) from checkpoint ─────────────────
    ckpt = torch.load(args.checkpoint_dopfn_bb, map_location='cpu', weights_only=False)
    edges_Y = ckpt['edges'].cpu().numpy()           # (J+1,) — e.g. [-1.0, -0.8, ..., 1.0]
    J = int(ckpt['config']['J'])
    bin_w_Y = float(edges_Y[1] - edges_Y[0])
    centers_Y = 0.5 * (edges_Y[:-1] + edges_Y[1:])

    # ── τ grid (for BB / Do-PFN) ─────────────────────────────────────
    tau_min = 2.0 * edges_Y[0]                       # e.g. -2.0
    tau_max = 2.0 * edges_Y[-1]                      # e.g. +2.0
    n_tau_bins = int(round((tau_max - tau_min) / bin_w_Y))
    edges_tau = np.linspace(tau_min, tau_max, n_tau_bins + 1)
    bin_w_tau = float(edges_tau[1] - edges_tau[0])   # = bin_w_Y = 0.2

    # ── J=100 grid (for fn=50 / UWYK) on the SAME [-1, 1] support ────
    J_100 = 100
    edges_Y_100 = np.linspace(edges_Y[0], edges_Y[-1], J_100 + 1)
    bin_w_Y_100 = float(edges_Y_100[1] - edges_Y_100[0])
    n_tau_bins_100 = int(round((tau_max - tau_min) / bin_w_Y_100))
    edges_tau_100 = np.linspace(tau_min, tau_max, n_tau_bins_100 + 1)
    bin_w_tau_100 = float(edges_tau_100[1] - edges_tau_100[0])

    # Per-joint-bin offset → τ bin index. τ_center(j0, j1) = centers_Y[j1] - centers_Y[j0].
    tau_bin_for_joint = np.zeros((J, J), dtype=np.int64)
    for j0 in range(J):
        for j1 in range(J):
            tau_c = centers_Y[j1] - centers_Y[j0]
            k = int(np.clip(np.searchsorted(edges_tau, tau_c, side='right') - 1,
                             0, n_tau_bins - 1))
            tau_bin_for_joint[j0, j1] = k

    # ── Helper: aggregate a native-grid density onto J=10 bins ───────
    def density_to_j10_probs(density, native_grid, native_bin_w):
        """density on native_grid → per-J=10-bin probability."""
        p = np.zeros(J)
        for j in range(J):
            mask = (native_grid >= edges_Y[j]) & (native_grid < edges_Y[j+1])
            p[j] = float(density[mask].sum() * native_bin_w)
        s = p.sum()
        return p / s if s > 0 else p

    def density_to_tau_probs(density_tau, native_tau_grid, native_bin_w):
        """τ density on native grid → per-τ-bin probability."""
        p = np.zeros(n_tau_bins)
        for k in range(n_tau_bins):
            mask = (native_tau_grid >= edges_tau[k]) & (native_tau_grid < edges_tau[k+1])
            p[k] = float(density_tau[mask].sum() * native_bin_w)
        s = p.sum()
        return p / s if s > 0 else p

    def truth_probs_y(mu, sigma, edges=None):
        e = edges_Y if edges is None else edges
        cdf = norm.cdf(e, loc=mu, scale=max(sigma, 1e-8))
        return np.diff(cdf)

    def truth_probs_tau(mu_tau, sigma_tau, edges=None):
        e = edges_tau if edges is None else edges
        cdf = norm.cdf(e, loc=mu_tau, scale=max(sigma_tau, 1e-8))
        return np.diff(cdf)

    def l2(p, t, bin_w):
        return float(np.sqrt(np.sum((np.asarray(p) - np.asarray(t))**2) / bin_w))

    # ── Load shards ──────────────────────────────────────────────────
    shards = sorted(glob.glob(args.shards_glob))
    dopfn_shards = sorted(glob.glob(args.dopfn_shards_glob)) if args.dopfn_shards_glob else []
    dopfn_by_r = {int(s.split('.r')[-1].split('.')[0]): s for s in dopfn_shards}
    fn50_shards = sorted(glob.glob(args.fn50_shards_glob)) if args.fn50_shards_glob else []
    fn50_by_r = {int(s.split('.r')[-1].split('.')[0]): s for s in fn50_shards}
    uwyk_shards = sorted(glob.glob(args.uwyk_shards_glob)) if args.uwyk_shards_glob else []
    uwyk_by_r = {int(s.split('.r')[-1].split('.')[0]): s for s in uwyk_shards}
    bb_2dmarg_shards = sorted(glob.glob(args.bb_2dmarg_shards_glob)) if args.bb_2dmarg_shards_glob else []
    bb_2dmarg_by_r = {int(s.split('.r')[-1].split('.')[0]): s for s in bb_2dmarg_shards}
    bb_b500_shards = sorted(glob.glob(args.bb_b500_shards_glob)) if args.bb_b500_shards_glob else []
    bb_b500_by_r = {int(s.split('.r')[-1].split('.')[0]): s for s in bb_b500_shards}
    bb_2dmarg_b500_shards = sorted(glob.glob(args.bb_2dmarg_b500_shards_glob)) if args.bb_2dmarg_b500_shards_glob else []
    bb_2dmarg_b500_by_r = {int(s.split('.r')[-1].split('.')[0]): s for s in bb_2dmarg_b500_shards}
    fn50_2dmarg_shards = sorted(glob.glob(args.fn50_2dmarg_shards_glob)) if args.fn50_2dmarg_shards_glob else []
    fn50_2dmarg_by_r = {int(s.split('.r')[-1].split('.')[0]): s for s in fn50_2dmarg_shards}
    fn50_b500_shards = sorted(glob.glob(args.fn50_b500_shards_glob)) if args.fn50_b500_shards_glob else []
    fn50_b500_by_r = {int(s.split('.r')[-1].split('.')[0]): s for s in fn50_b500_shards}
    fn50_2dmarg_b500_shards = sorted(glob.glob(args.fn50_2dmarg_b500_shards_glob)) if args.fn50_2dmarg_b500_shards_glob else []
    fn50_2dmarg_b500_by_r = {int(s.split('.r')[-1].split('.')[0]): s for s in fn50_2dmarg_b500_shards}
    bb_b1000_shards = sorted(glob.glob(args.bb_b1000_shards_glob)) if args.bb_b1000_shards_glob else []
    bb_b1000_by_r = {int(s.split('.r')[-1].split('.')[0]): s for s in bb_b1000_shards}
    bb_2dmarg_b1000_shards = sorted(glob.glob(args.bb_2dmarg_b1000_shards_glob)) if args.bb_2dmarg_b1000_shards_glob else []
    bb_2dmarg_b1000_by_r = {int(s.split('.r')[-1].split('.')[0]): s for s in bb_2dmarg_b1000_shards}
    fn50_b1000_shards = sorted(glob.glob(args.fn50_b1000_shards_glob)) if args.fn50_b1000_shards_glob else []
    fn50_b1000_by_r = {int(s.split('.r')[-1].split('.')[0]): s for s in fn50_b1000_shards}
    fn50_2dmarg_b1000_shards = sorted(glob.glob(args.fn50_2dmarg_b1000_shards_glob)) if args.fn50_2dmarg_b1000_shards_glob else []
    fn50_2dmarg_b1000_by_r = {int(s.split('.r')[-1].split('.')[0]): s for s in fn50_2dmarg_b1000_shards}
    print(f'[load] BB[B100:{len(shards)} B500:{len(bb_b500_shards)} B1000:{len(bb_b1000_shards)} '
          f'2d-B100:{len(bb_2dmarg_shards)} 2d-B500:{len(bb_2dmarg_b500_shards)} 2d-B1000:{len(bb_2dmarg_b1000_shards)}]  '
          f'fn50[1d-B100:{len(fn50_shards)} 1d-B500:{len(fn50_b500_shards)} 1d-B1000:{len(fn50_b1000_shards)} '
          f'2d-B100:{len(fn50_2dmarg_shards)} 2d-B500:{len(fn50_2dmarg_b500_shards)} 2d-B1000:{len(fn50_2dmarg_b1000_shards)}]  '
          f'dopfn:{len(dopfn_shards)}  uwyk:{len(uwyk_shards)}', flush=True)
    if not shards:
        sys.exit('no main (B=100) shards match')

    # Accumulators — pool across (realization, query)
    acc = {m: {q: [] for q in ('y0', 'y1', 'tau', 'ate')}
           for m in ('dopfn',
                       'bb_malc_b100',      'bb_2dmarg_b100',
                       'bb_malc_b500',      'bb_2dmarg_b500',
                       'bb_malc_b1000',     'bb_2dmarg_b1000',
                       'fn50_1d_b100',      'fn50_2d_b100',
                       'fn50_1d_b500',      'fn50_2d_b500',
                       'fn50_1d_b1000',     'fn50_2d_b1000',
                       'uwyk_noanc',        'uwyk_anc',
                       # kept for backwards-compat with old sweeps; unused rows
                       'bb_raw', 'bb_malc_indep')}
    # Per-method realization coverage — set of realization indices for which
    # any L2 value was appended. Reported as "X/N" per row where N is the
    # expected total (IHDP=100, ACIC=10).
    seen_r = {m: set() for m in acc}
    n_expected = 100 if args.dataset == 'ihdp' else 10

    # Union of all realizations across every input glob so we cover shards
    # from side sweeps that aren't in the main (--shards-glob) set.
    all_r = sorted(set(list(dopfn_by_r) + list(fn50_by_r) + list(uwyk_by_r)
                        + list(bb_2dmarg_by_r) + list(bb_b500_by_r)
                        + list(bb_2dmarg_b500_by_r) + list(bb_b1000_by_r)
                        + list(bb_2dmarg_b1000_by_r) + list(fn50_b500_by_r)
                        + list(fn50_2dmarg_by_r) + list(fn50_b500_by_r)
                        + list(fn50_2dmarg_b500_by_r) + list(fn50_b1000_by_r)
                        + list(fn50_2dmarg_b1000_by_r)
                        + [int(s.split('.r')[-1].split('.')[0]) for s in shards]))
    main_by_r = {int(s.split('.r')[-1].split('.')[0]): s for s in shards}

    for si, r in enumerate(all_r):
        shard_path = main_by_r.get(r, None)
        # Load truth for this realization
        if args.dataset == 'ihdp':
            cd, _ = IHDPDataset()[r]
            y_train_full = np.asarray(cd.y_train.detach().cpu()
                                      if hasattr(cd.y_train, 'detach') else cd.y_train)
            truth = load_ihdp_truth(r, args.causalpfn, y_train_full)
        else:
            from eval_realization import _install_dopfn_datasets_shim
            try: _install_dopfn_datasets_shim(args.dopfn)
            except Exception: pass
            cd, _ = ACIC2016Dataset()[r]
            y_train_full = np.asarray(cd.y_train.detach().cpu()
                                      if hasattr(cd.y_train, 'detach') else cd.y_train)
            truth = load_acic_truth(r, y_train_full, cache_dir=(args.acic_cache_dir or None))

        # Truth per-query params (mu, sigma in scaled Y)
        mu0 = truth.mu0_test_scaled if hasattr(truth, 'mu0_test_scaled') else None
        mu1 = truth.mu1_test_scaled if hasattr(truth, 'mu1_test_scaled') else None
        sigma_scaled = truth.sigma_scaled
        if mu0 is None or mu1 is None:
            print(f'  [warn] r={r} truth missing mu0/mu1; skipping'); continue
        n_q = mu0.shape[0]

        # Snapshot list lengths so we can detect which method rows got fed by
        # this realization → seen_r[m] accumulation happens at end of loop.
        _before = {m: len(acc[m]['y0']) for m in acc}

        # Load Do-PFN-bb shard (raw + MALC). If the main sweep is missing
        # this realization, use an empty stand-in so the block short-circuits
        # via `has_malc = False` etc. and reference sweeps still run below.
        import contextlib
        _mctx = np.load(shard_path) if shard_path is not None else contextlib.nullcontext({})
        with _mctx as z:
            keys = list(z.files) if shard_path is not None else []
            # Raw joint p_mat is NOT stored in current shards — we use p_y0_raw as the
            # J=10 raw marginal (stored via ours_dopfn_bb_rawmarg pseudo-method).
            # For CATE we'd need p_mat; fall back to independence-convolution using raw.
            has_rawmarg = 'ours_dopfn_bb_rawmarg__p_y0' in keys
            has_malc = 'ours_dopfn_bb__p_y0' in keys
            has_ate = 'ours_dopfn_bb__p_ate' in keys

            # From Y_CENTERS-space stored densities → J=10 probs (integrate over J=10 bins).
            # Y_CENTERS is fixed at 100 pts on [-1.5, 1.5], but shards store on that grid.
            Y_CENTERS = np.linspace(-1.5, 1.5, 101); Y_CENTERS = 0.5*(Y_CENTERS[:-1]+Y_CENTERS[1:])
            Y_BIN = float(Y_CENTERS[1] - Y_CENTERS[0])

            def y_probs_from_stored(d):
                """Integrate stored density (on Y_CENTERS) into J=10 bins."""
                p = np.zeros(J)
                for j in range(J):
                    mask = (Y_CENTERS >= edges_Y[j]) & (Y_CENTERS < edges_Y[j+1])
                    p[j] = float(d[mask].sum() * Y_BIN)
                s = p.sum()
                return p / s if s > 0 else p

            # Per-query loop
            for q in range(n_q):
                # ── TRUTH per-bin probs ───────────────────────────────
                t_y0 = truth_probs_y(mu0[q], sigma_scaled)
                t_y1 = truth_probs_y(mu1[q], sigma_scaled)
                mu_tau = float(mu1[q] - mu0[q])
                sigma_tau = float(np.sqrt(2.0) * sigma_scaled)
                t_tau = truth_probs_tau(mu_tau, sigma_tau)

                # ── DoPFN-bb MALC (2D-MALC diagonal τ) ─────────────────
                if has_malc:
                    p_bb_malc_y0 = y_probs_from_stored(z['ours_dopfn_bb__p_y0'][q])
                    p_bb_malc_y1 = y_probs_from_stored(z['ours_dopfn_bb__p_y1'][q])
                    p_bb_malc_tau = density_to_tau_probs(
                        z['ours_dopfn_bb__p_tau'][q],
                        np.linspace(-3.0, 3.0, 601)[1:] * 0 + 0.5 * (np.linspace(-3.0, 3.0, 601)[:-1] + np.linspace(-3.0, 3.0, 601)[1:]),
                        float(np.linspace(-3.0, 3.0, 601)[1] - np.linspace(-3.0, 3.0, 601)[0]))
                    acc['bb_malc_b100']['y0'].append(l2(p_bb_malc_y0, t_y0, bin_w_Y))
                    acc['bb_malc_b100']['y1'].append(l2(p_bb_malc_y1, t_y1, bin_w_Y))
                    acc['bb_malc_b100']['tau'].append(l2(p_bb_malc_tau, t_tau, bin_w_tau))
                    # ── DoPFN-bb MALC + INDEPENDENCE τ ────────────────
                    # Same marginals as bb_malc; τ via convolution of MALC marginals
                    # (assumes Y0 ⊥ Y1, ignoring the 2D joint fit).
                    p_bb_mi_tau_conv = np.convolve(p_bb_malc_y1, p_bb_malc_y0[::-1], mode='full')
                    tau_conv_offsets = np.arange(-(J-1), J) * bin_w_Y
                    p_bb_mi_tau = np.zeros(n_tau_bins)
                    for i, off in enumerate(tau_conv_offsets):
                        k = int(np.clip(np.searchsorted(edges_tau, off, side='right') - 1,
                                         0, n_tau_bins - 1))
                        p_bb_mi_tau[k] += p_bb_mi_tau_conv[i]
                    acc['bb_malc_indep']['y0'].append(l2(p_bb_malc_y0, t_y0, bin_w_Y))
                    acc['bb_malc_indep']['y1'].append(l2(p_bb_malc_y1, t_y1, bin_w_Y))
                    acc['bb_malc_indep']['tau'].append(l2(p_bb_mi_tau, t_tau, bin_w_tau))

                # ── DoPFN-bb raw marginals ────────────────────────────
                if has_rawmarg:
                    p_bb_raw_y0 = y_probs_from_stored(z['ours_dopfn_bb_rawmarg__p_y0'][q])
                    p_bb_raw_y1 = y_probs_from_stored(z['ours_dopfn_bb_rawmarg__p_y1'][q])
                    # Raw τ via independence convolution
                    p_bb_raw_tau_conv = np.convolve(p_bb_raw_y1, p_bb_raw_y0[::-1], mode='full')
                    tau_conv_offsets = np.arange(-(J-1), J) * bin_w_Y
                    # Map offsets to τ bins
                    p_bb_raw_tau = np.zeros(n_tau_bins)
                    for i, off in enumerate(tau_conv_offsets):
                        k = int(np.clip(np.searchsorted(edges_tau, off, side='right') - 1,
                                         0, n_tau_bins - 1))
                        p_bb_raw_tau[k] += p_bb_raw_tau_conv[i]
                    acc['bb_raw']['y0'].append(l2(p_bb_raw_y0, t_y0, bin_w_Y))
                    acc['bb_raw']['y1'].append(l2(p_bb_raw_y1, t_y1, bin_w_Y))
                    acc['bb_raw']['tau'].append(l2(p_bb_raw_tau, t_tau, bin_w_tau))

            # ATE (single per-realization value)
            if has_ate:
                # Truth ATE density: barycenter of per-query true τ densities.
                TAU_FINE = np.linspace(-3.0, 3.0, 601); TAU_FINE_C = 0.5*(TAU_FINE[:-1]+TAU_FINE[1:])
                TAU_FINE_BIN = TAU_FINE[1] - TAU_FINE[0]
                true_tau_dens = np.stack([
                    norm.pdf(TAU_FINE_C, loc=float(mu1[q]-mu0[q]),
                              scale=float(np.sqrt(2)*sigma_scaled))
                    for q in range(n_q)])
                true_ate = wasserstein_barycenter_1d(true_tau_dens, TAU_FINE_C)
                t_ate = density_to_tau_probs(true_ate, TAU_FINE_C, TAU_FINE_BIN)

                # Do-PFN-bb MALC ATE (from stored p_ate)
                p_bb_malc_ate = density_to_tau_probs(z['ours_dopfn_bb__p_ate'], TAU_FINE_C, TAU_FINE_BIN)
                acc['bb_malc_b100']['ate'].append(l2(p_bb_malc_ate, t_ate, bin_w_tau))

                # Do-PFN-bb RAW ATE: barycenter of per-query raw τ densities.
                # Raw τ per query = independence convolution of raw y0/y1 marginals
                # (on Y_CENTERS grid), yielding a τ density on a natural convolution grid.
                def _conv_ate(y0_key, y1_key):
                    dens_per_q = []
                    for q in range(n_q):
                        d_y0 = z[y0_key][q]
                        d_y1 = z[y1_key][q]
                        conv = np.convolve(d_y1, d_y0[::-1], mode='full') * Y_BIN
                        tau_grid_conv = np.arange(-(len(Y_CENTERS)-1), len(Y_CENTERS)) * Y_BIN + (Y_CENTERS[0]-Y_CENTERS[-1])
                        d_tau_q = np.interp(TAU_FINE_C, tau_grid_conv, conv, left=0, right=0)
                        s = d_tau_q.sum() * TAU_FINE_BIN
                        if s > 0: d_tau_q = d_tau_q / s
                        dens_per_q.append(d_tau_q)
                    return np.stack(dens_per_q)

                if has_rawmarg:
                    raw_tau_dens = _conv_ate('ours_dopfn_bb_rawmarg__p_y0',
                                              'ours_dopfn_bb_rawmarg__p_y1')
                    raw_ate = wasserstein_barycenter_1d(raw_tau_dens, TAU_FINE_C)
                    p_bb_raw_ate = density_to_tau_probs(raw_ate, TAU_FINE_C, TAU_FINE_BIN)
                    acc['bb_raw']['ate'].append(l2(p_bb_raw_ate, t_ate, bin_w_tau))

                # Do-PFN-bb MALC + INDEPENDENCE ATE: barycenter of convolutions of
                # MALC-smoothed marginals (same marginals as bb_malc, τ via independence).
                if has_malc:
                    malc_conv_tau_dens = _conv_ate('ours_dopfn_bb__p_y0',
                                                    'ours_dopfn_bb__p_y1')
                    malc_conv_ate = wasserstein_barycenter_1d(malc_conv_tau_dens, TAU_FINE_C)
                    p_bb_mi_ate = density_to_tau_probs(malc_conv_ate, TAU_FINE_C, TAU_FINE_BIN)
                    acc['bb_malc_indep']['ate'].append(l2(p_bb_mi_ate, t_ate, bin_w_tau))

        # ── Helper: read one method from a shard file, aggregate to J=10 ───
        TAU_FINE = np.linspace(-3.0, 3.0, 601); TAU_FINE_C = 0.5*(TAU_FINE[:-1]+TAU_FINE[1:])
        TAU_FINE_BIN = TAU_FINE[1] - TAU_FINE[0]
        # Precompute truth ATE density once per realization
        true_tau_dens = np.stack([
            norm.pdf(TAU_FINE_C, loc=float(mu1[q]-mu0[q]),
                      scale=float(np.sqrt(2)*sigma_scaled))
            for q in range(n_q)])
        true_ate_density = wasserstein_barycenter_1d(true_tau_dens, TAU_FINE_C)
        t_ate_bins = density_to_tau_probs(true_ate_density, TAU_FINE_C, TAU_FINE_BIN)

        def _read_reference_method(shard_file, key_prefix, acc_key,
                                    tau_via='indep_convolve', use_j100=False):
            """Read p_y0/p_y1 (+ p_tau, p_ate) from a shard for one method.
            use_j100=True → aggregate to J=100 bins on [-1, 1] (native for fn=50/UWYK);
            False → J=10 bins (default, for BB/Do-PFN).
            tau_via: 'indep_convolve' or 'stored_ptau'."""
            if shard_file is None: return
            e_Y  = edges_Y_100 if use_j100 else edges_Y
            bw_Y = bin_w_Y_100 if use_j100 else bin_w_Y
            e_τ  = edges_tau_100 if use_j100 else edges_tau
            bw_τ = bin_w_tau_100 if use_j100 else bin_w_tau
            n_τ  = n_tau_bins_100 if use_j100 else n_tau_bins
            J_y  = J_100 if use_j100 else J

            def _y_bin(density_on_Y):
                p = np.zeros(J_y)
                for j in range(J_y):
                    mask = (Y_CENTERS >= e_Y[j]) & (Y_CENTERS < e_Y[j+1])
                    p[j] = float(density_on_Y[mask].sum() * Y_BIN)
                s = p.sum()
                return p / s if s > 0 else p

            def _tau_bin_from_native(density_on_tau_native, tau_grid_native, tau_bw_native):
                p = np.zeros(n_τ)
                for k in range(n_τ):
                    mask = (tau_grid_native >= e_τ[k]) & (tau_grid_native < e_τ[k+1])
                    p[k] = float(density_on_tau_native[mask].sum() * tau_bw_native)
                s = p.sum()
                return p / s if s > 0 else p

            with np.load(shard_file) as z2:
                py0k = f'{key_prefix}__p_y0'; py1k = f'{key_prefix}__p_y1'
                ptauk = f'{key_prefix}__p_tau'; patek = f'{key_prefix}__p_ate'
                # Recipe-strict per-bin probabilities on the model's native
                # head, if the sweep stored them (2dmarg with exact-CDF
                # marginals). Works for both BB (J=10) and fn=50 (J=100)
                # so long as the stored width matches the target grid.
                py0_binsk = f'{key_prefix}__p_y0_bins_j10'
                py1_binsk = f'{key_prefix}__p_y1_bins_j10'
                _tgt_J   = J_100 if use_j100 else J
                use_stored_bins = (
                    py0_binsk in z2.files and py1_binsk in z2.files
                    and int(z2[py0_binsk].shape[-1]) == _tgt_J)
                if py0k not in z2.files: return
                for q in range(n_q):
                    t_y0 = truth_probs_y(mu0[q], sigma_scaled, edges=e_Y)
                    t_y1 = truth_probs_y(mu1[q], sigma_scaled, edges=e_Y)
                    mu_tau = float(mu1[q] - mu0[q])
                    sigma_tau = float(np.sqrt(2.0) * sigma_scaled)
                    t_tau = truth_probs_tau(mu_tau, sigma_tau, edges=e_τ)

                    if use_stored_bins:
                        p_y0 = np.asarray(z2[py0_binsk][q], dtype=np.float64)
                        p_y1 = np.asarray(z2[py1_binsk][q], dtype=np.float64)
                    else:
                        p_y0 = _y_bin(z2[py0k][q])
                        p_y1 = _y_bin(z2[py1k][q])
                    acc[acc_key]['y0'].append(l2(p_y0, t_y0, bw_Y))
                    acc[acc_key]['y1'].append(l2(p_y1, t_y1, bw_Y))

                    if tau_via == 'stored_ptau' and ptauk in z2.files:
                        p_tau = _tau_bin_from_native(z2[ptauk][q], TAU_FINE_C, TAU_FINE_BIN)
                    else:
                        conv = np.convolve(p_y1, p_y0[::-1], mode='full')
                        offs = np.arange(-(J_y-1), J_y) * bw_Y
                        p_tau = np.zeros(n_τ)
                        for i, off in enumerate(offs):
                            k = int(np.clip(np.searchsorted(e_τ, off, side='right') - 1, 0, n_τ - 1))
                            p_tau[k] += conv[i]
                    acc[acc_key]['tau'].append(l2(p_tau, t_tau, bw_τ))

                # ATE — truth ATE on the appropriate τ grid
                if patek in z2.files:
                    # Recompute truth ATE bin-probs for this row's τ grid
                    true_ate_dens_this = wasserstein_barycenter_1d(true_tau_dens, TAU_FINE_C)
                    t_ate_this = _tau_bin_from_native(true_ate_dens_this, TAU_FINE_C, TAU_FINE_BIN)
                    p_ate = _tau_bin_from_native(z2[patek], TAU_FINE_C, TAU_FINE_BIN)
                    acc[acc_key]['ate'].append(l2(p_ate, t_ate_this, bw_τ))

        # Do-PFN — J=10 grid (user request 2026-08-19)
        if r in dopfn_by_r:
            _read_reference_method(dopfn_by_r[r], 'dopfn', 'dopfn',
                                    tau_via='indep_convolve', use_j100=False)
        # fn=50 @ B=100, 1D-MALC marginals — J=100 grid, native head resolution
        if r in fn50_by_r:
            _read_reference_method(fn50_by_r[r], 'ours_fn50', 'fn50_1d_b100',
                                    tau_via='stored_ptau', use_j100=True)
        # fn=50 @ B=100, 2D-MALC marginals (sweep with MARGINALS_FROM_2D=1)
        if r in fn50_2dmarg_by_r:
            _read_reference_method(fn50_2dmarg_by_r[r], 'ours_fn50', 'fn50_2d_b100',
                                    tau_via='stored_ptau', use_j100=True)
        # fn=50 @ B=500, 1D-MALC marginals
        if r in fn50_b500_by_r:
            _read_reference_method(fn50_b500_by_r[r], 'ours_fn50', 'fn50_1d_b500',
                                    tau_via='stored_ptau', use_j100=True)
        # fn=50 @ B=500, 2D-MALC marginals
        if r in fn50_2dmarg_b500_by_r:
            _read_reference_method(fn50_2dmarg_b500_by_r[r], 'ours_fn50', 'fn50_2d_b500',
                                    tau_via='stored_ptau', use_j100=True)
        # UWYK — J=100 grid
        if r in uwyk_by_r:
            _read_reference_method(uwyk_by_r[r], 'uwyk_noanc', 'uwyk_noanc',
                                    tau_via='indep_convolve', use_j100=True)
            _read_reference_method(uwyk_by_r[r], 'uwyk_anc', 'uwyk_anc',
                                    tau_via='indep_convolve', use_j100=True)
        # BB @ B=100 with 2D-MALC marginals (separate sweep, MARGINALS_FROM_2D=1)
        if r in bb_2dmarg_by_r:
            _read_reference_method(bb_2dmarg_by_r[r], 'ours_dopfn_bb', 'bb_2dmarg_b100',
                                    tau_via='stored_ptau', use_j100=False)
        # BB @ B=500 with 1D-MALC marginals (main sweep with malc_B=500)
        if r in bb_b500_by_r:
            _read_reference_method(bb_b500_by_r[r], 'ours_dopfn_bb', 'bb_malc_b500',
                                    tau_via='stored_ptau', use_j100=False)
        # BB @ B=500 with 2D-MALC marginals (2dmarg sweep with malc_B=500)
        if r in bb_2dmarg_b500_by_r:
            _read_reference_method(bb_2dmarg_b500_by_r[r], 'ours_dopfn_bb', 'bb_2dmarg_b500',
                                    tau_via='stored_ptau', use_j100=False)
        # BB @ B=1000 with 1D-MALC marginals
        if r in bb_b1000_by_r:
            _read_reference_method(bb_b1000_by_r[r], 'ours_dopfn_bb', 'bb_malc_b1000',
                                    tau_via='stored_ptau', use_j100=False)
        # BB @ B=1000 with 2D-MALC marginals
        if r in bb_2dmarg_b1000_by_r:
            _read_reference_method(bb_2dmarg_b1000_by_r[r], 'ours_dopfn_bb', 'bb_2dmarg_b1000',
                                    tau_via='stored_ptau', use_j100=False)
        # fn=50 @ B=1000, 1D-MALC marginals
        if r in fn50_b1000_by_r:
            _read_reference_method(fn50_b1000_by_r[r], 'ours_fn50', 'fn50_1d_b1000',
                                    tau_via='stored_ptau', use_j100=True)
        # fn=50 @ B=1000, 2D-MALC marginals
        if r in fn50_2dmarg_b1000_by_r:
            _read_reference_method(fn50_2dmarg_b1000_by_r[r], 'ours_fn50', 'fn50_2d_b1000',
                                    tau_via='stored_ptau', use_j100=True)

        # Update per-method realization coverage: any acc[m][*] that grew this
        # iteration means realization r contributed to method m.
        for m in acc:
            if len(acc[m]['y0']) > _before[m] or len(acc[m]['ate']) > _before[m]:
                seen_r[m].add(r)

        if (si + 1) % 10 == 0:
            print(f'  processed {si+1}/{len(all_r)}', flush=True)

    # ── Report ───────────────────────────────────────────────────────
    def _stat(vs):
        arr = np.asarray(vs, dtype=np.float64); arr = arr[np.isfinite(arr)]
        if arr.size == 0: return 'na'
        m = arr.mean(); sem = arr.std(ddof=1)/np.sqrt(arr.size) if arr.size > 1 else 0.
        return f'{m:.4f}±{sem:.4f}'

    print()
    print(f'══ {args.dataset.upper()} — per-bin probability L2 '
          f'(BB/Do-PFN: J=10 y-bins, {n_tau_bins} τ-bins @{bin_w_tau:.2f}; '
          f'fn=50/UWYK: J={J_100} y-bins, {n_tau_bins_100} τ-bins @{bin_w_tau_100:.2f}) ══')
    print(f'{"method":52s}  {"cov":>8s}  {"y0":>16s}  {"y1":>16s}  {"τ (CATE)":>16s}  {"ATE":>16s}')
    rows = [
        ('dopfn',            'Do-PFN [J=10]'),
        ('__sep__',          '─── BB @ B=100 ──────────────────────────'),
        ('bb_malc_b100',     'Do-PFN-bb MALC (2D-τ) 1D-marg [B=100, J=10]'),
        ('bb_2dmarg_b100',   'Do-PFN-bb MALC (2D-τ) 2D-marg [B=100, J=10]'),
        ('__sep__',          '─── BB @ B=500 ──────────────────────────'),
        ('bb_malc_b500',     'Do-PFN-bb MALC (2D-τ) 1D-marg [B=500, J=10]'),
        ('bb_2dmarg_b500',   'Do-PFN-bb MALC (2D-τ) 2D-marg [B=500, J=10]'),
        ('__sep__',          '─── BB @ B=1000 ─────────────────────────'),
        ('bb_malc_b1000',    'Do-PFN-bb MALC (2D-τ) 1D-marg [B=1000, J=10]'),
        ('bb_2dmarg_b1000',  'Do-PFN-bb MALC (2D-τ) 2D-marg [B=1000, J=10]'),
        ('__sep__',          '─── fn=50 @ B=100 ───────────────────────'),
        ('fn50_1d_b100',     'fn=50 (2D-τ) 1D-marg [B=100, J=100]'),
        ('fn50_2d_b100',     'fn=50 (2D-τ) 2D-marg [B=100, J=100]'),
        ('__sep__',          '─── fn=50 @ B=500 ───────────────────────'),
        ('fn50_1d_b500',     'fn=50 (2D-τ) 1D-marg [B=500, J=100]'),
        ('fn50_2d_b500',     'fn=50 (2D-τ) 2D-marg [B=500, J=100]'),
        ('__sep__',          '─── fn=50 @ B=1000 ──────────────────────'),
        ('fn50_1d_b1000',    'fn=50 (2D-τ) 1D-marg [B=1000, J=100]'),
        ('fn50_2d_b1000',    'fn=50 (2D-τ) 2D-marg [B=1000, J=100]'),
        ('__sep__',          '─── UWYK ────────────────────────────────'),
        ('uwyk_noanc',       'UWYK-NoAnc [J=100]'),
        ('uwyk_anc',         'UWYK-FullAnc [J=100]'),
    ]
    for m_key, m_label in rows:
        if m_key == '__sep__':
            print(m_label); continue
        cov = f'{len(seen_r[m_key])}/{n_expected}'
        row = f'{m_label:52s}  {cov:>8s}'
        for metric in ['y0', 'y1', 'tau', 'ate']:
            row += f'  {_stat(acc[m_key][metric]):>16s}'
        print(row)


if __name__ == '__main__':
    main()
