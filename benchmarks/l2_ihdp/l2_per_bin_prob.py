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
                     help='Shards for Do-PFN-bb (raw + MALC come from these).')
    ap.add_argument('--dopfn-shards-glob', default='',
                     help='Optional separate shards for Do-PFN (if not in main shards).')
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

    # ── J=10 grid from checkpoint ────────────────────────────────────
    ckpt = torch.load(args.checkpoint_dopfn_bb, map_location='cpu', weights_only=False)
    edges_Y = ckpt['edges'].cpu().numpy()           # (J+1,) — e.g. [-1.0, -0.8, ..., 1.0]
    J = int(ckpt['config']['J'])
    bin_w_Y = float(edges_Y[1] - edges_Y[0])
    centers_Y = 0.5 * (edges_Y[:-1] + edges_Y[1:])

    # ── τ grid ───────────────────────────────────────────────────────
    tau_min = 2.0 * edges_Y[0]                       # e.g. -2.0
    tau_max = 2.0 * edges_Y[-1]                      # e.g. +2.0
    n_tau_bins = int(round((tau_max - tau_min) / bin_w_Y))
    edges_tau = np.linspace(tau_min, tau_max, n_tau_bins + 1)
    bin_w_tau = float(edges_tau[1] - edges_tau[0])   # = bin_w_Y = 0.2

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

    def truth_probs_y(mu, sigma):
        """Exact per-J=10-bin probability for Gaussian(μ, σ)."""
        cdf_edges = norm.cdf(edges_Y, loc=mu, scale=max(sigma, 1e-8))
        return np.diff(cdf_edges)

    def truth_probs_tau(mu_tau, sigma_tau):
        """Exact per-τ-bin probability for Gaussian(μ_τ, σ_τ)."""
        cdf_edges = norm.cdf(edges_tau, loc=mu_tau, scale=max(sigma_tau, 1e-8))
        return np.diff(cdf_edges)

    def l2(p, t, bin_w):
        return float(np.sqrt(np.sum((np.asarray(p) - np.asarray(t))**2) / bin_w))

    # ── Load shards ──────────────────────────────────────────────────
    shards = sorted(glob.glob(args.shards_glob))
    dopfn_shards = sorted(glob.glob(args.dopfn_shards_glob)) if args.dopfn_shards_glob else []
    dopfn_by_r = {int(s.split('.r')[-1].split('.')[0]): s for s in dopfn_shards}
    print(f'[load] {len(shards)} main shards, {len(dopfn_shards)} dopfn ref shards', flush=True)
    if not shards:
        sys.exit('no main shards match')

    # Accumulators — pool across (realization, query)
    acc = {m: {q: [] for q in ('y0', 'y1', 'tau', 'ate')}
           for m in ('dopfn', 'bb_raw', 'bb_malc', 'bb_malc_indep')}

    for si, shard_path in enumerate(shards):
        r = int(shard_path.split('.r')[-1].split('.')[0])
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

        # Load Do-PFN-bb shard (raw + MALC)
        with np.load(shard_path) as z:
            keys = z.files
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
                    acc['bb_malc']['y0'].append(l2(p_bb_malc_y0, t_y0, bin_w_Y))
                    acc['bb_malc']['y1'].append(l2(p_bb_malc_y1, t_y1, bin_w_Y))
                    acc['bb_malc']['tau'].append(l2(p_bb_malc_tau, t_tau, bin_w_tau))
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
                acc['bb_malc']['ate'].append(l2(p_bb_malc_ate, t_ate, bin_w_tau))

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

        # ── Do-PFN reference shard (separate file) ────────────────────
        if r in dopfn_by_r:
            with np.load(dopfn_by_r[r]) as z2:
                if 'dopfn__p_y0' in z2.files:
                    for q in range(n_q):
                        t_y0 = truth_probs_y(mu0[q], sigma_scaled)
                        t_y1 = truth_probs_y(mu1[q], sigma_scaled)
                        mu_tau = float(mu1[q] - mu0[q])
                        sigma_tau = float(np.sqrt(2.0) * sigma_scaled)
                        t_tau = truth_probs_tau(mu_tau, sigma_tau)

                        p_do_y0 = y_probs_from_stored(z2['dopfn__p_y0'][q])
                        p_do_y1 = y_probs_from_stored(z2['dopfn__p_y1'][q])
                        # Do-PFN τ via independence convolution of its marginals
                        p_do_tau_conv = np.convolve(p_do_y1, p_do_y0[::-1], mode='full')
                        tau_conv_offsets = np.arange(-(J-1), J) * bin_w_Y
                        p_do_tau = np.zeros(n_tau_bins)
                        for i, off in enumerate(tau_conv_offsets):
                            k = int(np.clip(np.searchsorted(edges_tau, off, side='right') - 1,
                                             0, n_tau_bins - 1))
                            p_do_tau[k] += p_do_tau_conv[i]
                        acc['dopfn']['y0'].append(l2(p_do_y0, t_y0, bin_w_Y))
                        acc['dopfn']['y1'].append(l2(p_do_y1, t_y1, bin_w_Y))
                        acc['dopfn']['tau'].append(l2(p_do_tau, t_tau, bin_w_tau))
                if 'dopfn__p_ate' in z2.files:
                    TAU_FINE = np.linspace(-3.0, 3.0, 601); TAU_FINE_C = 0.5*(TAU_FINE[:-1]+TAU_FINE[1:])
                    TAU_FINE_BIN = TAU_FINE[1] - TAU_FINE[0]
                    true_tau_dens = np.stack([
                        norm.pdf(TAU_FINE_C, loc=float(mu1[q]-mu0[q]),
                                  scale=float(np.sqrt(2)*sigma_scaled))
                        for q in range(n_q)])
                    true_ate = wasserstein_barycenter_1d(true_tau_dens, TAU_FINE_C)
                    t_ate = density_to_tau_probs(true_ate, TAU_FINE_C, TAU_FINE_BIN)
                    p_do_ate = density_to_tau_probs(z2['dopfn__p_ate'], TAU_FINE_C, TAU_FINE_BIN)
                    acc['dopfn']['ate'].append(l2(p_do_ate, t_ate, bin_w_tau))

        if (si + 1) % 10 == 0:
            print(f'  processed {si+1}/{len(shards)}', flush=True)

    # ── Report ───────────────────────────────────────────────────────
    def _stat(vs):
        arr = np.asarray(vs, dtype=np.float64); arr = arr[np.isfinite(arr)]
        if arr.size == 0: return 'na'
        m = arr.mean(); sem = arr.std(ddof=1)/np.sqrt(arr.size) if arr.size > 1 else 0.
        return f'{m:.4f}±{sem:.4f}'

    print()
    print(f'══ {args.dataset.upper()} — per-bin probability L2 (J=10 bins for y0/y1; '
          f'{n_tau_bins} τ bins width {bin_w_tau:.2f}) ══')
    print(f'{"method":18s}  {"y0":>16s}  {"y1":>16s}  {"τ (CATE)":>16s}  {"ATE":>16s}')
    for m_key, m_label in [('dopfn', 'Do-PFN'),
                            ('bb_raw', 'Do-PFN-bb raw'),
                            ('bb_malc', 'Do-PFN-bb MALC (2D-τ)'),
                            ('bb_malc_indep', 'Do-PFN-bb MALC (indep-τ)')]:
        row = f'{m_label:18s}'
        for metric in ['y0', 'y1', 'tau', 'ate']:
            row += f'  {_stat(acc[m_key][metric]):>16s}'
        print(row)


if __name__ == '__main__':
    main()
