"""Find a (realization, query) example where BOTH Do-PFN-bb beats Do-PFN AND
fn=50 beats the better UWYK arm, then plot 4-panel truth-vs-method densities.

Runs on IHDP OR ACIC. Selection criterion (on CATE τ per-bin L2):
    bb_beats_dopfn = L2(dopfn, τ)  >  L2(bb,   τ)
    fn50_beats_uwyk = min(L2(uwyk_noanc,τ), L2(uwyk_anc,τ))  >  L2(fn50, τ)
Only queries satisfying BOTH are candidates. Ranked by
    score = (L2_dopfn - L2_bb) + (L2_bestUWYK - L2_fn50)   (larger = better).

Writes 4 PNGs at <out>_{y0,y1,tau,ate}.png with truth + all 5 method
densities overlaid.

Usage:
    python benchmarks/plots/find_and_plot_example.py \\
      --dataset ihdp \\
      --bb-shards         "$DEPLOY_ROOT/ihdp_l2_dopfnbb_j10_s150k_2dmarg_B1000K1.r*.npz" \\
      --dopfn-uwyk-shards "$DEPLOY_ROOT/ihdp_l2_dopfn_uwyk_strict.r*.npz" \\
      --fn50-shards       "$DEPLOY_ROOT/ihdp_l2_fn50_2dmarg_B1000K1.r*.npz" \\
      --repo $DEPLOY_ROOT/R-PFN --causalpfn $DEPLOY_ROOT/external/causalpfn \\
      --checkpoint-dopfn-bb $CHECKPOINT_DOPFN_BB \\
      --out $DEPLOY_ROOT/ihdp_winning_example
"""
from __future__ import annotations
import argparse, glob, os, re, sys
import numpy as np


SHARD_RE = re.compile(r'\.r\d+\.npz$')


def _strict_shards(pattern):
    return sorted(f for f in glob.glob(pattern)
                   if SHARD_RE.search(os.path.basename(f)))


def _by_r(files):
    return {int(f.split('.r')[-1].split('.')[0]): f for f in files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['ihdp', 'acic'], required=True)
    ap.add_argument('--bb-shards',         required=True)
    ap.add_argument('--dopfn-uwyk-shards', required=True)
    ap.add_argument('--fn50-shards',       required=True)
    ap.add_argument('--repo',       required=True)
    ap.add_argument('--causalpfn',  required=True)
    ap.add_argument('--dopfn',      default='')
    ap.add_argument('--checkpoint-dopfn-bb', required=True)
    ap.add_argument('--acic-cache-dir', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--top-k', type=int, default=1,
                     help='Save up to K examples ranked by score (default 1).')
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'l2_ihdp'))
    sys.path.insert(0, os.path.join(args.repo, 'benchmarks', 'l2_acic'))
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
        from eval_realization import _install_dopfn_datasets_shim
        try: _install_dopfn_datasets_shim(args.dopfn)
        except Exception: pass
        from true_acic import load_acic_truth
        from benchmarks import ACIC2016Dataset

    # J=10 evaluation grid from DoPFN-bb checkpoint
    ckpt = torch.load(args.checkpoint_dopfn_bb, map_location='cpu', weights_only=False)
    edges_Y = ckpt['edges'].cpu().numpy()          # J=10 uniform on [-1,1]
    bin_wY  = float(edges_Y[1] - edges_Y[0])       # 0.2
    tau_min, tau_max = 2*edges_Y[0], 2*edges_Y[-1]
    n_tau = int(round((tau_max-tau_min) / bin_wY))
    edges_tau = np.linspace(tau_min, tau_max, n_tau+1)
    bin_wtau = float(edges_tau[1] - edges_tau[0])
    # Shard density grid (Y_CENTERS on [-1.5, 1.5], TAU_FINE_C on [-3, 3])
    Y_CENTERS = 0.5 * (np.linspace(-1.5, 1.5, 101)[:-1] + np.linspace(-1.5, 1.5, 101)[1:])
    Y_BIN = float(Y_CENTERS[1] - Y_CENTERS[0])
    TAU_FINE_C = 0.5 * (np.linspace(-3, 3, 601)[:-1] + np.linspace(-3, 3, 601)[1:])
    TAU_FINE_BIN = float(TAU_FINE_C[1] - TAU_FINE_C[0])
    # Fine plotting grid (scaled Y and scaled τ)
    Y_PLOT   = np.linspace(-1.5, 1.5, 601)
    TAU_PLOT = np.linspace(-2.0, 2.0, 801)

    def _bin_density_on_YCENTERS(d):
        p = np.zeros(len(edges_Y)-1)
        for j in range(len(edges_Y)-1):
            mask = (Y_CENTERS >= edges_Y[j]) & (Y_CENTERS < edges_Y[j+1])
            p[j] = float(d[mask].sum() * Y_BIN)
        s = p.sum()
        return p / s if s > 0 else p

    def _bin_tau_on_TAUFINE(d):
        p = np.zeros(n_tau)
        for k in range(n_tau):
            mask = (TAU_FINE_C >= edges_tau[k]) & (TAU_FINE_C < edges_tau[k+1])
            p[k] = float(d[mask].sum() * TAU_FINE_BIN)
        s = p.sum()
        return p / s if s > 0 else p

    def _truth_bin_y(mu, sigma):
        cdf = norm.cdf(edges_Y, loc=mu, scale=max(sigma, 1e-8))
        pb = np.diff(cdf); s = pb.sum()
        return pb/s if s > 0 else pb

    def _truth_bin_tau(mu_tau, sigma_tau):
        cdf = norm.cdf(edges_tau, loc=mu_tau, scale=max(sigma_tau, 1e-8))
        pb = np.diff(cdf); s = pb.sum()
        return pb/s if s > 0 else pb

    def _l2(p, t, bw):
        return float(np.sqrt(np.sum((np.asarray(p)-np.asarray(t))**2) / bw))

    bb_by_r  = _by_r(_strict_shards(args.bb_shards))
    du_by_r  = _by_r(_strict_shards(args.dopfn_uwyk_shards))
    fn_by_r  = _by_r(_strict_shards(args.fn50_shards))
    rs = sorted(set(bb_by_r) & set(du_by_r) & set(fn_by_r))
    print(f'[load] BB={len(bb_by_r)}  dopfn+uwyk={len(du_by_r)}  '
          f'fn50={len(fn_by_r)}  common r={len(rs)}', flush=True)

    candidates = []   # (score, r, q, L2s dict, densities dict)
    for r in rs:
        # truth
        if args.dataset == 'ihdp':
            cd, _ = IHDPDataset()[r]
            y = np.asarray(cd.y_train.detach().cpu() if hasattr(cd.y_train,'detach') else cd.y_train).reshape(-1)
            tr = load_ihdp_truth(r, args.causalpfn, y)
        else:
            cd, _ = ACIC2016Dataset()[r]
            y = np.asarray(cd.y_train.detach().cpu() if hasattr(cd.y_train,'detach') else cd.y_train).reshape(-1)
            tr = load_acic_truth(r, y, cache_dir=(args.acic_cache_dir or None))
        mu0 = tr.mu0_test_scaled; mu1 = tr.mu1_test_scaled; sigma = float(tr.sigma_scaled)
        n_q = mu0.shape[0]

        with np.load(bb_by_r[r]) as zb, np.load(du_by_r[r]) as zd, np.load(fn_by_r[r]) as zf:
            for q in range(n_q):
                # per-method densities (Y_CENTERS / TAU_FINE_C for tau)
                bb_y0 = zb['ours_dopfn_bb__p_y0'][q]; bb_y1 = zb['ours_dopfn_bb__p_y1'][q]
                bb_tau = zb['ours_dopfn_bb__p_tau'][q]
                dop_y0 = zd['dopfn__p_y0'][q];       dop_y1 = zd['dopfn__p_y1'][q]
                dop_tau = None  # dopfn tau via convolution of bins (below)
                un_y0 = zd['uwyk_noanc__p_y0'][q];   un_y1 = zd['uwyk_noanc__p_y1'][q]
                un_tau = None
                ua_y0 = zd['uwyk_anc__p_y0'][q];    ua_y1 = zd['uwyk_anc__p_y1'][q]
                ua_tau = None
                fn_y0 = zf['ours_fn50__p_y0'][q];   fn_y1 = zf['ours_fn50__p_y1'][q]
                fn_tau = zf['ours_fn50__p_tau'][q]

                # per-bin L2 on J=10 for τ using stored (or convolved for indep-tau methods)
                bb_p_tau  = _bin_tau_on_TAUFINE(bb_tau)
                fn_p_tau  = _bin_tau_on_TAUFINE(fn_tau)
                # Do-PFN/UWYK: convolve marginals to get τ under independence
                def _conv_bin(p_y0_bin, p_y1_bin):
                    p_conv = np.convolve(p_y1_bin, p_y0_bin[::-1], mode='full')
                    offsets = np.arange(-(len(edges_Y)-2), len(edges_Y)-1) * bin_wY
                    p_tau = np.zeros(n_tau)
                    for i, off in enumerate(offsets):
                        k = int(np.clip(np.searchsorted(edges_tau, off, side='right')-1, 0, n_tau-1))
                        p_tau[k] += p_conv[i]
                    return p_tau
                dop_p_y0_bin = _bin_density_on_YCENTERS(dop_y0)
                dop_p_y1_bin = _bin_density_on_YCENTERS(dop_y1)
                un_p_y0_bin  = _bin_density_on_YCENTERS(un_y0)
                un_p_y1_bin  = _bin_density_on_YCENTERS(un_y1)
                ua_p_y0_bin  = _bin_density_on_YCENTERS(ua_y0)
                ua_p_y1_bin  = _bin_density_on_YCENTERS(ua_y1)
                dop_p_tau = _conv_bin(dop_p_y0_bin, dop_p_y1_bin)
                un_p_tau  = _conv_bin(un_p_y0_bin,  un_p_y1_bin)
                ua_p_tau  = _conv_bin(ua_p_y0_bin,  ua_p_y1_bin)

                mu_tau = float(mu1[q] - mu0[q])
                sigma_tau = float(np.sqrt(2.0) * sigma)
                t_tau_bin = _truth_bin_tau(mu_tau, sigma_tau)

                L2_bb    = _l2(bb_p_tau,  t_tau_bin, bin_wtau)
                L2_dop   = _l2(dop_p_tau, t_tau_bin, bin_wtau)
                L2_fn    = _l2(fn_p_tau,  t_tau_bin, bin_wtau)
                L2_un    = _l2(un_p_tau,  t_tau_bin, bin_wtau)
                L2_ua    = _l2(ua_p_tau,  t_tau_bin, bin_wtau)
                L2_best_uwyk = min(L2_un, L2_ua)

                bb_beats  = L2_dop > L2_bb
                fn_beats  = L2_best_uwyk > L2_fn
                if not (bb_beats and fn_beats): continue
                score = (L2_dop - L2_bb) + (L2_best_uwyk - L2_fn)
                candidates.append((score, r, q,
                    dict(L2_bb=L2_bb, L2_dopfn=L2_dop, L2_fn50=L2_fn,
                         L2_uwyk_noanc=L2_un, L2_uwyk_anc=L2_ua),
                    dict(mu0=float(mu0[q]), mu1=float(mu1[q]), sigma=sigma,
                         bb_y0=np.asarray(bb_y0), bb_y1=np.asarray(bb_y1), bb_tau=np.asarray(bb_tau),
                         dop_y0=np.asarray(dop_y0), dop_y1=np.asarray(dop_y1),
                         un_y0=np.asarray(un_y0), un_y1=np.asarray(un_y1),
                         ua_y0=np.asarray(ua_y0), ua_y1=np.asarray(ua_y1),
                         fn_y0=np.asarray(fn_y0), fn_y1=np.asarray(fn_y1),
                         fn_tau=np.asarray(fn_tau))))
        print(f'  r={r}: total candidates so far = {len(candidates)}', flush=True)

    if not candidates:
        print('NO winning candidate found (BB not beating Do-PFN AND fn=50 not beating UWYK)')
        return

    candidates.sort(key=lambda t: -t[0])
    print(f'\n[top {min(args.top_k, len(candidates))} of {len(candidates)} candidates]')
    for i, (sc, r, q, l2s, _) in enumerate(candidates[:args.top_k]):
        print(f'  #{i+1}  seed={r} q={q}  score={sc:.4f}  '
              f'BB={l2s["L2_bb"]:.3f} Do-PFN={l2s["L2_dopfn"]:.3f}  '
              f'fn=50={l2s["L2_fn50"]:.3f}  UWYK-noanc={l2s["L2_uwyk_noanc"]:.3f}  '
              f'UWYK-anc={l2s["L2_uwyk_anc"]:.3f}')

    # Plot top candidate(s)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for rank, (sc, r, q, l2s, dens) in enumerate(candidates[:args.top_k], start=1):
        def _interp_Y(d):
            out = np.interp(Y_PLOT, Y_CENTERS, d, left=0, right=0)
            s = out.sum() * (Y_PLOT[1] - Y_PLOT[0])
            return out / s if s > 0 else out
        def _interp_TAU(d):
            out = np.interp(TAU_PLOT, TAU_FINE_C, d, left=0, right=0)
            s = out.sum() * (TAU_PLOT[1] - TAU_PLOT[0])
            return out / s if s > 0 else out
        def _conv_tau_fine(y0_YC, y1_YC):
            # Convolve the two Y_CENTERS densities → τ density on convolution grid;
            # re-interpolate onto TAU_PLOT
            conv = np.convolve(y1_YC, y0_YC[::-1], mode='full') * Y_BIN
            tau_conv_grid = np.arange(-(len(Y_CENTERS)-1), len(Y_CENTERS)) * Y_BIN
            out = np.interp(TAU_PLOT, tau_conv_grid, conv, left=0, right=0)
            s = out.sum() * (TAU_PLOT[1] - TAU_PLOT[0])
            return out / s if s > 0 else out

        truth_y0  = norm.pdf(Y_PLOT, dens['mu0'], dens['sigma'])
        truth_y1  = norm.pdf(Y_PLOT, dens['mu1'], dens['sigma'])
        truth_tau = norm.pdf(TAU_PLOT, dens['mu1']-dens['mu0'],
                              np.sqrt(2)*dens['sigma'])

        bb_y0_plt  = _interp_Y(dens['bb_y0'])
        bb_y1_plt  = _interp_Y(dens['bb_y1'])
        dop_y0_plt = _interp_Y(dens['dop_y0'])
        dop_y1_plt = _interp_Y(dens['dop_y1'])
        un_y0_plt  = _interp_Y(dens['un_y0'])
        un_y1_plt  = _interp_Y(dens['un_y1'])
        ua_y0_plt  = _interp_Y(dens['ua_y0'])
        ua_y1_plt  = _interp_Y(dens['ua_y1'])
        fn_y0_plt  = _interp_Y(dens['fn_y0'])
        fn_y1_plt  = _interp_Y(dens['fn_y1'])
        bb_tau_plt = _interp_TAU(dens['bb_tau'])
        fn_tau_plt = _interp_TAU(dens['fn_tau'])
        dop_tau_plt = _conv_tau_fine(dens['dop_y0'], dens['dop_y1'])
        un_tau_plt  = _conv_tau_fine(dens['un_y0'],  dens['un_y1'])
        ua_tau_plt  = _conv_tau_fine(dens['ua_y0'],  dens['ua_y1'])

        panels = [
            ('y0',  '$p(Y_{do(0)})$',       Y_PLOT, truth_y0,
             bb_y0_plt, dop_y0_plt, fn_y0_plt, un_y0_plt, ua_y0_plt),
            ('y1',  '$p(Y_{do(1)})$',       Y_PLOT, truth_y1,
             bb_y1_plt, dop_y1_plt, fn_y1_plt, un_y1_plt, ua_y1_plt),
            ('tau', r'$p(\tau|x)$  (CATE)', TAU_PLOT, truth_tau,
             bb_tau_plt, dop_tau_plt, fn_tau_plt, un_tau_plt, ua_tau_plt),
            ('ate', r'$p(\tau_{ATE})$  (single query = CATE)', TAU_PLOT, truth_tau,
             bb_tau_plt, dop_tau_plt, fn_tau_plt, un_tau_plt, ua_tau_plt),
        ]

        for tag, title, xg, tr, bb, dop, fn50, un, ua in panels:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(xg, tr, color='k', lw=2.2, label='truth')
            ax.plot(xg, bb, color='#d62728', lw=1.8, label='Do-PFN-bb (B=1000 2D-marg)')
            ax.plot(xg, dop, color='#1f77b4', lw=1.6, label='Do-PFN')
            ax.plot(xg, fn50, color='#2ca02c', lw=1.6, label='fn=50 (2D-marg B=1000)')
            ax.plot(xg, un, color='#ff7f0e', lw=1.2, ls='--', label='UWYK-NoAnc')
            ax.plot(xg, ua, color='#9467bd', lw=1.2, ls='--', label='UWYK-FullAnc')
            ax.set_title(f'{args.dataset.upper()} — {title}   r={r} q={q}   '
                          f'(BB={l2s["L2_bb"]:.2f} DoPFN={l2s["L2_dopfn"]:.2f} '
                          f'fn50={l2s["L2_fn50"]:.2f} UWYK={min(l2s["L2_uwyk_noanc"], l2s["L2_uwyk_anc"]):.2f})',
                          fontsize=9)
            ax.set_xlabel('scaled Y' if tag in ('y0','y1') else 'scaled τ')
            ax.set_ylabel('density')
            ax.legend(fontsize=8, frameon=False, loc='upper right')
            ax.axvline(0.0, color='k', lw=0.4, alpha=0.3)
            fig.tight_layout()
            suffix = f'_top{rank}' if args.top_k > 1 else ''
            outpng = f'{args.out}{suffix}_{tag}.png'
            fig.savefig(outpng, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'[saved] {outpng}')


if __name__ == '__main__':
    main()
