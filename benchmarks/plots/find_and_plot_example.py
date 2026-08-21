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
    ap.add_argument('--mode', choices=['winner', 'bb_worst'], default='winner',
                     help='winner: (r,q) where BB beats DoPFN AND fn=50 beats UWYK '
                          '(plots all 5 methods). bb_worst: (r,q) where BB has the '
                          'largest τ L2 (plots only Do-PFN and Do-PFN-bb).')
    ap.add_argument('--plot-joint', action='store_true',
                     help='Additionally plot joint p(Y_do0, Y_do1) contours for each '
                          'method + truth (independence for methods without a joint).')
    ap.add_argument('--r', type=int, default=None,
                     help='Skip candidate search and plot this realization directly. '
                          'Requires --q too.')
    ap.add_argument('--q', type=int, default=None,
                     help='Skip candidate search and plot this query directly. '
                          'Requires --r too.')
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

    # ── Fast path: skip candidate search if (r, q) explicitly given ──────
    if args.r is not None and args.q is not None:
        rs = [args.r] if args.r in rs else []
        if not rs:
            print(f'ERROR: --r {args.r} not in common realizations'); return
        print(f'[fast-path] plotting r={args.r} q={args.q} directly (skipping search)',
              flush=True)

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
            q_range = ([args.q] if args.q is not None
                        else range(n_q))
            for q in q_range:
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

                if args.r is not None and args.q is not None:
                    score = 0.0                  # fast path: don't filter, don't rank
                elif args.mode == 'winner':
                    bb_beats  = L2_dop > L2_bb
                    fn_beats  = L2_best_uwyk > L2_fn
                    if not (bb_beats and fn_beats): continue
                    score = (L2_dop - L2_bb) + (L2_best_uwyk - L2_fn)
                else:                           # bb_worst
                    score = L2_bb               # rank by BB τ L2, largest first
                # MEMORY-LITE: only keep score + (r, q, sigma, mu0, mu1) + L2s.
                # Densities are re-loaded on demand for the top-k winners.
                candidates.append((score, r, q,
                    dict(L2_bb=L2_bb, L2_dopfn=L2_dop, L2_fn50=L2_fn,
                         L2_uwyk_noanc=L2_un, L2_uwyk_anc=L2_ua),
                    dict(mu0=float(mu0[q]), mu1=float(mu1[q]), sigma=sigma)))
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
        # Re-load only THIS realization's shards to get densities for the winner
        with np.load(bb_by_r[r]) as zb, np.load(du_by_r[r]) as zd, np.load(fn_by_r[r]) as zf:
            dens['bb_y0']  = np.asarray(zb['ours_dopfn_bb__p_y0'][q])
            dens['bb_y1']  = np.asarray(zb['ours_dopfn_bb__p_y1'][q])
            dens['bb_tau'] = np.asarray(zb['ours_dopfn_bb__p_tau'][q])
            dens['dop_y0'] = np.asarray(zd['dopfn__p_y0'][q])
            dens['dop_y1'] = np.asarray(zd['dopfn__p_y1'][q])
            dens['un_y0']  = np.asarray(zd['uwyk_noanc__p_y0'][q])
            dens['un_y1']  = np.asarray(zd['uwyk_noanc__p_y1'][q])
            dens['ua_y0']  = np.asarray(zd['uwyk_anc__p_y0'][q])
            dens['ua_y1']  = np.asarray(zd['uwyk_anc__p_y1'][q])
            dens['fn_y0']  = np.asarray(zf['ours_fn50__p_y0'][q])
            dens['fn_y1']  = np.asarray(zf['ours_fn50__p_y1'][q])
            dens['fn_tau'] = np.asarray(zf['ours_fn50__p_tau'][q])

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

        # ═══ REFERENCE STYLE (matches benchmarks/plots/ihdp_n10/UWYK-2DMALC-VS-TRUE) ═══
        # One figure, one panel per method. Each panel:
        #   truth p(Y_do0), p(Y_do1) as red dotted bells + red dots at μ_true
        #   method p(Y_do0) solid blue + blue dot at method's E[Y_do0]
        #   method p(Y_do1) solid purple + purple dot at method's E[Y_do1]

        PALETTE = {'do0': '#2E7DAF', 'do1': '#7B3E9E'}   # reference blue / purple

        if args.mode == 'winner':
            method_list = [
                ('Do-PFN-bb', 'bb'),
                ('Do-PFN',    'dop'),
                ('fn=50',     'fn'),
                ('UWYK-NoAnc','un'),
                ('UWYK-FullAnc','ua'),
            ]
        else:  # bb_worst
            method_list = [
                ('Do-PFN-bb', 'bb'),
                ('Do-PFN',    'dop'),
            ]
        method_y0 = {'bb': bb_y0_plt, 'dop': dop_y0_plt, 'fn': fn_y0_plt,
                     'un': un_y0_plt, 'ua': ua_y0_plt}
        method_y1 = {'bb': bb_y1_plt, 'dop': dop_y1_plt, 'fn': fn_y1_plt,
                     'un': un_y1_plt, 'ua': ua_y1_plt}

        def _est_mean(x, p):
            dx = float(x[1] - x[0])
            s = p.sum() * dx
            return float((x * p).sum() * dx / s) if s > 0 else float('nan')
        def _pt_on(x, p, mu):
            return mu, float(np.interp(mu, x, p))

        n = len(method_list)
        n_cols = n if n <= 3 else 3
        n_rows = (n + n_cols - 1) // n_cols

        # ─── Panel 1: Marginals ────────────────────────────────────────
        fig, axes = plt.subplots(n_rows, n_cols,
                                   figsize=(5.4 * n_cols, 3.6 * n_rows),
                                   squeeze=False)
        for k, (label, key) in enumerate(method_list):
            ax = axes[k // n_cols][k % n_cols]
            ax.plot(Y_PLOT, truth_y0, color='red', ls=':', lw=1.7)
            ax.plot(Y_PLOT, truth_y1, color='red', ls=':', lw=1.7)
            for mu in (dens['mu0'], dens['mu1']):
                px, py = _pt_on(Y_PLOT, truth_y0 if mu == dens['mu0'] else truth_y1, mu)
                ax.plot(px, py, 'o', color='red', markersize=9,
                         markeredgecolor='white', markeredgewidth=1.0, zorder=6)
            p0 = method_y0[key]; p1 = method_y1[key]
            ax.plot(Y_PLOT, p0, color=PALETTE['do0'], lw=1.9,
                     label=r'$p(Y_{do0})$' if k == 0 else None)
            ax.plot(Y_PLOT, p1, color=PALETTE['do1'], lw=1.9,
                     label=r'$p(Y_{do1})$' if k == 0 else None)
            E0 = _est_mean(Y_PLOT, p0); E1 = _est_mean(Y_PLOT, p1)
            ax.plot(E0, float(np.interp(E0, Y_PLOT, p0)), 'o',
                     color=PALETTE['do0'], markersize=9,
                     markeredgecolor='white', markeredgewidth=1.0, zorder=5)
            ax.plot(E1, float(np.interp(E1, Y_PLOT, p1)), 'o',
                     color=PALETTE['do1'], markersize=9,
                     markeredgecolor='white', markeredgewidth=1.0, zorder=5)
            ax.set_title(label, fontsize=11)
            if k // n_cols == n_rows - 1: ax.set_xlabel(r'$Y$  (scaled)')
            if k %  n_cols == 0:          ax.set_ylabel('density')
            ax.set_xlim(-1.0, 1.0)
            ax.grid(alpha=0.25)
            if k == 0: ax.legend(fontsize=9, loc='upper right')
        for k in range(n, n_rows * n_cols):
            axes[k // n_cols][k % n_cols].set_visible(False)
        tau_true = dens['mu1'] - dens['mu0']
        fig.suptitle(f'{args.dataset.upper()} r={r} q={q}   '
                      f'$\\tau_{{true}}$={tau_true:+.2f}   —   marginals vs TRUE',
                      fontsize=12, y=1.0)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        suffix = f'_top{rank}' if args.top_k > 1 else ''
        outpng = f'{args.out}{suffix}_marginals.png'
        fig.savefig(outpng, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {outpng}')

        # ─── Panel 2: CATE (per query, single r/q) ─────────────────────
        # Orange fill for method, red dotted truth, dot at both means.
        ORANGE = '#C1420F'
        TAU_C_PLOT = np.linspace(-1.5, 1.5, 601)
        sigma_tau  = float(np.sqrt(2.0) * dens['sigma'])
        mu_tau     = float(dens['mu1'] - dens['mu0'])
        truth_tau_c = norm.pdf(TAU_C_PLOT, mu_tau, sigma_tau)

        # Method CATE densities on TAU_C_PLOT
        def _to_TAUC(d_native):
            out = np.interp(TAU_C_PLOT, TAU_FINE_C, d_native, left=0, right=0)
            s = out.sum() * (TAU_C_PLOT[1] - TAU_C_PLOT[0])
            return out / s if s > 0 else out
        def _conv_tau_c(y0_YC, y1_YC):
            conv = np.convolve(y1_YC, y0_YC[::-1], mode='full') * Y_BIN
            tau_conv_grid = np.arange(-(len(Y_CENTERS)-1), len(Y_CENTERS)) * Y_BIN
            out = np.interp(TAU_C_PLOT, tau_conv_grid, conv, left=0, right=0)
            s = out.sum() * (TAU_C_PLOT[1] - TAU_C_PLOT[0])
            return out / s if s > 0 else out
        method_tau = {
            'bb':  _to_TAUC(dens['bb_tau']),
            'dop': _conv_tau_c(dens['dop_y0'], dens['dop_y1']),
            'fn':  _to_TAUC(dens['fn_tau']),
            'un':  _conv_tau_c(dens['un_y0'],  dens['un_y1']),
            'ua':  _conv_tau_c(dens['ua_y0'],  dens['ua_y1']),
        }
        fig, axes = plt.subplots(n_rows, n_cols,
                                   figsize=(5.4 * n_cols, 3.6 * n_rows),
                                   squeeze=False)
        for k, (label, key) in enumerate(method_list):
            ax = axes[k // n_cols][k % n_cols]
            p_tau = method_tau[key]
            ax.fill_between(TAU_C_PLOT, p_tau, alpha=0.25, color=ORANGE)
            ax.plot(TAU_C_PLOT, p_tau, color=ORANGE, lw=2.0,
                     label=r'method $p(\tau)$' if k == 0 else None)
            ax.plot(TAU_C_PLOT, truth_tau_c, color='red', ls=':', lw=1.7,
                     label=r'true $p(\tau)$' if k == 0 else None)
            E_tau = _est_mean(TAU_C_PLOT, p_tau)
            ax.plot(E_tau, float(np.interp(E_tau, TAU_C_PLOT, p_tau)),
                     'o', color=ORANGE, markersize=9,
                     markeredgecolor='white', markeredgewidth=1.0, zorder=5)
            ax.plot(mu_tau, float(np.interp(mu_tau, TAU_C_PLOT, truth_tau_c)),
                     'o', color='red', markersize=9,
                     markeredgecolor='white', markeredgewidth=1.0, zorder=6)
            ax.set_title(f'{label}   $E={E_tau:+.2f}$   '
                          f'$\\tau_{{true}}={mu_tau:+.2f}$', fontsize=10)
            ax.set_xlim(-1.5, 1.5)
            if k // n_cols == n_rows - 1: ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$  (scaled)')
            if k %  n_cols == 0:          ax.set_ylabel(r'$p(\tau)$')
            ax.grid(alpha=0.25)
            if k == 0: ax.legend(fontsize=9, loc='upper right')
        for k in range(n, n_rows * n_cols):
            axes[k // n_cols][k % n_cols].set_visible(False)
        fig.suptitle(f'{args.dataset.upper()} r={r} q={q}   '
                      f'$\\tau_{{true}}$={mu_tau:+.2f}   —   CATE vs TRUE',
                      fontsize=12, y=1.0)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        outpng = f'{args.out}{suffix}_cate.png'
        fig.savefig(outpng, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {outpng}')

        # ─── Panel 3: ATE (aggregated over ALL queries in realization r) ─
        # Load per-realization p_ate stored in each shard, plus compute the
        # truth ATE as the barycentre of per-query truth CATE Gaussians.
        with np.load(bb_by_r[r]) as zb, np.load(du_by_r[r]) as zd, np.load(fn_by_r[r]) as zf:
            method_ate = {}
            method_ate['bb']  = _to_TAUC(zb['ours_dopfn_bb__p_ate'])
            method_ate['dop'] = _to_TAUC(zd['dopfn__p_ate'])
            method_ate['un']  = _to_TAUC(zd['uwyk_noanc__p_ate'])
            method_ate['ua']  = _to_TAUC(zd['uwyk_anc__p_ate'])
            method_ate['fn']  = _to_TAUC(zf['ours_fn50__p_ate'])
        # Truth ATE: barycentre of per-query truth CATE densities
        if args.dataset == 'ihdp':
            cd_full, _ = IHDPDataset()[r]
            y_full = np.asarray(cd_full.y_train.detach().cpu()
                                if hasattr(cd_full.y_train, 'detach') else cd_full.y_train).reshape(-1)
            tr_full = load_ihdp_truth(r, args.causalpfn, y_full)
        else:
            cd_full, _ = ACIC2016Dataset()[r]
            y_full = np.asarray(cd_full.y_train.detach().cpu()
                                if hasattr(cd_full.y_train, 'detach') else cd_full.y_train).reshape(-1)
            tr_full = load_acic_truth(r, y_full, cache_dir=(args.acic_cache_dir or None))
        mu_taus = np.asarray(tr_full.mu1_test_scaled).reshape(-1) - \
                  np.asarray(tr_full.mu0_test_scaled).reshape(-1)
        sigma_tau_r = float(np.sqrt(2.0) * tr_full.sigma_scaled)
        truth_cate_stack = np.stack([norm.pdf(TAU_C_PLOT, mt, sigma_tau_r) for mt in mu_taus])
        truth_ate = wasserstein_barycenter_1d(truth_cate_stack, TAU_C_PLOT)
        truth_ate /= max(truth_ate.sum() * (TAU_C_PLOT[1]-TAU_C_PLOT[0]), 1e-12)
        mu_ate_true = float(mu_taus.mean())

        fig, axes = plt.subplots(n_rows, n_cols,
                                   figsize=(5.4 * n_cols, 3.6 * n_rows),
                                   squeeze=False)
        for k, (label, key) in enumerate(method_list):
            ax = axes[k // n_cols][k % n_cols]
            p_ate = method_ate[key]
            ax.fill_between(TAU_C_PLOT, p_ate, alpha=0.25, color=ORANGE)
            ax.plot(TAU_C_PLOT, p_ate, color=ORANGE, lw=2.0,
                     label=r'method $p(\tau_{ATE})$' if k == 0 else None)
            ax.plot(TAU_C_PLOT, truth_ate, color='red', ls=':', lw=1.7,
                     label=r'true $p(\tau_{ATE})$' if k == 0 else None)
            E_ate = _est_mean(TAU_C_PLOT, p_ate)
            ax.plot(E_ate, float(np.interp(E_ate, TAU_C_PLOT, p_ate)),
                     'o', color=ORANGE, markersize=9,
                     markeredgecolor='white', markeredgewidth=1.0, zorder=5)
            ax.plot(mu_ate_true, float(np.interp(mu_ate_true, TAU_C_PLOT, truth_ate)),
                     'o', color='red', markersize=9,
                     markeredgecolor='white', markeredgewidth=1.0, zorder=6)
            ax.set_title(f'{label}   $E={E_ate:+.2f}$   '
                          f'$\\tau_{{ATE,true}}={mu_ate_true:+.2f}$', fontsize=10)
            ax.set_xlim(-1.5, 1.5)
            if k // n_cols == n_rows - 1: ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$  (scaled)')
            if k %  n_cols == 0:          ax.set_ylabel(r'$p(\tau_{ATE})$')
            ax.grid(alpha=0.25)
            if k == 0: ax.legend(fontsize=9, loc='upper right')
        for k in range(n, n_rows * n_cols):
            axes[k // n_cols][k % n_cols].set_visible(False)
        fig.suptitle(f'{args.dataset.upper()} r={r}   '
                      f'$\\tau_{{ATE,true}}$={mu_ate_true:+.2f}   —   ATE vs TRUE '
                      f'(barycentre over all queries)',
                      fontsize=12, y=1.0)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        outpng = f'{args.out}{suffix}_ate.png'
        fig.savefig(outpng, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {outpng}')

        # ─── Joint 2D heatmaps (matches ihdp_n10 UWYK-2DMALC/joint.png) ──
        if args.plot_joint:
            xs = np.linspace(-1.0, 1.0, 201)
            ys = xs
            XX, YY = np.meshgrid(xs, ys)
            sigma_sc = dens['sigma']
            def _marg_on(x, d):
                return np.interp(x, Y_CENTERS, d, left=0, right=0)

            # Truth = independent Gaussians in scaled Y (holds for IHDP/ACIC)
            joint_truth = (norm.pdf(XX, dens['mu0'], sigma_sc) *
                            norm.pdf(YY, dens['mu1'], sigma_sc))
            joint_truth /= max(joint_truth.sum() * (xs[1]-xs[0]) * (ys[1]-ys[0]), 1e-12)
            joints = [('Truth', joint_truth)]
            joints.append(('Do-PFN-bb',
                            np.outer(_marg_on(ys, dens['bb_y1']),
                                      _marg_on(xs, dens['bb_y0']))))
            joints.append(('Do-PFN',
                            np.outer(_marg_on(ys, dens['dop_y1']),
                                      _marg_on(xs, dens['dop_y0']))))
            if args.mode == 'winner':
                joints.append(('fn=50',
                                np.outer(_marg_on(ys, dens['fn_y1']),
                                          _marg_on(xs, dens['fn_y0']))))
                joints.append(('UWYK-NoAnc',
                                np.outer(_marg_on(ys, dens['un_y1']),
                                          _marg_on(xs, dens['un_y0']))))
                joints.append(('UWYK-FullAnc',
                                np.outer(_marg_on(ys, dens['ua_y1']),
                                          _marg_on(xs, dens['ua_y0']))))

            n = len(joints)
            n_cols = n if n <= 3 else 3
            n_rows = (n + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols,
                                       figsize=(4.6 * n_cols, 4.2 * n_rows),
                                       squeeze=False)
            extent = [-1.0, 1.0, -1.0, 1.0]
            for k, (label, J) in enumerate(joints):
                ax = axes[k // n_cols][k % n_cols]
                # Normalise to per-unit-area density for visual comparability
                dxdy = (xs[1] - xs[0]) * (ys[1] - ys[0])
                s = J.sum() * dxdy
                Jn = J / s if s > 0 else J
                im = ax.imshow(Jn, origin='lower', extent=extent,
                                aspect='equal', cmap='viridis')
                ax.plot([-1, 1], [-1, 1], color='red', ls=':', lw=1.0)
                ax.plot([dens['mu0']], [dens['mu1']], marker='o', color='red',
                         markersize=8, markeredgecolor='white', markeredgewidth=1.0,
                         zorder=6)
                ax.set_title(label, fontsize=11)
                if k // n_cols == n_rows - 1: ax.set_xlabel(r'$Y_{do0}$  (scaled)')
                if k %  n_cols == 0:          ax.set_ylabel(r'$Y_{do1}$  (scaled)')
                plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
            for k in range(n, n_rows * n_cols):
                axes[k // n_cols][k % n_cols].set_visible(False)
            tau_true = dens['mu1'] - dens['mu0']
            fig.suptitle(f'{args.dataset.upper()} r={r} q={q}   '
                          f'$\\tau_{{true}}$={tau_true:+.2f}   joint '
                          f'$p(Y_{{do0}}, Y_{{do1}})$',
                          fontsize=12, y=0.999)
            fig.tight_layout(rect=[0, 0, 1, 0.985])
            outpng = f'{args.out}{suffix}_joint.png'
            fig.savefig(outpng, dpi=140, bbox_inches='tight')
            plt.close(fig)
            print(f'[saved] {outpng}')


if __name__ == '__main__':
    main()
