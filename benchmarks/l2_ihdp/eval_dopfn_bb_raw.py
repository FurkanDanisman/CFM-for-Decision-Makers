"""Fast IHDP eval for a DoPFN-backbone Ours checkpoint — inner-only-mean AND
full 9-region mean CATE. No MALC 2D, no density L2/KL.

Two CATE estimators are reported side-by-side:

  * INNER-ONLY (aka 'raw' in output) — μ = Σ p_i · c_i on the J² inner bin
    centers only. Ignores the 8 tail regions of the 2D BarDist head, so
    any predicted mass beyond [-1, 1] in scaled Y is invisible. On IHDP
    realizations with large y_rng (outlier-heavy training Y), true CATE
    in scaled units can be << bin_width and the model puts its mass in
    the outer regions — inner-only mean systematically under-predicts.

  * FULL 9-REGION — the 2D BarDist output is a mixture of 9 regions
    (inner + 8 outer / mixed) with 9 mixture weights w_region and 4
    half-Gaussian tail scales sL0/sR0/sL1/sR1 (see losses/BarDistribution2D.py).
    For each arm t ∈ {0,1}:
        E[Y_t] = P_inner_t · E[Y_t | inner marginal]
                + P_L_t     · (−1 − σ_L·√(2/π))
                + P_R_t     · (+1 + σ_R·√(2/π))
    where P_inner/L/R_t sum the appropriate w_region entries and
    E[Y_t | inner marginal] is the bin-center-weighted mean of the inner
    marginal (p_mat.sum axis). Half-Gaussian expected values follow from
    the tail parametrisation used in training.
    Result: predictions can extend outside [-1, 1] naturally, matching
    what the head was trained to represent.

Usage:
    python eval_dopfn_bb_raw.py \
        --repo      $DEPLOY_ROOT/R-PFN \
        --dopfn     $DEPLOY_ROOT/external/dopfn \
        --causalpfn $DEPLOY_ROOT/external/causalpfn \
        --checkpoint $DEPLOY_ROOT/checkpoints_dopfn_backbone_realj10/step_50000.pt \
        --n-realizations 100
"""
from __future__ import annotations
import argparse, os, sys, time, types, traceback
import numpy as np
import torch
from scipy.stats import norm as _scipy_norm


# ── EM mean correction on binned Gaussian data ─────────────────────────────
# Copied verbatim from MALC/malc_2d.py::_em_mean_2d (which is 1D despite the
# name — used per-axis in the 2D MALC pipeline). Duplicated here so the eval
# doesn't need to pull in the whole MALC module (which drags Cython deps).
def _em_mean_1d(props, grid_edges, sigma, start,
                 max_step=1000, eps2=1e-10, eps1=1e-5):
    """One-dimensional truncated-normal EM for the latent Gaussian mean
    given binned probabilities.

    Args:
        props: (J,) probabilities per bin (sums to 1).
        grid_edges: (J+1,) bin edges.
        sigma: initial Gaussian std (kept fixed across iterations).
        start: initial μ guess (typically raw bin-center mean).
    """
    pn = props / max(props.sum(), 1e-300)
    mu = start
    for _ in range(max_step):
        a = (grid_edges - mu) / sigma
        G1 = _scipy_norm.cdf(a)
        G2 = _scipy_norm.pdf(a)
        temp = (np.diff(G2) + eps2) / (np.diff(G1) + eps2)
        mu_new = mu - sigma * float(np.sum(pn * temp))
        if abs(mu_new - mu) < eps1:
            return mu_new
        mu = mu_new
    return mu


def _init_sigma_1d(props, centers, mean_est, bin_width):
    """Sheppard-corrected σ² = Var_p[c] + Δ²/12. Guarantees positive."""
    var = float(np.sum(props * (centers - mean_est) ** 2)) + (bin_width ** 2) / 12.0
    s = float(np.sqrt(max(var, (bin_width / 12.0) ** 2)))
    return s if np.isfinite(s) and s > 0 else bin_width


def _install_dopfn_datasets_shim(dopfn_dir):
    if 'datasets' in sys.modules:
        return
    sys.path.insert(0, dopfn_dir)
    ds_mod = types.ModuleType('datasets')
    with open(os.path.join(dopfn_dir, 'datasets/__init__.py')) as fp:
        src = fp.read().split('def load_semi_real')[0]
    exec(src, ds_mod.__dict__)
    sys.modules['datasets'] = ds_mod


def _np(a):
    if isinstance(a, torch.Tensor): return a.numpy()
    return np.asarray(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo',            required=True)
    ap.add_argument('--dopfn',           required=True)
    ap.add_argument('--causalpfn',       required=True)
    ap.add_argument('--checkpoint',      required=True,
                    help='Path to DoPFN-backbone checkpoint (.pt with model_state_dict + config + edges).')
    ap.add_argument('--n-realizations',  type=int, default=100,
                    help='How many IHDP realizations to score (0..N-1). IHDP has 100 total.')
    ap.add_argument('--start-realization', type=int, default=0)
    ap.add_argument('--out', default='',
                    help='Optional .npz with per-realization PEHE / eps_ATE arrays.')
    ap.add_argument('--n-context', type=int, default=0,
                    help='If > 0, subsample this many context rows per realization.')
    ap.add_argument('--malc-upsample', action='store_true',
                    help='Also fit MALC 2D to the inner p_mat and re-sample on a fine '
                         'grid to compute an upsampled-mean CATE (inner MALC + tail '
                         'expectations). Slower (~0.2-1s per query).')
    ap.add_argument('--malc-n-eval', type=int, default=100,
                    help='Fine grid resolution when --malc-upsample is set (default 100).')
    ap.add_argument('--malc-max-K', type=int, default=1,
                    help='Max mixture components for MALC 2D fit (default 1 = single log-concave).')
    ap.add_argument('--malc-B',     type=int, default=100,
                    help='B_fit / B_select for MALC 2D fit (default 100 — smaller than the '
                         'main pipeline default 500 for speed; L2/KL not evaluated here so '
                         'precision matters less).')
    args = ap.parse_args()

    # ── Paths ─────────────────────────────────────────────────────────────
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.join(args.repo, 'training_dopfn_base'))
    sys.path.insert(0, _here)

    from true_ihdp import load_ihdp_truth
    from dopfn_backbone_head import DoPFNBackboneWith2DHead
    from losses.BarDistribution2D import unpack_pred

    fit_malc_inner = dmalc_2d = None
    fine_centers = fine_edges = fine_eval_pts = fine_bw = None
    if args.malc_upsample:
        from losses.BarDistribution2D import fit_malc_inner  # noqa: F811
        malc_dir = os.path.join(args.repo, 'MALC')
        if malc_dir not in sys.path: sys.path.insert(0, malc_dir)
        from malc_2d import dmalc_2d  # noqa: F811
        # fine grid on the same scaled [-1, 1] support
        n_ev = args.malc_n_eval
        fine_edges = np.linspace(-1.0, 1.0, n_ev + 1, dtype=np.float64)
        fine_centers = 0.5 * (fine_edges[:-1] + fine_edges[1:])
        fine_bw = float(fine_edges[1] - fine_edges[0])
        XX, YY = np.meshgrid(fine_centers, fine_centers, indexing='xy')
        fine_eval_pts = np.column_stack([XX.ravel(), YY.ravel()])
        print(f'[malc] upsample enabled: n_eval={n_ev}  max_K={args.malc_max_K}  '
              f'B={args.malc_B}', flush=True)

    _install_dopfn_datasets_shim(args.dopfn)
    sys.path.insert(0, args.causalpfn)
    from benchmarks import IHDPDataset

    # ── Load checkpoint ───────────────────────────────────────────────────
    print(f'[load] {args.checkpoint}', flush=True)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['config']; J = int(cfg['J'])
    edges_np = ckpt['edges'].cpu().numpy()
    bin_width = float(edges_np[1] - edges_np[0])
    centers_scaled = 0.5 * (edges_np[:-1] + edges_np[1:])   # (J,) in scaled [-1, 1]
    step = int(ckpt.get('step', -1))
    print(f'[ckpt] J={J}  edges=[{edges_np[0]:+.2f}, {edges_np[-1]:+.2f}]  '
          f'bw={bin_width:.4f}  step={step}', flush=True)

    model = DoPFNBackboneWith2DHead(dopfn_root=args.dopfn, K=J).eval()
    model.load_state_dict(ckpt['model_state_dict'])

    # ── Iterate over realizations ─────────────────────────────────────────
    pehe_list, eps_ate_list = [], []
    pehe_em_k1_list, eps_em_k1_list = [], []
    pehe_malc_list, eps_malc_list = [], []
    pehe_malc_em_list, eps_malc_em_list = [], []
    std_y0_list, std_y1_list = [], []
    t_all = time.time()
    end = min(args.start_realization + args.n_realizations, 100)
    for r in range(args.start_realization, end):
        t_r = time.time()
        cd, _ = IHDPDataset()[r]
        y_train_full = _np(cd.y_train)
        truth = load_ihdp_truth(r, args.causalpfn, y_train_full)
        y_min = float(truth.y_min); y_rng = float(truth.y_rng)

        # Subsample context if requested
        if args.n_context > 0 and args.n_context < cd.X_train.shape[0]:
            rng = np.random.default_rng(r)
            idx = rng.choice(cd.X_train.shape[0], args.n_context, replace=False)
            X_ctx = _np(cd.X_train)[idx]
            T_ctx = _np(cd.t_train)[idx]
            Y_ctx = _np(cd.y_train)[idx]
        else:
            X_ctx = _np(cd.X_train)
            T_ctx = _np(cd.t_train)
            Y_ctx = _np(cd.y_train)
        X_qry = _np(cd.X_test)

        # Rescale Y to [-1, 1] using truth's y_min / y_rng
        Y_ctx_s = ((Y_ctx.reshape(-1) - y_min) / y_rng * 2.0 - 1.0).astype(np.float32)

        X_ctx_t = torch.from_numpy(X_ctx.astype(np.float32)).unsqueeze(0)
        T_ctx_t = torch.from_numpy(T_ctx.astype(np.float32).reshape(-1, 1)).unsqueeze(0)
        Y_ctx_t = torch.from_numpy(Y_ctx_s.reshape(-1, 1)).unsqueeze(0)
        X_qry_t = torch.from_numpy(X_qry.astype(np.float32)).unsqueeze(0)

        with torch.no_grad():
            pred = model(X_ctx_t, T_ctx_t, Y_ctx_t, X_qry_t)['predictions'][0]

        n_test = X_qry.shape[0]
        cate_raw_scaled      = np.zeros(n_test, dtype=np.float64)   # inner-only Σ p·c
        cate_em_scaled       = np.zeros(n_test, dtype=np.float64)   # inner + tails (9 regions)
        cate_malc_raw_scaled = np.zeros(n_test, dtype=np.float64)   # MALC-upsampled Σ p·c (no tails)
        cate_malc_em_scaled  = np.zeros(n_test, dtype=np.float64)   # MALC-upsampled + truncnorm EM
        sigma_em_y0_scaled = np.zeros(n_test, dtype=np.float64)
        sigma_em_y1_scaled = np.zeros(n_test, dtype=np.float64)
        n_malc_fail = 0
        bin_w_scaled = float(centers_scaled[1] - centers_scaled[0])
        lo = float(edges_np[0])   # -1 in scaled space
        hi = float(edges_np[-1])  # +1
        _sqrt_2_over_pi = float(np.sqrt(2.0 / np.pi))

        # Region layout (see losses/BarDistribution2D.py):
        # 0=inner-inner  1=L0-inner  2=R0-inner  3=inner-L1  4=inner-R1
        # 5=L0-L1  6=L0-R1  7=R0-L1  8=R0-R1
        # Y_do0 side:  inner={0,3,4}  L0={1,5,6}  R0={2,7,8}
        # Y_do1 side:  inner={0,1,2}  L1={3,5,7}  R1={4,6,8}
        Y0_INNER = [0, 3, 4]; Y0_L = [1, 5, 6]; Y0_R = [2, 7, 8]
        Y1_INNER = [0, 1, 2]; Y1_L = [3, 5, 7]; Y1_R = [4, 6, 8]

        for q in range(n_test):
            p_mat, w_region, sL0, sR0, sL1, sR1 = unpack_pred(pred[q], J, bin_width)
            pm = p_mat.detach().cpu().numpy().astype(np.float64)
            s = pm.sum()
            if s > 0: pm /= s
            w = w_region.detach().cpu().numpy().astype(np.float64)
            sL0_v = float(sL0); sR0_v = float(sR0)
            sL1_v = float(sL1); sR1_v = float(sR1)
            p_marg0 = pm.sum(axis=1)  # inner marg for Y0 (conditional on Y0∈inner)
            p_marg1 = pm.sum(axis=0)  # inner marg for Y1

            # ── OLD raw bin-center mean (inner region only, ignores tails) ─
            mean0_inner_s = float((centers_scaled * p_marg0).sum())
            mean1_inner_s = float((centers_scaled * p_marg1).sum())
            cate_raw_scaled[q] = mean1_inner_s - mean0_inner_s

            # ── FULL 9-region mean (inner + tail regions) ────────────────
            # Y_do0 side:
            P0_inner = float(w[Y0_INNER].sum())
            P0_L     = float(w[Y0_L].sum())
            P0_R     = float(w[Y0_R].sum())
            # Half-Gaussian tail expectations at boundaries ±1 with scales σ:
            #   E[Y | Y < lo] = lo - σ · √(2/π)
            #   E[Y | Y > hi] = hi + σ · √(2/π)
            E0_L = lo - sL0_v * _sqrt_2_over_pi
            E0_R = hi + sR0_v * _sqrt_2_over_pi
            mean0_full_s = (P0_inner * mean0_inner_s
                             + P0_L     * E0_L
                             + P0_R     * E0_R)

            # Y_do1 side:
            P1_inner = float(w[Y1_INNER].sum())
            P1_L     = float(w[Y1_L].sum())
            P1_R     = float(w[Y1_R].sum())
            E1_L = lo - sL1_v * _sqrt_2_over_pi
            E1_R = hi + sR1_v * _sqrt_2_over_pi
            mean1_full_s = (P1_inner * mean1_inner_s
                             + P1_L     * E1_L
                             + P1_R     * E1_R)

            cate_em_scaled[q] = mean1_full_s - mean0_full_s

            # ── Pure MALC-upsampled means (raw + EM), no tails ───────────
            # User's literal proposal: fit MALC 2D to the discrete J=10 inner
            # p_mat, evaluate on a fine grid (default J=100), marginalize,
            # compute BOTH raw and EM mean on the fine grid. Tails are NOT
            # added — this isolates the effect of MALC upsampling itself.
            if args.malc_upsample:
                try:
                    fit = fit_malc_inner(
                        pm.T, edges_np, edges_np,
                        B_fit=args.malc_B, B_select=args.malc_B,
                        max_K=args.malc_max_K,
                        seed=int((q + 1) * 1_000_003 + r) & 0x7fffffff,
                        parallel=False,
                    )
                    dens = dmalc_2d(fit, fine_eval_pts).reshape(
                        args.malc_n_eval, args.malc_n_eval)
                    # convert density → prob mass by · bin_area, renormalize
                    p_fine = dens * (fine_bw * fine_bw)
                    ps = p_fine.sum()
                    if ps > 0: p_fine = p_fine / ps
                    # AXIS CONVENTION: after dmalc_2d(...).reshape(n_ev, n_ev),
                    # first axis (rows) is Y1 index, second axis (cols) is Y0
                    # index (see methods_densities.py:345 for reference).
                    # So Y0 marginal = sum over rows (axis=0);
                    #    Y1 marginal = sum over cols (axis=1).
                    p_marg0_fine = p_fine.sum(axis=0)   # Y0 marginal
                    p_marg1_fine = p_fine.sum(axis=1)   # Y1 marginal
                    # Raw mean on fine grid
                    m0_raw_f = float((fine_centers * p_marg0_fine).sum())
                    m1_raw_f = float((fine_centers * p_marg1_fine).sum())
                    cate_malc_raw_scaled[q] = m1_raw_f - m0_raw_f
                    # EM mean on fine grid (truncnorm fixed-point)
                    sig0_f = _init_sigma_1d(p_marg0_fine, fine_centers, m0_raw_f, fine_bw)
                    sig1_f = _init_sigma_1d(p_marg1_fine, fine_centers, m1_raw_f, fine_bw)
                    m0_em_f = _em_mean_1d(p_marg0_fine, fine_edges, sig0_f, m0_raw_f)
                    m1_em_f = _em_mean_1d(p_marg1_fine, fine_edges, sig1_f, m1_raw_f)
                    cate_malc_em_scaled[q] = m1_em_f - m0_em_f
                except Exception:
                    n_malc_fail += 1
                    cate_malc_raw_scaled[q] = cate_raw_scaled[q]  # fallback to inner-only
                    cate_malc_em_scaled[q]  = cate_raw_scaled[q]

            sigma_em_y0_scaled[q] = 0.5 * (sL0_v + sR0_v)
            sigma_em_y1_scaled[q] = 0.5 * (sL1_v + sR1_v)

        cate_pred_raw      = cate_raw_scaled      * (y_rng / 2.0)  # inner-only
        cate_pred_em       = cate_em_scaled       * (y_rng / 2.0)  # inner + tails
        cate_pred_malc_raw = cate_malc_raw_scaled * (y_rng / 2.0)  # MALC upsamp raw
        cate_pred_malc_em  = cate_malc_em_scaled  * (y_rng / 2.0)  # MALC upsamp + EM
        sigma_em_y0_raw = sigma_em_y0_scaled * (y_rng / 2.0)
        sigma_em_y1_raw = sigma_em_y1_scaled * (y_rng / 2.0)

        true_cate_raw = _np(cd.true_cate).reshape(-1).astype(np.float64)
        pehe_raw       = float(np.sqrt(np.mean((cate_pred_raw      - true_cate_raw) ** 2)))
        pehe_full      = float(np.sqrt(np.mean((cate_pred_em       - true_cate_raw) ** 2)))
        pehe_malc_raw  = float(np.sqrt(np.mean((cate_pred_malc_raw - true_cate_raw) ** 2))) if args.malc_upsample else float('nan')
        pehe_malc_em   = float(np.sqrt(np.mean((cate_pred_malc_em  - true_cate_raw) ** 2))) if args.malc_upsample else float('nan')
        eps_ate_raw       = float(abs(cate_pred_raw.mean()      - true_cate_raw.mean()))
        eps_ate_full      = float(abs(cate_pred_em.mean()       - true_cate_raw.mean()))
        eps_ate_malc_raw  = float(abs(cate_pred_malc_raw.mean() - true_cate_raw.mean())) if args.malc_upsample else float('nan')
        eps_ate_malc_em   = float(abs(cate_pred_malc_em.mean()  - true_cate_raw.mean())) if args.malc_upsample else float('nan')
        pehe_list.append(pehe_raw); eps_ate_list.append(eps_ate_raw)
        pehe_em_k1_list.append(pehe_full); eps_em_k1_list.append(eps_ate_full)
        pehe_malc_list.append(pehe_malc_raw); eps_malc_list.append(eps_ate_malc_raw)
        # New: also collect malc_em
        try:
            pehe_malc_em_list.append(pehe_malc_em); eps_malc_em_list.append(eps_ate_malc_em)
        except NameError:
            pass
        std_y0_list.append(float(sigma_em_y0_raw.mean()))
        std_y1_list.append(float(sigma_em_y1_raw.mean()))
        malc_note = f'  malc_fail={n_malc_fail}/{n_test}' if args.malc_upsample else ''
        malc_extra = (f'   PEHE malc_raw={pehe_malc_raw:.4f}  malc_em={pehe_malc_em:.4f}   '
                       f'eps_ATE malc_raw={eps_ate_malc_raw:.4f}  malc_em={eps_ate_malc_em:.4f}'
                       if args.malc_upsample else '')
        print(f'  r={r:3d}  PEHE inner={pehe_raw:.4f}  full={pehe_full:.4f}   '
              f'eps_ATE inner={eps_ate_raw:.4f}  full={eps_ate_full:.4f}{malc_extra}'
              f'   <σ_tail>=({sigma_em_y0_raw.mean():.3f},{sigma_em_y1_raw.mean():.3f}){malc_note}   '
              f'({time.time()-t_r:.1f}s)', flush=True)

    def _summary(arr, label):
        arr = np.asarray(arr)
        n = len(arr)
        m = arr.mean() if n else float('nan')
        s = arr.std(ddof=1) if n > 1 else 0.0
        sem = s / np.sqrt(n) if n > 1 else 0.0
        med = float(np.median(arr)) if n else float('nan')
        print(f'{label:<20s}  mean={m:.4f}  std={s:.4f}  median={med:.4f}  sem={sem:.4f}')

    pehe_arr        = np.array(pehe_list)
    eps_arr         = np.array(eps_ate_list)
    pehe_em_k1_arr  = np.array(pehe_em_k1_list)
    eps_em_k1_arr   = np.array(eps_em_k1_list)
    pehe_malc_arr   = np.array(pehe_malc_list)
    eps_malc_arr    = np.array(eps_malc_list)
    std_y0_arr      = np.array(std_y0_list)
    std_y1_arr      = np.array(std_y1_list)
    print('')
    print(f'== step={step}  n={len(pehe_arr)}  total={time.time()-t_all:.1f}s ==')
    _summary(pehe_arr,        'PEHE (inner)')
    _summary(pehe_em_k1_arr,  'PEHE (full 9-reg)')
    if args.malc_upsample:
        _summary(pehe_malc_arr,        'PEHE (malc raw)')
        _summary(np.array(pehe_malc_em_list), 'PEHE (malc em)')
    _summary(eps_arr,         'eps_ATE (inner)')
    _summary(eps_em_k1_arr,   'eps_ATE (full 9-reg)')
    if args.malc_upsample:
        _summary(eps_malc_arr,         'eps_ATE (malc raw)')
        _summary(np.array(eps_malc_em_list), 'eps_ATE (malc em)')
    _summary(std_y0_arr,      'σ_tail Y_do(0)')
    _summary(std_y1_arr,      'σ_tail Y_do(1)')

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        save_kw = dict(
            pehe=pehe_arr, eps_ate=eps_arr,
            pehe_em=pehe_em_k1_arr, eps_ate_em=eps_em_k1_arr,
            sigma_em_y0=std_y0_arr, sigma_em_y1=std_y1_arr,
            step=step, checkpoint=args.checkpoint,
        )
        if args.malc_upsample:
            save_kw['pehe_malc_raw'] = pehe_malc_arr
            save_kw['eps_ate_malc_raw'] = eps_malc_arr
            save_kw['pehe_malc_em'] = np.array(pehe_malc_em_list)
            save_kw['eps_ate_malc_em'] = np.array(eps_malc_em_list)
        np.savez(args.out, **save_kw)
        print(f'[save] {args.out}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
