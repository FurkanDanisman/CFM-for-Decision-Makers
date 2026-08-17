"""Fast IHDP eval for a DoPFN-backbone Ours checkpoint — raw-mean AND
truncated-normal EM-corrected mean CATE. No MALC 2D, no density L2/KL.

Two CATE estimators are reported side-by-side:

  * RAW MEAN — μ = Σ p_i · c_i on bin centers c_i. Simple, but for small J
    the bin-center quantization biases μ by up to bin_width/2, which
    dominates on J=10 (IHDP bin width ≈ 3 raw units).

  * EM MEAN — truncated-normal EM correction on the binned Gaussian data
    (MALC/malc_2d.py::_em_mean_2d). Method-of-moments start (raw mean +
    Sheppard-corrected σ), then fixed-point iteration on
        μ ← μ − σ · Σ p_j · [φ(a_{j+1}) − φ(a_j)] / [Φ(a_{j+1}) − Φ(a_j)],
        a_k = (grid_k − μ)/σ.
    This is E-step: E[Y|bin, μ, σ]; M-step: match observed E[Y]. Recovers
    the true latent Gaussian mean without paying the bin-quantization cost.
    Converges within ~1 bin width of truth even at J=10.

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
    args = ap.parse_args()

    # ── Paths ─────────────────────────────────────────────────────────────
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.join(args.repo, 'training_dopfn_base'))
    sys.path.insert(0, _here)

    from true_ihdp import load_ihdp_truth
    from dopfn_backbone_head import DoPFNBackboneWith2DHead
    from losses.BarDistribution2D import unpack_pred

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
        cate_raw_scaled  = np.zeros(n_test, dtype=np.float64)   # Σ p·c on bin centers
        cate_em_scaled   = np.zeros(n_test, dtype=np.float64)   # truncnorm EM-corrected
        sigma_em_y0_scaled = np.zeros(n_test, dtype=np.float64)
        sigma_em_y1_scaled = np.zeros(n_test, dtype=np.float64)
        bin_w_scaled = float(centers_scaled[1] - centers_scaled[0])

        for q in range(n_test):
            p_mat, *_ = unpack_pred(pred[q], J, bin_width)
            pm = p_mat.detach().cpu().numpy().astype(np.float64)
            s = pm.sum()
            if s > 0: pm /= s
            p_marg0 = pm.sum(axis=1)  # marg over Y1 → gives Y0 marginal
            p_marg1 = pm.sum(axis=0)  # marg over Y0 → gives Y1 marginal

            # ── raw bin-center mean ────────────────────────────────────
            mean0_raw_s = float((centers_scaled * p_marg0).sum())
            mean1_raw_s = float((centers_scaled * p_marg1).sum())
            cate_raw_scaled[q] = mean1_raw_s - mean0_raw_s

            # ── EM-corrected mean (truncated-normal fixed-point) ───────
            # Method-of-moments start + Sheppard-corrected σ, then iterate.
            sigma0 = _init_sigma_1d(p_marg0, centers_scaled, mean0_raw_s, bin_w_scaled)
            sigma1 = _init_sigma_1d(p_marg1, centers_scaled, mean1_raw_s, bin_w_scaled)
            mean0_em_s = _em_mean_1d(p_marg0, edges_np, sigma=sigma0, start=mean0_raw_s)
            mean1_em_s = _em_mean_1d(p_marg1, edges_np, sigma=sigma1, start=mean1_raw_s)
            cate_em_scaled[q] = mean1_em_s - mean0_em_s
            sigma_em_y0_scaled[q] = sigma0
            sigma_em_y1_scaled[q] = sigma1

        cate_pred_raw = cate_raw_scaled * (y_rng / 2.0)
        cate_pred_em  = cate_em_scaled  * (y_rng / 2.0)
        sigma_em_y0_raw = sigma_em_y0_scaled * (y_rng / 2.0)
        sigma_em_y1_raw = sigma_em_y1_scaled * (y_rng / 2.0)

        true_cate_raw = _np(cd.true_cate).reshape(-1).astype(np.float64)
        pehe_raw = float(np.sqrt(np.mean((cate_pred_raw - true_cate_raw) ** 2)))
        pehe_em  = float(np.sqrt(np.mean((cate_pred_em  - true_cate_raw) ** 2)))
        eps_ate_raw = float(abs(cate_pred_raw.mean() - true_cate_raw.mean()))
        eps_ate_em  = float(abs(cate_pred_em.mean()  - true_cate_raw.mean()))
        pehe_list.append(pehe_raw); eps_ate_list.append(eps_ate_raw)
        pehe_em_k1_list.append(pehe_em); eps_em_k1_list.append(eps_ate_em)
        std_y0_list.append(float(sigma_em_y0_raw.mean()))
        std_y1_list.append(float(sigma_em_y1_raw.mean()))
        print(f'  r={r:3d}  PEHE raw={pehe_raw:.4f}  em={pehe_em:.4f}   '
              f'eps_ATE raw={eps_ate_raw:.4f}  em={eps_ate_em:.4f}   '
              f'<σ_Y0>={sigma_em_y0_raw.mean():.3f}  <σ_Y1>={sigma_em_y1_raw.mean():.3f}   '
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
    std_y0_arr      = np.array(std_y0_list)
    std_y1_arr      = np.array(std_y1_list)
    print('')
    print(f'== step={step}  n={len(pehe_arr)}  total={time.time()-t_all:.1f}s ==')
    _summary(pehe_arr,        'PEHE (raw)')
    _summary(pehe_em_k1_arr,  'PEHE (em_corr)')
    _summary(eps_arr,         'eps_ATE (raw)')
    _summary(eps_em_k1_arr,   'eps_ATE (em_corr)')
    _summary(std_y0_arr,      'σ_init Y_do(0)')
    _summary(std_y1_arr,      'σ_init Y_do(1)')

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        np.savez(args.out,
                  pehe=pehe_arr, eps_ate=eps_arr,
                  pehe_em=pehe_em_k1_arr, eps_ate_em=eps_em_k1_arr,
                  sigma_em_y0=std_y0_arr, sigma_em_y1=std_y1_arr,
                  step=step, checkpoint=args.checkpoint)
        print(f'[save] {args.out}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
