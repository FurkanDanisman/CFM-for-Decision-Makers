"""Two diagnostics in one script:

1. Synthetic sanity check that _em_mean_1d actually corrects bin-quantization
   bias (i.e., that our EM implementation isn't broken). Constructs a
   known Gaussian on J=10 bins, prints raw-mean error vs EM-mean error
   for a range of σ. Expected: EM error ≪ raw error when σ < bin_width.

2. Per-realization diagnostic on the actual model. For a given checkpoint,
   dumps per-realization {true_ATE, pred_ATE_raw, pred_ATE_em, bias}.
   Shows whether the ATE bias is a uniform shift across all realizations
   (⇒ prior mismatch or systematic under-confidence) or a few outliers
   (⇒ specific bad queries).

Usage:
    # Just the EM sanity check:
    python verify_em_and_ate.py --sanity

    # Full per-realization diagnostic (needs checkpoint):
    python verify_em_and_ate.py \
        --repo $DEPLOY_ROOT/R-PFN \
        --dopfn $DEPLOY_ROOT/external/dopfn \
        --causalpfn $DEPLOY_ROOT/external/causalpfn \
        --checkpoint $DEPLOY_ROOT/checkpoints_dopfn_backbone_realj10/step_100000.pt \
        --n-realizations 20
"""
from __future__ import annotations
import argparse, os, sys, types, traceback
import numpy as np
from scipy.stats import norm as _scipy_norm


def _em_mean_1d(props, grid_edges, sigma, start,
                 max_step=1000, eps2=1e-10, eps1=1e-5):
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
    var = float(np.sum(props * (centers - mean_est) ** 2)) + (bin_width ** 2) / 12.0
    return float(np.sqrt(max(var, (bin_width / 12.0) ** 2)))


def sanity_check():
    """EM should meaningfully outperform raw when σ_true < bin_width."""
    J = 10
    edges = np.linspace(-1.0, 1.0, J + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_w = float(edges[1] - edges[0])
    print(f'J={J}  bin_width={bin_w}\n')

    print(f'{"μ_true":>8s}  {"σ_true":>8s}    {"raw":>10s}   {"em":>10s}    '
          f'{"raw_err":>10s}   {"em_err":>10s}   {"σ_init":>8s}')
    print('-' * 100)
    for mu_true in [0.00, 0.05, 0.10, 0.15]:
        for sig_true in [0.05, 0.10, 0.20, 0.30, 0.50]:
            p = _scipy_norm.cdf(edges[1:], mu_true, sig_true) - \
                _scipy_norm.cdf(edges[:-1], mu_true, sig_true)
            p /= p.sum()
            mu_raw = float((centers * p).sum())
            sig0 = _init_sigma_1d(p, centers, mu_raw, bin_w)
            mu_em = _em_mean_1d(p, edges, sig0, mu_raw)
            print(f'{mu_true:8.3f}  {sig_true:8.3f}    {mu_raw:10.4f}   {mu_em:10.4f}    '
                  f'{mu_raw-mu_true:+10.4f}   {mu_em-mu_true:+10.4f}   {sig0:8.4f}')
        print()
    print('If EM works, em_err ≪ raw_err when σ_true is between 0.05 and 0.30 '
          '(σ ~ ≤ bin_width). At σ_true ≥ 0.5 both are small.\n')


def per_realization(args):
    import torch
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.join(args.repo, 'training_dopfn_base'))
    sys.path.insert(0, _here)
    from true_ihdp import load_ihdp_truth
    from dopfn_backbone_head import DoPFNBackboneWith2DHead
    from losses.BarDistribution2D import unpack_pred

    def _install_ds(dopfn_dir):
        if 'datasets' in sys.modules: return
        sys.path.insert(0, dopfn_dir)
        ds = types.ModuleType('datasets')
        with open(os.path.join(dopfn_dir, 'datasets/__init__.py')) as fp:
            src = fp.read().split('def load_semi_real')[0]
        exec(src, ds.__dict__); sys.modules['datasets'] = ds
    _install_ds(args.dopfn)
    sys.path.insert(0, args.causalpfn)
    from benchmarks import IHDPDataset

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['config']; J = int(cfg['J'])
    edges_np = ckpt['edges'].cpu().numpy()
    bin_width = float(edges_np[1] - edges_np[0])
    centers_scaled = 0.5 * (edges_np[:-1] + edges_np[1:])
    bin_w_scaled = float(centers_scaled[1] - centers_scaled[0])
    model = DoPFNBackboneWith2DHead(dopfn_root=args.dopfn, K=J).eval()
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'[ckpt] J={J} step={ckpt.get("step","?")} bin_w_scaled={bin_w_scaled:.4f}\n')

    print(f'{"r":>3s}  {"true_ATE":>10s}  {"pred_ATE_raw":>14s}  {"pred_ATE_em":>14s}  '
          f'{"bias_raw":>10s}  {"bias_em":>10s}  {"y_rng":>8s}  {"σ_med":>8s}')
    print('-' * 105)
    biases_raw, biases_em = [], []
    for r in range(min(args.n_realizations, 100)):
        cd, _ = IHDPDataset()[r]
        y_train = cd.y_train.numpy()
        truth = load_ihdp_truth(r, args.causalpfn, y_train)
        y_min = float(truth.y_min); y_rng = float(truth.y_rng)
        Y_ctx_s = ((y_train.reshape(-1) - y_min) / y_rng * 2 - 1).astype(np.float32)
        X_ctx = cd.X_train.numpy().astype(np.float32)
        T_ctx = cd.t_train.numpy().astype(np.float32).reshape(-1, 1)
        X_qry = cd.X_test.numpy().astype(np.float32)
        with torch.no_grad():
            pred = model(
                torch.from_numpy(X_ctx).unsqueeze(0),
                torch.from_numpy(T_ctx).unsqueeze(0),
                torch.from_numpy(Y_ctx_s.reshape(-1, 1)).unsqueeze(0),
                torch.from_numpy(X_qry).unsqueeze(0),
            )['predictions'][0]
        n_test = X_qry.shape[0]
        cate_raw = np.zeros(n_test); cate_em = np.zeros(n_test); sigs = []
        for q in range(n_test):
            p_mat, *_ = unpack_pred(pred[q], J, bin_width)
            pm = p_mat.detach().cpu().numpy().astype(np.float64); pm /= max(pm.sum(), 1e-12)
            p0 = pm.sum(axis=1); p1 = pm.sum(axis=0)
            m0_r = float((centers_scaled * p0).sum()); m1_r = float((centers_scaled * p1).sum())
            cate_raw[q] = m1_r - m0_r
            s0 = _init_sigma_1d(p0, centers_scaled, m0_r, bin_w_scaled)
            s1 = _init_sigma_1d(p1, centers_scaled, m1_r, bin_w_scaled)
            m0_e = _em_mean_1d(p0, edges_np, s0, m0_r)
            m1_e = _em_mean_1d(p1, edges_np, s1, m1_r)
            cate_em[q] = m1_e - m0_e
            sigs.append(0.5 * (s0 + s1))
        pred_ate_raw = cate_raw.mean() * (y_rng / 2)
        pred_ate_em  = cate_em.mean()  * (y_rng / 2)
        true_ate = float(cd.true_cate.numpy().mean())
        b_raw = pred_ate_raw - true_ate
        b_em  = pred_ate_em  - true_ate
        biases_raw.append(b_raw); biases_em.append(b_em)
        print(f'{r:3d}  {true_ate:10.3f}  {pred_ate_raw:14.3f}  {pred_ate_em:14.3f}  '
              f'{b_raw:+10.3f}  {b_em:+10.3f}  {y_rng:8.2f}  '
              f'{np.median(sigs) * (y_rng / 2):8.3f}')

    br = np.array(biases_raw); be = np.array(biases_em)
    print('-' * 105)
    print(f'raw bias:   mean={br.mean():+.4f}  median={np.median(br):+.4f}  '
          f'std={br.std(ddof=1):.4f}  n_positive={(br > 0).sum()}/{len(br)}')
    print(f'em  bias:   mean={be.mean():+.4f}  median={np.median(be):+.4f}  '
          f'std={be.std(ddof=1):.4f}  n_positive={(be > 0).sum()}/{len(be)}')
    print()
    print('If n_positive is ~50% and mean bias ≈ 0 → no systematic shift, ε_ATE '
          'is from per-realization variance.')
    print('If n_positive is ≫50% (say 80+/100) and mean bias > 0 → SYSTEMATIC OVER-'
          'PREDICTION. Points to prior mismatch or scaling bug.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sanity', action='store_true')
    ap.add_argument('--repo')
    ap.add_argument('--dopfn')
    ap.add_argument('--causalpfn')
    ap.add_argument('--checkpoint')
    ap.add_argument('--n-realizations', type=int, default=20)
    args = ap.parse_args()

    if args.sanity or not args.checkpoint:
        sanity_check()
    if args.checkpoint:
        per_realization(args)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
