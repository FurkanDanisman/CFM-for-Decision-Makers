"""Does Ours' 2D BarDist head actually output a joint with ρ, or has it
collapsed to independence?

Runs a single forward on a polynomial SCM with high true ρ (default 0.8),
pulls out p_mat for each test query, and computes for each query:

  frobenius_dist(p_mat, outer(p_y0, p_y1))  — 0 iff exact independence
  KL(p_mat || outer)                        — 0 iff exact independence
  implied_rho(p_mat)                        — corr coefficient under p_mat

If implied_rho ≈ 0.8 (matching truth) → the model captures ρ.
If implied_rho ≈ 0                    → the model collapsed to independence.

Also compares the model's p_mat to the ANALYTIC 2D Gaussian truth p_mat
directly. Numerical answer regardless of MALC / L2 downstream.

Usage:
    python diagnose_joint_collapse.py \
        --repo $DEPLOY_ROOT/R-PFN \
        --checkpoint50 $DEPLOY_ROOT/R-PFN/checkpoints/step_50000_final.pt \
        --rho 0.8 --K 5 --N-context 200 --N-test 50
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch


def _pad(X, n_feat):
    if n_feat <= 0: return np.asarray(X, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1: X = X.reshape(-1, 1)
    d = X.shape[1]
    if d < n_feat:
        pad = np.full((X.shape[0], n_feat - d), np.nan, dtype=np.float32)
        return np.concatenate([X, pad], axis=1)
    return X[:, :n_feat]


def make_polynomial_scm(seed, n_context, n_test, rho, x_dim=5, degree=3,
                          sigma_eps=1.0):
    rng = np.random.default_rng(seed)
    N = n_context + n_test
    X = rng.standard_normal((N, x_dim)).astype(np.float32)
    feats = np.concatenate([X ** k for k in range(1, degree + 1)], axis=1)
    F = feats.shape[1]
    w_T  = rng.standard_normal(F) / np.sqrt(F)
    w_Y0 = rng.standard_normal(F) / np.sqrt(F)
    w_Y1 = rng.standard_normal(F) / np.sqrt(F)
    mu0 = feats @ w_Y0
    mu1 = feats @ w_Y1
    Sigma = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(2))
    z = rng.standard_normal((N, 2))
    eta = z @ L.T
    y0 = (mu0 + sigma_eps * eta[:, 0]).astype(np.float32)
    y1 = (mu1 + sigma_eps * eta[:, 1]).astype(np.float32)
    logits = feats @ w_T
    logits = (logits - logits.mean()) / (logits.std() + 1e-9)
    p_T = 1.0 / (1.0 + np.exp(-logits))
    T = rng.binomial(1, p_T).astype(np.float32)
    Y_obs = np.where(T > 0.5, y1, y0)
    idx = rng.permutation(N)
    ctx = idx[:n_context]; tst = idx[n_context:]
    class _CD: pass
    cd = _CD()
    cd.X_train = torch.from_numpy(X[ctx])
    cd.t_train = torch.from_numpy(T[ctx])
    cd.y_train = torch.from_numpy(Y_obs[ctx])
    cd.X_test  = torch.from_numpy(X[tst])
    cd._mu0_test = mu0[tst]; cd._mu1_test = mu1[tst]
    cd._sigma_eps = float(sigma_eps); cd._rho = float(rho)
    return cd


def _analytic_2d_gauss_p_mat(mu0, mu1, sigma, rho, edges):
    """Discretise the 2D Gaussian into J×J bin masses (mass = density × bin_area)."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    bw = float(edges[1] - edges[0])
    Y0, Y1 = np.meshgrid(centers, centers, indexing='ij')
    inv_det = 1.0 / (2 * np.pi * sigma * sigma * np.sqrt(1 - rho ** 2))
    z0 = (Y0 - mu0) / sigma; z1 = (Y1 - mu1) / sigma
    q = z0 ** 2 - 2 * rho * z0 * z1 + z1 ** 2
    dens_2d = inv_det * np.exp(-q / (2 * (1 - rho ** 2)))
    m = dens_2d * (bw ** 2)
    return m / m.sum()


def _implied_rho(p_mat, centers):
    """Correlation coefficient under the discrete joint p_mat[i, j]."""
    p = p_mat / p_mat.sum()
    mu0 = float((p.sum(axis=1) * centers).sum())
    mu1 = float((p.sum(axis=0) * centers).sum())
    var0 = float((p.sum(axis=1) * (centers - mu0) ** 2).sum())
    var1 = float((p.sum(axis=0) * (centers - mu1) ** 2).sum())
    cov = 0.0
    for i in range(len(centers)):
        for j in range(len(centers)):
            cov += p[i, j] * (centers[i] - mu0) * (centers[j] - mu1)
    denom = float(np.sqrt(var0 * var1))
    return cov / max(denom, 1e-12), mu0, mu1


def _kl(p, q, eps=1e-12):
    p = np.asarray(p) + eps; q = np.asarray(q) + eps
    p /= p.sum(); q /= q.sum()
    return float((p * np.log(p / q)).sum())


def _frob(p, q):
    return float(np.linalg.norm(p - q))


def load_ours_ipfn(repo, checkpoint_path):
    sys.path.insert(0, repo); sys.path.insert(0, os.path.join(repo, 'MALC'))
    from models.InterventionalPFN import InterventionalPFN
    _orig = torch.load
    def _p_load(*a, **kw): kw.setdefault('weights_only', False); return _orig(*a, **kw)
    torch.load = _p_load
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    torch.load = _orig
    cfg = ckpt['config']; J = cfg['J']
    edges = ckpt['edges'].cpu().numpy()
    bin_width = float(edges[1] - edges[0])
    NF = cfg['num_features']
    model = InterventionalPFN(
        num_features=NF, d_model=cfg['d_model'], depth=cfg['depth'],
        heads_feat=cfg['heads'], heads_samp=cfg['heads'], dropout=0.0,
        output_dim=J*J + 9 + 4, hidden_mult=cfg['hidden_mult'],
        normalize_features=True, normalize_treatment=False,
        use_treatment_in_query=False, use_checkpoint=False,
    ).eval()
    model.load_state_dict(ckpt['model_state_dict'])
    return model, edges, J, bin_width, NF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo',         required=True)
    ap.add_argument('--checkpoint50', required=True)
    ap.add_argument('--rho',          type=float, default=0.8)
    ap.add_argument('--K',            type=int,   default=5,
                    help='number of queries to diagnose (default: first 5)')
    ap.add_argument('--N-context',    type=int,   default=200)
    ap.add_argument('--N-test',       type=int,   default=50)
    ap.add_argument('--seed',         type=int,   default=42)
    args = ap.parse_args()

    print(f'Setup: ρ_true={args.rho}  N_ctx={args.N_context}  N_test={args.N_test}  K={args.K}')

    print('[load] Ours(fn=50) InterventionalPFN')
    model, edges, J, bw, NF = load_ours_ipfn(args.repo, args.checkpoint50)
    centers_scaled = 0.5 * (edges[:-1] + edges[1:])
    print(f'  J={J}  edges=[{edges[0]:.2f}, {edges[-1]:.2f}]  bin_width_scaled={bw:.4f}')

    print(f'[sample] polynomial SCM at ρ={args.rho}')
    cd = make_polynomial_scm(args.seed, args.N_context, args.N_test, args.rho)

    # Y-scaling (Ours rescales training Y to [-1, 1])
    y = cd.y_train.numpy()
    y_min = float(y.min()); y_max = float(y.max())
    y_rng = max(y_max - y_min, 1e-6)
    def _rescale(y_raw): return (y_raw - y_min) / y_rng * 2.0 - 1.0
    Y_ctx = _rescale(y).reshape(-1, 1).astype(np.float32)
    T_ctx = cd.t_train.numpy().astype(np.float32).reshape(-1, 1)
    X_ctx = _pad(cd.X_train.numpy(), NF)
    X_qry = _pad(cd.X_test.numpy(),  NF)

    from losses.BarDistribution2D import unpack_pred
    print(f'[forward] running model on {args.N_context} ctx + {args.N_test} queries...')
    t0 = time.time()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_ctx).unsqueeze(0),
                      torch.from_numpy(T_ctx).unsqueeze(0),
                      torch.from_numpy(Y_ctx).unsqueeze(0),
                      torch.from_numpy(X_qry).unsqueeze(0))['predictions'][0]
    print(f'  done in {time.time()-t0:.1f}s\n')

    print(f'{"query":>5} {"true_ρ":>7} {"impl_ρ":>7} {"|impl-true|":>12} '
          f'{"KL(p||outer)":>14} {"Frob(p,outer)":>14} '
          f'{"KL(p||truth)":>14}')
    print('-' * 90)

    # Scaled truth means (in scaled y ∈ [-1, 1])
    for q in range(min(args.K, args.N_test)):
        p_mat_tensor, *_ = unpack_pred(pred[q], J, bw)
        p_mat = p_mat_tensor.detach().cpu().numpy()  # (J, J), sums to ~1

        p0 = p_mat.sum(axis=1)      # marginal Y_do0
        p1 = p_mat.sum(axis=0)      # marginal Y_do1
        outer = np.outer(p0, p1)    # what independence would give

        impl_rho, impl_mu0, impl_mu1 = _implied_rho(p_mat, centers_scaled)
        kl_indep = _kl(p_mat, outer)
        fro_indep = _frob(p_mat, outer)

        # Analytic truth p_mat on scaled y grid
        mu0_scaled = float(_rescale(cd._mu0_test[q]))
        mu1_scaled = float(_rescale(cd._mu1_test[q]))
        sigma_scaled = cd._sigma_eps * (2.0 / y_rng)
        p_true = _analytic_2d_gauss_p_mat(mu0_scaled, mu1_scaled,
                                            sigma_scaled, args.rho, edges)
        kl_truth = _kl(p_mat, p_true)

        print(f'{q:>5d} {args.rho:>7.3f} {impl_rho:>7.3f} '
              f'{abs(impl_rho - args.rho):>12.3f} '
              f'{kl_indep:>14.4f} {fro_indep:>14.4f} '
              f'{kl_truth:>14.4f}')

    print('\nRead-out:')
    print('  impl_ρ ≈ true_ρ  ⇒  model captures correlation.')
    print('  impl_ρ ≈ 0       ⇒  model collapsed to independence.')
    print('  KL(p||outer) ≈ 0 ⇒  p_mat is essentially outer product (indep).')
    print('  KL(p||truth) small ⇒ p_mat matches the analytic 2D Gaussian.')


if __name__ == '__main__':
    try: main()
    except Exception:
        import traceback; traceback.print_exc(); sys.exit(1)
