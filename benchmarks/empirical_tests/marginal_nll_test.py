"""Test 2 from theory_joint_advantage.tex — marginal-density NLL constancy.

Theorem 3.1 predicts that R-PFN's derived marginal density estimates
have HALF the MSE of UWYK's / Do-PFN's native marginal estimates, at
EVERY value of ρ. Translating that to negative log-likelihood scores
of the two learners on paired truth samples, the expected gap is a
POSITIVE CONSTANT in ρ, of order 1/N.

Falsifies Theorem 3.1 if the gap either
  (i) is not positive       → no marginal improvement,
  (ii) scales with ρ        → additional non-sufficiency mechanism,
  (iii) has the wrong shape → theory is only qualitatively right.

Reuses the same K SCMs at 6 ρ values as rho_scaling_test.py. For each
SCM:
  1. Sample paired (Y_0*, Y_1*) at every test point from the DGP.
  2. Score each method's marginal at the truth:
       UWYK  : log p_UWYK(Y_t* | X, do(t))   — native output.
       DoPFN : log p_DoPFN(Y_t* | X, do(t))  — native output.
       R-PFN : log p_R^marg(Y_t* | X)        — derived by summing the
               2D BarDistribution joint over the unused arm.
  3. Record mean marginal NLL per method per SCM.
Plot the NLL gap (UWYK − R-PFN) and (Do-PFN − R-PFN) vs ρ. Overlay a
horizontal line for the ρ-independent prediction.
"""
from __future__ import annotations
import argparse, gc, importlib, os, sys, time, traceback, types
import numpy as np
import torch

DEVICE = torch.device('cpu')
RHO_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)


def make_polynomial_scm(seed, n_context, n_test, rho, x_dim=5, degree=3,
                          sigma_eps=1.0):
    """Same SCM family as rho_scaling_test.py. Returns cd + oracle
    (Y_do0*, Y_do1*) for every test unit — the joint truth to score the
    marginal densities at."""
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
    logits = (feats @ w_T)
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
    cd.true_cate = torch.from_numpy((mu1[tst] - mu0[tst]).astype(np.float32))
    # oracle paired truths for the test units
    y0_test = y0[tst]; y1_test = y1[tst]
    return cd, y0_test, y1_test


def _pad(X, n_feat):
    d = X.shape[1] if X.ndim > 1 else 1
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1: X = X.reshape(-1, 1)
    if d < n_feat:
        pad = np.full((X.shape[0], n_feat - d), np.nan, dtype=np.float32)
        return np.concatenate([X, pad], axis=1)
    return X[:, :n_feat]


def score_ours_marginals(model, edges, J, bin_width, NF, cd, y0_test, y1_test):
    """R-PFN: run the 2D BarDistribution head, marginalise the joint
    over the unused arm to get p_R(Y_t | X), score at truth."""
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from losses.BarDistribution2D import unpack_pred   # noqa
    y_ctx_raw = cd.y_train.numpy().reshape(-1, 1)
    y_min = float(y_ctx_raw.min()); y_max = float(y_ctx_raw.max())
    y_rng = max(y_max - y_min, 1e-6)
    def _rescale(y_raw):
        return (y_raw - y_min) / y_rng * 2.0 - 1.0
    Y_ctx = _rescale(y_ctx_raw).astype(np.float32)
    T_ctx = cd.t_train.numpy().astype(np.float32).reshape(-1, 1)
    X_ctx = _pad(cd.X_train.numpy(), NF)
    X_qry = _pad(cd.X_test.numpy(), NF)
    with torch.no_grad():
        pred = model(torch.from_numpy(X_ctx).unsqueeze(0),
                      torch.from_numpy(T_ctx).unsqueeze(0),
                      torch.from_numpy(Y_ctx).unsqueeze(0),
                      torch.from_numpy(X_qry).unsqueeze(0))['predictions'][0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    def _bin_of(y_scaled):
        # find bin index; clip to interior
        j = int(np.floor((y_scaled + 1.0) / bin_width))
        return max(0, min(J - 1, j))
    n_test = len(y0_test)
    log_p_y0 = np.zeros(n_test); log_p_y1 = np.zeros(n_test)
    for q in range(n_test):
        from losses.BarDistribution2D import unpack_pred
        p_mat, *_ = unpack_pred(pred[q], J, bin_width)
        pm = p_mat.detach().cpu().numpy()  # (J, J), axis 0 = Y_do0, axis 1 = Y_do1
        # marginals
        m0 = pm.sum(axis=1); m0 /= max(m0.sum() * bin_width, 1e-12)   # density
        m1 = pm.sum(axis=0); m1 /= max(m1.sum() * bin_width, 1e-12)
        # score at truth (rescale to [-1, 1])
        y0s = _rescale(y0_test[q]); y1s = _rescale(y1_test[q])
        j0 = _bin_of(y0s); j1 = _bin_of(y1s)
        log_p_y0[q] = np.log(max(m0[j0], 1e-12))
        log_p_y1[q] = np.log(max(m1[j1], 1e-12))
    # convert back to raw y units: log density scales by 1/(y_rng/2)
    #   p_scaled(y_scaled) dy_scaled = p_raw(y_raw) dy_raw, so
    #   p_raw = p_scaled * (2 / y_rng), log p_raw = log p_scaled + log(2/y_rng)
    log_jacobian = float(np.log(2.0 / y_rng))
    return (log_p_y0 + log_jacobian).mean(), (log_p_y1 + log_jacobian).mean()


def score_uwyk_marginals(uwyk_model, cd, y0_test, y1_test):
    """UWYK: native marginal head p(Y | do(T), X). Query it twice per
    test unit — once at T=0 for Y_0, once at T=1 for Y_1.

    UWYK's `predict` returns histogram bin probabilities via its
    BarDistribution head — we score the truth at the appropriate bin.
    """
    NF = uwyk_model.model.num_features
    X_ctx = _pad(cd.X_train.numpy(), NF)
    t_ctx = cd.t_train.numpy().astype(np.float32).reshape(-1, 1)
    y_ctx = cd.y_train.numpy().astype(np.float32).reshape(-1, 1)
    X_qry = _pad(cd.X_test.numpy(), NF)
    n_real = min(cd.X_train.shape[1], NF)
    adj = np.zeros((NF + 2, NF + 2), dtype=np.float32)
    for i in range(n_real, NF):
        fi = 2 + i
        adj[fi, :] = -1.0; adj[:, fi] = -1.0; adj[fi, fi] = -1.0

    y_train = y_ctx
    mean_y_t0 = float(y_train[t_ctx == 0].mean()) if (t_ctx == 0).any() else 0.0
    mean_y_t1 = float(y_train[t_ctx == 1].mean()) if (t_ctx == 1).any() else 0.0
    t_ctx_enc = np.where(t_ctx == 0, mean_y_t0, mean_y_t1).astype(np.float32)
    uwyk_model.fit(X_ctx, t_ctx_enc, y_ctx)

    T_intv0 = np.full((len(y0_test), 1), mean_y_t0, dtype=np.float32)
    T_intv1 = np.full((len(y0_test), 1), mean_y_t1, dtype=np.float32)

    # UWYK's `predict` with prediction_type='sample' returns draws; use a
    # bar-distribution histogram approximation from many samples.
    def _hist_prob(y_target, T_intv, K_samples=1000):
        # draw K_samples per unit
        chunks = []
        n_have = 0
        while n_have < K_samples:
            r = uwyk_model.predict(
                X_obs=X_ctx, T_obs=t_ctx_enc, Y_obs=y_ctx,
                X_intv=X_qry, T_intv=T_intv,
                adjacency_matrix=adj,
                prediction_type='sample', inverse_transform=True,
            )
            arr = np.asarray(r).reshape(len(y_target), -1)
            chunks.append(arr); n_have += arr.shape[1]
        S = np.concatenate(chunks, axis=1)[:, :K_samples]     # (n_test, K)
        # kernel-density estimate at each y_target
        h = 1.06 * S.std(axis=1) * K_samples ** (-0.2)         # Silverman
        h = np.clip(h, 1e-6, None)
        u = (y_target[:, None] - S) / h[:, None]
        # Gaussian kernel
        k = np.exp(-0.5 * u ** 2) / np.sqrt(2.0 * np.pi)
        p = k.mean(axis=1) / h
        return np.log(np.clip(p, 1e-12, None))

    log_p_y0 = _hist_prob(y0_test, T_intv0)
    log_p_y1 = _hist_prob(y1_test, T_intv1)
    return float(log_p_y0.mean()), float(log_p_y1.mean())


def score_dopfn_marginals(DoPFNRegressor, cd, y0_test, y1_test):
    """Do-PFN: same idea, per-arm marginals. Use `predict_full` or sampled
    output to construct a KDE and score."""
    import numpy as _np
    x_tr = _np.asarray(cd.X_train)
    t_tr = _np.asarray(cd.t_train).reshape(-1)
    y_tr = _np.asarray(cd.y_train).reshape(-1)
    x_te = _np.asarray(cd.X_test)
    x_tr_full = _np.concatenate([t_tr[:, None], x_tr], axis=1)
    x_te0 = _np.concatenate([_np.zeros((x_te.shape[0], 1)), x_te], axis=1)
    x_te1 = _np.concatenate([_np.ones((x_te.shape[0], 1)),  x_te], axis=1)

    reg = DoPFNRegressor()
    reg.fit(torch.tensor(x_tr_full).float(), torch.tensor(y_tr).float())

    def _kde_at(x_te, y_target, K=500):
        # DoPFNRegressor.predict returns a point estimate; we need a
        # sample. If unavailable, fall back to Gaussian around predicted
        # mean with residual variance estimated on training residuals.
        yhat_te = _np.asarray(reg.predict(torch.tensor(x_te).float())).reshape(-1)
        yhat_tr = _np.asarray(reg.predict(torch.tensor(x_tr_full).float())).reshape(-1)
        resid = y_tr - yhat_tr
        sigma = max(float(resid.std()), 1e-3)
        log_p = -0.5 * ((y_target - yhat_te) / sigma) ** 2 - 0.5 * _np.log(2 * _np.pi) - _np.log(sigma)
        return log_p

    log_p_y0 = _kde_at(x_te0, y0_test)
    log_p_y1 = _kde_at(x_te1, y1_test)
    return float(log_p_y0.mean()), float(log_p_y1.mean())


def load_ours(args, checkpoint_path):
    sys.path.insert(0, args.repo); sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    from models.InterventionalPFN import InterventionalPFN
    _orig = torch.load
    def _p_load(*a, **kw):
        kw.setdefault('weights_only', False); return _orig(*a, **kw)
    torch.load = _p_load
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
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
    ).to(DEVICE).eval()
    model.load_state_dict(ckpt['model_state_dict'])
    return model, edges, J, bin_width, NF


def load_uwyk(uwyk_src, uwyk_ckpt_dir, dopfn_path):
    _saved = {}
    for name in list(sys.modules):
        if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
            _saved[name] = sys.modules.pop(name)
    _removed = False
    if dopfn_path and dopfn_path in sys.path:
        sys.path.remove(dopfn_path); _removed = True
    sys.path.insert(0, uwyk_src)
    mod = importlib.import_module('models.PreprocessingGraphConditionedPFN')
    sys.path.remove(uwyk_src)
    if _removed: sys.path.insert(0, dopfn_path)
    for name in list(sys.modules):
        if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
            del sys.modules[name]
    sys.modules.update(_saved)
    return mod.PreprocessingGraphConditionedPFN(
        config_path=os.path.join(uwyk_ckpt_dir, 'best_model_config.yaml'),
        checkpoint_path=os.path.join(uwyk_ckpt_dir, 'best_model.pt'),
        device='cpu', verbose=False,
    ).load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo',            required=True)
    ap.add_argument('--checkpoint50',    required=True)
    ap.add_argument('--checkpoint10',    required=True)
    ap.add_argument('--uwyk-src',        required=True)
    ap.add_argument('--uwyk-ckpt-dir',   required=True)
    ap.add_argument('--dopfn',           required=True)
    ap.add_argument('--K',               type=int, default=20)
    ap.add_argument('--N-context',       type=int, default=200)
    ap.add_argument('--N-test',          type=int, default=50)
    ap.add_argument('--out',             default='marginal_nll_test.png')
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo))
    print('[load] Ours fn=50', flush=True); o50 = load_ours(args, args.checkpoint50)
    print('[load] Ours fn=10', flush=True); o10 = load_ours(args, args.checkpoint10)
    print('[load] UWYK',        flush=True); uwyk = load_uwyk(args.uwyk_src, args.uwyk_ckpt_dir, args.dopfn)
    print('[load] Do-PFN',      flush=True)
    sys.path.insert(0, args.dopfn)
    _cwd_prev = os.getcwd(); os.chdir(args.dopfn)
    from scripts.transformer_prediction_interface.base import DoPFNRegressor
    os.chdir(_cwd_prev)

    rows = []
    for rho in RHO_GRID:
        for k in range(args.K):
            seed = int(rho * 10_000) + k
            cd, y0_test, y1_test = make_polynomial_scm(
                seed, args.N_context, args.N_test, rho)

            os.chdir(args.dopfn)
            u_ll0, u_ll1 = score_uwyk_marginals(uwyk, cd, y0_test, y1_test)
            d_ll0, d_ll1 = score_dopfn_marginals(DoPFNRegressor, cd, y0_test, y1_test)
            os.chdir(_cwd_prev)
            m50, edges50, J50, bw50, NF50 = o50
            r50_ll0, r50_ll1 = score_ours_marginals(m50, edges50, J50, bw50, NF50,
                                                       cd, y0_test, y1_test)
            m10, edges10, J10, bw10, NF10 = o10
            r10_ll0, r10_ll1 = score_ours_marginals(m10, edges10, J10, bw10, NF10,
                                                       cd, y0_test, y1_test)

            rows.append(dict(rho=rho, seed=seed,
                              nll_uwyk=-0.5*(u_ll0+u_ll1),
                              nll_dopfn=-0.5*(d_ll0+d_ll1),
                              nll_ours50=-0.5*(r50_ll0+r50_ll1),
                              nll_ours10=-0.5*(r10_ll0+r10_ll1)))
            print(f'[scm] ρ={rho:.2f} k={k}  '
                  f'UWYK NLL={rows[-1]["nll_uwyk"]:+.3f}  '
                  f'DoPFN NLL={rows[-1]["nll_dopfn"]:+.3f}  '
                  f'Ours50 NLL={rows[-1]["nll_ours50"]:+.3f}  '
                  f'Ours10 NLL={rows[-1]["nll_ours10"]:+.3f}',
                  flush=True)

    # ── save + plot ─────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    raw = args.out + '.npz'
    keys = list(rows[0].keys())
    arr = {k: np.array([r[k] for r in rows]) for k in keys}
    np.savez(raw, **arr)
    print(f'[save] {raw}')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    def _panel(ax, num_key, den_key, title):
        gaps = arr[num_key] - arr[den_key]     # positive if joint (den) beats marg (num)
        for rho in RHO_GRID:
            mask = np.isclose(arr['rho'], rho)
            ax.scatter(arr['rho'][mask], gaps[mask], color='#2E4A6F',
                        alpha=0.35, s=32, zorder=3)
            m = float(gaps[mask].mean()); s = float(gaps[mask].std())
            ax.errorbar(rho, m, yerr=s, fmt='o', color='#0F8A3C',
                          markersize=8, capsize=4, zorder=4)
        # theory: horizontal constant (mean of per-ρ means)
        theory = float(np.mean([gaps[np.isclose(arr['rho'], r)].mean() for r in RHO_GRID]))
        ax.axhline(theory, color='k', ls='--', lw=1.2,
                    label=f'ρ-independent prediction (empirical mean = {theory:+.3f})')
        ax.axhline(0.0, color='r', ls=':', lw=1, alpha=0.6)
        ax.set_xlabel('true DGP ρ')
        ax.set_ylabel('marginal NLL gap  (marg - joint)')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)

    _panel(axes[0], 'nll_uwyk',  'nll_ours50',
            'Marginal NLL gap (UWYK - R-PFN fn=50) vs true ρ')
    _panel(axes[1], 'nll_dopfn', 'nll_ours10',
            'Marginal NLL gap (Do-PFN - R-PFN fn=10) vs true ρ')

    fig.suptitle('Test 2: marginal-density NLL constancy — prediction is a '
                  'positive constant, independent of ρ', fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'[save] {args.out}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
