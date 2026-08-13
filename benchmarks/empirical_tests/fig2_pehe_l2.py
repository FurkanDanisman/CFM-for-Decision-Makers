"""Fig 2 — PEHE / marginal L2 / CATE L2 / ATE L2 vs ρ on polynomial SCM.

Four methods evaluated at each ρ ∈ {0, 0.2, 0.4, 0.6, 0.8, ~1}:
    UWYK-NoAnc            — samples for Y_do0, Y_do1 (independent draws)
    Ours(fn=50)           — 2D BarDist joint p(Y0,Y1|X)
    Do-PFN                — Gaussian around per-arm point estimate
    Ours-DoPFN-bb(200K)   — same 2D BarDist head, DoPFN backbone

Under UWYK / Do-PFN's independence assumption:
    p_method(τ | x)  =  p_method(Y1|x)  ⊛  p_method(-Y0|x)

The 2D-joint methods (Ours) obtain p(τ|x) by projecting the joint along
τ = y1 - y0 diagonals; this captures the DGP's ρ. UWYK / Do-PFN cannot.

ATE density = 1D Wasserstein barycenter over per-query CATE densities
(matches how l2_ihdp/eval_realization.py aggregates).

Truth (closed-form for the polynomial SCM):
    p(Y_dot | x_q)  =  N(μ_t(x_q), σ²)
    p(τ    | x_q)  =  N(μ1(x_q) - μ0(x_q), 2σ²(1-ρ))
    p_ATE(τ)       =  W-barycenter over q of p(τ | x_q)

At ρ = 1 the paired-noise variance is 0 → truth CATE density is a Dirac.
Any smooth estimate has L2 dominated by its own norm — not meaningful.
So we internally cap ρ at 0.99 for evaluation.

Layout: 2x2 summary PNG + one 4-panel density-diagnostic PNG per ρ.

Usage (single-task = all rhos):
    python fig2_pehe_l2.py \
        --repo $DEPLOY_ROOT/R-PFN --dopfn $DEPLOY_ROOT/external/dopfn \
        --causalpfn $DEPLOY_ROOT/external/causalpfn \
        --uwyk-src $DEPLOY_ROOT/external/uwyk/src \
        --uwyk-ckpt-dir $DEPLOY_ROOT/external/uwyk/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0 \
        --checkpoint50 $DEPLOY_ROOT/R-PFN/checkpoints/step_50000_final.pt \
        --checkpoint-dopfn-bb $DEPLOY_ROOT/checkpoints_dopfn_backbone_j10/step_200000.pt \
        --out fig2_pehe_l2

SLURM-style per-ρ mode: pass --rho-index 0..5 (writes one shard per ρ),
then --plot on the aggregator to build the summary + diagnostics.
"""
from __future__ import annotations
import argparse, gc, importlib, os, sys, time, traceback, types
import numpy as np
import torch


_here = os.path.dirname(os.path.abspath(__file__))
_bench = os.path.dirname(_here)
if _bench not in sys.path: sys.path.insert(0, _bench)
from methods import dopfn as _dopfn_shim   # noqa: F401  — sklearn shim

if _here not in sys.path: sys.path.insert(0, _here)
from dopfn_helpers import (load_dopfn_bb, load_dopfn, dopfn_predict_cate,
                            dopfn_predict_ymean)

DEVICE = torch.device('cpu')
RHO_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
RHO_EFF  = tuple(min(r, 0.99) for r in RHO_GRID)   # cap ρ=1 → 0.99 internally

# Shared grids for density comparison.
Y_MIN, Y_MAX, Y_BINS = -8.0, 8.0, 201
TAU_MIN, TAU_MAX, TAU_BINS = -10.0, 10.0, 201
Y_GRID   = np.linspace(Y_MIN, Y_MAX, Y_BINS)
TAU_GRID = np.linspace(TAU_MIN, TAU_MAX, TAU_BINS)
Y_DX     = float(Y_GRID[1] - Y_GRID[0])
TAU_DX   = float(TAU_GRID[1] - TAU_GRID[0])


# ── SCM (mirrors marginal_nll_test.py) ────────────────────────────────────
def make_polynomial_scm(seed, n_context, n_test, rho_eff, x_dim=5, degree=3,
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
    Sigma = np.array([[1.0, rho_eff], [rho_eff, 1.0]], dtype=np.float64)
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
    cd.true_cate = torch.from_numpy((mu1[tst] - mu0[tst]).astype(np.float32))
    cd._mu0_test = mu0[tst].astype(np.float32)
    cd._mu1_test = mu1[tst].astype(np.float32)
    cd._sigma_eps = float(sigma_eps)
    cd._rho_eff = float(rho_eff)
    return cd


# ── Truth densities ───────────────────────────────────────────────────────
def _gauss_on_grid(mu, sigma, grid):
    """(n,) means → (n, len(grid)) normalised Gaussian densities."""
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    z = (grid[None, :] - mu[:, None]) / max(sigma, 1e-12)
    p = np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * max(sigma, 1e-12))
    s = p.sum(axis=1, keepdims=True) * (grid[1] - grid[0])
    return p / np.maximum(s, 1e-12)


def true_marginals(cd):
    """Per-query true p(Y_do0) and p(Y_do1) on Y_GRID."""
    p0 = _gauss_on_grid(cd._mu0_test, cd._sigma_eps, Y_GRID)
    p1 = _gauss_on_grid(cd._mu1_test, cd._sigma_eps, Y_GRID)
    return p0, p1


def true_cate(cd):
    """Per-query true p(τ | x) on TAU_GRID. Var = 2σ²(1-ρ)."""
    var_tau = 2.0 * cd._sigma_eps ** 2 * (1.0 - cd._rho_eff)
    sd_tau = float(np.sqrt(max(var_tau, 1e-12)))
    mu_tau = cd._mu1_test - cd._mu0_test
    return _gauss_on_grid(mu_tau, sd_tau, TAU_GRID)


# ── L2 helpers ────────────────────────────────────────────────────────────
def l2_1d(f, g, dx):
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.sum((f - g) ** 2) * dx))


def wass_bary_of_grid(p_matrix, grid, wb_fn):
    """1D W-barycenter over queries. p_matrix: (n_queries, len(grid))."""
    # wasserstein_barycenter_1d expects densities on the same grid.
    p_ate = wb_fn(p_matrix, grid)
    s = p_ate.sum() * (grid[1] - grid[0])
    if s > 0: p_ate = p_ate / s
    return p_ate


# ── Method density derivations ────────────────────────────────────────────
def _pad(X, n_feat):
    if n_feat <= 0:  # DoPFN-bb: no cap
        return np.asarray(X, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1: X = X.reshape(-1, 1)
    d = X.shape[1]
    if d < n_feat:
        pad = np.full((X.shape[0], n_feat - d), np.nan, dtype=np.float32)
        return np.concatenate([X, pad], axis=1)
    return X[:, :n_feat]


def ours_densities(model, edges, J, bin_width, NF, cd):
    """Per-query 2D BarDist forward → marginals + CATE densities on
    the shared grids, plus per-query CATE point estimates."""
    from losses.BarDistribution2D import unpack_pred
    y_ctx_raw = cd.y_train.numpy().reshape(-1)
    y_min = float(y_ctx_raw.min()); y_max = float(y_ctx_raw.max())
    y_rng = max(y_max - y_min, 1e-6)
    def _rescale(y_raw): return (y_raw - y_min) / y_rng * 2.0 - 1.0
    def _unscale(y_scaled): return (y_scaled + 1.0) * 0.5 * y_rng + y_min

    Y_ctx = _rescale(y_ctx_raw).reshape(-1, 1).astype(np.float32)
    T_ctx = cd.t_train.numpy().astype(np.float32).reshape(-1, 1)
    X_ctx = _pad(cd.X_train.numpy(), NF)
    X_qry = _pad(cd.X_test.numpy(), NF)
    with torch.no_grad():
        pred = model(torch.from_numpy(X_ctx).unsqueeze(0),
                      torch.from_numpy(T_ctx).unsqueeze(0),
                      torch.from_numpy(Y_ctx).unsqueeze(0),
                      torch.from_numpy(X_qry).unsqueeze(0))['predictions'][0]

    centers_scaled = 0.5 * (edges[:-1] + edges[1:])
    centers_raw = _unscale(centers_scaled)                 # (J,)
    # τ centers = y1_raw_center - y0_raw_center for each (j0, j1) pair
    # Diagonals of the joint at fixed (j1 - j0) give the raw τ.
    # We resample the per-query joint onto (Y_GRID, TAU_GRID) via interp.

    n_test = cd.X_test.shape[0]
    p_y0_out = np.zeros((n_test, Y_BINS))
    p_y1_out = np.zeros((n_test, Y_BINS))
    p_tau_out = np.zeros((n_test, TAU_BINS))
    cate_pt = np.zeros(n_test)

    # Per-bin τ_ij = centers_raw[j] - centers_raw[i]; sum joint mass over
    # (i, j) pairs by τ bin.
    tau_ij = centers_raw[None, :] - centers_raw[:, None]  # (J, J)

    for q in range(n_test):
        p_mat, *_ = unpack_pred(pred[q], J, bin_width)
        pm = p_mat.detach().cpu().numpy()                  # (J, J)
        pm = pm / max(pm.sum(), 1e-12)                     # normalise mass

        # Raw-space marginals (probabilities over J bins → density on Y_GRID)
        m0_raw = pm.sum(axis=1); m1_raw = pm.sum(axis=0)   # (J,)
        # Convert to density on the raw centers_raw grid, then interp to Y_GRID.
        bin_raw = float(centers_raw[1] - centers_raw[0])
        d0_raw = m0_raw / bin_raw
        d1_raw = m1_raw / bin_raw
        p_y0_out[q] = np.interp(Y_GRID, centers_raw, d0_raw, left=0.0, right=0.0)
        p_y1_out[q] = np.interp(Y_GRID, centers_raw, d1_raw, left=0.0, right=0.0)
        # Renormalise on the target grid.
        s0 = p_y0_out[q].sum() * Y_DX
        s1 = p_y1_out[q].sum() * Y_DX
        if s0 > 0: p_y0_out[q] /= s0
        if s1 > 0: p_y1_out[q] /= s1

        # CATE density from the joint: bin τ_ij into TAU_GRID centers.
        # Convert τ_ij values to bin indices on TAU_GRID; accumulate probability.
        idx = np.round((tau_ij - TAU_MIN) / TAU_DX).astype(int)
        valid = (idx >= 0) & (idx < TAU_BINS)
        hist = np.zeros(TAU_BINS)
        np.add.at(hist, idx[valid], pm[valid])
        p_tau = hist / max(TAU_DX, 1e-12)
        s = p_tau.sum() * TAU_DX
        if s > 0: p_tau /= s
        p_tau_out[q] = p_tau

        # CATE point estimate = E[τ] under this discrete joint.
        cate_pt[q] = float((tau_ij * pm).sum())

    return dict(p_y0=p_y0_out, p_y1=p_y1_out, p_tau=p_tau_out, cate=cate_pt)


def uwyk_densities(uwyk_model, cd, n_samples=500):
    """UWYK-NoAnc: sample Y_do0 and Y_do1 independently, KDE for marginals,
    compute CATE as (Y1-Y0) samples KDE (equivalent to convolution under
    independence)."""
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

    mean_y_t0 = float(y_ctx[t_ctx == 0].mean()) if (t_ctx == 0).any() else 0.0
    mean_y_t1 = float(y_ctx[t_ctx == 1].mean()) if (t_ctx == 1).any() else 0.0
    t_ctx_enc = np.where(t_ctx == 0, mean_y_t0, mean_y_t1).astype(np.float32)
    uwyk_model.fit(X_ctx, t_ctx_enc, y_ctx)

    n_test = cd.X_test.shape[0]
    T0 = np.full((n_test, 1), mean_y_t0, dtype=np.float32)
    T1 = np.full((n_test, 1), mean_y_t1, dtype=np.float32)

    def _draw(T_intv):
        chunks = []; got = 0
        while got < n_samples:
            r = uwyk_model.predict(
                X_obs=X_ctx, T_obs=t_ctx_enc, Y_obs=y_ctx,
                X_intv=X_qry, T_intv=T_intv,
                adjacency_matrix=adj,
                prediction_type='sample', inverse_transform=True,
            )
            arr = np.asarray(r).reshape(n_test, -1)
            chunks.append(arr); got += arr.shape[1]
        return np.concatenate(chunks, axis=1)[:, :n_samples]

    Y0 = _draw(T0)                # (n_test, n_samples)
    Y1 = _draw(T1)
    TAU = Y1 - Y0                 # element-wise pairing → independent τ

    def _kde_on_grid(samples, grid):
        n_test = samples.shape[0]
        out = np.zeros((n_test, grid.shape[0]))
        for q in range(n_test):
            s = samples[q]; sd = float(s.std())
            h = max(1.06 * sd * s.size ** (-0.2), 1e-3)
            u = (grid[None, :] - s[:, None]) / h
            k = np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
            p = k.mean(axis=0) / h
            z = p.sum() * (grid[1] - grid[0])
            if z > 0: p /= z
            out[q] = p
        return out

    p_y0 = _kde_on_grid(Y0, Y_GRID)
    p_y1 = _kde_on_grid(Y1, Y_GRID)
    p_tau = _kde_on_grid(TAU, TAU_GRID)
    cate_pt = TAU.mean(axis=1)
    return dict(p_y0=p_y0, p_y1=p_y1, p_tau=p_tau, cate=cate_pt)


def dopfn_densities(DoPFNRegressor, cd):
    """Do-PFN: Gaussian around per-arm point estimate. Under independence
    p(τ|x) = N(μ1̂-μ0̂, 2σ̂²)."""
    yhat0, yhat1, sigma = dopfn_predict_ymean(DoPFNRegressor, cd)
    p_y0 = _gauss_on_grid(yhat0, sigma, Y_GRID)
    p_y1 = _gauss_on_grid(yhat1, sigma, Y_GRID)
    sd_tau = float(np.sqrt(2.0) * sigma)
    p_tau = _gauss_on_grid(yhat1 - yhat0, sd_tau, TAU_GRID)
    cate_pt = yhat1 - yhat0
    return dict(p_y0=p_y0, p_y1=p_y1, p_tau=p_tau, cate=cate_pt)


# ── Per-SCM scoring ───────────────────────────────────────────────────────
def _pehe(true_cate, pred_cate):
    t = np.asarray(true_cate).reshape(-1)
    p = np.asarray(pred_cate).reshape(-1)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def score_one_scm(cd, method_densities, wb_fn):
    """Given per-method density dict from *_densities(...), compute:
        pehe, marg_l2 (avg over queries and arms), cate_l2 (avg over queries),
        ate_l2 (single value).
    Returns also the raw densities so caller can save diagnostic plots."""
    p0_true, p1_true = true_marginals(cd)
    p_tau_true = true_cate(cd)
    p_ate_true = wass_bary_of_grid(p_tau_true, TAU_GRID, wb_fn)

    metrics = {}
    diag = {'true_p_y0_q0': p0_true[0], 'true_p_y1_q0': p1_true[0],
             'true_p_tau_q0': p_tau_true[0], 'true_p_ate': p_ate_true,
             'true_cate': cd.true_cate.numpy()}

    for method_name, d in method_densities.items():
        pehe = _pehe(cd.true_cate.numpy(), d['cate'])
        # per-query, per-arm marginal L2; average over both arms and queries.
        n_q = d['p_y0'].shape[0]
        marg_l2 = 0.0
        for q in range(n_q):
            marg_l2 += l2_1d(d['p_y0'][q], p0_true[q], Y_DX)
            marg_l2 += l2_1d(d['p_y1'][q], p1_true[q], Y_DX)
        marg_l2 /= (2.0 * n_q)
        cate_l2 = float(np.mean([l2_1d(d['p_tau'][q], p_tau_true[q], TAU_DX)
                                   for q in range(n_q)]))
        p_ate_method = wass_bary_of_grid(d['p_tau'], TAU_GRID, wb_fn)
        ate_l2 = l2_1d(p_ate_method, p_ate_true, TAU_DX)

        metrics[f'pehe_{method_name}']    = pehe
        metrics[f'marg_l2_{method_name}'] = marg_l2
        metrics[f'cate_l2_{method_name}'] = cate_l2
        metrics[f'ate_l2_{method_name}']  = ate_l2

        diag[f'{method_name}_p_y0_q0']  = d['p_y0'][0]
        diag[f'{method_name}_p_y1_q0']  = d['p_y1'][0]
        diag[f'{method_name}_p_tau_q0'] = d['p_tau'][0]
        diag[f'{method_name}_p_ate']    = p_ate_method

    return metrics, diag


# ── Loaders ───────────────────────────────────────────────────────────────
def load_ours_ipfn(args, checkpoint_path):
    sys.path.insert(0, args.repo); sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    from models.InterventionalPFN import InterventionalPFN
    _orig = torch.load
    def _p_load(*a, **kw):
        kw.setdefault('weights_only', False); return _orig(*a, **kw)
    torch.load = _p_load
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
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
    ).to(DEVICE).eval()
    model.load_state_dict(ckpt['model_state_dict'])
    ot_dir = os.path.join(args.repo, 'MALC', 'Optimal_Transport')
    if ot_dir not in sys.path: sys.path.insert(0, ot_dir)
    from ot_barycenter import wasserstein_barycenter_1d
    return model, edges, J, bin_width, NF, wasserstein_barycenter_1d


def load_uwyk(uwyk_src, uwyk_ckpt_dir):
    _saved = {}
    for name in list(sys.modules):
        if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
            _saved[name] = sys.modules.pop(name)
    sys.path.insert(0, uwyk_src)
    mod = importlib.import_module('models.PreprocessingGraphConditionedPFN')
    sys.path.remove(uwyk_src)
    for name in list(sys.modules):
        if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
            del sys.modules[name]
    sys.modules.update(_saved)
    return mod.PreprocessingGraphConditionedPFN(
        config_path=os.path.join(uwyk_ckpt_dir, 'best_model_config.yaml'),
        checkpoint_path=os.path.join(uwyk_ckpt_dir, 'best_model.pt'),
        device='cpu', verbose=False, random_state=42,
    ).load()


# ── Aggregation / plotting ────────────────────────────────────────────────
METHODS = ('uwyk', 'ours50', 'dopfn', 'dopfnbb')
METHOD_LABEL = {
    'uwyk':    'UWYK-NoAnc',
    'ours50':  'Ours (fn=50)',
    'dopfn':   'Do-PFN',
    'dopfnbb': 'Ours-DoPFN-bb (200K)',
}
METHOD_COLOR = {
    'uwyk':    '#B84A2A',
    'ours50':  '#0F8A3C',
    'dopfn':   '#8A4FBE',
    'dopfnbb': '#2E4A6F',
}


def _plot_aggregate(args):
    import matplotlib.pyplot as plt
    all_arr = {}
    for idx in range(len(RHO_GRID)):
        shard = f'{args.out}.rho{idx}.npz'
        if not os.path.exists(shard):
            print(f'[warn] missing shard {shard}'); continue
        with np.load(shard, allow_pickle=True) as f:
            for k in f.files:
                all_arr.setdefault(k, []).append(f[k])
    if not all_arr:
        raise SystemExit('[error] no shards found')
    arr = {k: np.concatenate(v) for k, v in all_arr.items()}
    np.savez(args.out + '.npz', **arr)
    print(f'[save] {args.out}.npz  ({len(arr["rho"])} rows)')

    # ── Printed tables (one per metric) so user can copy-paste to paper ──
    METRIC_LABELS = [('pehe', 'PEHE'), ('marg_l2', 'Marginal-L2'),
                      ('cate_l2', 'CATE-L2'), ('ate_l2', 'ATE-L2')]
    for prefix, mlabel in METRIC_LABELS:
        header = f'{"ρ":>6s}   ' + '   '.join(
            f'{METHOD_LABEL[m]:>22s}' for m in METHODS)
        print(f'\n── {mlabel}  (mean ± SEM,  lower is better) ──')
        print(header)
        print('-' * len(header))
        for rho in RHO_GRID:
            mask = np.isclose(arr['rho'], rho)
            if not mask.any(): continue
            cells = []
            for m in METHODS:
                key = f'{prefix}_{m}'
                if key not in arr:
                    cells.append(f'{"—":>22s}'); continue
                v = arr[key][mask]; v = v[np.isfinite(v)]
                if v.size == 0:
                    cells.append(f'{"—":>22s}'); continue
                mean = v.mean()
                sem  = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
                cells.append(f'{mean:10.4f} ± {sem:8.4f}   ')
            print(f'{rho:>6.2f}   ' + ''.join(cells))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8))
    panels = [
        ('pehe',    axes[0, 0], 'PEHE',                    False),
        ('marg_l2', axes[0, 1], 'Marginal-density L2',     False),
        ('cate_l2', axes[1, 0], 'CATE-density L2',         False),
        ('ate_l2',  axes[1, 1], 'ATE-density L2',          False),
    ]
    for prefix, ax, title, logy in panels:
        for m in METHODS:
            key = f'{prefix}_{m}'
            if key not in arr:
                continue
            means, sems, xs = [], [], []
            for rho in RHO_GRID:
                mask = np.isclose(arr['rho'], rho)
                if not mask.any(): continue
                v = arr[key][mask]; v = v[np.isfinite(v)]
                if v.size == 0: continue
                xs.append(rho); means.append(v.mean())
                sems.append(v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)
            ax.errorbar(xs, means, yerr=sems,
                          fmt='o-', color=METHOD_COLOR[m], lw=1.7,
                          markersize=6, capsize=3, label=METHOD_LABEL[m])
        ax.set_xlabel(r'true DGP $\rho$')
        ax.set_ylabel(title + '   (lower is better)')
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if logy: ax.set_yscale('log')
        if prefix == 'pehe':
            ax.legend(loc='best', fontsize=9)

    fig.suptitle('Fig 2 — PEHE + density-L2 metrics vs ρ '
                  f'(K={arr["rho"].size // len(RHO_GRID)} SCMs / ρ, '
                  f'N-ctx={int(arr["n_context"][0]) if "n_context" in arr else "?"})',
                  fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out + '.png', dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'[save] {args.out}.png')


def _plot_diagnostics(args):
    """One 2x2 diagnostic PNG per ρ: p(Y_do0|x_q0), p(Y_do1|x_q0), p(τ|x_q0), p(τ_ATE).
    Curves for all 4 methods with true density as dashed reference."""
    import matplotlib.pyplot as plt
    diag_dir = os.path.join(os.path.dirname(args.out) or '.', 'density_diagnostics')
    os.makedirs(diag_dir, exist_ok=True)
    for idx, rho in enumerate(RHO_GRID):
        diag_shard = f'{args.out}.rho{idx}.diag.npz'
        if not os.path.exists(diag_shard):
            print(f'[warn] missing diagnostic shard {diag_shard}'); continue
        with np.load(diag_shard, allow_pickle=True) as f:
            d = {k: f[k] for k in f.files}

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        panels = [
            ('p_y0_q0',  Y_GRID,   axes[0, 0], r'$p(Y_{do(0)} \mid x_{q=0})$'),
            ('p_y1_q0',  Y_GRID,   axes[0, 1], r'$p(Y_{do(1)} \mid x_{q=0})$'),
            ('p_tau_q0', TAU_GRID, axes[1, 0], r'$p(\tau \mid x_{q=0})$'),
            ('p_ate',    TAU_GRID, axes[1, 1], r'$p(\tau_{ATE})$'),
        ]
        for key, grid, ax, title in panels:
            truth = d.get(f'true_{key}')
            if truth is not None:
                ax.plot(grid, truth, 'k--', lw=2, label='truth', alpha=0.85)
            for m in METHODS:
                p = d.get(f'{m}_{key}')
                if p is None: continue
                ax.plot(grid, p, color=METHOD_COLOR[m], lw=1.4,
                         label=METHOD_LABEL[m], alpha=0.9)
            ax.set_title(title)
            ax.set_xlabel('value'); ax.set_ylabel('density')
            ax.grid(alpha=0.3)
        axes[0, 0].legend(loc='upper right', fontsize=8)
        fig.suptitle(f'Density diagnostics at ρ = {rho:.2f} '
                      f'(first SCM, first query for marginals/CATE; barycenter for ATE)',
                      fontsize=11, y=1.02)
        fig.tight_layout()
        out = os.path.join(diag_dir, f'rho_{idx}_{str(rho).replace(".", "p")}.png')
        fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
        print(f'[save] {out}')


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo',                 required=True)
    ap.add_argument('--dopfn',                required=True)
    ap.add_argument('--causalpfn',            required=True)
    ap.add_argument('--uwyk-src',             required=True)
    ap.add_argument('--uwyk-ckpt-dir',        required=True)
    ap.add_argument('--checkpoint50',         required=True)
    ap.add_argument('--checkpoint-dopfn-bb',  required=True)
    ap.add_argument('--K',                    type=int, default=20)
    ap.add_argument('--N-context',            type=int, default=200)
    ap.add_argument('--N-test',               type=int, default=50)
    ap.add_argument('--uwyk-n-samples',       type=int, default=500)
    ap.add_argument('--out',                  default='fig2_pehe_l2')
    ap.add_argument('--rho-index',            type=int, default=-1)
    ap.add_argument('--plot',                 action='store_true')
    args = ap.parse_args()

    if args.plot:
        _plot_aggregate(args)
        _plot_diagnostics(args)
        return

    _out_dir = os.path.dirname(args.out)
    if _out_dir:
        os.makedirs(_out_dir, exist_ok=True)

    if args.rho_index >= 0:
        rho_targets = [(args.rho_index, RHO_GRID[args.rho_index], RHO_EFF[args.rho_index])]
        shard_path = f'{args.out}.rho{args.rho_index}.npz'
    else:
        rho_targets = list(zip(range(len(RHO_GRID)), RHO_GRID, RHO_EFF))
        shard_path = args.out + '.npz'

    sys.path.insert(0, args.causalpfn)
    print('[load] Ours fn=50 (InterventionalPFN)', flush=True)
    ipfn_model, ipfn_edges, ipfn_J, ipfn_bw, ipfn_NF, wb_fn = load_ours_ipfn(args, args.checkpoint50)
    print('[load] Ours-DoPFN-bb (200K)', flush=True)
    bb_model, bb_edges, bb_J, bb_bw, bb_centers, bb_NF, _ = load_dopfn_bb(
        args, args.checkpoint_dopfn_bb)
    print('[load] UWYK-NoAnc', flush=True)
    uwyk = load_uwyk(args.uwyk_src, args.uwyk_ckpt_dir)
    print('[load] Do-PFN', flush=True)
    DoPFNRegressor = load_dopfn(args)
    _dopfn_cwd = os.getcwd()

    rows = []

    for rho_idx, rho, rho_eff in rho_targets:
        diag_this_rho = {}          # first-SCM diagnostics for this ρ only
        for k in range(args.K):
            seed = rho_idx * 10_000 + k
            cd = make_polynomial_scm(seed, args.N_context, args.N_test, rho_eff)

            t0 = time.time()
            os.chdir(args.dopfn)
            d_dopfn = dopfn_densities(DoPFNRegressor, cd)
            os.chdir(_dopfn_cwd)
            d_uwyk  = uwyk_densities(uwyk, cd, n_samples=args.uwyk_n_samples)
            d_ours50 = ours_densities(ipfn_model, ipfn_edges, ipfn_J, ipfn_bw, ipfn_NF, cd)
            d_bb     = ours_densities(bb_model, bb_edges, bb_J, bb_bw, bb_NF, cd)

            method_densities = {'uwyk': d_uwyk, 'ours50': d_ours50,
                                  'dopfn': d_dopfn, 'dopfnbb': d_bb}
            metrics, diag = score_one_scm(cd, method_densities, wb_fn)
            metrics['rho'] = rho; metrics['seed'] = seed
            metrics['n_context'] = args.N_context; metrics['n_test'] = args.N_test
            rows.append(metrics)
            if k == 0:
                diag_this_rho = {**diag, 'rho': rho}
            dt = time.time() - t0
            print(f'[scm] ρ={rho:.2f}  k={k:2d}  '
                  f'PEHE(uwyk={metrics["pehe_uwyk"]:.3f} '
                  f'ours50={metrics["pehe_ours50"]:.3f} '
                  f'dopfn={metrics["pehe_dopfn"]:.3f} '
                  f'dopfnbb={metrics["pehe_dopfnbb"]:.3f})  '
                  f'CATE-L2(ours50={metrics["cate_l2_ours50"]:.3f} '
                  f'dopfnbb={metrics["cate_l2_dopfnbb"]:.3f})  '
                  f'{dt:.1f}s', flush=True)
            gc.collect()

        # Save one diagnostic shard per ρ so both per-task and single-process
        # runs agree with the aggregator's per-ρ filename convention.
        if diag_this_rho:
            diag_path = f'{args.out}.rho{rho_idx}.diag.npz'
            np.savez(diag_path,
                      **{k: np.asarray(v) for k, v in diag_this_rho.items()})
            print(f'[save] {diag_path}')

    if not rows:
        print('[skip] nothing to save'); return

    keys = sorted({k for r in rows for k in r.keys()})
    arr = {k: np.array([r.get(k, np.nan) for r in rows]) for k in keys}
    np.savez(shard_path, **arr)
    print(f'[save] {shard_path}  ({len(rows)} rows)')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
