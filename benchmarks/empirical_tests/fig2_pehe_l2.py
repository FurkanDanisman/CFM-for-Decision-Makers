"""Fig 2 v2 — PEHE / marginal-L2 / CATE-L2 / ATE-L2 vs ρ, discrete bar probs.

Density derivation is done directly from each model's own BarDistribution
head (no KDE, no Gaussian approximation, no sampling+smoothing):

    Ours (2D BarDist head)
        p_mat[i, j]  = raw joint density on the (J × J) bin grid.
        marginals    : p0[i] = Σ_j p_mat[i,j]   ;   p1[j] = Σ_i p_mat[i,j]
        CATE         : p_τ[k] = Σ_{i,j: c[j]-c[i]∈bin k} p_mat[i,j]
                       (captures ρ; the joint sums along τ diagonals).
        MALC variant : replace CATE p_τ with _fit_and_marginalize(p_mat, ...)
                       from benchmarks/methods/ours.py. Marginals stay raw.

    UWYK-NoAnc (BarDist head, per-arm)
        p_arm[i]     = softmax over the BarDist head's interior K bins,
                       obtained by monkey-patching bar_distribution.mode to
                       capture raw_preds during a single predict() call.
        CATE         : p_τ[k] = Σ_{i,j: c[j]-c[i]∈bin k} p1[j] · p0[i]
                       (independence-assumption convolution).

    Do-PFN (FullSupportBarDistribution head)
        p_arm[i]     = predict_full(X)['buckets'][i] (already softmax'd).
        CATE         : independence convolution (as UWYK).

All densities land on each model's own K/J bin centers, then are
interpolated to a shared τ-grid and Y-grid ONLY for L2 comparison.
Interpolation preserves shape; direct L2 numbers within a comparison
pair (Ours vs UWYK  or  Ours-DoPFN-bb vs Do-PFN) are meaningful.

Comparison pairs (paper-relevant; no cross-method rows):
    Pair A: {UWYK-NoAnc}  vs  {Ours(fn=50) raw, Ours(fn=50) MALC}
    Pair B: {Do-PFN}      vs  {Ours-DoPFN-bb raw, Ours-DoPFN-bb MALC}

ρ grid: {0, 0.2, 0.4, 0.6, 0.8, 1.0}. ρ=1 is evaluated at 0.99 internally
(truth CATE density becomes a Dirac at ρ=1 exactly).
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
from dopfn_helpers import load_dopfn_bb, load_dopfn
# 1D MALC for Ours marginals (Python port via CVXPY; density_calc.md §3b)
_l2ihdp = os.path.abspath(os.path.join(_here, '..', 'l2_ihdp'))
if _l2ihdp not in sys.path: sys.path.insert(0, _l2ihdp)
from methods_densities import malc_1d_cvxpy  # noqa: E402

DEVICE = torch.device('cpu')
RHO_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
RHO_EFF  = tuple(min(r, 0.99) for r in RHO_GRID)

# Shared τ and Y grids used ONLY for L2 comparison across methods that
# have different native K/J. Fine enough that interpolation error is small.
Y_MIN, Y_MAX, Y_BINS = -8.0, 8.0, 501
TAU_MIN, TAU_MAX, TAU_BINS = -10.0, 10.0, 501
Y_GRID   = np.linspace(Y_MIN, Y_MAX, Y_BINS)
TAU_GRID = np.linspace(TAU_MIN, TAU_MAX, TAU_BINS)
Y_DX     = float(Y_GRID[1] - Y_GRID[0])
TAU_DX   = float(TAU_GRID[1] - TAU_GRID[0])


# ── Polynomial SCM ────────────────────────────────────────────────────────
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


# ── Truth (analytic) ──────────────────────────────────────────────────────
def _gauss_on_grid(mu, sigma, grid):
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    z = (grid[None, :] - mu[:, None]) / max(sigma, 1e-12)
    p = np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * max(sigma, 1e-12))
    s = p.sum(axis=1, keepdims=True) * (grid[1] - grid[0])
    return p / np.maximum(s, 1e-12)


def truth_marginals(cd):
    p0 = _gauss_on_grid(cd._mu0_test, cd._sigma_eps, Y_GRID)
    p1 = _gauss_on_grid(cd._mu1_test, cd._sigma_eps, Y_GRID)
    return p0, p1


def truth_cate(cd):
    var_tau = 2.0 * cd._sigma_eps ** 2 * (1.0 - cd._rho_eff)
    sd_tau = float(np.sqrt(max(var_tau, 1e-12)))
    mu_tau = cd._mu1_test - cd._mu0_test
    return _gauss_on_grid(mu_tau, sd_tau, TAU_GRID)


# ── L2 + resampling to common grid ────────────────────────────────────────
def _to_common_grid(density_native, centers_native, common_grid):
    """Linear-interpolate a density from its native centers onto the shared
    common_grid. Zeros outside native support. Renormalises."""
    p = np.interp(common_grid, centers_native, density_native, left=0.0, right=0.0)
    dx = common_grid[1] - common_grid[0]
    s = float(p.sum() * dx)
    return p / s if s > 0 else p


def _discrete_to_density(p_discrete, centers):
    """Bar probabilities → density on the same bar centers."""
    bw = float(centers[1] - centers[0])
    return p_discrete / max(bw, 1e-12)


def l2_1d(f, g, dx):
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.sum((f - g) ** 2) * dx))


def kl_1d(p, q, dx, eps=1e-12):
    """KL(p || q) = ∫ p log(p/q) dx  (Riemann sum on common grid).
    Both p and q must be densities integrating to 1 on the same grid."""
    p = np.asarray(p, dtype=np.float64).reshape(-1) + eps
    q = np.asarray(q, dtype=np.float64).reshape(-1) + eps
    return float(np.sum(p * np.log(p / q)) * dx)


def wass_bary_of_grid(p_matrix, grid, wb_fn):
    p_ate = wb_fn(p_matrix, grid)
    dx = grid[1] - grid[0]
    s = p_ate.sum() * dx
    return p_ate / s if s > 0 else p_ate


# ── CATE derivation ───────────────────────────────────────────────────────
def _joint_to_ptau(p_mat, centers, tau_grid):
    """CATE density from a 2D joint p_mat[i,j] on (centers × centers) bins.
    Sums along τ = c[j] - c[i] diagonals into tau_grid bins."""
    # Precompute τ_ij (J,J) and their tau_grid bin indices.
    tau_ij = centers[None, :] - centers[:, None]
    tau_min = float(tau_grid[0]); tau_dx = float(tau_grid[1] - tau_grid[0])
    idx = np.round((tau_ij - tau_min) / tau_dx).astype(int)
    valid = (idx >= 0) & (idx < tau_grid.shape[0])
    hist = np.zeros(tau_grid.shape[0])
    np.add.at(hist, idx[valid], p_mat[valid])
    return hist / max(tau_dx, 1e-12)


def _marginals_to_ptau(p0, p1, centers, tau_grid):
    """CATE density under independence: p_τ[k] = Σ p1[j]·p0[i] on the same
    τ_ij bins as _joint_to_ptau. Equivalent to a discrete convolution."""
    joint = np.outer(p0, p1)                # (J,J), p0=row, p1=col
    return _joint_to_ptau(joint, centers, tau_grid)


# ── Ours (fn=50 or DoPFN-bb): raw p_mat via forward pass ──────────────────
def _pad(X, n_feat):
    if n_feat <= 0: return np.asarray(X, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1: X = X.reshape(-1, 1)
    d = X.shape[1]
    if d < n_feat:
        pad = np.full((X.shape[0], n_feat - d), np.nan, dtype=np.float32)
        return np.concatenate([X, pad], axis=1)
    return X[:, :n_feat]


def ours_forward(model, edges, J, bin_width, NF, cd):
    """Returns per-query 2D joint p_mat on raw y centers.
        p_mat_all: (n_test, J, J)
        centers_raw: (J,)  in raw y units
    Rescale/unscale uses train Y min/max as reference. """
    from losses.BarDistribution2D import unpack_pred
    y_ctx_raw = cd.y_train.numpy().reshape(-1)
    y_min = float(y_ctx_raw.min()); y_max = float(y_ctx_raw.max())
    y_rng = max(y_max - y_min, 1e-6)
    def _rescale(y): return (y - y_min) / y_rng * 2.0 - 1.0
    def _unscale(y): return (y + 1.0) * 0.5 * y_rng + y_min

    Y_ctx = _rescale(y_ctx_raw).reshape(-1, 1).astype(np.float32)
    T_ctx = cd.t_train.numpy().astype(np.float32).reshape(-1, 1)
    X_ctx = _pad(cd.X_train.numpy(), NF)
    X_qry = _pad(cd.X_test.numpy(),  NF)

    with torch.no_grad():
        pred = model(torch.from_numpy(X_ctx).unsqueeze(0),
                      torch.from_numpy(T_ctx).unsqueeze(0),
                      torch.from_numpy(Y_ctx).unsqueeze(0),
                      torch.from_numpy(X_qry).unsqueeze(0))['predictions'][0]

    centers_scaled = 0.5 * (edges[:-1] + edges[1:])
    centers_raw = _unscale(centers_scaled).astype(np.float64)

    n_test = cd.X_test.shape[0]
    p_mat_all = np.zeros((n_test, J, J), dtype=np.float64)
    for q in range(n_test):
        p_mat, *_ = unpack_pred(pred[q], J, bin_width)
        pm = p_mat.detach().cpu().numpy().astype(np.float64)
        # Sums to ~1 already; normalise for numerical safety.
        s = pm.sum()
        p_mat_all[q] = pm / s if s > 0 else pm
    return p_mat_all, centers_raw


# ── UWYK: raw bar probs via bar_distribution monkey-patch ─────────────────
def _uwyk_capture_raw_preds(uwyk_model, X_ctx, T_ctx, Y_ctx, X_qry, T_intv,
                              adjacency):
    """Fires uwyk_model.predict(prediction_type='mode') but grabs raw_preds
    from bar_distribution.mode via a monkey-patch."""
    captured = {'raw': None}
    orig_mode = uwyk_model.bar_distribution.mode

    def _mode_capture(raw_preds):
        captured['raw'] = raw_preds.detach().cpu()
        return orig_mode(raw_preds)

    uwyk_model.bar_distribution.mode = _mode_capture
    try:
        _ = uwyk_model.predict(
            X_obs=X_ctx, T_obs=T_ctx, Y_obs=Y_ctx,
            X_intv=X_qry, T_intv=T_intv,
            adjacency_matrix=adjacency,
            prediction_type='mode', inverse_transform=False,
        )
    finally:
        uwyk_model.bar_distribution.mode = orig_mode
    return captured['raw']


def uwyk_forward_raw_probs(uwyk_model, cd, adjacency_kind='noanc'):
    """Returns per-arm bar probabilities on UWYK's INTERIOR K bin centers,
    in the model's TARGET-SCALED space [-1, 1]. Caller must inverse-scale
    to raw y if it wants to compare in raw units.

    adjacency_kind:
      'noanc' — zero adjacency (no graph hint at inference)
      'anc'   — full-graph adjacency (T→Y, X→T, X→Y for real features)

        p0_all: (n_test, K)   p1_all: (n_test, K)
        centers_scaled: (K,)  in [-1, 1] scaled y space
    """
    NF = uwyk_model.model.num_features
    X_ctx = _pad(cd.X_train.numpy(), NF)
    t_ctx = cd.t_train.numpy().astype(np.float32).reshape(-1, 1)
    y_ctx = cd.y_train.numpy().astype(np.float32).reshape(-1, 1)
    X_qry = _pad(cd.X_test.numpy(), NF)
    n_real = min(cd.X_train.shape[1], NF)
    adj = np.zeros((NF + 2, NF + 2), dtype=np.float32)
    if adjacency_kind == 'anc':
        T_idx, Y_idx, off = 0, 1, 2
        adj[T_idx, Y_idx] = 1.0
        for i in range(n_real):
            adj[off + i, T_idx] = 1.0
            adj[off + i, Y_idx] = 1.0
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

    raw0 = _uwyk_capture_raw_preds(uwyk_model, X_ctx, t_ctx_enc, y_ctx, X_qry, T0, adj)
    raw1 = _uwyk_capture_raw_preds(uwyk_model, X_ctx, t_ctx_enc, y_ctx, X_qry, T1, adj)

    # raw shape: (B=1, n_test_padded, K+4). First K+2 are logits: [pL, bars..., pR]
    def _to_bar_probs(raw):
        w_logits = raw[..., :-2]                              # (1, n_test, K+2)
        probs = torch.softmax(w_logits, dim=-1)
        pBars = probs[..., 1:-1].squeeze(0).numpy()            # (n_test, K)
        return pBars

    p0_padded = _to_bar_probs(raw0)                            # (n_test_padded, K)
    p1_padded = _to_bar_probs(raw1)
    p0_all = p0_padded[:n_test]                                # trim padding
    p1_all = p1_padded[:n_test]

    # Renormalise (interior bars only; tails were excluded — mass slightly < 1).
    for arr in (p0_all, p1_all):
        s = arr.sum(axis=1, keepdims=True)
        arr /= np.maximum(s, 1e-12)

    centers_scaled = uwyk_model.bar_distribution.centers.detach().cpu().numpy().astype(np.float64)
    return p0_all, p1_all, centers_scaled, (mean_y_t0, mean_y_t1)


def uwyk_inverse_scale_centers(centers_scaled, cd):
    """UWYK scales targets to [-1, 1] using train Y range at .fit() time.
    Recover raw-y centers using the same formula."""
    y = cd.y_train.numpy().reshape(-1)
    y_min = float(y.min()); y_max = float(y.max())
    y_rng = max(y_max - y_min, 1e-6)
    return (centers_scaled + 1.0) * 0.5 * y_rng + y_min


# ── Do-PFN: raw bar probs via predict_full()['buckets'] ────────────────────
def dopfn_forward_raw_probs(DoPFNRegressor, cd):
    """Returns per-arm bar probabilities from Do-PFN's BarDistribution head.
        p0_all: (n_test, K)   p1_all: (n_test, K)
        centers_raw: (K,)     in raw y units (borders already rescaled by DoPFN)"""
    def _np(a):
        if isinstance(a, torch.Tensor): return a.numpy()
        return np.asarray(a)
    X_train = _np(cd.X_train).astype(np.float32)
    t_train = _np(cd.t_train).astype(np.float32).reshape(-1)
    y_train = _np(cd.y_train).astype(np.float32).reshape(-1)
    X_test  = _np(cd.X_test).astype(np.float32)
    x_tr = np.concatenate([t_train[:, None], X_train], axis=1)
    x_te0 = np.concatenate([np.zeros((X_test.shape[0], 1), dtype=np.float32), X_test], axis=1)
    x_te1 = np.concatenate([np.ones((X_test.shape[0], 1),  dtype=np.float32), X_test], axis=1)

    reg = DoPFNRegressor()
    reg.fit(torch.tensor(x_tr), torch.tensor(y_train))
    pred0 = reg.predict_full(torch.tensor(x_te0))
    pred1 = reg.predict_full(torch.tensor(x_te1))

    p0 = np.asarray(pred0['buckets'], dtype=np.float64)       # (n_test, num_bars)
    p1 = np.asarray(pred1['buckets'], dtype=np.float64)
    borders = np.asarray(pred0['criterion'].borders.numpy(), dtype=np.float64)
    centers_raw = 0.5 * (borders[:-1] + borders[1:])          # (num_bars,)

    # Renormalise for numerical safety.
    p0 /= np.maximum(p0.sum(axis=1, keepdims=True), 1e-12)
    p1 /= np.maximum(p1.sum(axis=1, keepdims=True), 1e-12)
    return p0, p1, centers_raw


# ── MALC setup (in-process) ───────────────────────────────────────────────
_MALC_STATE = {}


def setup_malc(edges, J, bin_width, N_EVAL, MALC_B, MALC_MAX_K, repo):
    """Populate ours.py::_GLOBAL directly (no worker pool needed)."""
    malc_dir = os.path.join(repo, 'MALC')
    if repo not in sys.path: sys.path.insert(0, repo)
    if malc_dir not in sys.path: sys.path.insert(0, malc_dir)
    from methods import ours as ours_mod
    from losses.BarDistribution2D import fit_malc_inner
    from malc_2d import dmalc_2d
    G = ours_mod._GLOBAL
    G['fit'] = fit_malc_inner; G['dmalc'] = dmalc_2d
    G['edges'] = edges; G['J'] = J; G['bw'] = bin_width
    G['MALC_B'] = MALC_B; G['MALC_MAX_K'] = MALC_MAX_K
    xs = np.linspace(edges[0], edges[-1], N_EVAL)
    ys = np.linspace(edges[0], edges[-1], N_EVAL)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    G['xs'] = xs; G['ys'] = ys
    G['eval_pts'] = np.column_stack([XX.ravel(), YY.ravel()])
    G['dy0'] = xs[1] - xs[0]; G['dy1'] = ys[1] - ys[0]
    G['tau'] = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], 401)
    G['dtau'] = G['tau'][1] - G['tau'][0]
    _MALC_STATE['fam_mod'] = ours_mod
    _MALC_STATE['tau_scaled'] = G['tau']
    _MALC_STATE['edges'] = edges


def malc_cate_density_scaled(p_mat_scaled, seed):
    """Runs _fit_and_marginalize on a (J,J) raw p_mat. Returns:
        p_tau_scaled: (401,) density on _MALC_STATE['tau_scaled']  (scaled τ ∈ [-2, 2])"""
    ours = _MALC_STATE['fam_mod']
    p_tau = ours._fit_and_marginalize(p_mat_scaled, seed)     # normalised density
    return np.asarray(p_tau, dtype=np.float64)


def scaled_tau_to_raw_centers(cd):
    """Convert MALC's scaled τ-grid ([-2, 2]) to raw-τ centers.
    Ours' scaled y ∈ [-1, 1] corresponds to raw y ∈ [y_min, y_max].
    So scaled τ = τ_raw · (2 / y_rng). Inverse: τ_raw = τ_scaled · (y_rng / 2)."""
    y = cd.y_train.numpy().reshape(-1)
    y_rng = max(float(y.max()) - float(y.min()), 1e-6)
    return _MALC_STATE['tau_scaled'] * (y_rng / 2.0), y_rng


# ── One-SCM scoring ───────────────────────────────────────────────────────
def _pehe(true_cate, pred_cate):
    t = np.asarray(true_cate).reshape(-1)
    p = np.asarray(pred_cate).reshape(-1)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def _cate_point_from_density(p_tau, tau_grid):
    dx = tau_grid[1] - tau_grid[0]
    return float((tau_grid * p_tau).sum() * dx)


def score_one_scm(cd, method_data, wb_fn):
    """method_data: dict{method_name -> {p_y0_all, p_y1_all, centers_y_raw,
                                          p_tau_all, tau_centers_raw,
                                          cate_pred}}.
    Everything expressed in raw-y and raw-τ units.
    Returns per-method dict of metrics + per-method densities interpolated
    onto the shared grids for diagnostics.
    """
    p0_true, p1_true = truth_marginals(cd)          # (n_test, Y_BINS)
    p_tau_true = truth_cate(cd)                     # (n_test, TAU_BINS)
    p_ate_true = wass_bary_of_grid(p_tau_true, TAU_GRID, wb_fn)

    metrics = {}
    diag = {'true_p_y0_q0': p0_true[0], 'true_p_y1_q0': p1_true[0],
             'true_p_tau_q0': p_tau_true[0], 'true_p_ate': p_ate_true,
             'true_cate': cd.true_cate.numpy()}

    for name, d in method_data.items():
        # ─ interpolate to shared Y_GRID / TAU_GRID for L2 ─
        n_test = d['p_y0_all'].shape[0]
        p0_common = np.zeros((n_test, Y_BINS));  p1_common = np.zeros((n_test, Y_BINS))
        p_tau_common = np.zeros((n_test, TAU_BINS))
        for q in range(n_test):
            d0 = _discrete_to_density(d['p_y0_all'][q], d['centers_y_raw'])
            d1 = _discrete_to_density(d['p_y1_all'][q], d['centers_y_raw'])
            p0_common[q] = _to_common_grid(d0, d['centers_y_raw'], Y_GRID)
            p1_common[q] = _to_common_grid(d1, d['centers_y_raw'], Y_GRID)
            dtau = _discrete_to_density(d['p_tau_all'][q], d['tau_centers_raw'])
            p_tau_common[q] = _to_common_grid(dtau, d['tau_centers_raw'], TAU_GRID)

        # ─ PEHE ─
        pehe = _pehe(cd.true_cate.numpy(), d['cate_pred'])

        # ─ Marginal L2 + KL (NaN-safe mean over both arms and queries) ─
        _mv = lambda vs: float(np.nanmean(vs)) if np.any(np.isfinite(vs)) else float('nan')
        marg_l2_all      = []
        marg_kl_fwd_all  = []
        marg_kl_rev_all  = []
        for q in range(n_test):
            for tside in (0, 1):
                if tside == 0:
                    pe, pt = p0_common[q], p0_true[q]
                else:
                    pe, pt = p1_common[q], p1_true[q]
                if not (np.all(np.isfinite(pe)) and np.all(np.isfinite(pt))):
                    continue
                marg_l2_all.append(l2_1d(pe, pt, Y_DX))
                marg_kl_fwd_all.append(kl_1d(pt, pe, Y_DX))
                marg_kl_rev_all.append(kl_1d(pe, pt, Y_DX))
        marg_l2      = _mv(marg_l2_all)
        marg_kl_fwd  = _mv(marg_kl_fwd_all)
        marg_kl_rev  = _mv(marg_kl_rev_all)

        # ─ CATE L2 + KL (NaN-safe mean over queries) ─
        cate_l2_all, cate_kl_fwd_all, cate_kl_rev_all = [], [], []
        for q in range(n_test):
            pe, pt = p_tau_common[q], p_tau_true[q]
            if not (np.all(np.isfinite(pe)) and np.all(np.isfinite(pt))):
                continue
            cate_l2_all.append(l2_1d(pe, pt, TAU_DX))
            cate_kl_fwd_all.append(kl_1d(pt, pe, TAU_DX))
            cate_kl_rev_all.append(kl_1d(pe, pt, TAU_DX))
        cate_l2     = _mv(cate_l2_all)
        cate_kl_fwd = _mv(cate_kl_fwd_all)
        cate_kl_rev = _mv(cate_kl_rev_all)

        # ─ ATE L2 + KL (single value) ─
        p_ate_method = wass_bary_of_grid(p_tau_common, TAU_GRID, wb_fn)
        ate_l2       = l2_1d(p_ate_method, p_ate_true, TAU_DX)
        ate_kl_fwd   = kl_1d(p_ate_true, p_ate_method, TAU_DX)
        ate_kl_rev   = kl_1d(p_ate_method, p_ate_true, TAU_DX)

        metrics[f'pehe_{name}']         = pehe
        metrics[f'marg_l2_{name}']      = marg_l2
        metrics[f'marg_kl_fwd_{name}']  = marg_kl_fwd
        metrics[f'marg_kl_rev_{name}']  = marg_kl_rev
        metrics[f'cate_l2_{name}']      = cate_l2
        metrics[f'cate_kl_fwd_{name}']  = cate_kl_fwd
        metrics[f'cate_kl_rev_{name}']  = cate_kl_rev
        metrics[f'ate_l2_{name}']       = ate_l2
        metrics[f'ate_kl_fwd_{name}']   = ate_kl_fwd
        metrics[f'ate_kl_rev_{name}']   = ate_kl_rev

        diag[f'{name}_p_y0_q0']  = p0_common[0]
        diag[f'{name}_p_y1_q0']  = p1_common[0]
        diag[f'{name}_p_tau_q0'] = p_tau_common[0]
        diag[f'{name}_p_ate']    = p_ate_method

    return metrics, diag


# ── Loaders ───────────────────────────────────────────────────────────────
def load_ours_ipfn(args, checkpoint_path):
    sys.path.insert(0, args.repo); sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    from models.InterventionalPFN import InterventionalPFN
    _orig = torch.load
    def _p_load(*a, **kw): kw.setdefault('weights_only', False); return _orig(*a, **kw)
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
        use_clustering=False,   # sample/raw shape bug in clustered path
    ).load()


# ── Aggregation / plotting ────────────────────────────────────────────────
# Method list agreed 2026-08-16 (density_calc.md §3, §4):
#   UWYK-NoAnc, UWYK-FullAnc, Do-PFN, Ours(fn=50)
# Ours-DoPFN-bb dropped this round.
METHODS = ('uwyk_noanc', 'uwyk_anc', 'dopfn', 'ours_fn50')
METHOD_LABEL = {
    'uwyk_noanc': 'UWYK-NoAnc',
    'uwyk_anc':   'UWYK-FullAnc',
    'dopfn':      'Do-PFN',
    'ours_fn50':  'Ours(fn=50) MALC',
}
METHOD_COLOR = {
    'uwyk_noanc': '#B84A2A',
    'uwyk_anc':   '#DC7A5A',
    'dopfn':      '#8A4FBE',
    'ours_fn50':  '#0F8A3C',
}
# Single 4-method table (no pair split — per user 2026-08-16).
PAIRS = [('All 4 methods', list(METHODS))]


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

    # ── Tables per pair — all metrics (PEHE, L2, KL_fwd, KL_rev) ──
    METRIC_PREFIXES = [
        ('pehe',        'PEHE'),
        ('marg_l2',     'Marg-L2'),
        ('marg_kl_fwd', 'Marg-KL_fwd'),
        ('marg_kl_rev', 'Marg-KL_rev'),
        ('cate_l2',     'CATE-L2'),
        ('cate_kl_fwd', 'CATE-KL_fwd'),
        ('cate_kl_rev', 'CATE-KL_rev'),
        ('ate_l2',      'ATE-L2'),
        ('ate_kl_fwd',  'ATE-KL_fwd'),
        ('ate_kl_rev',  'ATE-KL_rev'),
    ]
    for pair_label, pair_methods in PAIRS:
        print(f'\n╔══ {pair_label}  ({"  ".join(METHOD_LABEL[m] for m in pair_methods)}) ══')
        for prefix, mlabel in METRIC_PREFIXES:
            header = f'{mlabel:>12s}  ρ→' + '   '.join(f'{METHOD_LABEL[m]:>26s}'
                                                        for m in pair_methods)
            print()
            print(header)
            print('-' * len(header))
            for rho in RHO_GRID:
                mask = np.isclose(arr['rho'], rho)
                if not mask.any(): continue
                cells = []
                for m in pair_methods:
                    key = f'{prefix}_{m}'
                    if key not in arr: cells.append(f'{"—":>26s}'); continue
                    v = arr[key][mask]; v = v[np.isfinite(v)]
                    if v.size == 0: cells.append(f'{"—":>26s}'); continue
                    mean = v.mean()
                    sem = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
                    cells.append(f'{mean:11.4f} ± {sem:8.4f}   ')
                print(f'ρ={rho:>4.2f}  ' + ''.join(cells))

    # ── 2x2 summary plot per pair (L2 panels; KL is only in tables) ──
    for pair_label, pair_methods in PAIRS:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.8))
        for (prefix, title), ax in zip([('pehe','PEHE (CATE point estimate)'),
                                          ('marg_l2','Marginal density L2'),
                                          ('cate_l2','CATE density L2'),
                                          ('ate_l2','ATE density L2')],
                                         axes.flat):
            for m in pair_methods:
                key = f'{prefix}_{m}'
                if key not in arr: continue
                xs, means, sems = [], [], []
                for rho in RHO_GRID:
                    mask = np.isclose(arr['rho'], rho)
                    if not mask.any(): continue
                    v = arr[key][mask]; v = v[np.isfinite(v)]
                    if v.size == 0: continue
                    xs.append(rho); means.append(v.mean())
                    sems.append(v.std(ddof=1)/np.sqrt(v.size) if v.size > 1 else 0.0)
                ax.errorbar(xs, means, yerr=sems, fmt='o-',
                              color=METHOD_COLOR[m], lw=1.7, markersize=6,
                              capsize=3, label=METHOD_LABEL[m])
            ax.set_xlabel(r'true DGP $\rho$')
            ax.set_ylabel(title + '  (lower is better)')
            ax.set_title(title); ax.grid(alpha=0.3)
            if prefix == 'pehe':
                ax.legend(loc='best', fontsize=8)
        fig.suptitle(f'Fig 2 — {pair_label}', fontsize=11, y=1.02)
        fig.tight_layout()
        suffix = pair_label.split(' ')[1][1:-1]  # 'fn=50' or 'DoPFN-bb'
        out = f'{args.out}_{suffix.replace("=","").replace("-","_")}.png'
        fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
        print(f'[save] {out}')


def _plot_diagnostics(args):
    import matplotlib.pyplot as plt
    diag_dir = os.path.join(os.path.dirname(args.out) or '.', 'density_diagnostics')
    os.makedirs(diag_dir, exist_ok=True)
    for idx, rho in enumerate(RHO_GRID):
        diag_shard = f'{args.out}.rho{idx}.diag.npz'
        if not os.path.exists(diag_shard):
            print(f'[warn] missing diag shard {diag_shard}'); continue
        with np.load(diag_shard, allow_pickle=True) as f:
            d = {k: f[k] for k in f.files}
        for pair_label, pair_methods in PAIRS:
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
                for m in pair_methods:
                    p = d.get(f'{m}_{key}')
                    if p is None: continue
                    ax.plot(grid, p, color=METHOD_COLOR[m], lw=1.4,
                             label=METHOD_LABEL[m], alpha=0.9)
                ax.set_title(title); ax.grid(alpha=0.3)
                ax.set_xlabel('value'); ax.set_ylabel('density')
            axes[0, 0].legend(loc='upper right', fontsize=8)
            fig.suptitle(f'Density diagnostics — {pair_label} — ρ = {rho:.2f}',
                          fontsize=11, y=1.02)
            fig.tight_layout()
            suffix = pair_label.split(' ')[1][1:-1]
            tag = suffix.replace('=','').replace('-','_')
            out = os.path.join(diag_dir,
                f'rho_{idx}_{str(rho).replace(".","p")}_{tag}.png')
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
    ap.add_argument('--checkpoint-dopfn-bb',  default='',
                    help='OPTIONAL — kept for backward-compat; ignored in v3 '
                         '(Ours-DoPFN-bb dropped from Fig 2 per user 2026-08-16).')
    ap.add_argument('--K',                    type=int, default=20)
    ap.add_argument('--N-context',            type=int, default=200)
    ap.add_argument('--N-test',               type=int, default=50)
    ap.add_argument('--malc-B',               type=int, default=200)
    ap.add_argument('--malc-max-K',           type=int, default=3)
    ap.add_argument('--malc-n-eval',          type=int, default=100)
    ap.add_argument('--out',                  default='fig2_pehe_l2_v2')
    ap.add_argument('--rho-index',            type=int, default=-1)
    ap.add_argument('--plot',                 action='store_true')
    args = ap.parse_args()

    _out_dir = os.path.dirname(args.out)
    if _out_dir:
        os.makedirs(_out_dir, exist_ok=True)

    if args.plot:
        _plot_aggregate(args); _plot_diagnostics(args); return

    if args.rho_index >= 0:
        rho_targets = [(args.rho_index, RHO_GRID[args.rho_index], RHO_EFF[args.rho_index])]
    else:
        rho_targets = list(zip(range(len(RHO_GRID)), RHO_GRID, RHO_EFF))

    sys.path.insert(0, args.causalpfn)
    print('[load] Ours fn=50 (InterventionalPFN)', flush=True)
    ipfn_model, ipfn_edges, ipfn_J, ipfn_bw, ipfn_NF, wb_fn = load_ours_ipfn(args, args.checkpoint50)
    print('[load] UWYK (single ckpt — used for both NoAnc and FullAnc)', flush=True)
    uwyk = load_uwyk(args.uwyk_src, args.uwyk_ckpt_dir)
    print('[load] Do-PFN', flush=True)
    DoPFNRegressor = load_dopfn(args)

    # MALC uses Ours(fn=50)'s edges — set up ONCE (no more DoPFN-bb swap).
    setup_malc(ipfn_edges, ipfn_J, ipfn_bw,
                args.malc_n_eval, args.malc_B, args.malc_max_K, args.repo)
    print('[setup] MALC in-process globals set for fn=50 edges', flush=True)

    _dopfn_cwd = os.getcwd()
    rows = []

    for rho_idx, rho, rho_eff in rho_targets:
        diag_this_rho = {}
        for k in range(args.K):
            seed = rho_idx * 10_000 + k
            cd = make_polynomial_scm(seed, args.N_context, args.N_test, rho_eff)
            n_test = cd.X_test.shape[0]
            t0 = time.time()

            # ─── forward passes ───
            os.chdir(args.dopfn)
            dopfn_p0, dopfn_p1, dopfn_centers = dopfn_forward_raw_probs(DoPFNRegressor, cd)
            os.chdir(_dopfn_cwd)
            uwyk_noanc_p0, uwyk_noanc_p1, uwyk_centers_scaled, _ = \
                uwyk_forward_raw_probs(uwyk, cd, adjacency_kind='noanc')
            uwyk_anc_p0,   uwyk_anc_p1,   _,                   _ = \
                uwyk_forward_raw_probs(uwyk, cd, adjacency_kind='anc')
            uwyk_centers = uwyk_inverse_scale_centers(uwyk_centers_scaled, cd)

            p_mat_50, centers_50 = ours_forward(ipfn_model, ipfn_edges, ipfn_J, ipfn_bw, ipfn_NF, cd)

            # ─── derived densities per method ───
            # Ours(fn=50) — marginals via 1D MALC on raw p_mat marginal
            # (density_calc.md §3b). CATE via 2D MALC + diagonal integration.
            _bw_50 = float(centers_50[1] - centers_50[0])
            p0_marg_raw_50 = p_mat_50.sum(axis=2)     # (n_test, J) raw marginal probs
            p1_marg_raw_50 = p_mat_50.sum(axis=1)
            p0_50 = np.stack([malc_1d_cvxpy(p0_marg_raw_50[q]) for q in range(n_test)])
            p1_50 = np.stack([malc_1d_cvxpy(p1_marg_raw_50[q]) for q in range(n_test)])

            # Ours(fn=50) CATE — 2D MALC on p_mat, diagonal-integrate to 1D p(τ).
            tau_raw_50_malc, y_rng_50 = scaled_tau_to_raw_centers(cd)
            p_tau_50 = np.zeros((n_test, tau_raw_50_malc.shape[0]))
            for q in range(n_test):
                p_scaled = malc_cate_density_scaled(p_mat_50[q], seed=seed * 1000 + q)
                p_tau_50[q] = p_scaled * (2.0 / y_rng_50)
            cate_50 = np.array([_cate_point_from_density(p_tau_50[q], tau_raw_50_malc)
                                   for q in range(n_test)])

            # UWYK-NoAnc (independence-outer for CATE)
            tau_grid_uwyk = np.linspace(uwyk_centers[0]-uwyk_centers[-1],
                                          uwyk_centers[-1]-uwyk_centers[0], 401)
            p_tau_uwyk_noanc = np.stack([_marginals_to_ptau(uwyk_noanc_p0[q], uwyk_noanc_p1[q],
                                                                 uwyk_centers, tau_grid_uwyk)
                                            for q in range(n_test)])
            cate_uwyk_noanc = np.array([_cate_point_from_density(p_tau_uwyk_noanc[q], tau_grid_uwyk)
                                           for q in range(n_test)])

            # UWYK-FullAnc (independence-outer for CATE)
            p_tau_uwyk_anc = np.stack([_marginals_to_ptau(uwyk_anc_p0[q], uwyk_anc_p1[q],
                                                              uwyk_centers, tau_grid_uwyk)
                                          for q in range(n_test)])
            cate_uwyk_anc = np.array([_cate_point_from_density(p_tau_uwyk_anc[q], tau_grid_uwyk)
                                         for q in range(n_test)])

            # Do-PFN (independence-outer for CATE)
            tau_grid_dopfn = np.linspace(dopfn_centers[0]-dopfn_centers[-1],
                                            dopfn_centers[-1]-dopfn_centers[0], 401)
            p_tau_dopfn = np.stack([_marginals_to_ptau(dopfn_p0[q], dopfn_p1[q],
                                                            dopfn_centers, tau_grid_dopfn)
                                        for q in range(n_test)])
            cate_dopfn = np.array([_cate_point_from_density(p_tau_dopfn[q], tau_grid_dopfn)
                                       for q in range(n_test)])

            method_data = {
                'uwyk_noanc': dict(p_y0_all=uwyk_noanc_p0, p_y1_all=uwyk_noanc_p1,
                                    centers_y_raw=uwyk_centers,
                                    p_tau_all=p_tau_uwyk_noanc, tau_centers_raw=tau_grid_uwyk,
                                    cate_pred=cate_uwyk_noanc),
                'uwyk_anc':   dict(p_y0_all=uwyk_anc_p0,   p_y1_all=uwyk_anc_p1,
                                    centers_y_raw=uwyk_centers,
                                    p_tau_all=p_tau_uwyk_anc,   tau_centers_raw=tau_grid_uwyk,
                                    cate_pred=cate_uwyk_anc),
                'dopfn':      dict(p_y0_all=dopfn_p0, p_y1_all=dopfn_p1,
                                    centers_y_raw=dopfn_centers,
                                    p_tau_all=p_tau_dopfn, tau_centers_raw=tau_grid_dopfn,
                                    cate_pred=cate_dopfn),
                'ours_fn50':  dict(p_y0_all=p0_50, p_y1_all=p1_50,
                                    centers_y_raw=centers_50,
                                    p_tau_all=p_tau_50, tau_centers_raw=tau_raw_50_malc,
                                    cate_pred=cate_50),
            }
            metrics, diag = score_one_scm(cd, method_data, wb_fn)
            metrics['rho'] = rho; metrics['seed'] = seed
            metrics['n_context'] = args.N_context; metrics['n_test'] = args.N_test
            rows.append(metrics)
            if k == 0:
                diag_this_rho = {**diag, 'rho': rho}

            dt = time.time() - t0
            print(f'[scm] ρ={rho:.2f} k={k:2d}   '
                  f'PEHE (uwyk_noanc={metrics["pehe_uwyk_noanc"]:.3f}  '
                  f'uwyk_anc={metrics["pehe_uwyk_anc"]:.3f}  '
                  f'dopfn={metrics["pehe_dopfn"]:.3f}  '
                  f'ours_fn50={metrics["pehe_ours_fn50"]:.3f})  '
                  f'{dt:.1f}s', flush=True)
            gc.collect()

        if diag_this_rho:
            diag_path = f'{args.out}.rho{rho_idx}.diag.npz'
            np.savez(diag_path,
                      **{k: np.asarray(v) for k, v in diag_this_rho.items()})
            print(f'[save] {diag_path}')

    if not rows:
        print('[skip] nothing to save'); return

    shard_path = (f'{args.out}.rho{args.rho_index}.npz'
                   if args.rho_index >= 0 else args.out + '.npz')
    keys = sorted({k for r in rows for k in r.keys()})
    arr = {k: np.array([r.get(k, np.nan) for r in rows]) for k in keys}
    np.savez(shard_path, **arr)
    print(f'[save] {shard_path}  ({len(rows)} rows)')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
