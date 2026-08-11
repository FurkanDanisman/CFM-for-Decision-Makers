"""Per-method density adapters for the IHDP L2 evaluation.

Every adapter returns, for a single realization, three arrays shaped
(n_queries, len(grid)):

    p_y0[i, :]  density of Y_do(0) | X_i  on true_ihdp.Y_CENTERS
    p_y1[i, :]  density of Y_do(1) | X_i  on true_ihdp.Y_CENTERS
    p_tau[i, :] density of tau     | X_i  on true_ihdp.TAU_CENTERS

Adapters implemented:
    ours_densities(cd, model, ckpt_meta, args)  — joint head + MALC + diag integration
    uwyk_noanc_densities(cd, uwyk_model, args)  — sample -> histogram + independence convolution
    dopfn_densities(cd, DoPFNRegressor, args)   — predict_full logits/borders + resample + independence conv

All adapters use the SAME rescaling (y_min, y_rng from training) as
plot_ihdp_n10*.py so their outputs share a coordinate frame with
true_ihdp.load_ihdp_truth().

Independence convolution for baselines: given per-arm marginals p0, p1 on the
common Y_CENTERS grid,
    p_tau(t) = sum_{y0} p0(y0) * p1(y0 + t) * dy   ~= diagonal sums of outer.
"""
from __future__ import annotations

import hashlib
import os
import sys
from typing import Any, Callable

import numpy as np
import torch

from l2 import resample_onto
from true_ihdp import TAU_BIN, TAU_CENTERS, Y_BIN, Y_CENTERS


# ── Independence-convolution helper (shared by UWYK-NoAnc and DoPFN) ─────
def naive_p_tau_from_marginals(p_y0: np.ndarray, p_y1: np.ndarray) -> np.ndarray:
    """CATE density under independence: p(tau) = ∫ p1(y0 + tau) p0(y0) dy0.

    Inputs on Y_CENTERS (uniform). Output on TAU_CENTERS.

    Uses np.correlate (equivalent to np.outer + diagonal sums, but with the
    correct sign convention: for outer[i, j] = p1[i] * p0[j] and
    trace(offset=off) = sum_k p1[k] * p0[k + off], each summand corresponds
    to tau = Y_CENTERS[k] - Y_CENTERS[k+off] = -off * Y_BIN. So the trace
    at offset off contributes to tau = -off * dY, not +off * dY. The
    sibling script plot_ihdp_n10_uwyk_noanc.py::_naive_p_tau uses the
    wrong sign convention (tau = +off * dY) and produces a mirrored p(tau).
    """
    p_y0 = np.asarray(p_y0, dtype=np.float64).reshape(-1)
    p_y1 = np.asarray(p_y1, dtype=np.float64).reshape(-1)
    n = p_y0.shape[0]
    # correlate(p1, p0)[k] = sum_i p1[i + (k - (n-1))] * p0[i]
    #   lag = k - (n-1)  <->  tau = lag * Y_BIN
    density = np.correlate(p_y1, p_y0, mode='full')            # length 2n-1
    tau_native = np.arange(-(n - 1), n) * Y_BIN
    out = np.interp(TAU_CENTERS, tau_native, density, left=0.0, right=0.0)
    s = out.sum() * TAU_BIN
    if s > 0:
        out = out / s
    return out


def _rescale_and_pad(X: np.ndarray, num_features: int) -> np.ndarray:
    """Pad X with NaN columns up to num_features (matches sibling scripts).

    If num_features is -1 (or 0) the backbone accepts any feature count
    (e.g., DoPFN's PerFeatureTransformer): return X unchanged.
    """
    if num_features is None or num_features <= 0:
        return X.astype(np.float32)
    d = X.shape[1]
    if d < num_features:
        pad = np.full((X.shape[0], num_features - d), np.nan, dtype=np.float32)
        return np.concatenate([X.astype(np.float32), pad], axis=1)
    return X.astype(np.float32)[:, :num_features]


# ─────────────────────────────────────────────────────────────────────────
# Ours (fn=50 or fn=10) — joint head + 2D MALC + diagonal integration
# ─────────────────────────────────────────────────────────────────────────
def ours_densities(cd,
                    model,
                    edges_np: np.ndarray,
                    J: int,
                    bin_width: float,
                    num_features: int,
                    y_min: float,
                    y_rng: float,
                    malc_B: int = 60,
                    malc_max_K: int = 3,
                    n_eval: int = 200,
                    n_context: int | None = None,
                    fit_malc_inner: Callable[..., Any] = None,
                    dmalc_2d: Callable[..., Any] = None,
                    ) -> dict[str, np.ndarray]:
    """Joint-head forward pass, per-query MALC fit, marginalise + diagonal
    integrate onto the common (Y_CENTERS, TAU_CENTERS) grids.

    fit_malc_inner / dmalc_2d are passed in to avoid a hard import at
    package-import time (the caller supplies them once).
    """
    X_train_full = _np(cd.X_train)
    t_train_full = _np(cd.t_train)
    y_train_full = _np(cd.y_train)
    X_test = _np(cd.X_test)

    N = X_train_full.shape[0] if n_context is None else min(n_context, X_train_full.shape[0])
    X_context = X_train_full[:N].astype(np.float32)
    T_context = t_train_full[:N].astype(np.float32).reshape(-1, 1)
    Y_context = y_train_full[:N].astype(np.float32).reshape(-1, 1)
    Y_context = ((Y_context - y_min) / y_rng * 2.0 - 1.0).astype(np.float32)

    X_context = _rescale_and_pad(X_context, num_features)
    X_test_p = _rescale_and_pad(X_test.astype(np.float32), num_features)

    Xc = torch.from_numpy(X_context).unsqueeze(0)
    Tc = torch.from_numpy(T_context).unsqueeze(0)
    Yc = torch.from_numpy(Y_context).unsqueeze(0)
    Xq = torch.from_numpy(X_test_p).unsqueeze(0)

    from losses.BarDistribution2D import unpack_pred  # cluster-side import
    with torch.no_grad():
        pred = model(Xc, Tc, Yc, Xq)['predictions'][0]           # (n_test, D)

    n_test = X_test_p.shape[0]
    p_mats = np.zeros((n_test, J, J), dtype=np.float32)
    for q in range(n_test):
        p_mat, *_ = unpack_pred(pred[q], J, bin_width)
        p_mats[q] = p_mat.detach().cpu().numpy()

    # Evaluate MALC once per query on a fine (n_eval, n_eval) grid then
    # (a) marginalise -> p_y0, p_y1 on Y_CENTERS and
    # (b) diagonal-integrate -> p_tau on TAU_CENTERS.
    xs = np.linspace(edges_np[0], edges_np[-1], n_eval)
    ys = np.linspace(edges_np[0], edges_np[-1], n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    eval_pts = np.column_stack([XX.ravel(), YY.ravel()])
    dxs = float(xs[1] - xs[0]); dys = float(ys[1] - ys[0])

    p_y0 = np.zeros((n_test, len(Y_CENTERS)), dtype=np.float64)
    p_y1 = np.zeros((n_test, len(Y_CENTERS)), dtype=np.float64)
    p_tau = np.zeros((n_test, len(TAU_CENTERS)), dtype=np.float64)

    for q in range(n_test):
        seed = int(hashlib.md5(f'q{q}'.encode()).hexdigest()[:8], 16) % (10 ** 8)
        # MALC's log-concave fitter can raise RuntimeError on certain p_mat
        # shapes (e.g., mass too concentrated to fit a log-concave). Try
        # requested max_K, then a small ladder of retries, then fall back to
        # the raw piecewise-constant p_mat so evaluation of a single query
        # never kills the whole realization.
        density_2d = None
        _tried = []
        for _K in {malc_max_K, max(malc_max_K + 1, 2), 3, 1}:
            try:
                fit = fit_malc_inner(p_mats[q].T, edges_np, edges_np,
                                     B_fit=malc_B, B_select=malc_B,
                                     max_K=_K, seed=seed, parallel=False)
                density_2d = dmalc_2d(fit, eval_pts).reshape(n_eval, n_eval)
                break
            except Exception as e:
                _tried.append(f'K={_K}:{type(e).__name__}')
        if density_2d is None:
            # Fallback: piecewise-constant density from the raw p_mat.
            # Nearest-neighbour lookup onto the fine (xs, ys) grid.
            print(f'[warn] q={q} MALC failed for all K ({" ".join(_tried)}); '
                  f'using raw p_mat fallback', flush=True)
            p_norm = p_mats[q] / max(p_mats[q].sum(), 1e-12)
            j0 = np.clip(np.searchsorted(edges_np, xs) - 1, 0, J - 1)
            j1 = np.clip(np.searchsorted(edges_np, ys) - 1, 0, J - 1)
            # density value = probability_mass / bin_area
            density_2d = (p_norm[np.ix_(j0, j1)] / (bin_width * bin_width)).T

        # Marginals over the fine grid
        m_y0_fine = density_2d.sum(axis=0) * dys                          # (n_eval,)
        m_y1_fine = density_2d.sum(axis=1) * dxs                          # (n_eval,)
        p_y0[q] = resample_onto(xs, m_y0_fine, Y_CENTERS)
        p_y1[q] = resample_onto(ys, m_y1_fine, Y_CENTERS)

        # CATE via diagonal integration.
        # For each tau, integrate p(y0, y0+tau) over y0. y0 runs over xs
        # (the fine grid); we look up the density_2d value at that y0
        # column, and bilinearly interpolate in y1 = y0 + tau across rows.
        #
        # NOTE: earlier revisions (and the sibling plot_ihdp_n10.py) used
        #     col = np.clip(np.searchsorted(xs, xs[valid]) - 1, 0, len(xs)-1)
        # which is off-by-one for exact matches: searchsorted(side='left')
        # returns k for xs[valid][i] == xs[k], so -1 gives k-1 (wrong).
        # For smooth densities this shifts sampling by dxs ~ 1.5%; for
        # spiky joint outputs (fn=10 on hard queries) it can miss the peak
        # column entirely. Corrected form: use searchsorted output directly.
        p_tau_native = np.zeros(len(TAU_CENTERS))
        for k, t in enumerate(TAU_CENTERS):
            y1_target = xs + t
            valid = (y1_target >= ys[0]) & (y1_target <= ys[-1])
            if not np.any(valid):
                continue
            col = np.clip(np.searchsorted(xs, xs[valid]), 0, len(xs) - 1)
            row_f = (y1_target[valid] - ys[0]) / dys
            row_lo = np.clip(np.floor(row_f).astype(int), 0, len(ys) - 2)
            row_hi = row_lo + 1
            w_hi = row_f - row_lo
            w_lo = 1.0 - w_hi
            f_diag = w_lo * density_2d[row_lo, col] + w_hi * density_2d[row_hi, col]
            p_tau_native[k] = f_diag.sum() * dxs
        s = p_tau_native.sum() * TAU_BIN
        if s > 0:
            p_tau_native = p_tau_native / s
        p_tau[q] = p_tau_native

    return dict(p_y0=p_y0, p_y1=p_y1, p_tau=p_tau)


# ─────────────────────────────────────────────────────────────────────────
# UWYK No-Ancestral — sample + histogram + independence convolution
# ─────────────────────────────────────────────────────────────────────────
def uwyk_noanc_densities(cd,
                          uwyk_model,
                          num_features: int,
                          y_min: float,
                          y_rng: float,
                          n_context: int | None = None,
                          n_samples: int = 1024,
                          ) -> dict[str, np.ndarray]:
    """UWYK-NoAnc marginals via histogram of samples; CATE under independence.

    Mirrors plot_ihdp_n10_uwyk_noanc.py's construction.
    """
    X_train_full = _np(cd.X_train)
    t_train_full = _np(cd.t_train)
    y_train_full = _np(cd.y_train)
    X_test = _np(cd.X_test).astype(np.float32)

    N = X_train_full.shape[0] if n_context is None else min(n_context, X_train_full.shape[0])
    X_context = X_train_full[:N].astype(np.float32)
    t_train_orig = t_train_full[:N].astype(np.float32).reshape(-1, 1)
    y_train_ctx = y_train_full[:N].astype(np.float32).reshape(-1, 1)

    X_train_p = _rescale_and_pad(X_context, num_features)
    X_test_p = _rescale_and_pad(X_test, num_features)

    mean_y_t0 = float(y_train_ctx[t_train_orig == 0].mean())
    mean_y_t1 = float(y_train_ctx[t_train_orig == 1].mean())
    t_train_enc = np.where(t_train_orig == 0, mean_y_t0, mean_y_t1).astype(np.float32)
    uwyk_model.fit(X_train_p, t_train_enc, y_train_ctx)

    # No-Ancestral adjacency: real features connected, padded features masked -1
    n_real = X_context.shape[1]
    adj = np.zeros((num_features + 2, num_features + 2), dtype=np.float32)
    for i in range(n_real, num_features):
        fi = 2 + i
        adj[fi, :] = -1.0; adj[:, fi] = -1.0; adj[fi, fi] = -1.0

    n_test = X_test_p.shape[0]
    T_intv_0 = np.full((n_test, 1), mean_y_t0, dtype=np.float32)
    T_intv_1 = np.full((n_test, 1), mean_y_t1, dtype=np.float32)

    def _sample(T_intv):
        chunks = []; got = 0
        while got < n_samples:
            r = uwyk_model.predict(
                X_obs=X_train_p, T_obs=t_train_enc, Y_obs=y_train_ctx,
                X_intv=X_test_p, T_intv=T_intv,
                adjacency_matrix=adj,
                prediction_type='sample', inverse_transform=True,
            )
            arr = np.asarray(r).reshape(n_test, -1)
            chunks.append(arr); got += arr.shape[1]
        return np.concatenate(chunks, axis=1)[:, :n_samples]

    Y0_raw = _sample(T_intv_0)
    Y1_raw = _sample(T_intv_1)

    Y0_scaled = (Y0_raw - y_min) / y_rng * 2.0 - 1.0
    Y1_scaled = (Y1_raw - y_min) / y_rng * 2.0 - 1.0

    y_edges = np.concatenate([Y_CENTERS - 0.5 * Y_BIN, [Y_CENTERS[-1] + 0.5 * Y_BIN]])
    p_y0 = np.zeros((n_test, len(Y_CENTERS)), dtype=np.float64)
    p_y1 = np.zeros((n_test, len(Y_CENTERS)), dtype=np.float64)
    p_tau = np.zeros((n_test, len(TAU_CENTERS)), dtype=np.float64)
    for q in range(n_test):
        h0, _ = np.histogram(Y0_scaled[q], bins=y_edges, density=True)
        h1, _ = np.histogram(Y1_scaled[q], bins=y_edges, density=True)
        p_y0[q] = h0
        p_y1[q] = h1
        p_tau[q] = naive_p_tau_from_marginals(h0, h1)
    return dict(p_y0=p_y0, p_y1=p_y1, p_tau=p_tau)


# ─────────────────────────────────────────────────────────────────────────
# Do-PFN — predict_full logits/borders + independence convolution
# ─────────────────────────────────────────────────────────────────────────
def dopfn_densities(cd,
                     DoPFNRegressor,
                     y_min: float,
                     y_rng: float,
                     dopfn_root: str,
                     n_context: int | None = None,
                     ) -> dict[str, np.ndarray]:
    """DoPFN per-arm density via predict_full()['logits'] over criterion.borders.

    Do-PFN convention (from benchmarks/methods/dopfn.py + upstream base.py):
    the treatment goes in column 0 of the covariate matrix. We call
    predict_full twice, once with column 0 = 0 (control) and once with
    column 0 = 1 (treatment); softmax(logits)/bar_width gives a density on
    criterion.borders, which we then resample onto Y_CENTERS.

    predict_full returns a dict with 'logits' (n_test, num_bars),
    'criterion' (a FullSupportBarDistribution with .borders), plus
    mean/median/mode/quantiles.

    DoPFNRegressor.__init__ opens 'artifacts/dopfn_config.pkl' by relative
    path, so we chdir to dopfn_root for the duration of the instantiation
    (matches the existing benchmark pipeline; see submit.sbatch:54).
    """
    X_train = _np(cd.X_train).astype(np.float32)
    t_train = _np(cd.t_train).astype(np.float32).reshape(-1)
    y_train = _np(cd.y_train).astype(np.float32).reshape(-1)
    X_test = _np(cd.X_test).astype(np.float32)

    N = X_train.shape[0] if n_context is None else min(n_context, X_train.shape[0])
    X_train_ctx = X_train[:N]
    t_train_ctx = t_train[:N]
    y_train_ctx = y_train[:N]

    # DoPFN convention (mirrors methods/dopfn.py::dopfn_pipeline). Both
    # __init__ and fit() open files with relative paths (artifacts/*.pkl,
    # artifacts/*.cpkt), so CWD must stay at dopfn_root through both calls
    # and through predict_full below (safest to keep it for the whole block).
    x_tr = np.concatenate([t_train_ctx[:, None], X_train_ctx], axis=1)
    x_te = np.concatenate([np.zeros((X_test.shape[0], 1), dtype=np.float32), X_test], axis=1)
    n_test = X_test.shape[0]

    _cwd = os.getcwd()
    try:
        os.chdir(dopfn_root)
        reg = DoPFNRegressor()
        reg.fit(torch.tensor(x_tr), torch.tensor(y_train_ctx))

        def _arm_density(arm: int) -> np.ndarray:
            x = x_te.copy()
            x[:, 0] = float(arm)
            full = reg.predict_full(torch.tensor(x))
            logits = np.asarray(full['logits'])                     # (n_test, num_bars)
            borders = np.asarray(full['criterion'].borders.cpu())    # (num_bars + 1,) in raw Y units
            bar_widths = np.diff(borders)                            # (num_bars,)
            # Softmax + convert to density on the (raw-Y) bar centres
            z = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(z); probs /= probs.sum(axis=1, keepdims=True)
            density_raw = probs / bar_widths[None, :]                # (n_test, num_bars)
            bar_centres_raw = 0.5 * (borders[:-1] + borders[1:])
            # Rescale border centres to [-1, 1] Y frame
            bar_centres_scaled = (bar_centres_raw - y_min) / y_rng * 2.0 - 1.0
            # Density transforms as (dy_raw / dy_scaled) = y_rng / 2
            density_scaled = density_raw * (y_rng / 2.0)
            out = np.zeros((n_test, len(Y_CENTERS)), dtype=np.float64)
            for q in range(n_test):
                out[q] = resample_onto(bar_centres_scaled, density_scaled[q], Y_CENTERS)
            return out

        p_y0 = _arm_density(0)
        p_y1 = _arm_density(1)
    finally:
        os.chdir(_cwd)

    p_tau = np.zeros((n_test, len(TAU_CENTERS)), dtype=np.float64)
    for q in range(n_test):
        p_tau[q] = naive_p_tau_from_marginals(p_y0[q], p_y1[q])
    return dict(p_y0=p_y0, p_y1=p_y1, p_tau=p_tau)


def _np(a):
    if isinstance(a, torch.Tensor):
        return a.numpy()
    return np.asarray(a)
