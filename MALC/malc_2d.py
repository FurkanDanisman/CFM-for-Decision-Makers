"""MALC_2D: Mixture of log-concave densities for 2D binned data.

Pure-Python port of MALC_2D_Algorithm.R. Same algorithm:

  Per-component fit (4 steps):
    1. Marginal EM mean correction per dimension
    2. Beta jitter parameters calibrated from EM means
    3. Sample B bins from the joint p_mat, add within-bin Beta jitter
    4. Fit 2D log-concave MLE on the B synthetic points (mlelcd_2d)

  Mixture EM:
    - Init: K modal peaks of p_mat, Voronoi assignment
    - E-step: integrate each f_k over bins, form responsibilities gamma_jk
    - M-step: refit each f_k on p_mat * gamma_k; update pi_k
    - Convergence: patience-based (5 non-improving iterations)
    - K=1 special case: single fit, no EM

  Two-stage K selection:
    Stage 1: BIC scan K=1..max_K at B_select=100
    Stage 2: refit at K* with B_fit=300
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import beta as beta_dist, norm

from log_concave_2d_fast import LogConcaveDensity2D, dlcd_2d_fast as dlcd_2d, mlelcd_2d_fast as mlelcd_2d


# ---- EM mean correction for one marginal -------------------------------------


def _em_mean_2d(
    props: np.ndarray,
    grid: np.ndarray,
    sigma: float,
    start: float,
    max_step: int = 1000,
    eps2: float = 1e-10,
    eps1: float = 1e-5,
) -> float:
    pn = props / props.sum()
    mu = start
    for _ in range(max_step):
        a = (grid - mu) / sigma
        G1 = norm.cdf(a)
        G2 = norm.pdf(a)
        temp = (np.diff(G2) + eps2) / (np.diff(G1) + eps2)
        mu_new = mu - sigma * float(np.sum(pn * temp))
        if abs(mu_new - mu) < eps1:
            return mu_new
        mu = mu_new
    return mu


# ---- Fit one MALC_2D component ----------------------------------------------


@dataclass
class ComponentFit2D:
    fhatn: LogConcaveDensity2D
    mu_hat: np.ndarray  # (2,)
    alpha: float
    beta: np.ndarray  # (2,)
    grid_x: np.ndarray
    grid_y: np.ndarray
    p_mat: np.ndarray  # the (weighted) p_mat used for this component
    xstar: np.ndarray  # (B, 2) synthetic points


def _fit_component_2d(
    p_mat: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    B: int,
    alpha: float,
    rng: np.random.Generator,
    tol_gap: float = 1e-8,
) -> ComponentFit2D | None:
    p_mat = np.maximum(p_mat, 0.0)
    s = p_mat.sum()
    if s < 1e-10 or np.sum(p_mat > 1e-10) < 2:
        return None
    p_mat = p_mat / s

    n_y, n_x = p_mat.shape
    delta_x = grid_x[1] - grid_x[0]
    delta_y = grid_y[1] - grid_y[0]

    p_x = p_mat.sum(axis=0)
    p_y = p_mat.sum(axis=1)

    grid_left_x = grid_x[:-1]
    grid_left_y = grid_y[:-1]
    centers_x = 0.5 * (grid_left_x + grid_x[1:])
    centers_y = 0.5 * (grid_left_y + grid_y[1:])

    mu_low_x = float(np.sum(p_x * grid_left_x))
    mu_low_y = float(np.sum(p_y * grid_left_y))
    mu_mid_x = 0.5 * (mu_low_x + float(np.sum(p_x * grid_x[1:])))
    mu_mid_y = 0.5 * (mu_low_y + float(np.sum(p_y * grid_y[1:])))

    sigma_x = float(np.sqrt(np.sum(p_x * (centers_x - mu_mid_x) ** 2) + delta_x ** 2 / 12.0))
    sigma_y = float(np.sqrt(np.sum(p_y * (centers_y - mu_mid_y) ** 2) + delta_y ** 2 / 12.0))
    if not np.isfinite(sigma_x) or sigma_x <= 0:
        sigma_x = delta_x
    if not np.isfinite(sigma_y) or sigma_y <= 0:
        sigma_y = delta_y

    mu_n_x = _em_mean_2d(p_x, grid_x, sigma=sigma_x, start=mu_mid_x)
    mu_n_y = _em_mean_2d(p_y, grid_y, sigma=sigma_y, start=mu_mid_y)

    beta_x = 2.0 * alpha * ((mu_n_x - mu_low_x) / delta_x - 0.5)
    beta_y = 2.0 * alpha * ((mu_n_y - mu_low_y) / delta_y - 0.5)

    if not np.isfinite(beta_x) or min(alpha + beta_x, alpha - beta_x) <= 0:
        return None
    if not np.isfinite(beta_y) or min(alpha + beta_y, alpha - beta_y) <= 0:
        return None

    # Sample B bins from joint p_mat. R uses column-major flattening on a (n_y, n_x)
    # matrix; in numpy with a row-major (n_y, n_x) array, the same indexing yields:
    #   bin_idx in 1..n_y*n_x;  j = (bin_idx-1) %/% n_y + 1;  i = (bin_idx-1) %% n_y + 1
    # We reproduce that exact mapping so the synthetic points match the R run.
    prob_vec = p_mat.flatten(order="F")  # column-major like R as.vector
    bin_idx = rng.choice(len(prob_vec), size=B, p=prob_vec, replace=True)
    bi_j = bin_idx // n_y  # 0-based column index
    bi_i = bin_idx % n_y  # 0-based row index

    zstar_x = delta_x * rng.beta(alpha + beta_x, alpha - beta_x, size=B)
    zstar_y = delta_y * rng.beta(alpha + beta_y, alpha - beta_y, size=B)
    xstar = np.column_stack([grid_left_x[bi_j] + zstar_x, grid_left_y[bi_i] + zstar_y])

    try:
        fhatn = mlelcd_2d(
            xstar,
            jitter=1e-10,
            seed=int(rng.integers(2**31 - 1)),
            tol_gap=tol_gap,
            tol_feas=tol_gap,
        )
    except Exception:
        return None

    return ComponentFit2D(
        fhatn=fhatn,
        mu_hat=np.array([mu_n_x, mu_n_y]),
        alpha=alpha,
        beta=np.array([beta_x, beta_y]),
        grid_x=grid_x,
        grid_y=grid_y,
        p_mat=p_mat,
        xstar=xstar,
    )


# ---- Integrate component density over bins (E-step) -------------------------


def _bin_probs_2d(fit_k: ComponentFit2D, grid_x: np.ndarray, grid_y: np.ndarray, n_eval: int = 24) -> np.ndarray:
    x_seq = np.linspace(grid_x.min(), grid_x.max(), n_eval)
    y_seq = np.linspace(grid_y.min(), grid_y.max(), n_eval)
    XX, YY = np.meshgrid(x_seq, y_seq, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel()])

    dens = dlcd_2d(pts, fit_k.fhatn)
    dens = np.where(np.isnan(dens), 0.0, dens)
    dens = np.maximum(dens, 0.0)

    dx = x_seq[1] - x_seq[0]
    dy = y_seq[1] - y_seq[0]
    n_x = len(grid_x) - 1
    n_y = len(grid_y) - 1

    # Assign each eval point to a bin
    bx = np.clip(np.searchsorted(grid_x, pts[:, 0], side="right") - 1, 0, n_x - 1)
    by = np.clip(np.searchsorted(grid_y, pts[:, 1], side="right") - 1, 0, n_y - 1)

    probs = np.zeros((n_y, n_x))
    contrib = dens * dx * dy
    np.add.at(probs, (by, bx), contrib)

    s = probs.sum()
    if s > 0:
        return probs / s
    return np.full((n_y, n_x), 1.0 / (n_x * n_y))


# ---- Init: modal Voronoi partition -----------------------------------------


def _init_assignments_2d(p_mat: np.ndarray, K: int) -> np.ndarray:
    n_y, n_x = p_mat.shape
    pmax = float(p_mat.max())
    rtol = 1e-12
    floor = 1e-10 * pmax
    is_peak = np.zeros_like(p_mat, dtype=bool)
    for i in range(n_y):
        for j in range(n_x):
            nb = p_mat[max(0, i - 1):min(n_y, i + 2), max(0, j - 1):min(n_x, j + 2)]
            nb_max = float(nb.max())
            tol = rtol * nb_max
            if p_mat[i, j] + tol >= nb_max and p_mat[i, j] > floor:
                is_peak[i, j] = True

    candidates = np.argwhere(is_peak)
    if len(candidates) == 0:
        candidates = np.argwhere(p_mat == p_mat.max())[:1]

    # Non-maximum suppression: merge candidates within 8-connected adjacency
    # into a single representative (the highest-value cell, ties broken by scan
    # order). This collapses plateau peaks introduced by tolerance ties.
    vals = p_mat[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(-vals, kind="stable")
    kept: list[int] = []
    nms_radius = 1
    for idx in order:
        i, j = candidates[idx]
        ok = True
        for k in kept:
            ki, kj = candidates[k]
            if abs(i - ki) <= nms_radius and abs(j - kj) <= nms_radius:
                ok = False
                break
        if ok:
            kept.append(int(idx))
    peak_locs = candidates[kept]

    if len(peak_locs) >= K:
        vals = p_mat[peak_locs[:, 0], peak_locs[:, 1]]
        order = np.argsort(-vals)
        peak_locs = peak_locs[order[:K]]
    else:
        # K-means++-style deterministic augmentation: add seeds at the
        # high-density cell farthest from existing seeds.
        II, JJ = np.meshgrid(np.arange(n_y), np.arange(n_x), indexing="ij")
        coords = np.stack([II, JJ], axis=-1).astype(float)
        seeds = list(peak_locs)
        while len(seeds) < K:
            min_d2 = np.full((n_y, n_x), np.inf)
            for s in seeds:
                d2 = (coords[..., 0] - s[0]) ** 2 + (coords[..., 1] - s[1]) ** 2
                min_d2 = np.minimum(min_d2, d2)
            score = p_mat * min_d2
            idx = np.unravel_index(int(np.argmax(score)), p_mat.shape)
            seeds.append(np.array(idx))
        peak_locs = np.array(seeds)

    asgn = np.zeros((n_y, n_x), dtype=int)
    II, JJ = np.meshgrid(np.arange(n_y), np.arange(n_x), indexing="ij")
    for i in range(n_y):
        for j in range(n_x):
            d = (peak_locs[:, 0] - i) ** 2 + (peak_locs[:, 1] - j) ** 2
            asgn[i, j] = int(np.argmin(d))
    return asgn


# ---- Fit object -------------------------------------------------------------


@dataclass
class MALC2DFit:
    fits: list  # list[ComponentFit2D | None]
    pi: np.ndarray  # (K,)
    K: int
    loglik: float
    grid_x: np.ndarray
    grid_y: np.ndarray
    p_mat: np.ndarray
    bic_table: np.ndarray | None = None  # (max_K, 2)  [K, BIC]


# ---- EM fit at fixed K and B ------------------------------------------------


def MALC_2D_fit(
    p_mat: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    K: int,
    B: int = 100,
    alpha: float = 2.0,
    max_iter: int = 30,
    tol: float = 1e-4,
    seed: int = 20180621,
    verbose: bool = False,
    tol_gap: float = 1e-8,
    bin_n_eval: int = 24,
) -> MALC2DFit:
    rng = np.random.default_rng(seed)

    p_mat = np.maximum(p_mat, 0.0)
    p_mat = p_mat / p_mat.sum()
    n_y, n_x = p_mat.shape

    if K == 1:
        fit = _fit_component_2d(p_mat, grid_x, grid_y, B=B, alpha=alpha, rng=rng, tol_gap=tol_gap)
        if fit is None:
            raise RuntimeError("MALC_2D_fit: K=1 component fit failed")
        pb = _bin_probs_2d(fit, grid_x, grid_y, n_eval=bin_n_eval)
        loglik = float(np.sum(p_mat * np.log(np.maximum(pb, 1e-300))))
        return MALC2DFit(
            fits=[fit],
            pi=np.array([1.0]),
            K=1,
            loglik=loglik,
            grid_x=grid_x,
            grid_y=grid_y,
            p_mat=p_mat,
        )

    asgn = _init_assignments_2d(p_mat, K)
    pi_k = np.array([np.sum(asgn == k) for k in range(K)], dtype=float) / (n_y * n_x)
    pi_k = pi_k / pi_k.sum()

    fits: list = []
    for k in range(K):
        mask = (asgn == k).astype(float)
        fits.append(_fit_component_2d(p_mat * mask, grid_x, grid_y, B=B, alpha=alpha, rng=rng, tol_gap=tol_gap))

    best_loglik = -np.inf
    best_fits = list(fits)
    best_pi = pi_k.copy()
    no_improve = 0

    for it in range(max_iter):
        p_arr = np.zeros((n_y, n_x, K))
        for k in range(K):
            if fits[k] is not None:
                p_arr[:, :, k] = _bin_probs_2d(fits[k], grid_x, grid_y, n_eval=bin_n_eval)

        mix_p = np.zeros((n_y, n_x))
        for k in range(K):
            mix_p += pi_k[k] * p_arr[:, :, k]

        gam = np.zeros((n_y, n_x, K))
        mix_safe = np.maximum(mix_p, 1e-300)
        for k in range(K):
            gam[:, :, k] = pi_k[k] * p_arr[:, :, k] / mix_safe

        p_k_list = [p_mat * gam[:, :, k] for k in range(K)]
        loglik = float(np.sum(p_mat * np.log(mix_safe)))
        if verbose:
            print(f"  iter {it + 1}  loglik = {loglik:.4f}")

        if loglik > best_loglik + tol:
            best_loglik = loglik
            best_fits = list(fits)
            best_pi = pi_k.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 5:
                break

        pi_k = np.maximum(np.array([pk.sum() for pk in p_k_list]), 1e-6)
        pi_k = pi_k / pi_k.sum()
        for k in range(K):
            nf = _fit_component_2d(p_k_list[k], grid_x, grid_y, B=B, alpha=alpha, rng=rng, tol_gap=tol_gap)
            if nf is not None:
                fits[k] = nf

    return MALC2DFit(
        fits=best_fits,
        pi=best_pi,
        K=K,
        loglik=best_loglik,
        grid_x=grid_x,
        grid_y=grid_y,
        p_mat=p_mat,
    )


# ---- BIC --------------------------------------------------------------------


def MALC_2D_bic(obj: MALC2DFit) -> float:
    n_eff = 1.0 / float(np.sum(obj.p_mat ** 2))
    num_params = 3 * obj.K - 1
    return -2.0 * obj.loglik * n_eff + num_params * np.log(n_eff)


# ---- Main entry point -------------------------------------------------------


def _bic_worker(args):
    """Stage 1 worker for ProcessPoolExecutor. Returns (K, BIC)."""
    # Limit BLAS threads per worker to avoid oversubscription.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    p_mat, grid_x, grid_y, K, B, alpha, max_iter, tol, seed, tol_gap, bin_n_eval = args
    try:
        fit = MALC_2D_fit(
            p_mat, grid_x, grid_y, K=K, B=B, alpha=alpha,
            max_iter=max_iter, tol=tol, seed=seed, verbose=False,
            tol_gap=tol_gap, bin_n_eval=bin_n_eval,
        )
        return K, float(MALC_2D_bic(fit))
    except Exception:
        return K, float("inf")


def MALC_2D(
    p_mat: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    K: int | None = None,
    max_K: int = 5,
    B_select: int = 500,
    B_fit: int = 500,
    alpha: float = 2.0,
    max_iter: int = 30,
    tol: float = 1e-4,
    seed: int = 20180621,
    verbose: bool = False,
    tol_gap_select: float = 1e-5,
    tol_gap_fit: float = 1e-8,
    parallel: bool = True,
    n_workers: int | None = None,
    bin_n_eval: int = 24,
) -> MALC2DFit:
    if K is not None:
        if verbose:
            print(f"K fixed at {K}, fitting with B={B_fit}")
        return MALC_2D_fit(
            p_mat, grid_x, grid_y, K=K, B=B_fit, alpha=alpha,
            max_iter=max_iter, tol=tol, seed=seed, verbose=verbose,
            tol_gap=tol_gap_fit, bin_n_eval=bin_n_eval,
        )

    if verbose:
        print(f"Stage 1: BIC scan K=1..{max_K} at B={B_select} (parallel={parallel}, tol={tol_gap_select})")
    bics = np.full(max_K, np.inf)

    worker_args = [
        (p_mat, grid_x, grid_y, k, B_select, alpha, max_iter, tol, seed, tol_gap_select, bin_n_eval)
        for k in range(1, max_K + 1)
    ]

    if parallel and max_K > 1:
        nw = n_workers if n_workers is not None else min(max_K, mp.cpu_count())
        # 'spawn' is the default on macOS/Windows; safe for numpy/scipy state.
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=nw, mp_context=ctx) as ex:
            for K_done, bic in ex.map(_bic_worker, worker_args):
                bics[K_done - 1] = bic
                if verbose:
                    print(f"  K={K_done}  BIC={bic:.2f}")
    else:
        for args in worker_args:
            K_done, bic = _bic_worker(args)
            bics[K_done - 1] = bic
            if verbose:
                print(f"  K={K_done}  BIC={bic:.2f}")

    K_star = int(np.argmin(bics)) + 1
    if verbose:
        print(f"Stage 1 selected K={K_star}")
        print(f"Stage 2: refit K={K_star} at B={B_fit} (tol={tol_gap_fit})")

    obj = MALC_2D_fit(
        p_mat, grid_x, grid_y, K=K_star, B=B_fit, alpha=alpha,
        max_iter=max_iter, tol=tol, seed=seed, verbose=verbose,
        tol_gap=tol_gap_fit, bin_n_eval=bin_n_eval,
    )
    obj.bic_table = np.column_stack([np.arange(1, max_K + 1), bics])
    return obj


# ---- Evaluate mixture density ----------------------------------------------


def dmalc_2d(obj: MALC2DFit, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    total = np.zeros(len(pts))
    for k in range(obj.K):
        if obj.fits[k] is not None:
            total += obj.pi[k] * np.maximum(dlcd_2d(pts, obj.fits[k].fhatn), 0.0)
    return total


def eval_grid_2d(obj: MALC2DFit, n_eval: int = 60):
    xs = np.linspace(obj.grid_x.min(), obj.grid_x.max(), n_eval)
    ys = np.linspace(obj.grid_y.min(), obj.grid_y.max(), n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    dens = dmalc_2d(obj, pts)
    return xs, ys, dens.reshape(n_eval, n_eval)


# ---- Plot ------------------------------------------------------------------


def plot_malc_2d(
    obj: MALC2DFit,
    true_pdf=None,
    n_eval: int = 60,
    col_est: str = "navy",
    col_true: str = "firebrick",
    main: str = "",
    show: bool = True,
    savepath: str | None = None,
):
    import matplotlib.pyplot as plt

    xs, ys, Z = eval_grid_2d(obj, n_eval=n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    if true_pdf is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ax = axes[0]
    cs = ax.contour(XX, YY, Z, colors=col_est, linewidths=2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{main} - estimated" if main else "MALC-2D")
    comp_colors = ["#E41A1C", "#4DAF4A", "#FF7F00", "#984EA3", "#A65628"]
    for k in range(obj.K):
        if obj.fits[k] is not None:
            ax.scatter(
                obj.fits[k].xstar[:, 0],
                obj.fits[k].xstar[:, 1],
                s=4,
                c=comp_colors[k % len(comp_colors)],
                alpha=0.4,
            )

    if true_pdf is not None:
        pts_t = np.column_stack([XX.ravel(), YY.ravel()])
        Zt = true_pdf(pts_t).reshape(n_eval, n_eval)
        ax2 = axes[1]
        ax2.contour(XX, YY, Zt, colors=col_true, linewidths=2)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title(f"{main} - true")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120)
    if show:
        plt.show()
    return fig
