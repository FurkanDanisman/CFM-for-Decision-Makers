"""Evaluate cpfn2d checkpoint on IHDP with MALC-based CATE means.

Two methods per query — same model forward, different smoothing pipeline:

  1) 1D MALC on marginals (per density_calc.md § 3b):
       - Marginalise p_mat → p_marg0, p_marg1 (raw 1D).
       - `malc_1d_cvxpy(...)` → log-concave discrete MLE per marginal.
       - E[Y_do_t] = Σ_j centre[j] · smoothed_marg[j].
       - CATE = E[Y_do1] − E[Y_do0].
       - Cost: ~2 CVXPY solves per query. Fast (~100ms/query on CPU).

  2) 2D MALC (per density_calc.md § 4a, but for the MEAN):
       - `MALC_2D(p_mat.T, edges, edges, B_fit=B, B_select=B)` → mixture of
         log-concave 2D densities.
       - Evaluate on n_eval × n_eval fine grid.
       - Marginalise the smoothed density along each axis → E[Y_do_t].
       - CATE = E[Y_do1] − E[Y_do0].
       - By linearity of expectation this equals ∫τ·p(τ)dτ from diagonal
         projection — same mean, no need for the extra diagonal step.
       - Cost: ~5–15 s per query with B=1000, max_K=5 (BIC scan).
       - Parallelised across queries via multiprocessing.

Env vars:
  CKPT             checkpoint path
  OUT              per-realization NPZ output dir
  CAUSALPFN        path to external/causalpfn
  MALC_B           2D MALC B_fit=B_select   (default 1000)
  MALC_MAX_K       2D MALC max_K            (default 5, per density_calc.md)
  N_WORKERS        multiprocessing workers  (default = SLURM_CPUS_PER_TASK)
  N_REAL           # IHDP realizations      (default 100 — full)
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch


CKPT      = os.environ['CKPT']
OUT       = os.environ.get('OUT', './results_cpfn2d_ihdp_malc')
CAUSALPFN = os.environ['CAUSALPFN']
MALC_B    = int(os.environ.get('MALC_B',    1000))
MALC_MAX_K = int(os.environ.get('MALC_MAX_K', 5))
N_EVAL    = int(os.environ.get('N_EVAL',    100))    # MALC density-eval grid size
N_WORKERS = int(os.environ.get('N_WORKERS', os.environ.get('SLURM_CPUS_PER_TASK', 8)))
# Range of realizations this job evaluates: [REAL_START, REAL_END).
# Default = [0, 100) — full IHDP. Split across many jobs for wall-clock speedup.
REAL_START = int(os.environ.get('REAL_START', 0))
REAL_END   = int(os.environ.get('REAL_END',   100))

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, CAUSALPFN)
sys.path.insert(0, CAUSALPFN + '/src')

# MALC lives at $REPO/MALC/malc_2d.py — add explicitly for the workers.
_MALC_DIR = os.path.join(REPO_SRC, 'MALC')
if _MALC_DIR not in sys.path:
    sys.path.insert(0, _MALC_DIR)

from benchmarks import IHDPDataset  # noqa: E402  — CausalPFN's benchmarks pkg
from training_causalpfn2d.model_causalpfn_2d import CausalPFN2DHead  # noqa: E402
from malc_2d import MALC_2D, eval_grid_2d  # noqa: E402

# `benchmarks` here resolves to CausalPFN's package (path order), so
# `from benchmarks.l2_ihdp.methods_densities import malc_1d_cvxpy` fails
# with ModuleNotFoundError. Load our copy directly by file path.
# Also add benchmarks/l2_ihdp/ to sys.path so methods_densities' sibling
# imports (`from l2 import resample_onto`) resolve.
import importlib.util as _iutil  # noqa: E402
_l2_ihdp_dir = os.path.join(REPO_SRC, 'benchmarks', 'l2_ihdp')
if _l2_ihdp_dir not in sys.path:
    sys.path.insert(0, _l2_ihdp_dir)
_methods_path = os.path.join(_l2_ihdp_dir, 'methods_densities.py')
_spec = _iutil.spec_from_file_location('rpfn_methods_densities', _methods_path)
_methods_densities = _iutil.module_from_spec(_spec)
_spec.loader.exec_module(_methods_densities)
malc_1d_cvxpy = _methods_densities.malc_1d_cvxpy


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_state_dict_safe(model, sd):
    if any('_orig_mod.' in k for k in sd):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    ref = model.state_dict()
    kept = {k: v for k, v in sd.items() if k in ref and ref[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(kept, strict=False)
    print(f'[eval] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}',
          flush=True)
    if len(missing) > 20:
        raise RuntimeError(f'ABORT: {len(missing)} missing keys — refusing to eval random init.')


def _pad(X, F):
    if X.shape[1] == F: return X
    if X.shape[1] > F:  return X[:, :F]
    return np.hstack([X, np.zeros((X.shape[0], F - X.shape[1]), dtype=X.dtype)])


def _std_X(Xtr, Xte, eps=1e-8):
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + eps
    return (Xtr - mu) / sd, (Xte - mu) / sd


# ── Method 1: 1D MALC on marginals ──────────────────────────────────────
def cate_1d_malc(p_mat, centers):
    """1D MALC on the two marginals of p_mat → CATE mean.

    p_mat: (J, J) with axis 0 = Y_do0, axis 1 = Y_do1 (our convention).
    centers: (J,) bin centres on the standardised scale.
    Returns scalar CATE_std.
    """
    p_marg0 = p_mat.sum(axis=1)    # Y_do0 marginal (Y_do1 summed out)
    p_marg1 = p_mat.sum(axis=0)    # Y_do1 marginal (Y_do0 summed out)
    # Renormalise (numerical safety) before MALC.
    p_marg0 = p_marg0 / max(p_marg0.sum(), 1e-45)
    p_marg1 = p_marg1 / max(p_marg1.sum(), 1e-45)
    smooth0 = malc_1d_cvxpy(p_marg0)
    smooth1 = malc_1d_cvxpy(p_marg1)
    e_y0 = float((centers * smooth0).sum())
    e_y1 = float((centers * smooth1).sum())
    return e_y1 - e_y0


# ── Method 2: 2D MALC (parallel worker) ─────────────────────────────────
def _cate_2d_malc_worker(args):
    """Worker: fit 2D MALC on p_mat, return TWO CATE means from the same fit:

    (a) MARGINAL path: marginalise smoothed joint on each axis → E[Y_do_t].
        CATE_marg = E[Y_do1] − E[Y_do0].
    (b) DIAGONAL path: p(τ) = ∫ f(y0, y0+τ) dy0 via linear interp on the
        fine (y0, y1) grid, then E[τ] = ∫ τ · p(τ) dτ.

    By linearity of expectation the two should agree exactly. Numerically
    they can differ because (a) does two 1D integrations and (b) does a
    2D shear+integration. Any residual gap tells us how much
    discretization bias each path carries at the chosen n_eval.

    n_eval is the grid size for evaluating the fitted continuous density.
    Higher n_eval → tighter numerical integration (reduces discretisation
    bias on marginal means too).
    """
    i, p_mat_np, edges_np, malc_B, malc_max_K, seed, n_eval = args

    # Reimport paths inside worker (Pool.spawn does not inherit sys.path).
    import os, sys
    import numpy as np
    _repo_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    _malc_dir = os.path.join(_repo_src, 'MALC')
    for _p in (_malc_dir, _repo_src):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from malc_2d import MALC_2D, eval_grid_2d

    # We pass p_mat.T to match the reference impl in benchmarks/methods/ours.py
    # (rows = Y_do1, cols = Y_do0 after transpose). See _fit_and_marginalize.
    try:
        fit = MALC_2D(
            p_mat_np.T, edges_np, edges_np,
            B_fit=malc_B, B_select=malc_B, max_K=malc_max_K,
            seed=seed, parallel=False,
        )
    except Exception as e:
        return i, float('nan'), float('nan'), str(e)

    # Fine density grid: rows = Y_do1 (ys), cols = Y_do0 (xs) per the transpose.
    xs, ys, dens = eval_grid_2d(fit, n_eval=n_eval)
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]

    # ── (a) MARGINAL PATH ──
    marg_x = dens.sum(axis=0) * dy       # marginal of Y_do0 (cols)
    marg_y = dens.sum(axis=1) * dx       # marginal of Y_do1 (rows)
    s0 = marg_x.sum() * dx
    s1 = marg_y.sum() * dy
    if s0 > 0: marg_x = marg_x / s0
    if s1 > 0: marg_y = marg_y / s1
    e_y0 = float((xs * marg_x).sum() * dx)
    e_y1 = float((ys * marg_y).sum() * dy)
    cate_marg = e_y1 - e_y0

    # ── (b) DIAGONAL PATH ──
    # τ grid spans the full possible range [ymin - ymax, ymax - ymin] with the
    # same resolution as the y-axis grids.
    tau_min = float(ys[0] - xs[-1])
    tau_max = float(ys[-1] - xs[0])
    tau_grid = np.linspace(tau_min, tau_max, n_eval)
    dtau = tau_grid[1] - tau_grid[0]

    # For each τ, integrate along y0 with y1 = y0 + τ (linear interp in y1).
    p_tau = np.zeros_like(tau_grid)
    for k, t in enumerate(tau_grid):
        y1_at = xs + t
        valid = (y1_at >= ys[0]) & (y1_at <= ys[-1])
        if not valid.any():
            continue
        col_idx = np.arange(len(xs))[valid]       # cols we're using
        rf = (y1_at[valid] - ys[0]) / dy
        rlo = np.clip(np.floor(rf).astype(int), 0, len(ys) - 2)
        rhi = rlo + 1
        whi = rf - rlo; wlo = 1.0 - whi
        # dens[row, col] = f(y0=xs[col], y1=ys[row]); interp along y1 axis:
        f_interp = wlo * dens[rlo, col_idx] + whi * dens[rhi, col_idx]
        p_tau[k] = f_interp.sum() * dx

    s_tau = p_tau.sum() * dtau
    if s_tau > 0: p_tau = p_tau / s_tau
    cate_diag = float((tau_grid * p_tau).sum() * dtau)

    return i, cate_marg, cate_diag, None


@torch.no_grad()
def forward_pmats(model, X_ctx, T_ctx, Y_ctx_raw, X_q, J, edges,
                   y_scaling_mode='pooled_std'):
    """Return (p_mats: (N_q, J, J) numpy, y_shift, y_scale)."""
    X_ctx = torch.from_numpy(X_ctx.astype(np.float32)).unsqueeze(0).to(DEVICE)
    T_ctx = torch.from_numpy(T_ctx.astype(np.float32)).unsqueeze(0).to(DEVICE)
    y_ctx_raw = torch.from_numpy(Y_ctx_raw.astype(np.float32)).unsqueeze(0).to(DEVICE)
    X_q = torch.from_numpy(X_q.astype(np.float32)).unsqueeze(0).to(DEVICE)

    if y_scaling_mode == 'uwyk_minmax':
        y_lo = y_ctx_raw.amin(dim=1, keepdim=True)
        y_hi = y_ctx_raw.amax(dim=1, keepdim=True)
        y_shift = 0.5 * (y_lo + y_hi)
        y_scale = (0.5 * (y_hi - y_lo)).clamp(min=1e-6)
    else:
        y_shift = y_ctx_raw.mean(dim=1, keepdim=True)
        y_scale = y_ctx_raw.std(dim=1, keepdim=True).clamp(min=1e-6)
    y_ctx_std = (y_ctx_raw - y_shift) / y_scale

    logits   = model._forward_logits(X_ctx, T_ctx, y_ctx_std, X_q)   # (1, N_q, nbins_2d)
    interior = logits[..., : J * J]
    p_mats   = torch.softmax(interior, dim=-1).reshape(1, -1, J, J).squeeze(0).cpu().numpy()
    return p_mats, float(y_shift.item()), float(y_scale.item())


def evaluate(realization, model, edges, J, F, y_scaling_mode, pool):
    ds = IHDPDataset()
    cate_ds = ds[realization][0]
    X_tr = cate_ds.X_train.astype(np.float32)
    T_tr = cate_ds.t_train.astype(np.float32).reshape(-1)
    y_tr = cate_ds.y_train.astype(np.float32).reshape(-1)
    X_te = cate_ds.X_test.astype(np.float32)
    true_cate = cate_ds.true_cate.astype(np.float32)
    true_ate  = float(true_cate.mean())

    X_tr_s, X_te_s = _std_X(X_tr, X_te)
    X_tr_p = _pad(X_tr_s, F); X_te_p = _pad(X_te_s, F)

    p_mats, y_shift, y_scale = forward_pmats(
        model, X_tr_p, T_tr, y_tr, X_te_p, J, edges,
        y_scaling_mode=y_scaling_mode,
    )
    edges_np = edges.cpu().numpy().astype(np.float64)
    centres  = 0.5 * (edges_np[:-1] + edges_np[1:])
    N_q      = p_mats.shape[0]

    # ── Method 1: 1D MALC per query (fast, sequential) ────────────
    cate_1d = np.empty(N_q)
    t_1d0 = time.time()
    for q in range(N_q):
        cate_1d[q] = cate_1d_malc(p_mats[q].astype(np.float64), centres)
    t_1d = time.time() - t_1d0

    # ── Method 2: 2D MALC per query (slow, parallelised) ──────────
    # Worker returns BOTH marginal-path CATE and diagonal-path CATE.
    import hashlib
    seeds = [int(hashlib.md5(f'r{realization}q{q}'.encode()).hexdigest()[:8], 16) % (10**8)
             for q in range(N_q)]
    tasks = [(q, p_mats[q].astype(np.float64), edges_np, MALC_B, MALC_MAX_K,
              seeds[q], N_EVAL)
             for q in range(N_q)]
    cate_2d_marg = np.full(N_q, np.nan)
    cate_2d_diag = np.full(N_q, np.nan)
    n_fail  = 0
    t_2d0 = time.time()
    for i, cate_m, cate_d, err in pool.imap_unordered(_cate_2d_malc_worker, tasks, chunksize=1):
        cate_2d_marg[i] = cate_m
        cate_2d_diag[i] = cate_d
        if err is not None or not np.isfinite(cate_m):
            n_fail += 1
    t_2d = time.time() - t_2d0

    # Un-standardise to raw Y units.
    cate_1d_raw       = cate_1d * y_scale
    cate_2d_marg_raw  = cate_2d_marg * y_scale
    cate_2d_diag_raw  = cate_2d_diag * y_scale

    def _pehe(cate):
        m = np.isfinite(cate)
        if not m.any(): return float('nan'), float('nan'), float('nan')
        pehe = float(np.sqrt(np.mean((cate[m] - true_cate[m]) ** 2)))
        ate  = float(cate[m].mean())
        err  = abs(ate - true_ate) / max(abs(true_ate), 1e-9)
        return pehe, err, ate

    pehe_1d, err_1d, ate_1d = _pehe(cate_1d_raw)
    pehe_2d, err_2d, ate_2d = _pehe(cate_2d_raw)

    return {
        'dataset': 'IHDP', 'realization': realization,
        'true_ate': true_ate,
        'pehe_1d_malc': pehe_1d, 'err_1d_malc': err_1d, 'ate_1d_malc': ate_1d,
        'pehe_2d_malc': pehe_2d, 'err_2d_malc': err_2d, 'ate_2d_malc': ate_2d,
        't_1d_sec': t_1d, 't_2d_sec': t_2d, 'n_2d_fail': int(n_fail),
    }


def main():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    cfg = ck['config']; edges = ck['edges']; step = ck.get('step', '?')
    y_scaling_mode = cfg.get('y_scaling_mode', 'pooled_std')
    loss_type      = cfg.get('loss_type',      'density')
    hlgauss_sigma  = float(cfg.get('hlgauss_sigma', 0.2))
    print(f'[bootstrap] ckpt={CKPT}')
    print(f'[bootstrap] step={step}  J={cfg["J"]}  y_scaling={y_scaling_mode}  '
          f'loss={loss_type}  MALC_B={MALC_B}  MALC_MAX_K={MALC_MAX_K}  workers={N_WORKERS}  '
          f'realizations=[{REAL_START}, {REAL_END})')
    print(f'[bootstrap] edges: [{edges[0].item():.3f}, {edges[-1].item():.3f}]  '
          f'bw={((edges[-1]-edges[0])/cfg["J"]).item():.4f}')

    model = CausalPFN2DHead(
        J=cfg['J'], num_features=cfg['num_features'],
        ninp=cfg['ninp'], nhid=cfg['nhid'], nhead=cfg['nhead'],
        nlayers=cfg['nlayers'], dropout=cfg.get('dropout', 0.0),
        n_out=cfg.get('n_out', 10),
        y_scaling_mode=y_scaling_mode,
        loss_type=loss_type,
        hlgauss_sigma=hlgauss_sigma,
    ).to(DEVICE)
    _load_state_dict_safe(model, ck['model_state_dict'])
    model.eval()

    os.makedirs(OUT, exist_ok=True)

    # One shared multiprocessing pool for all realizations (avoid Pool restart cost).
    import multiprocessing as mp
    ctx = mp.get_context('spawn')  # 'spawn' avoids fork issues with torch + numpy
    pool = ctx.Pool(N_WORKERS)
    try:
        rows = []
        t0 = time.time()
        for r in range(REAL_START, min(REAL_END, 100)):
            row = evaluate(r, model, edges, cfg['J'], cfg['num_features'],
                           y_scaling_mode, pool)
            np.savez(os.path.join(OUT, f'r{r:03d}.npz'),
                     **{k: np.array(v) for k, v in row.items()})
            rows.append(row)
            print(
                f'r={r:03d}  '
                f'1D-MALC: pehe={row["pehe_1d_malc"]:6.3f} err={row["err_1d_malc"]:5.3f}  |  '
                f'2D-MALC: pehe={row["pehe_2d_malc"]:6.3f} err={row["err_2d_malc"]:5.3f}  '
                f'(t_1d={row["t_1d_sec"]:.1f}s  t_2d={row["t_2d_sec"]:.1f}s  '
                f'fail={row["n_2d_fail"]}/{int(row.get("t_2d_sec", 0) > 0)}  '
                f'elapsed={time.time()-t0:.0f}s)',
                flush=True,
            )
    finally:
        pool.close(); pool.join()

    def _ms(k):
        v = np.array([r[k] for r in rows if np.isfinite(r[k])])
        if v.size == 0: return float('nan'), float('nan')
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

    print(f'\n══ IHDP summary (n={len(rows)}, step={step}, MALC_B={MALC_B}) ══')
    for k in ('pehe_1d_malc', 'err_1d_malc', 'pehe_2d_malc', 'err_2d_malc'):
        m, s = _ms(k)
        print(f'  {k:15s} = {m:8.3f} ± {s:6.3f}')


if __name__ == '__main__':
    main()
