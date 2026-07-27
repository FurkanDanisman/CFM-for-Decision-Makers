"""IHDP demonstration — small-context (N=10) joint / marginals / TE / OT.

Runs on killarney where CausalPFN's benchmarks package + Do-PFN's dataset
module are on disk. Locally the script exits cleanly with a `[skip]` message.

Given a single IHDP realization we:
  1. Take the FIRST 10 training observations as context (a very small
     context that highlights the model's remaining uncertainty).
  2. Pick 6 test queries by true-τ percentile so we span the CATE range.
  3. Run OURS inference and cache the per-query 2D p(Y_do0, Y_do1) grids.
  4. For each query render the joint density, the marginals, and the
     MALC-smoothed p(τ) via diagonal integration.
  5. Aggregate the six per-query p(τ) into a Wasserstein-2 barycenter
     to visualise the OT-based population ATE.

Four PNGs land in benchmarks/plots/ihdp_n10/:
  ihdp_n10_joint.png
  ihdp_n10_marginals.png
  ihdp_n10_te.png
  ihdp_n10_ot.png

Environment
-----------
  REPO        (default the R-PFN repo containing this script)
  CAUSALPFN   path to CausalPFN repo — contains benchmarks/IHDPDataset
  DOPFN       path to Do-PFN repo — needed for its datasets/ shim
  UWYK_SRC    ignored here but read for symmetry with other scripts
  CHECKPOINT  our checkpoint (default checkpoints/step_50000_final.pt)
  REALIZATION realization index to use  (default 0)
  N_CONTEXT   number of training observations to keep  (default 10)
  N_QUERIES   number of test queries to render         (default 6)
  MALC_B      MALC bootstrap size                       (default 60)
  N_EVAL      grid resolution for joint / MALC          (default 200)
"""
from __future__ import annotations
import os, sys, types
import numpy as np
import torch
import matplotlib.pyplot as plt

_HERE   = os.path.dirname(os.path.abspath(__file__))
_BENCH  = os.path.dirname(_HERE)
_REPO   = os.environ.get('REPO', os.path.dirname(_BENCH))
# Two output subfolders — Ours (2DMALC joint+MALC+OT) and the ablation
# UWYK-2DMALC-NAIVE (same marginals we predict, but the treatment-effect
# distribution is computed under independence rather than from the joint).
_ROOTDIR      = os.path.join(_HERE, 'ihdp_n10')
_OUTDIR_JOINT = os.path.join(_ROOTDIR, 'UWYK-2DMALC')
_OUTDIR_NAIVE = os.path.join(_ROOTDIR, 'UWYK-2DMALC-NAIVE')
os.makedirs(_OUTDIR_JOINT, exist_ok=True)
os.makedirs(_OUTDIR_NAIVE, exist_ok=True)
_OUTDIR = _OUTDIR_JOINT   # legacy alias kept for older figure-1/-2 code below

CKPT       = os.environ.get('CHECKPOINT', os.path.join(_REPO, 'checkpoints', 'step_50000_final.pt'))
CAUSALPFN  = os.environ.get('CAUSALPFN', '')
DOPFN      = os.environ.get('DOPFN', '')
REALIZATION = int(os.environ.get('REALIZATION', 0))
N_CONTEXT  = int(os.environ.get('N_CONTEXT', 200))
N_QUERIES  = int(os.environ.get('N_QUERIES', 10))
MALC_B     = int(os.environ.get('MALC_B', 60))
N_EVAL     = int(os.environ.get('N_EVAL', 200))

if not (os.path.isfile(CKPT) and os.path.isdir(CAUSALPFN) and os.path.isdir(DOPFN)):
    print('[skip] IHDP demonstration needs CAUSALPFN and DOPFN paths + a valid checkpoint.')
    print(f'   CHECKPOINT  = {CKPT}   exists={os.path.isfile(CKPT)}')
    print(f'   CAUSALPFN   = {CAUSALPFN!r}  exists={os.path.isdir(CAUSALPFN)}')
    print(f'   DOPFN       = {DOPFN!r}      exists={os.path.isdir(DOPFN)}')
    sys.exit(0)

sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'MALC'))
sys.path.insert(0, os.path.join(_REPO, 'MALC', 'Optimal_Transport'))
from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import unpack_pred, fit_malc_inner
from malc_2d import dmalc_2d
from ot_barycenter import wasserstein_barycenter_1d

# Do-PFN's datasets module is needed by CausalPFN's IHDP loader
sys.path.insert(0, DOPFN)
ds_mod = types.ModuleType('datasets')
with open(os.path.join(DOPFN, 'datasets/__init__.py')) as fp:
    _src = fp.read().split('def load_semi_real')[0]
exec(_src, ds_mod.__dict__)
sys.modules['datasets'] = ds_mod
sys.path.insert(0, CAUSALPFN)

DEVICE = torch.device('cpu')
_orig_torch_load = torch.load
def _p_load(*a, **kw):
    kw.setdefault('weights_only', False); return _orig_torch_load(*a, **kw)
torch.load = _p_load


# ── Load model ────────────────────────────────────────────────────────────
ckpt = torch.load(CKPT, map_location=DEVICE)
cfg = ckpt['config']; J = cfg['J']
edges_np = ckpt['edges'].cpu().numpy()
bin_width = float(edges_np[1] - edges_np[0])
centers = 0.5 * (edges_np[:-1] + edges_np[1:])
NUM_FEATURES = cfg['num_features']
model = InterventionalPFN(
    num_features=NUM_FEATURES, d_model=cfg['d_model'], depth=cfg['depth'],
    heads_feat=cfg['heads'], heads_samp=cfg['heads'], dropout=0.0,
    output_dim=J*J + 9 + 4, hidden_mult=cfg['hidden_mult'],
    normalize_features=True, normalize_treatment=False,
    use_treatment_in_query=False, use_checkpoint=False,
).to(DEVICE).eval()
model.load_state_dict(ckpt['model_state_dict'])
print(f'[load] OURS  J={J}', flush=True)


# ── Load IHDP realization ─────────────────────────────────────────────────
from benchmarks import IHDPDataset
cd, ad = IHDPDataset()[REALIZATION]
X_train_full = cd.X_train.numpy() if hasattr(cd.X_train, 'numpy') else np.asarray(cd.X_train)
t_train_full = cd.t_train.numpy() if hasattr(cd.t_train, 'numpy') else np.asarray(cd.t_train)
y_train_full = cd.y_train.numpy() if hasattr(cd.y_train, 'numpy') else np.asarray(cd.y_train)
X_test = cd.X_test.numpy() if hasattr(cd.X_test, 'numpy') else np.asarray(cd.X_test)
true_cate = cd.true_cate.numpy() if hasattr(cd.true_cate, 'numpy') else np.asarray(cd.true_cate)
true_cate = true_cate.reshape(-1)

# Small context — first N observations
X_context = X_train_full[:N_CONTEXT].astype(np.float32)
T_context = t_train_full[:N_CONTEXT].astype(np.float32).reshape(-1, 1)
Y_context = y_train_full[:N_CONTEXT].astype(np.float32).reshape(-1, 1)

# Normalise Y to model's [-1, 1] using training-Y range
y_min = float(y_train_full.min()); y_max = float(y_train_full.max())
y_rng = max(y_max - y_min, 1e-6)
Y_context = ((Y_context - y_min) / y_rng * 2.0 - 1.0).astype(np.float32)
true_cate_scaled = true_cate * (2.0 / y_rng)

# Pad X to model's num_features
d = X_context.shape[1]
if d < NUM_FEATURES:
    pad = np.full((X_context.shape[0], NUM_FEATURES - d), np.nan, dtype=np.float32)
    X_context = np.concatenate([X_context, pad], axis=1)
    Xt_pad    = np.full((X_test.shape[0], NUM_FEATURES - d), np.nan, dtype=np.float32)
    X_test_p  = np.concatenate([X_test.astype(np.float32), Xt_pad], axis=1)
else:
    X_context = X_context[:, :NUM_FEATURES]
    X_test_p  = X_test.astype(np.float32)[:, :NUM_FEATURES]

# Pick queries by true-τ percentile
order = np.argsort(true_cate_scaled)
qs = np.linspace(0.05, 0.95, N_QUERIES)
QUERY_IDXS = order[(qs * (len(true_cate_scaled) - 1)).astype(int)].tolist()
print(f'[data] IHDP r={REALIZATION}  N_context={N_CONTEXT}  '
      f'n_queries={N_QUERIES}  queries={QUERY_IDXS}', flush=True)


# ── Inference ─────────────────────────────────────────────────────────────
Xc = torch.from_numpy(X_context).unsqueeze(0)
Tc = torch.from_numpy(T_context).unsqueeze(0)
Yc = torch.from_numpy(Y_context).unsqueeze(0)
Xq = torch.from_numpy(X_test_p).unsqueeze(0)
with torch.no_grad():
    pred = model(Xc, Tc, Yc, Xq)['predictions'][0]   # (n_test, D)

# Cache the 2D p_mat for each selected query
p_mats = np.zeros((N_QUERIES, J, J), dtype=np.float32)
for k, q in enumerate(QUERY_IDXS):
    p_mat, *_ = unpack_pred(pred[q], J, bin_width)
    p_mats[k] = p_mat.detach().cpu().numpy()
print('[infer] done', flush=True)


# ── Per-query MALC-smoothed p(τ) via diagonal integration ─────────────────
xs = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
ys = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
XX, YY = np.meshgrid(xs, ys, indexing='xy')
eval_pts = np.column_stack([XX.ravel(), YY.ravel()])
dy0_ev = xs[1] - xs[0]; dy1_ev = ys[1] - ys[0]
N_TAU = 401
tau_centers = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], N_TAU)


def _p_tau_from_pmat(pm, seed):
    fit = fit_malc_inner(pm.T, edges_np, edges_np,
                          B_fit=MALC_B, B_select=MALC_B,
                          max_K=3, seed=seed, parallel=False)
    density = dmalc_2d(fit, eval_pts).reshape(N_EVAL, N_EVAL)
    out = np.zeros_like(tau_centers)
    for k, t in enumerate(tau_centers):
        y1_target = xs + t
        valid = (y1_target >= ys[0]) & (y1_target <= ys[-1])
        if not np.any(valid): continue
        col = np.clip(np.searchsorted(xs, xs[valid]) - 1, 0, len(xs) - 1)
        row_f  = (y1_target[valid] - ys[0]) / dy1_ev
        row_lo = np.clip(np.floor(row_f).astype(int), 0, len(ys) - 2)
        row_hi = row_lo + 1
        w_hi   = row_f - row_lo
        w_lo   = 1.0 - w_hi
        f_diag = w_lo * density[row_lo, col] + w_hi * density[row_hi, col]
        out[k] = f_diag.sum() * dy0_ev
    s = out.sum() * (tau_centers[1] - tau_centers[0])
    if s > 0: out /= s
    return out


p_taus = np.zeros((N_QUERIES, N_TAU), dtype=np.float64)
for k in range(N_QUERIES):
    p_taus[k] = _p_tau_from_pmat(p_mats[k], seed=1000 + k)
print('[malc] per-query p(τ) done', flush=True)


# ── Figure layout: 5 cols if 10 queries (2×5), else 3 cols ───────────────
n_cols = 5 if N_QUERIES == 10 else 3
n_rows = (N_QUERIES + n_cols - 1) // n_cols
extent = [edges_np[0], edges_np[-1], edges_np[0], edges_np[-1]]

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 4.0 * n_rows),
                          squeeze=False)
for k, q in enumerate(QUERY_IDXS):
    ax = axes[k // n_cols][k % n_cols]
    im = ax.imshow(p_mats[k].T, origin='lower', extent=extent,
                    cmap='viridis', aspect='auto')
    ax.plot([edges_np[0], edges_np[-1]], [edges_np[0], edges_np[-1]],
             'r--', lw=0.7, alpha=0.55)
    ax.set_title(f'query {q}   $\\tau_{{true}}$={true_cate_scaled[q]:+.2f}',
                  fontsize=10)
    if k // n_cols == n_rows - 1: ax.set_xlabel(r'$Y_{do0}$  (scaled)')
    if k %  n_cols == 0:          ax.set_ylabel(r'$Y_{do1}$  (scaled)')
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
for k in range(N_QUERIES, n_rows * n_cols):
    axes[k // n_cols][k % n_cols].set_visible(False)
fig.suptitle(f'IHDP r={REALIZATION}   joint $p(Y_{{do0}}, Y_{{do1}})$   at N={N_CONTEXT}',
              fontsize=12, y=0.999)
fig.tight_layout(rect=[0, 0, 1, 0.985])
# Joint plot is only meaningful for the 2DMALC version; UWYK-2DMALC-NAIVE
# uses the *same* joint predictions but ignores the coupling, so we skip it there.
out = os.path.join(_OUTDIR_JOINT, 'ihdp_n10_joint.png')
fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
print(f'[save] {out}')


# ── Figure 2: marginals p(Y_do0), p(Y_do1) — same in BOTH folders ─────────
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.6 * n_rows),
                          squeeze=False)
palette = {'do0': '#2E7DAF', 'do1': '#7B3E9E'}
for k, q in enumerate(QUERY_IDXS):
    ax = axes[k // n_cols][k % n_cols]
    pm = p_mats[k]
    p_y0 = pm.sum(axis=1); p_y0 = p_y0 / max(p_y0.sum() * bin_width, 1e-12)
    p_y1 = pm.sum(axis=0); p_y1 = p_y1 / max(p_y1.sum() * bin_width, 1e-12)
    ax.plot(centers, p_y0, color=palette['do0'], lw=1.9, label=r'$p(Y_{do0})$')
    ax.plot(centers, p_y1, color=palette['do1'], lw=1.9, label=r'$p(Y_{do1})$')
    E_y0 = float((centers * p_y0).sum() * bin_width)
    E_y1 = float((centers * p_y1).sum() * bin_width)
    ax.plot(E_y0, float(np.interp(E_y0, centers, p_y0)), 'o',
             color=palette['do0'], markersize=9, markeredgecolor='white',
             markeredgewidth=1.0, zorder=5)
    ax.plot(E_y1, float(np.interp(E_y1, centers, p_y1)), 'o',
             color=palette['do1'], markersize=9, markeredgecolor='white',
             markeredgewidth=1.0, zorder=5)
    ax.set_title(f'query {q}   $\\tau_{{true}}$={true_cate_scaled[q]:+.2f}',
                  fontsize=10)
    if k // n_cols == n_rows - 1: ax.set_xlabel(r'$Y$  (scaled)')
    if k %  n_cols == 0:          ax.set_ylabel('density')
    ax.grid(alpha=0.25)
    if k == 0: ax.legend(fontsize=9, loc='upper right')
for k in range(N_QUERIES, n_rows * n_cols):
    axes[k // n_cols][k % n_cols].set_visible(False)
fig.suptitle(f'IHDP r={REALIZATION}   marginal potential-outcome densities at N={N_CONTEXT}',
              fontsize=12, y=0.999)
fig.tight_layout(rect=[0, 0, 1, 0.985])
for _dir in (_OUTDIR_JOINT, _OUTDIR_NAIVE):
    out = os.path.join(_dir, 'ihdp_n10_marginals.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'[save] {out}')
plt.close(fig)


# ── Naive TE distribution: p(τ) from marginals under independence ─────────
# For τ = Y_do1 - Y_do0 with Y_do1 ⊥⊥ Y_do0, the density is the difference
# convolution   p_τ(t) = ∫ p_{Y1}(y0 + t) p_{Y0}(y0) dy0. Discretely, we
# sum along diagonals of the outer product p_{Y1} ⊗ p_{Y0}. Grid resolution
# is the bin width h; we then interpolate to the MALC tau grid so the naive
# curve is on the same grid as the joint-based one.
def _naive_p_tau_from_marginals(pm):
    p_y0 = pm.sum(axis=1); p_y0 = p_y0 / max(p_y0.sum(), 1e-12)
    p_y1 = pm.sum(axis=0); p_y1 = p_y1 / max(p_y1.sum(), 1e-12)
    outer = np.outer(p_y1, p_y0)                              # (J, J)
    Jn = outer.shape[0]
    diag_sums = np.array([np.trace(outer, offset=off)
                           for off in range(-(Jn - 1), Jn)])
    tau_naive_grid = np.arange(-(Jn - 1), Jn) * bin_width
    density = diag_sums / bin_width                           # mass → density
    return np.interp(tau_centers, tau_naive_grid, density,
                      left=0.0, right=0.0)


p_taus_naive = np.zeros_like(p_taus)
for k in range(N_QUERIES):
    p_taus_naive[k] = _naive_p_tau_from_marginals(p_mats[k])
print('[naive] per-query naive-independence p(τ) computed', flush=True)


# ── Figure 3: per-query TE distributions ──────────────────────────────────
# 3a. UWYK-2DMALC: joint p(τ) (Ours) with naive overlay for comparison
# 3b. UWYK-2DMALC-NAIVE: naive p(τ) only (same marginals, ignore joint)
def _plot_te_figure(include_joint: bool, outpath: str):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.6 * n_rows),
                              squeeze=False)
    tau_step = tau_centers[1] - tau_centers[0]
    for k, q in enumerate(QUERY_IDXS):
        ax = axes[k // n_cols][k % n_cols]
        p_tau       = p_taus[k]
        p_tau_naive = p_taus_naive[k]
        E_tau       = float((tau_centers * p_tau).sum() * tau_step)
        E_tau_naive = float((tau_centers * p_tau_naive).sum() * tau_step)
        if include_joint:
            ax.fill_between(tau_centers, p_tau, alpha=0.25, color='#2E4A6F')
            ax.plot(tau_centers, p_tau, color='#2E4A6F', lw=2.0,
                     label=f'joint $p(\\tau)$  E={E_tau:+.2f}')
            ax.axvline(E_tau, color='#2E4A6F', ls='--', lw=1.4, alpha=0.85)
            ax.plot(tau_centers, p_tau_naive, color='#C1420F', lw=1.8, ls='--',
                     label=f'naive $p(\\tau)$  E={E_tau_naive:+.2f}')
            ax.axvline(E_tau_naive, color='#C1420F', ls=':', lw=1.4, alpha=0.85)
        else:
            ax.fill_between(tau_centers, p_tau_naive, alpha=0.25, color='#C1420F')
            ax.plot(tau_centers, p_tau_naive, color='#C1420F', lw=2.0,
                     label=f'naive $p(\\tau)$  E={E_tau_naive:+.2f}')
            ax.axvline(E_tau_naive, color='#C1420F', ls='--', lw=1.4, alpha=0.85)
        ax.axvline(true_cate_scaled[q], color='red', ls='--', lw=1.4,
                    label=f'true $\\tau$={true_cate_scaled[q]:+.2f}')
        ax.plot(true_cate_scaled[q], 0, 'o', color='red', markersize=9,
                 clip_on=False, zorder=6)
        ax.set_xlim(-1.0, 1.0)
        ax.set_title(f'query {q}', fontsize=10)
        if k // n_cols == n_rows - 1: ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$  (scaled)')
        if k %  n_cols == 0:          ax.set_ylabel(r'$p(\tau)$')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc='upper right')
    for k in range(N_QUERIES, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_visible(False)
    if include_joint:
        title = (f'IHDP r={REALIZATION}   per-query TE distributions at N={N_CONTEXT}   '
                 '(joint = Ours, naive = marginal-only independence)')
    else:
        title = (f'IHDP r={REALIZATION}   naive per-query TE at N={N_CONTEXT}   '
                 '(from marginals under independence)')
    fig.suptitle(title, fontsize=12, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(outpath, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'[save] {outpath}')


_plot_te_figure(include_joint=True,
                 outpath=os.path.join(_OUTDIR_JOINT, 'ihdp_n10_te.png'))
_plot_te_figure(include_joint=False,
                 outpath=os.path.join(_OUTDIR_NAIVE, 'ihdp_n10_te.png'))


# ── Figure 4: OT (W2 barycenter) aggregation ─────────────────────────────
# One barycenter per set of per-query densities.
def _plot_ot_figure(densities: np.ndarray, kind: str, outpath: str):
    tau_step = tau_centers[1] - tau_centers[0]
    bary = wasserstein_barycenter_1d(densities, tau_centers)
    bary_norm = bary / max(bary.sum() * tau_step, 1e-12)
    bary_mode = float(tau_centers[int(np.argmax(bary_norm))])
    bary_mean = float((tau_centers * bary_norm).sum() * tau_step)
    per_q_means = (tau_centers[None, :] * densities).sum(axis=1) * tau_step
    mean_of_means_local = float(per_q_means.mean())
    true_ate_local = float(true_cate_scaled.mean())

    fig, ax = plt.subplots(figsize=(9, 4.6))
    palette_Q = plt.cm.tab10(np.linspace(0, 0.9, N_QUERIES))
    for k in range(N_QUERIES):
        ax.plot(tau_centers, densities[k], color=palette_Q[k], lw=1.1, alpha=0.35)
    ax.fill_between(tau_centers, bary_norm, alpha=0.20, color='#0F8A3C')
    ax.plot(tau_centers, bary_norm, color='#0F8A3C', lw=2.6,
             label='W₂ barycenter (OT)')
    ax.axvline(true_ate_local, color='red', ls='--', lw=1.6,
                label=f'true population ATE = {true_ate_local:+.2f}')
    ax.axvline(bary_mode, color='#0F8A3C', ls='--', lw=1.6,
                label=f'OT-mode = {bary_mode:+.2f}')
    ax.axvline(mean_of_means_local, color='#C1420F', ls='--', lw=1.6,
                label=f'mean-of-means = {mean_of_means_local:+.2f}')
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$  (scaled)')
    ax.set_ylabel('density')
    ax.set_title(f'IHDP r={REALIZATION}   OT aggregation ({kind}) at N={N_CONTEXT}',
                  fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'[save] {outpath}')
    return true_ate_local, bary_mode, bary_mean, mean_of_means_local


true_ate, bary_mode, bary_mean, mean_of_means = _plot_ot_figure(
    p_taus, 'joint p(τ)',
    os.path.join(_OUTDIR_JOINT, 'ihdp_n10_ot.png'))
true_ate_n, bary_mode_n, bary_mean_n, mean_of_means_n = _plot_ot_figure(
    p_taus_naive, 'naive p(τ)',
    os.path.join(_OUTDIR_NAIVE, 'ihdp_n10_ot.png'))


print(f'\n[summary — UWYK-2DMALC   (joint p(τ))]')
print(f'  true population ATE (scaled) = {true_ate:+.3f}')
print(f'  mean-of-per-query-means      = {mean_of_means:+.3f}')
print(f'  W2 barycenter mode           = {bary_mode:+.3f}')
print(f'  W2 barycenter mean           = {bary_mean:+.3f}')

print(f'\n[summary — UWYK-2DMALC-NAIVE  (independence-derived p(τ))]')
print(f'  true population ATE (scaled) = {true_ate_n:+.3f}')
print(f'  mean-of-per-query-means      = {mean_of_means_n:+.3f}')
print(f'  W2 barycenter mode           = {bary_mode_n:+.3f}')
print(f'  W2 barycenter mean           = {bary_mean_n:+.3f}')
