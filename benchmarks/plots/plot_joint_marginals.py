"""Three-figure context-sweep set drawn from a fresh, fully deterministic SCM.

  <OUT_PREFIX>_TE_distribution.png       — p(τ) marginal per (query, N)
  <OUT_PREFIX>_joint_by_context.png      — joint p(Y_do0, Y_do1) per (query, N)
  <OUT_PREFIX>_marginals_by_context.png  — p(Y_do0) and p(Y_do1) per (query, N)

All three figures share the SAME SCM sample and the SAME 6 queries.

──────────────────────────────────────────────────────────────────────────────
Environment knobs (all optional — defaults reproduce the shipped plots):

  SCM_SEED       (default 2)         which SCM instance to draw
  N_TRAIN        (default 2000)      size of the context pool sampled once
  N_TEST         (default 500)       size of the test pool sampled once
  CONTEXT_SIZES  (default 500,1000,2000)   the N values shown as columns
  QUERY_IDXS     (default 357,419,472,339,64,105)  the 6 queries plotted
  OUT_PREFIX     (default 'scm')     filename prefix — change this when you
                                      want to keep existing PNGs and add a
                                      new set alongside them
  CHECKPOINT     (default checkpoints/step_50000_final.pt)
  UWYK_SRC       (default /tmp/g4cfm_uwyk/src)
  PYTHONHASHSEED (default 0)         must stay 0 for reproducibility

──────────────────────────────────────────────────────────────────────────────
Examples:

  # A completely different set of 6 queries, saved as newset_TE_distribution.png etc.
  UWYK_SRC=/tmp/g4cfm_uwyk/src PYTHONHASHSEED=0 \\
      QUERY_IDXS=12,34,56,78,90,111 OUT_PREFIX=newset \\
      python benchmarks/plots/plot_joint_marginals.py

  # A different SCM (seed=7), same 6 queries, saved as scm7_*.png
  UWYK_SRC=/tmp/g4cfm_uwyk/src PYTHONHASHSEED=0 \\
      SCM_SEED=7 OUT_PREFIX=scm7 \\
      python benchmarks/plots/plot_joint_marginals.py

The historical `context_sweep_TE_distribution.png` was produced before we knew
UWYK's SCMSampler leaked state through torch's global RNG. Its exact SCM draw
cannot be reproduced from disk. All plots below use a fresh SCM sample with
the RNG explicitly seeded — they are reproducible bit-for-bit.
"""
from __future__ import annotations
import os, sys
# Determinism guards — MUST run before importing torch. UWYK's SCMSampler
# consumes torch's global RNG in undocumented spots, so we lock every RNG
# source we can before any consumer touches them.
os.environ.setdefault('PYTHONHASHSEED', '0')
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_HERE  = os.path.dirname(os.path.abspath(__file__))               # benchmarks/plots
_BENCH = os.path.dirname(_HERE)                                    # benchmarks
_REPO  = os.path.dirname(_BENCH)                                   # R-PFN
_CSWEEP = os.path.join(_BENCH, 'context_sweep')                    # for scm_prior
_OUTDIR = os.path.join(_HERE, 'context_sweep')                     # PNG destination subfolder
os.makedirs(_OUTDIR, exist_ok=True)

# CLI knobs (env-driven so the script stays as a plain `python plot_...py` call)
CKPT       = os.environ.get('CHECKPOINT', os.path.join(_REPO, 'checkpoints', 'step_50000_final.pt'))
UWYK_SRC   = os.environ.get('UWYK_SRC',   '/tmp/g4cfm_uwyk/src')
SCM_SEED   = int(os.environ.get('SCM_SEED', 2))
N_TRAIN    = int(os.environ.get('N_TRAIN', 2000))    # sample this many context rows once
N_TEST     = int(os.environ.get('N_TEST',  500))     # then subsample from this pool
CONTEXT_SIZES = [int(x) for x in os.environ.get('CONTEXT_SIZES', '500,1000,2000').split(',')]
# QUERY_IDXS = 'auto' picks queries by true-τ percentiles (5%, 22%, 39%, 56%,
# 73%, 90%) — matches how the historical context_sweep_TE_distribution.png
# selected queries. Override with a comma list to force specific indices.
QUERY_IDXS_ENV = os.environ.get('QUERY_IDXS', 'auto')
# Filename prefix. Default embeds SCM_SEED so `SCM_SEED=X python plot_...py`
# produces distinct scm_seedX_TE_distribution.png / scm_seedX_joint... /
# scm_seedX_marginals... without overwriting the previous seed's plots.
OUT_PREFIX = os.environ.get('OUT_PREFIX', f'scm_seed{SCM_SEED}')
MALC_B     = int(os.environ.get('MALC_B', 100))
N_EVAL     = int(os.environ.get('N_EVAL', 200))

# EXACT import order from the original eval_context_sweep.py (UWYK_SRC first),
# so RNG state at the moment we call generate_paired_sample_with_raw matches
# what produced the reference context_sweep_TE_distribution.png.
sys.path.insert(0, UWYK_SRC); sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, 'MALC'))
sys.path.insert(0, _CSWEEP)

from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import unpack_pred, fit_malc_inner
from malc_2d import dmalc_2d
from scm_prior import generate_paired_sample_with_raw

DEVICE = torch.device('cpu')


def _to_np(a):
    if isinstance(a, torch.Tensor): return a.detach().cpu().numpy()
    return np.asarray(a)


# ── Load model ──────────────────────────────────────────────────────────────
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
cfg = ckpt['config']; J = cfg['J']
edges_np = ckpt['edges'].cpu().numpy()
bin_width = float(edges_np[1] - edges_np[0])
model = InterventionalPFN(
    num_features=cfg['num_features'], d_model=cfg['d_model'], depth=cfg['depth'],
    heads_feat=cfg['heads'], heads_samp=cfg['heads'], dropout=0.0,
    output_dim=J*J + 9 + 4, hidden_mult=cfg['hidden_mult'],
    normalize_features=True, normalize_treatment=False,
    use_treatment_in_query=False, use_checkpoint=False,
).to(DEVICE).eval()
model.load_state_dict(ckpt['model_state_dict'])
print(f"Model loaded (J={J})", flush=True)


# ── Sample ONE SCM with a large context pool, then subsample ────────────────
# Re-seed one more time right before the sample so results are reproducible
# even if the model-loading step above consumed random state.
random.seed(0); np.random.seed(0); torch.manual_seed(0)
print(f"Sampling SCM seed={SCM_SEED} with n_train={N_TRAIN}, n_test={N_TEST}…", flush=True)
sample = generate_paired_sample_with_raw(
    scm_seed=SCM_SEED, idx=0, n_train=N_TRAIN, n_test=N_TEST,
)

X_obs_full = sample['X_obs']       # (N_TRAIN, F)  scaled [-1,1]
T_obs_full = sample['T_obs']       # (N_TRAIN, 1)
Y_obs_full = sample['Y_obs']       # (N_TRAIN, 1)  scaled [-1,1]
X_intv     = sample['X_intv']      # (N_TEST, F)
Y_do0      = sample['Y_do0'].numpy().reshape(-1)   # scaled true τ pieces
Y_do1      = sample['Y_do1'].numpy().reshape(-1)

true_tau_scaled = Y_do1 - Y_do0
print(f"SCM sampled. True τ mean={true_tau_scaled.mean():+.3f}, std={true_tau_scaled.std():.3f}", flush=True)


# ── Pick queries by true-τ percentile (matches historical plot's method) ────
if QUERY_IDXS_ENV == 'auto':
    order = np.argsort(true_tau_scaled)
    quantiles = np.linspace(0.05, 0.95, 6)
    QUERY_IDXS = order[(quantiles * (len(true_tau_scaled) - 1)).astype(int)].tolist()
else:
    QUERY_IDXS = [int(x) for x in QUERY_IDXS_ENV.split(',')]
print(f"Queries: {QUERY_IDXS}", flush=True)
print(f"  true τ per query: {[float(f'{true_tau_scaled[q]:+.3f}') for q in QUERY_IDXS]}", flush=True)


# ── Run inference at each context size, cache p_mat for the picked queries ──
centers = 0.5 * (edges_np[:-1] + edges_np[1:])

# For each N in CONTEXT_SIZES, subsample the first N context rows and infer
p_mats_by_N = {}     # {N: (Q, J, J) — p_mat for each picked query}
for N in CONTEXT_SIZES:
    Xc = X_obs_full[:N].unsqueeze(0)
    Tc = T_obs_full[:N].unsqueeze(0)
    Yc = Y_obs_full[:N].unsqueeze(0)
    Xq = X_intv.unsqueeze(0)
    with torch.no_grad():
        pred = model(Xc, Tc, Yc, Xq)['predictions'][0]  # (N_TEST, D)
    pm = np.zeros((len(QUERY_IDXS), J, J), dtype=np.float32)
    for k, q in enumerate(QUERY_IDXS):
        p_mat, *_ = unpack_pred(pred[q], J, bin_width)
        pm[k] = p_mat.detach().cpu().numpy()
    p_mats_by_N[N] = pm
    print(f"  N={N} done", flush=True)


Q = len(QUERY_IDXS)
NN = len(CONTEXT_SIZES)

# ── Figure 0: TE distribution p(τ) per query, all N overlaid  ────────────────
# Matches the historical context_sweep_TE_distribution.png: MALC-smoothed p(τ)
# via diagonal integration of the 2D density on a fine (N_EVAL × N_EVAL) grid.
xs = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
ys = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
XX, YY = np.meshgrid(xs, ys, indexing='xy')
eval_pts = np.column_stack([XX.ravel(), YY.ravel()])
dy0_ev = xs[1] - xs[0]; dy1_ev = ys[1] - ys[0]
N_TAU = 401
tau_centers = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], N_TAU)


def _p_tau_from_pmat(pm, seed):
    """MALC-smooth the JxJ p_mat and integrate along the diagonal for p(τ)."""
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

n_cols = 3
n_rows = (Q + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 4.2 * n_rows), squeeze=False)
HISTORICAL_COLORS = ['#FFAA55', '#3388DD', '#003366']   # orange, blue, navy
if NN == 3:
    palette_N = HISTORICAL_COLORS
else:
    palette_N = plt.cm.viridis(np.linspace(0.15, 0.85, NN))

tau_step = tau_centers[1] - tau_centers[0]
for k, q in enumerate(QUERY_IDXS):
    ax = axes[k // n_cols][k % n_cols]
    for c, N in enumerate(CONTEXT_SIZES):
        pm = p_mats_by_N[N][k]
        p_tau = _p_tau_from_pmat(pm, seed=1000 * (c + 1) + q)
        E_tau  = float((tau_centers * p_tau).sum() * tau_step)
        M_tau  = float(tau_centers[int(np.argmax(p_tau))])
        color = palette_N[c]
        ax.plot(tau_centers, p_tau, color=color, lw=1.8,
                label=f'N={N}  E[τ]={E_tau:+.2f}  mode={M_tau:+.2f}')
        # Mean = dashed vertical in the N's color (thicker so it's readable
        # against the density curve of the same colour)
        ax.axvline(E_tau, color=color, ls='--', lw=1.8, alpha=0.95, zorder=4)
        # Mode = dotted vertical in the N's color
        ax.axvline(M_tau, color=color, ls=':',  lw=1.8, alpha=0.95, zorder=4)
    ax.axvline(true_tau_scaled[q], color='red', ls='--', lw=1.3,
               label=f'true τ = {true_tau_scaled[q]:+.3f}')
    ax.plot(true_tau_scaled[q], 0, 'o', color='red', markersize=9, clip_on=False, zorder=5)
    ax.set_title(f'query {q}   true τ = {true_tau_scaled[q]:+.3f}', fontsize=11)
    ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$  (scaled)')
    ax.set_xlim(-1.0, 1.0)
    if k % n_cols == 0: ax.set_ylabel(r'$p(\tau)$')
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='upper right')

fig.suptitle(
    f'Per-query TE distribution at increasing context size  '
    f'(SCM seed={SCM_SEED}, true τ mean={true_tau_scaled.mean():+.2f}, '
    f'std={true_tau_scaled.std():.2f})',
    fontsize=12, y=0.998)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out_te = os.path.join(_OUTDIR, f'{OUT_PREFIX}_TE_distribution.png')
fig.savefig(out_te, dpi=130, bbox_inches='tight'); plt.close(fig)
print(f'Saved: {out_te}', flush=True)


# ── Figure 1: joint distributions ───────────────────────────────────────────
extent = [edges_np[0], edges_np[-1], edges_np[0], edges_np[-1]]

fig, axes = plt.subplots(Q, NN, figsize=(4 * NN, 3.4 * Q), squeeze=False)
for r, q in enumerate(QUERY_IDXS):
    for c, N in enumerate(CONTEXT_SIZES):
        ax = axes[r][c]
        pm = p_mats_by_N[N][r]
        im = ax.imshow(pm.T, origin='lower', extent=extent, cmap='viridis', aspect='auto')
        # diagonal for reference
        ax.plot([edges_np[0], edges_np[-1]], [edges_np[0], edges_np[-1]],
                'r--', lw=0.7, alpha=0.55)
        ax.set_title(f"query {q}   τ_true={true_tau_scaled[q]:+.2f}   N={N}", fontsize=10)
        if r == Q - 1: ax.set_xlabel(r'$Y_{do0}$ (scaled)')
        if c == 0:      ax.set_ylabel(r'$Y_{do1}$ (scaled)')
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

fig.suptitle(
    f'Joint p($Y_{{do0}}$, $Y_{{do1}}$) sharpens with more context — '
    f'SCM seed={SCM_SEED}, true τ mean={true_tau_scaled.mean():+.2f}, '
    f'std={true_tau_scaled.std():.2f}',
    fontsize=12, y=0.999)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out_joint = os.path.join(_OUTDIR, f'{OUT_PREFIX}_joint_by_context.png')
fig.savefig(out_joint, dpi=130, bbox_inches='tight'); plt.close(fig)
print(f'Saved: {out_joint}', flush=True)


# ── Figure 2: marginals p(Y_do0), p(Y_do1) ──────────────────────────────────
fig, axes = plt.subplots(Q, NN, figsize=(4.5 * NN, 3 * Q), squeeze=False)
palette = {'do0': 'steelblue', 'do1': 'darkorange'}
for r, q in enumerate(QUERY_IDXS):
    for c, N in enumerate(CONTEXT_SIZES):
        ax = axes[r][c]
        pm = p_mats_by_N[N][r]
        p_y0 = pm.sum(axis=1)   # marginal over Y_do0
        p_y1 = pm.sum(axis=0)   # marginal over Y_do1
        # normalize so integrals ≈ 1
        p_y0 = p_y0 / max(p_y0.sum() * bin_width, 1e-12)
        p_y1 = p_y1 / max(p_y1.sum() * bin_width, 1e-12)
        ax.plot(centers, p_y0, color=palette['do0'], lw=1.8, label=r'$p(Y_{do0})$')
        ax.plot(centers, p_y1, color=palette['do1'], lw=1.8, label=r'$p(Y_{do1})$')
        # Predicted means: E[Y_do?] = ∑ y·p(y)·Δy — red dots on the density
        E_y0 = float((centers * p_y0).sum() * bin_width)
        E_y1 = float((centers * p_y1).sum() * bin_width)
        # y-heights for the dots: place ON the curve at that mean so the reader
        # sees where the peak-weighted average lands vertically as well
        y_at_Ey0 = float(np.interp(E_y0, centers, p_y0))
        y_at_Ey1 = float(np.interp(E_y1, centers, p_y1))
        ax.plot(E_y0, y_at_Ey0, 'o', color='red', markersize=8, zorder=5,
                label=r'$\mathbb{E}[Y_{do0}]$, $\mathbb{E}[Y_{do1}]$' if (r == 0 and c == 0) else None)
        ax.plot(E_y1, y_at_Ey1, 'o', color='red', markersize=8, zorder=5)
        ax.axvline(Y_do0[q], color=palette['do0'], ls=':', lw=1.2,
                    alpha=0.7, label=r'true $Y_{do0}$' if (r == 0 and c == 0) else None)
        ax.axvline(Y_do1[q], color=palette['do1'], ls=':', lw=1.2,
                    alpha=0.7, label=r'true $Y_{do1}$' if (r == 0 and c == 0) else None)
        ax.set_title(f"query {q}   τ_true={true_tau_scaled[q]:+.2f}   N={N}", fontsize=10)
        if r == Q - 1: ax.set_xlabel(r'$Y$ (scaled)')
        if c == 0:      ax.set_ylabel(r'density')
        ax.grid(alpha=0.3)
        if r == 0 and c == 0: ax.legend(fontsize=7, loc='upper right')

fig.suptitle(
    f'Marginal potential-outcome densities p($Y_{{do0}}$) and p($Y_{{do1}}$) — '
    f'SCM seed={SCM_SEED}',
    fontsize=12, y=0.999)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out_marg = os.path.join(_OUTDIR, f'{OUT_PREFIX}_marginals_by_context.png')
fig.savefig(out_marg, dpi=130, bbox_inches='tight'); plt.close(fig)
print(f'Saved: {out_marg}', flush=True)
