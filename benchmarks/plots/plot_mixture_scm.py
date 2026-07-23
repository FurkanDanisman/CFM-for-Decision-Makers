"""Hand-crafted 3-cluster mixture SCM — pedagogical trimodal-τ example.

The default UWYK prior and the polynomial prior rarely produce queries where
the treatment-effect distribution p(τ) is genuinely multimodal (see
find_multimodal_queries.py — 21/15 000 candidates at best, and MALC smooths
those into single wide bumps). To show what a truly multimodal-CATE case
looks like — and to prove our model captures it faithfully — we construct a
diagnostic SCM by hand:

  Z ∈ {0, 1, 2}                         latent cluster, uniform prior
  (X_1, X_2) = μ_z[Z] + N(0, σ_x² · I)  first two covariates encode cluster
  X_{3..d} = NaN                         remaining features unused
  T | Z ~ Bernoulli(0.5)                 treatment ignorable given Z
  Y_do0 | Z = a_z + N(0, σ_y²)
  Y_do1 | Z = b_z + N(0, σ_y²)
  τ_z = b_z − a_z ∈ {−0.5, 0.0, +0.5}   distinct treatment effects
  Y_obs = T·Y_do1 + (1−T)·Y_do0

Cluster centres are placed on an **equilateral triangle** in the (X_1, X_2)
plane so the centroid (0, 0) is equidistant from all three — the query at the
centroid then has p(Z|X) = (1/3, 1/3, 1/3) → **trimodal p(τ) by construction**.

At test time we probe six queries at increasingly ambiguous X positions:
  q0: cluster-0 centre    → Z=0 dominant → single-mode τ ≈ −0.5
  q1: cluster-1 centre    → Z=1 dominant → single-mode τ ≈  0.0
  q2: cluster-2 centre    → Z=2 dominant → single-mode τ ≈ +0.5
  q3: midpoint of 0 and 1 → bimodal τ  ∈ {−0.5, 0}
  q4: midpoint of 1 and 2 → bimodal τ  ∈ {0, +0.5}
  q5: triangle centroid   → **trimodal τ ∈ {−0.5, 0, +0.5}**

The whole figure — TE distribution + joint + marginals — is produced with
the SAME infrastructure as plot_joint_marginals.py (MALC-smoothed p(τ) via
diagonal integration, same annotations), so the plots compose visually with
the SCM-prior figures.

Environment knobs
-----------------
  N_CONTEXTS   (default 500,1000,2000)   which context sizes to render
  N_CONTEXT_MAX (default 2000)           size of the context pool to sample once
  MU           (default -0.7,0.0,+0.7)   cluster centres for X_1
  A            (default +0.3,-0.1,-0.4)  Y_do0 cluster means
  B            (default -0.2,-0.1,+0.1)  Y_do1 cluster means; τ_z = B_z − A_z
  SIGMA_X      (default 0.10)            X within-cluster std
  SIGMA_Y      (default 0.05)            Y observational noise
  SEED         (default 0)               numpy RNG seed
  OUT_PREFIX   (default 'mixture3')      filename prefix (mixture3_TE_distribution.png etc.)
  CHECKPOINT   (default checkpoints/step_50000_final.pt)
  MALC_B       (default 100)
  N_EVAL       (default 200)             tau-grid resolution for diagonal integration
"""
from __future__ import annotations
import os, sys
os.environ.setdefault('PYTHONHASHSEED', '0')
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import matplotlib.pyplot as plt

_HERE  = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
_REPO  = os.path.dirname(_BENCH)
_OUTDIR = os.path.join(_HERE, 'mixture_scm')
os.makedirs(_OUTDIR, exist_ok=True)

CKPT       = os.environ.get('CHECKPOINT', os.path.join(_REPO, 'checkpoints', 'step_50000_final.pt'))
CONTEXT_SIZES = [int(x) for x in os.environ.get('N_CONTEXTS', '1000').split(',')]
N_CONTEXT_MAX = int(os.environ.get('N_CONTEXT_MAX', max(CONTEXT_SIZES) + 100))
# Equilateral-triangle cluster centres in (X_1, X_2) — radius r puts each
# cluster at (r·cosθ, r·sinθ) with θ ∈ {90°, 210°, 330°}. The centroid is
# equidistant from all three, so probing the centroid yields uniform p(Z|X).
R      = float(os.environ.get('R',      0.7))
SIGMA_X = float(os.environ.get('SIGMA_X', 0.35))
# Cluster-specific potential-outcome centres. Default: spatially symmetric
# arrangement so no single cluster dominates — three joint modes at
# (+0.4, -0.4), (-0.4, -0.4), (-0.4, +0.4) with τ ∈ {−0.8, 0.0, +0.8}.
A_VEC  = [float(x) for x in os.environ.get('A',  '0.4,-0.4,-0.4').split(',')]
B_VEC  = [float(x) for x in os.environ.get('B',  '-0.4,-0.4,0.4').split(',')]
SIGMA_Y = float(os.environ.get('SIGMA_Y', 0.05))
SEED    = int(os.environ.get('SEED', 0))
OUT_PREFIX = os.environ.get('OUT_PREFIX', 'mixture3')
MALC_B = int(os.environ.get('MALC_B', 100))
N_EVAL = int(os.environ.get('N_EVAL', 200))
K = len(A_VEC)
assert K == len(B_VEC), "A and B must have the same length"
# 2D cluster centres on an equilateral triangle (K=3 assumed by geometry)
_ANGLES_DEG = [90.0, 210.0, 330.0][:K]
MU_2D = np.array([[R * np.cos(np.deg2rad(a)), R * np.sin(np.deg2rad(a))]
                   for a in _ANGLES_DEG], dtype=np.float32)
TAU = [B_VEC[z] - A_VEC[z] for z in range(K)]

sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'MALC'))
from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import unpack_pred, fit_malc_inner
from malc_2d import dmalc_2d

DEVICE = torch.device('cpu')


# ── Load model ──────────────────────────────────────────────────────────────
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
cfg = ckpt['config']; J = cfg['J']
edges_np = ckpt['edges'].cpu().numpy()
bin_width = float(edges_np[1] - edges_np[0])
centers = 0.5 * (edges_np[:-1] + edges_np[1:])
num_features = cfg['num_features']
model = InterventionalPFN(
    num_features=num_features, d_model=cfg['d_model'], depth=cfg['depth'],
    heads_feat=cfg['heads'], heads_samp=cfg['heads'], dropout=0.0,
    output_dim=J*J + 9 + 4, hidden_mult=cfg['hidden_mult'],
    normalize_features=True, normalize_treatment=False,
    use_treatment_in_query=False, use_checkpoint=False,
).to(DEVICE).eval()
model.load_state_dict(ckpt['model_state_dict'])
print(f'[load] model J={J} num_features={num_features}', flush=True)
print(f'[scm] K={K}  R={R}  μ_2D={MU_2D.tolist()}  τ={TAU}', flush=True)


# ── Sample mixture-SCM context ──────────────────────────────────────────────
rng = np.random.default_rng(SEED)
Z_ctx = rng.integers(0, K, size=N_CONTEXT_MAX)
X_ctx = np.zeros((N_CONTEXT_MAX, num_features), dtype=np.float32)
# (X_1, X_2) encode cluster identity; rest are NaN (unused)
noise2d = rng.normal(0, SIGMA_X, size=(N_CONTEXT_MAX, 2)).astype(np.float32)
X_ctx[:, :2] = MU_2D[Z_ctx] + noise2d
X_ctx[:, 2:] = np.nan
T_ctx = rng.binomial(1, 0.5, size=N_CONTEXT_MAX).astype(np.float32)
Y0_ctx = np.array(A_VEC)[Z_ctx] + rng.normal(0, SIGMA_Y, size=N_CONTEXT_MAX)
Y1_ctx = np.array(B_VEC)[Z_ctx] + rng.normal(0, SIGMA_Y, size=N_CONTEXT_MAX)
Y_ctx = (T_ctx * Y1_ctx + (1 - T_ctx) * Y0_ctx).astype(np.float32)

X_obs_full = torch.from_numpy(X_ctx)
T_obs_full = torch.from_numpy(T_ctx.reshape(-1, 1))
Y_obs_full = torch.from_numpy(Y_ctx.reshape(-1, 1))
print(f'[sample] context pool: N_max={N_CONTEXT_MAX}  '
      f'cluster counts={[int((Z_ctx==z).sum()) for z in range(K)]}', flush=True)


# ── Test queries at six 2D X positions probing cluster ambiguity ────────────
QUERY_LABELS_XY = [
    ('q0  cluster-0 centre',        MU_2D[0]),
    ('q1  cluster-1 centre',        MU_2D[1]),
    ('q2  cluster-2 centre',        MU_2D[2]),
    ('q3  midpoint of 0↔1',         0.5 * (MU_2D[0] + MU_2D[1])),
    ('q4  midpoint of 1↔2',         0.5 * (MU_2D[1] + MU_2D[2])),
    ('q5  triangle centroid',       MU_2D.mean(axis=0)),
]
X_intv = np.zeros((len(QUERY_LABELS_XY), num_features), dtype=np.float32)
for k, (_, xy) in enumerate(QUERY_LABELS_XY):
    X_intv[k, 0] = xy[0]
    X_intv[k, 1] = xy[1]
    X_intv[k, 2:] = np.nan
X_intv_t = torch.from_numpy(X_intv)

# Under uniform prior + isotropic Gaussian N(μ_z, σ_x² I) the posterior is
# proportional to the per-cluster Gaussian likelihood.
def posterior_z(xy):
    diffs = MU_2D - np.asarray(xy)[None, :]      # (K, 2)
    log_p = -0.5 * (diffs ** 2).sum(axis=1) / (SIGMA_X ** 2)
    log_p -= log_p.max()
    p = np.exp(log_p); p /= p.sum()
    return p


# ── Run inference at each context size ──────────────────────────────────────
p_mats_by_N = {}
posts_by_q = {}
for k, (_, xy) in enumerate(QUERY_LABELS_XY):
    posts_by_q[k] = posterior_z(xy)

for N in CONTEXT_SIZES:
    Xc = X_obs_full[:N].unsqueeze(0)
    Tc = T_obs_full[:N].unsqueeze(0)
    Yc = Y_obs_full[:N].unsqueeze(0)
    Xq = X_intv_t.unsqueeze(0)
    with torch.no_grad():
        pred = model(Xc, Tc, Yc, Xq)['predictions'][0]
    pm = np.zeros((len(QUERY_LABELS_XY), J, J), dtype=np.float32)
    for k in range(len(QUERY_LABELS_XY)):
        p_mat, *_ = unpack_pred(pred[k], J, bin_width)
        pm[k] = p_mat.detach().cpu().numpy()
    p_mats_by_N[N] = pm
    print(f'[infer] N={N} done', flush=True)


# ── p(τ) via MALC + diagonal integration (same as plot_joint_marginals.py) ──
xs = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
ys = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
XX, YY = np.meshgrid(xs, ys, indexing='xy')
eval_pts = np.column_stack([XX.ravel(), YY.ravel()])
dy0_ev = xs[1] - xs[0]; dy1_ev = ys[1] - ys[0]
N_TAU = 401
tau_centers = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], N_TAU)


def p_tau_from_pmat(pm, seed):
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


# ── Figure 0: TE distribution p(τ) per query, N-overlaid ────────────────────
Q = len(QUERY_LABELS_XY)
n_cols = 3
n_rows = (Q + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 4.2 * n_rows), squeeze=False)
palette_N = ['#FFAA55', '#3388DD', '#003366'][:len(CONTEXT_SIZES)]
tau_step = tau_centers[1] - tau_centers[0]

for k, (label, xy) in enumerate(QUERY_LABELS_XY):
    ax = axes[k // n_cols][k % n_cols]
    for c, N in enumerate(CONTEXT_SIZES):
        pm = p_mats_by_N[N][k]
        p_tau = p_tau_from_pmat(pm, seed=1000 * (c + 1) + k)
        E_tau = float((tau_centers * p_tau).sum() * tau_step)
        color = palette_N[c]
        ax.plot(tau_centers, p_tau, color=color, lw=1.8,
                label=f'N={N}  E[τ]={E_tau:+.2f}')
        # Only annotate the predicted MEAN — mode line removed per current
        # framing (joint-vs-marginals is the argument, not mode-vs-mean).
        ax.axvline(E_tau, color=color, ls='--', lw=1.8, alpha=0.95, zorder=4,
                    label=f'predicted mean E[τ] = {E_tau:+.2f}')
    # SINGLE ground-truth τ — posterior-weighted expected true τ over clusters.
    # This is the oracle Bayes-optimal point estimate under squared loss and
    # matches the mixture posterior at ambiguous X.
    post = posts_by_q[k]
    true_tau_query = float(sum(p * t for p, t in zip(post, TAU)))
    ax.axvline(true_tau_query, color='red', ls='--', lw=1.6, alpha=0.95, zorder=4,
                label=f'true τ = {true_tau_query:+.2f}')
    ax.plot(true_tau_query, 0, 'o', color='red', markersize=9,
             clip_on=False, zorder=5)
    post_str = '[' + ', '.join(f'{p:.2f}' for p in post) + ']'
    ax.set_title(f'{label}   p(Z|X)={post_str}', fontsize=10)
    ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$  (scaled)')
    ax.set_xlim(-1.0, 1.0)
    if k % n_cols == 0: ax.set_ylabel(r'$p(\tau)$')
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='upper right')

fig.suptitle(
    f'Mixture-SCM diagnostic: p(τ) becomes multi-modal at ambiguous X — '
    f'K={K} clusters, τ per cluster = {TAU}',
    fontsize=12, y=0.998)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out_te = os.path.join(_OUTDIR, f'{OUT_PREFIX}_TE_distribution.png')
fig.savefig(out_te, dpi=130, bbox_inches='tight'); plt.close(fig)
print(f'[save] {out_te}', flush=True)


# ── Figure 1: joint distributions per query ─────────────────────────────────
# With a single N the layout would be Q×1 (very tall); fold to n_cols=3.
extent = [edges_np[0], edges_np[-1], edges_np[0], edges_np[-1]]
NN = len(CONTEXT_SIZES)
if NN == 1:
    n_cols_grid = 3
    n_rows_grid = (Q + n_cols_grid - 1) // n_cols_grid
    N_only = CONTEXT_SIZES[0]
    fig, axes = plt.subplots(n_rows_grid, n_cols_grid,
                              figsize=(4.6 * n_cols_grid, 4.0 * n_rows_grid),
                              squeeze=False)
    for k, (label, xy) in enumerate(QUERY_LABELS_XY):
        ax = axes[k // n_cols_grid][k % n_cols_grid]
        pm = p_mats_by_N[N_only][k]
        im = ax.imshow(pm.T, origin='lower', extent=extent, cmap='viridis', aspect='auto')
        ax.plot([edges_np[0], edges_np[-1]], [edges_np[0], edges_np[-1]],
                 'r--', lw=0.7, alpha=0.55)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(r'$Y_{do0}$ (scaled)')
        ax.set_ylabel(r'$Y_{do1}$ (scaled)')
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    # Hide any unused subplots on the last row
    for k in range(Q, n_rows_grid * n_cols_grid):
        axes[k // n_cols_grid][k % n_cols_grid].set_visible(False)
    fig.suptitle(f'Joint p(Y_do0, Y_do1) at N={N_only} — mixture SCM (K={K})',
                  fontsize=12, y=0.999)
else:
    fig, axes = plt.subplots(Q, NN, figsize=(4 * NN, 3.4 * Q), squeeze=False)
    for r, (label, xy) in enumerate(QUERY_LABELS_XY):
        for c, N in enumerate(CONTEXT_SIZES):
            ax = axes[r][c]
            pm = p_mats_by_N[N][r]
            im = ax.imshow(pm.T, origin='lower', extent=extent, cmap='viridis', aspect='auto')
            ax.plot([edges_np[0], edges_np[-1]], [edges_np[0], edges_np[-1]],
                     'r--', lw=0.7, alpha=0.55)
            ax.set_title(f'{label}   N={N}', fontsize=9)
            if r == Q - 1: ax.set_xlabel(r'$Y_{do0}$ (scaled)')
            if c == 0:      ax.set_ylabel(r'$Y_{do1}$ (scaled)')
            plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle(f'Joint p(Y_do0, Y_do1) — mixture SCM (K={K})', fontsize=12, y=0.999)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out_joint = os.path.join(_OUTDIR, f'{OUT_PREFIX}_joint_by_context.png')
fig.savefig(out_joint, dpi=130, bbox_inches='tight'); plt.close(fig)
print(f'[save] {out_joint}', flush=True)


# ── Figure 2: marginals p(Y_do0), p(Y_do1) per query ────────────────────────
# Blue + purple so the marginals plot doesn't clash with the TE plot's
# orange N=1000 palette.
palette = {'do0': '#2E7DAF',   # steely blue
             'do1': '#7B3E9E'}   # medium purple

def _plot_marg(ax, k, N, show_legend):
    label, xy = QUERY_LABELS_XY[k]
    post = posts_by_q[k]
    pm = p_mats_by_N[N][k]
    p_y0 = pm.sum(axis=1); p_y0 = p_y0 / max(p_y0.sum() * bin_width, 1e-12)
    p_y1 = pm.sum(axis=0); p_y1 = p_y1 / max(p_y1.sum() * bin_width, 1e-12)
    ax.plot(centers, p_y0, color=palette['do0'], lw=1.8, label=r'$p(Y_{do0})$')
    ax.plot(centers, p_y1, color=palette['do1'], lw=1.8, label=r'$p(Y_{do1})$')
    E_y0 = float((centers * p_y0).sum() * bin_width)
    E_y1 = float((centers * p_y1).sum() * bin_width)
    y_at_Ey0 = float(np.interp(E_y0, centers, p_y0))
    y_at_Ey1 = float(np.interp(E_y1, centers, p_y1))
    ax.plot(E_y0, y_at_Ey0, 'o', color=palette['do0'], markersize=9,
             markeredgecolor='white', markeredgewidth=1.0, zorder=5,
             label=r'$\mathbb{E}[Y_{do0}]$' if show_legend else None)
    ax.plot(E_y1, y_at_Ey1, 'o', color=palette['do1'], markersize=9,
             markeredgecolor='white', markeredgewidth=1.0, zorder=5,
             label=r'$\mathbb{E}[Y_{do1}]$' if show_legend else None)
    for z in range(K):
        ax.axvline(A_VEC[z], color=palette['do0'], ls=':', lw=1.0, alpha=0.3 + 0.6 * post[z])
        ax.axvline(B_VEC[z], color=palette['do1'], ls=':', lw=1.0, alpha=0.3 + 0.6 * post[z])
    ax.set_title(label, fontsize=10)
    ax.grid(alpha=0.3)
    if show_legend: ax.legend(fontsize=8, loc='upper right')

if NN == 1:
    n_cols_grid = 3
    n_rows_grid = (Q + n_cols_grid - 1) // n_cols_grid
    N_only = CONTEXT_SIZES[0]
    fig, axes = plt.subplots(n_rows_grid, n_cols_grid,
                              figsize=(5.5 * n_cols_grid, 3.6 * n_rows_grid),
                              squeeze=False)
    for k in range(Q):
        ax = axes[k // n_cols_grid][k % n_cols_grid]
        _plot_marg(ax, k, N_only, show_legend=(k == 0))
        if k // n_cols_grid == n_rows_grid - 1: ax.set_xlabel(r'$Y$ (scaled)')
        if k %  n_cols_grid == 0:              ax.set_ylabel(r'density')
    for k in range(Q, n_rows_grid * n_cols_grid):
        axes[k // n_cols_grid][k % n_cols_grid].set_visible(False)
    fig.suptitle(f'Marginal potential-outcome densities at N={N_only} — mixture SCM (K={K})',
                  fontsize=12, y=0.999)
else:
    fig, axes = plt.subplots(Q, NN, figsize=(4.5 * NN, 3 * Q), squeeze=False)
    for r in range(Q):
        for c, N in enumerate(CONTEXT_SIZES):
            ax = axes[r][c]
            _plot_marg(ax, r, N, show_legend=(r == 0 and c == 0))
            if r == Q - 1: ax.set_xlabel(r'$Y$ (scaled)')
            if c == 0:      ax.set_ylabel(r'density')
    fig.suptitle(f'Marginal potential-outcome densities — mixture SCM (K={K})',
                  fontsize=12, y=0.999)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out_marg = os.path.join(_OUTDIR, f'{OUT_PREFIX}_marginals_by_context.png')
fig.savefig(out_marg, dpi=130, bbox_inches='tight'); plt.close(fig)
print(f'[save] {out_marg}', flush=True)

print(f'\nGround truth by cluster:')
for z in range(K):
    print(f'  Z={z}: μ_2D=({MU_2D[z, 0]:+.2f},{MU_2D[z, 1]:+.2f})  '
          f'a={A_VEC[z]:+.2f}  b={B_VEC[z]:+.2f}  τ={TAU[z]:+.2f}')
print(f'\nQuery posterior weights p(Z|X):')
for k, (label, xy) in enumerate(QUERY_LABELS_XY):
    post = posts_by_q[k]
    print(f'  {label}   X=({xy[0]:+.2f},{xy[1]:+.2f})   '
          f'p(Z|X) = {np.round(post, 3).tolist()}')
