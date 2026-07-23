"""Visual: what the OT (Wasserstein-barycenter) population ATE is doing.

Per-query, our model produces p(τ | X_i) — a distribution over the individual
treatment effect for that query. If we want a SINGLE POPULATION-LEVEL ATE,
we need to aggregate across queries. Two ways:

  * ``ours_mean`` — average of per-query means: E_i[E[τ|X_i]]
  * ``ours_ot_*`` — first compute the Wasserstein-2 barycenter of the per-query
                    distributions, then take mode / mean of THAT barycenter.

The barycenter is not the same as the pointwise average of densities. It is
the distribution that minimises the sum of squared Wasserstein distances to
each per-query distribution — it PRESERVES SHAPE (modes stay sharp) rather
than getting flattened by superposition. Under multimodal p(τ|X_i) the
pointwise-average will smear across modes; the barycenter will not.

This figure draws the same 6-query mixture SCM used in
``plot_mixture_scm.py`` and shows, for one context size:

  Top row  — the six per-query p(τ|X_i) densities (via MALC + diagonal
             integration of the joint p(Y_do0, Y_do1 | X_i))
  Bottom   — three aggregation strategies overlaid:
                (a) pointwise mean of the six densities (blue)
                (b) W2 barycenter (green)  ← what our OT method uses
                (c) predicted-mean-of-per-query-means Ours (dashed vertical, orange)
             Plus the SCM's true population ATE (dashed vertical, red).

Environment knobs mirror plot_mixture_scm.py — see that file's docstring.

Run:
    python benchmarks/plots/plot_ot_demonstration.py
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
_OUTDIR = os.path.join(_HERE, 'ot_demonstration')
os.makedirs(_OUTDIR, exist_ok=True)

CKPT       = os.environ.get('CHECKPOINT', os.path.join(_REPO, 'checkpoints', 'step_50000_final.pt'))
N_CONTEXT  = int(os.environ.get('N_CONTEXT', 1000))
N_CONTEXT_MAX = int(os.environ.get('N_CONTEXT_MAX', max(2000, N_CONTEXT + 100)))
R          = float(os.environ.get('R',      0.7))
SIGMA_X    = float(os.environ.get('SIGMA_X', 0.35))
A_VEC  = [float(x) for x in os.environ.get('A',  '0.4,-0.4,-0.4').split(',')]
B_VEC  = [float(x) for x in os.environ.get('B',  '-0.4,-0.4,0.4').split(',')]
SIGMA_Y = float(os.environ.get('SIGMA_Y', 0.05))
SEED    = int(os.environ.get('SEED', 0))
MALC_B  = int(os.environ.get('MALC_B', 100))
N_EVAL  = int(os.environ.get('N_EVAL', 200))
OUT_PREFIX = os.environ.get('OUT_PREFIX', 'mixture3_ot')
K = len(A_VEC)
_ANGLES_DEG = [90.0, 210.0, 330.0][:K]
MU_2D = np.array([[R * np.cos(np.deg2rad(a)), R * np.sin(np.deg2rad(a))]
                   for a in _ANGLES_DEG], dtype=np.float32)
TAU = [B_VEC[z] - A_VEC[z] for z in range(K)]

sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'MALC'))
sys.path.insert(0, os.path.join(_REPO, 'MALC', 'Optimal_Transport'))
from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import unpack_pred, fit_malc_inner
from malc_2d import dmalc_2d
from ot_barycenter import wasserstein_barycenter_1d

DEVICE = torch.device('cpu')


ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
cfg = ckpt['config']; J = cfg['J']
edges_np = ckpt['edges'].cpu().numpy()
bin_width = float(edges_np[1] - edges_np[0])
num_features = cfg['num_features']
model = InterventionalPFN(
    num_features=num_features, d_model=cfg['d_model'], depth=cfg['depth'],
    heads_feat=cfg['heads'], heads_samp=cfg['heads'], dropout=0.0,
    output_dim=J*J + 9 + 4, hidden_mult=cfg['hidden_mult'],
    normalize_features=True, normalize_treatment=False,
    use_treatment_in_query=False, use_checkpoint=False,
).to(DEVICE).eval()
model.load_state_dict(ckpt['model_state_dict'])
print(f'[load] model J={J}', flush=True)


# ── Sample context + queries from the mixture SCM (same as plot_mixture_scm.py)
rng = np.random.default_rng(SEED)
Z_ctx = rng.integers(0, K, size=N_CONTEXT_MAX)
X_ctx = np.zeros((N_CONTEXT_MAX, num_features), dtype=np.float32)
noise2d = rng.normal(0, SIGMA_X, size=(N_CONTEXT_MAX, 2)).astype(np.float32)
X_ctx[:, :2] = MU_2D[Z_ctx] + noise2d
X_ctx[:, 2:] = np.nan
T_ctx = rng.binomial(1, 0.5, size=N_CONTEXT_MAX).astype(np.float32)
Y0_ctx = np.array(A_VEC)[Z_ctx] + rng.normal(0, SIGMA_Y, size=N_CONTEXT_MAX)
Y1_ctx = np.array(B_VEC)[Z_ctx] + rng.normal(0, SIGMA_Y, size=N_CONTEXT_MAX)
Y_ctx  = (T_ctx * Y1_ctx + (1 - T_ctx) * Y0_ctx).astype(np.float32)

X_obs_full = torch.from_numpy(X_ctx)
T_obs_full = torch.from_numpy(T_ctx.reshape(-1, 1))
Y_obs_full = torch.from_numpy(Y_ctx.reshape(-1, 1))

QUERY_LABELS_XY = [
    ('q0  cluster-0 centre',        MU_2D[0]),
    ('q1  cluster-1 centre',        MU_2D[1]),
    ('q2  cluster-2 centre',        MU_2D[2]),
    ('q3  midpoint of 0↔1',         0.5 * (MU_2D[0] + MU_2D[1])),
    ('q4  midpoint of 1↔2',         0.5 * (MU_2D[1] + MU_2D[2])),
    ('q5  triangle centroid',       MU_2D.mean(axis=0)),
]
Q = len(QUERY_LABELS_XY)
X_intv = np.zeros((Q, num_features), dtype=np.float32)
for k, (_, xy) in enumerate(QUERY_LABELS_XY):
    X_intv[k, 0] = xy[0]; X_intv[k, 1] = xy[1]; X_intv[k, 2:] = np.nan
X_intv_t = torch.from_numpy(X_intv)

# Population-level TRUE ATE for THIS SCM sample. Since Z is uniform:
#   E[τ] = (1/K) · Σ τ_z
TRUE_ATE = float(np.mean(TAU))
print(f'[scm] true population ATE = mean over clusters = {TRUE_ATE:+.3f}', flush=True)


# ── Inference at chosen N ───────────────────────────────────────────────────
Xc = X_obs_full[:N_CONTEXT].unsqueeze(0)
Tc = T_obs_full[:N_CONTEXT].unsqueeze(0)
Yc = Y_obs_full[:N_CONTEXT].unsqueeze(0)
Xq = X_intv_t.unsqueeze(0)
with torch.no_grad():
    pred = model(Xc, Tc, Yc, Xq)['predictions'][0]
pms = np.zeros((Q, J, J), dtype=np.float32)
for k in range(Q):
    p_mat, *_ = unpack_pred(pred[k], J, bin_width)
    pms[k] = p_mat.detach().cpu().numpy()
print(f'[infer] N={N_CONTEXT}  Q={Q}', flush=True)


# ── Compute MALC-smoothed p(τ|X_i) per query ────────────────────────────────
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


p_taus = np.zeros((Q, N_TAU), dtype=np.float64)
for k in range(Q):
    p_taus[k] = _p_tau_from_pmat(pms[k], seed=1000 + k)
    print(f'  q{k} p(τ|X) computed', flush=True)


# ── Aggregation strategies ─────────────────────────────────────────────────
dtau_ = tau_centers[1] - tau_centers[0]
# (a) per-query means → averaged (ours mean-of-means)
per_query_means = (tau_centers[None, :] * p_taus).sum(axis=1) * dtau_
mean_of_means   = float(per_query_means.mean())

# (b) pointwise mean of the six densities
pw_mean_dens    = p_taus.mean(axis=0)

# (c) W2 barycenter
bary            = wasserstein_barycenter_1d(p_taus, tau_centers)
# Barycenter is a density on tau_centers — normalise + read mode + mean
bary_norm       = bary / max(bary.sum() * dtau_, 1e-12)
bary_mode       = float(tau_centers[int(np.argmax(bary_norm))])
bary_mean       = float((tau_centers * bary_norm).sum() * dtau_)

print(f'\n[agg] mean-of-per-query-means   = {mean_of_means:+.3f}   (Ours mean)')
print(f'[agg] pointwise-mean-of-densities mode = '
      f'{float(tau_centers[int(np.argmax(pw_mean_dens))]):+.3f}   (naïve superposition)')
print(f'[agg] W2 barycenter mode        = {bary_mode:+.3f}   (Ours OT-mode)')
print(f'[agg] W2 barycenter mean        = {bary_mean:+.3f}   (Ours OT-mean)')
print(f'[agg] true population ATE       = {TRUE_ATE:+.3f}\n')


# ── Figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.35],
                       hspace=0.42, wspace=0.28)

# Top row: 6 per-query p(τ|X_i) as small stacked traces in ONE axis (or 6 tiny axes)
# Use ONE axis with each query offset vertically for clarity
ax_top = fig.add_subplot(gs[0, :])
palette_Q = plt.cm.tab10(np.linspace(0, 0.9, Q))
for k in range(Q):
    ax_top.plot(tau_centers, p_taus[k] + k * 0.6, color=palette_Q[k], lw=1.6,
                 label=QUERY_LABELS_XY[k][0])
ax_top.axvline(TRUE_ATE, color='red', ls='--', lw=1.6,
                label=f'true population ATE = {TRUE_ATE:+.2f}')
ax_top.set_xlabel(r'$\tau$')
ax_top.set_ylabel(r'$p(\tau \mid X_i)$   (queries stacked)')
ax_top.set_title(f'Per-query treatment-effect distributions at N={N_CONTEXT}',
                  fontsize=11)
ax_top.set_xlim(-1.0, 1.0)
ax_top.legend(fontsize=9, ncol=4, loc='upper center')
ax_top.grid(alpha=0.3)

# Bottom row: three aggregation strategies overlaid (single wide panel)
ax_bot = fig.add_subplot(gs[1, :])
ax_bot.fill_between(tau_centers, pw_mean_dens, alpha=0.15, color='#2E7DAF')
ax_bot.plot(tau_centers, pw_mean_dens, color='#2E7DAF', lw=2.0,
             label='pointwise mean of densities (naïve superposition)')
ax_bot.fill_between(tau_centers, bary_norm, alpha=0.20, color='#0F8A3C')
ax_bot.plot(tau_centers, bary_norm, color='#0F8A3C', lw=2.4,
             label='W₂ barycenter (Ours OT)')

# Vertical estimates
ax_bot.axvline(TRUE_ATE,   color='red',    ls='--', lw=1.8,
                label=f'true ATE = {TRUE_ATE:+.2f}')
ax_bot.axvline(mean_of_means, color='#C1420F', ls=':',  lw=2.0,
                label=f'Ours mean-of-means = {mean_of_means:+.2f}')
ax_bot.axvline(bary_mode,  color='#0F8A3C', ls='--', lw=1.8,
                label=f'Ours OT-mode = {bary_mode:+.2f}')
ax_bot.axvline(bary_mean,  color='#0F8A3C', ls=':',  lw=2.0,
                label=f'Ours OT-mean = {bary_mean:+.2f}')

ax_bot.set_xlabel(r'$\tau$')
ax_bot.set_ylabel('density')
ax_bot.set_title('Aggregation strategies: the barycenter preserves shape; '
                  'superposition smears it', fontsize=11)
ax_bot.set_xlim(-1.0, 1.0)
ax_bot.legend(fontsize=9, loc='upper right')
ax_bot.grid(alpha=0.3)

fig.suptitle(f'Optimal-transport aggregation demonstration (mixture-SCM diagnostic, '
              f'K={K} clusters, τ ∈ {TAU})', fontsize=12, y=0.998)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out_path = os.path.join(_OUTDIR, f'{OUT_PREFIX}.png')
fig.savefig(out_path, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'[save] {out_path}')
