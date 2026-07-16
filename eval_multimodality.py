"""
Two experiments to test the multi-hypothesis / identifiability picture:

  (A) Context sweep at N = 50, 250, 1000, 5000, 10000.
      If the model captures identifiability uncertainty, small-N should show
      MORE modes (multiple SCMs plausible); large-N should collapse to fewer
      modes (SCM identified). Same 6 spread queries.

  (B) Raw p_mat vs MALC-smoothed density at MALC B = 100 and B = 500.
      For the same 6 queries at N=1000, plot p(τ) from three sources:
        - raw p_mat via discrete diagonal sum       (no smoothing)
        - MALC fit with B_select=B_fit=100         (light smoothing)
        - MALC fit with B_select=B_fit=500         (heavier fit)
      If raw p_mat shows sharp bimodal structure that MALC B=100 collapses,
      MALC is the bottleneck.

Notes:
  - Model was trained at N=1000; N=5000 and N=10000 are out-of-distribution
    for the transformer. Results at those sizes reflect what a model trained
    at N=1000 does when handed more context, not what training at scale would
    give you.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch
import matplotlib.pyplot as plt

UWYK_SRC   = os.environ.get('UWYK_SRC', '/tmp/g4cfm_uwyk/src')
CHECKPOINT = os.environ.get('CHECKPOINT', 'checkpoints/step_50000_final.pt')
OUT_DIR    = os.environ.get('OUT_DIR', 'eval_multimodality')
SCM_SEED   = int(os.environ.get('SCM_SEED', 2))
CONTEXT_SIZES_A = [int(x) for x in os.environ.get(
    'CONTEXT_SIZES_A', '50,250,1000,5000,10000').split(',')]
MALC_BS_B  = [int(x) for x in os.environ.get('MALC_BS_B', '100,500').split(',')]
N_EVAL     = int(os.environ.get('N_EVAL', 200))
N_TAU      = int(os.environ.get('N_TAU', 401))
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, UWYK_SRC); sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, 'MALC'))

from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import unpack_pred, fit_malc_inner
from malc_2d import dmalc_2d
from eval_scm_gen import generate_paired_sample_with_raw

os.makedirs(OUT_DIR, exist_ok=True)


def _log(msg, t0=None):
    t = f"[{time.time() - t0:6.1f}s]" if t0 is not None else "[  0.0s]"
    print(f"{t} {msg}", flush=True)


# ── Model ────────────────────────────────────────────────────────────────────
t0 = time.time()
ckpt   = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
config = ckpt['config']
J      = config['J']
edges  = ckpt['edges'].to(DEVICE)
edges_np = edges.detach().cpu().numpy()
bin_width = float(edges[1] - edges[0])
centers = 0.5 * (edges_np[:-1] + edges_np[1:])

model = InterventionalPFN(
    num_features=config['num_features'],
    d_model=config['d_model'], depth=config['depth'],
    heads_feat=config['heads'], heads_samp=config['heads'],
    dropout=0.0, output_dim=J*J + 9 + 4,
    hidden_mult=config['hidden_mult'],
    normalize_features=True, normalize_treatment=False,
    use_treatment_in_query=False, use_checkpoint=False,
).to(DEVICE).eval()
model.load_state_dict(ckpt['model_state_dict'])
_log(f"Model J={J} bin_width={bin_width:.4f}", t0)


# ── One SCM with enough context for the biggest N ─────────────────────────────
n_ctx_max = max(CONTEXT_SIZES_A)
_log(f"Generating SCM seed={SCM_SEED} with n_train={n_ctx_max}, n_test=500…", t0)
sample = generate_paired_sample_with_raw(
    scm_seed=SCM_SEED, idx=0, n_train=n_ctx_max, n_test=500,
)
true_tau_all = (sample['Y_do1'] - sample['Y_do0']).reshape(-1).numpy()
_log(f"  true τ mean={true_tau_all.mean():+.3f}  std={true_tau_all.std():.3f}  "
     f"min={true_tau_all.min():+.3f}  max={true_tau_all.max():+.3f}", t0)

order = np.argsort(true_tau_all)
query_idx = order[(np.linspace(0.05, 0.95, 6) * (len(true_tau_all) - 1)).astype(int)]
_log(f"Selected queries: {list(map(int, query_idx))}", t0)
_log(f"  true τ: {true_tau_all[query_idx].round(3).tolist()}", t0)


# ── Common grids ──────────────────────────────────────────────────────────────
xs = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
ys = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
XX, YY = np.meshgrid(xs, ys, indexing='xy')
eval_pts = np.column_stack([XX.ravel(), YY.ravel()])
dy0 = xs[1] - xs[0]; dy1 = ys[1] - ys[0]

# τ grid for smooth densities (MALC) — matches the fine evaluation grid
tau_smooth = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], N_TAU)

# τ grid for raw p_mat: the natural τ values are (j1-j0)*bin_width, one per k
k_range = np.arange(-J + 1, J)                     # -99 .. +99 for J=100
tau_raw  = k_range * bin_width                     # length 2J-1 = 199


def p_tau_from_malc_grid(density):
    """MALC diagonal integration on the smooth grid."""
    out = np.zeros_like(tau_smooth)
    for k, t in enumerate(tau_smooth):
        y1_target = xs + t
        valid = (y1_target >= ys[0]) & (y1_target <= ys[-1])
        if not np.any(valid):
            continue
        col_idx = np.clip(np.searchsorted(xs, xs[valid]) - 1, 0, len(xs) - 1)
        row_f  = (y1_target[valid] - ys[0]) / dy1
        row_lo = np.clip(np.floor(row_f).astype(int), 0, len(ys) - 2)
        row_hi = row_lo + 1
        w_hi   = row_f - row_lo
        w_lo   = 1.0 - w_hi
        f_diag = w_lo * density[row_lo, col_idx] + w_hi * density[row_hi, col_idx]
        out[k] = f_diag.sum() * dy0
    return out


def p_tau_from_raw_p_mat(p_mat_np):
    """Discrete diagonal sum on the J×J grid.
    p_mat[j0, j1] = P(Y_do0 in bin j0, Y_do1 in bin j1)
    P(τ = (j1-j0)*bw) = Σ_{j0} p_mat[j0, j0 + k]
    Density = that sum / bin_width.
    """
    P = np.zeros(len(k_range))
    for i, k in enumerate(k_range):
        if k >= 0:
            P[i] = np.trace(p_mat_np, offset=k)     # sum j0=0..J-1-k, j1 = j0+k
        else:
            P[i] = np.trace(p_mat_np, offset=k)     # np.trace handles negative offset
    density = P / bin_width
    return density


# ── (A) Context sweep ─────────────────────────────────────────────────────────
_log("\n=== (A) Context sweep ===", t0)
X_obs_full  = sample['X_obs']
T_obs_full  = sample['T_obs']
Y_obs_full  = sample['Y_obs']
X_intv_sel  = sample['X_intv'][query_idx]

p_tau_A = {N: [] for N in CONTEXT_SIZES_A}   # smooth (MALC B=100)
E_tau_A = {N: [] for N in CONTEXT_SIZES_A}
K_A     = {N: [] for N in CONTEXT_SIZES_A}
raw_p_taus_N1000 = []
raw_p_mats_N1000 = []
for N in CONTEXT_SIZES_A:
    _log(f"N={N}", t0)
    X_obs  = X_obs_full[:N].unsqueeze(0).to(DEVICE)
    T_obs  = T_obs_full[:N].unsqueeze(0).to(DEVICE)
    Y_obs  = Y_obs_full[:N].unsqueeze(0).to(DEVICE)
    X_intv = X_intv_sel.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]

    for i, qi in enumerate(query_idx):
        p_mat, *_ = unpack_pred(pred[i], J, bin_width)
        p_mat_np  = p_mat.detach().cpu().numpy()
        fit = fit_malc_inner(
            p_mat_np.T, edges_np, edges_np,
            B_select=100, B_fit=100, seed=SCM_SEED + int(qi),
            parallel=False,
        )
        density = dmalc_2d(fit, eval_pts).reshape(N_EVAL, N_EVAL)
        p_tau_A[N].append(p_tau_from_malc_grid(density))
        E_tau_A[N].append(float((tau_smooth * p_tau_A[N][-1]).sum() * (tau_smooth[1] - tau_smooth[0])))
        K_A[N].append(int(fit.K))
        if N == 1000:
            raw_p_taus_N1000.append(p_tau_from_raw_p_mat(p_mat_np))
            raw_p_mats_N1000.append(p_mat_np)

# ── Plot A ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False)
axes = axes.reshape(-1)
cmap = plt.get_cmap('viridis')
colors = {N: cmap(k / (len(CONTEXT_SIZES_A) - 1)) for k, N in enumerate(CONTEXT_SIZES_A)}

for i, qi in enumerate(query_idx):
    ax = axes[i]
    true_t = true_tau_all[int(qi)]
    for N in CONTEXT_SIZES_A:
        ax.plot(tau_smooth, p_tau_A[N][i], color=colors[N], lw=1.6,
                label=f'N={N}  E[τ]={E_tau_A[N][i]:+.2f}  K={K_A[N][i]}')
    ax.axvline(true_t, color='red', ls='--', lw=1.2, alpha=0.7)
    ax.plot(true_t, 0, 'o', color='red', markersize=10, zorder=5, clip_on=False)
    ax.set_title(f"query {int(qi)}  true τ = {true_t:+.3f}")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper right')
    if i % 3 == 0: ax.set_ylabel(r'$p(\tau)$')
    if i // 3 == 1: ax.set_xlabel(r'$\tau = Y_{do(1)} - Y_{do(0)}$  (scaled)')

fig.suptitle(
    f"(A) Per-query TE distribution across context sizes  "
    f"(SCM seed={SCM_SEED}, true τ mean={true_tau_all.mean():+.2f}, std={true_tau_all.std():.2f})",
    y=1.00,
)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/A_context_sweep.png', dpi=140, bbox_inches='tight')
plt.close(fig)
_log(f"Saved: {OUT_DIR}/A_context_sweep.png", t0)


# ── (B) Raw p_mat vs MALC(B=100) vs MALC(B=500) at N=1000 ─────────────────────
_log("\n=== (B) Raw p_mat vs MALC (B=100, 500) at N=1000 ===", t0)
X_obs  = X_obs_full[:1000].unsqueeze(0).to(DEVICE)
T_obs  = T_obs_full[:1000].unsqueeze(0).to(DEVICE)
Y_obs  = Y_obs_full[:1000].unsqueeze(0).to(DEVICE)
X_intv = X_intv_sel.unsqueeze(0).to(DEVICE)
with torch.no_grad():
    pred1k = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]

p_tau_B_raw   = []
p_tau_B_malc  = {B: [] for B in MALC_BS_B}
K_B_malc      = {B: [] for B in MALC_BS_B}
for i, qi in enumerate(query_idx):
    p_mat, *_ = unpack_pred(pred1k[i], J, bin_width)
    p_mat_np  = p_mat.detach().cpu().numpy()
    p_tau_B_raw.append(p_tau_from_raw_p_mat(p_mat_np))
    for B in MALC_BS_B:
        q_t0 = time.time()
        fit = fit_malc_inner(
            p_mat_np.T, edges_np, edges_np,
            B_select=B, B_fit=B, seed=SCM_SEED + int(qi),
            parallel=False,
        )
        density = dmalc_2d(fit, eval_pts).reshape(N_EVAL, N_EVAL)
        p_tau_B_malc[B].append(p_tau_from_malc_grid(density))
        K_B_malc[B].append(int(fit.K))
        _log(f"  q{i+1}/6 idx={int(qi):3d}  B={B}  K={fit.K}  {time.time() - q_t0:.1f}s", t0)

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False)
axes = axes.reshape(-1)
for i, qi in enumerate(query_idx):
    ax = axes[i]
    true_t = true_tau_all[int(qi)]
    ax.plot(tau_raw, p_tau_B_raw[i], color='black', lw=1.6,
            label='raw p_mat (discrete)', drawstyle='steps-mid')
    for B, c in zip(MALC_BS_B, ['#FF7F0E', '#1F77B4']):
        ax.plot(tau_smooth, p_tau_B_malc[B][i], color=c, lw=1.6,
                label=f'MALC B={B}  K={K_B_malc[B][i]}')
    ax.axvline(true_t, color='red', ls='--', lw=1.2, alpha=0.7)
    ax.plot(true_t, 0, 'o', color='red', markersize=10, zorder=5, clip_on=False)
    ax.set_title(f"query {int(qi)}  true τ = {true_t:+.3f}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='upper right')
    if i % 3 == 0: ax.set_ylabel(r'$p(\tau)$')
    if i // 3 == 1: ax.set_xlabel(r'$\tau$  (scaled)')

fig.suptitle(
    f"(B) Raw p_mat vs MALC-smoothed at N=1000  "
    f"(SCM seed={SCM_SEED})",
    y=1.00,
)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/B_raw_vs_malc.png', dpi=140, bbox_inches='tight')
plt.close(fig)
_log(f"Saved: {OUT_DIR}/B_raw_vs_malc.png", t0)


# ── Save arrays ───────────────────────────────────────────────────────────────
np.save(f'{OUT_DIR}/tau_smooth.npy', tau_smooth)
np.save(f'{OUT_DIR}/tau_raw.npy',    tau_raw)
np.save(f'{OUT_DIR}/true_TE.npy',    true_tau_all[query_idx])
np.save(f'{OUT_DIR}/query_idx.npy',  query_idx)
for N in CONTEXT_SIZES_A:
    np.save(f'{OUT_DIR}/A_p_tau_N{N}.npy', np.stack(p_tau_A[N]))
np.save(f'{OUT_DIR}/B_p_tau_raw.npy', np.stack(p_tau_B_raw))
for B in MALC_BS_B:
    np.save(f'{OUT_DIR}/B_p_tau_malc_B{B}.npy', np.stack(p_tau_B_malc[B]))
_log(f"Saved arrays to {OUT_DIR}/", t0)
