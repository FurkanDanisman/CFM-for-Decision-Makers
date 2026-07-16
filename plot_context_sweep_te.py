"""
Plot per-query TE distributions at multiple context sizes.

For one SCM, 6 spread queries: overlay p(τ) at N_context = 500 / 1000 / 2000
as colored lines. Red vertical marks true τ.

If more context shifts / sharpens the density toward the true τ, the model is
learning. If all three lines overlap, context size doesn't matter.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch
import matplotlib.pyplot as plt

UWYK_SRC   = os.environ.get('UWYK_SRC', '/tmp/g4cfm_uwyk/src')
CHECKPOINT = os.environ.get('CHECKPOINT', 'checkpoints/step_50000_final.pt')
OUT_DIR    = os.environ.get('OUT_DIR', 'eval_context_sweep')
SCM_SEED   = int(os.environ.get('SCM_SEED', 2))
CONTEXT_SIZES = [int(x) for x in os.environ.get('CONTEXT_SIZES', '500,1000,2000').split(',')]
MALC_B     = int(os.environ.get('MALC_B', 100))
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
_log(f"Loading checkpoint {CHECKPOINT}…")
ckpt   = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
config = ckpt['config']
J      = config['J']
edges  = ckpt['edges'].to(DEVICE)
edges_np = edges.detach().cpu().numpy()
bin_width = float(edges[1] - edges[0])

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


# ── One SCM with n_train >= max(CONTEXT_SIZES) ────────────────────────────────
n_ctx_max = max(CONTEXT_SIZES)
_log(f"Generating SCM seed={SCM_SEED} with n_train={n_ctx_max}, n_test=500…", t0)
sample = generate_paired_sample_with_raw(
    scm_seed=SCM_SEED, idx=0, n_train=n_ctx_max, n_test=500,
)
true_tau_all = (sample['Y_do1'] - sample['Y_do0']).reshape(-1).numpy()
_log(f"  true τ mean={true_tau_all.mean():+.3f}  std={true_tau_all.std():.3f}  "
     f"min={true_tau_all.min():+.3f}  max={true_tau_all.max():+.3f}", t0)


# ── Pick 6 spread queries by true τ percentile ───────────────────────────────
order = np.argsort(true_tau_all)
quantiles = np.linspace(0.05, 0.95, 6)
query_idx = order[(quantiles * (len(true_tau_all) - 1)).astype(int)]
_log(f"Selected queries: {list(map(int, query_idx))}", t0)
_log(f"  true τ per query: {true_tau_all[query_idx].round(3).tolist()}", t0)


# ── Set up MALC density evaluation grid & τ grid ──────────────────────────────
xs = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
ys = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
XX, YY = np.meshgrid(xs, ys, indexing='xy')
eval_pts = np.column_stack([XX.ravel(), YY.ravel()])
dy0 = xs[1] - xs[0]; dy1 = ys[1] - ys[0]

tau = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], N_TAU)

def te_density_from_grid(density):
    out = np.zeros_like(tau)
    for k, t in enumerate(tau):
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


# ── For each context size, run model + MALC per query ─────────────────────────
X_obs_full  = sample['X_obs']
T_obs_full  = sample['T_obs']
Y_obs_full  = sample['Y_obs']
X_intv_sel  = sample['X_intv'][query_idx]

# results[N][q_i] = p_tau array
results = {N: [] for N in CONTEXT_SIZES}
E_tau_per_N = {N: [] for N in CONTEXT_SIZES}

for N in CONTEXT_SIZES:
    _log(f"\nN_context = {N}", t0)
    X_obs  = X_obs_full[:N].unsqueeze(0).to(DEVICE)
    T_obs  = T_obs_full[:N].unsqueeze(0).to(DEVICE)
    Y_obs  = Y_obs_full[:N].unsqueeze(0).to(DEVICE)
    X_intv = X_intv_sel.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]

    for i, qi in enumerate(query_idx):
        q_t0 = time.time()
        p_mat, *_ = unpack_pred(pred[i], J, bin_width)
        # transpose so MALC's (x, y) = (Y_do0, Y_do1) matches downstream math
        p_mat_np = p_mat.detach().cpu().numpy().T
        fit = fit_malc_inner(
            p_mat_np, edges_np, edges_np,
            B_select=MALC_B, B_fit=MALC_B, seed=SCM_SEED + int(qi),
            parallel=False,
        )
        density = dmalc_2d(fit, eval_pts).reshape(N_EVAL, N_EVAL)
        p_tau = te_density_from_grid(density)
        E_tau = float((tau * p_tau).sum() * (tau[1] - tau[0]))
        results[N].append(p_tau)
        E_tau_per_N[N].append(E_tau)
        _log(f"  q{i+1}/6 idx={int(qi):3d} true={true_tau_all[int(qi)]:+.3f} "
             f"E[τ]={E_tau:+.3f} K={fit.K}  {time.time() - q_t0:.1f}s", t0)


# ── Plot 2×3 grid, overlay context sizes per subplot ─────────────────────────
_log("\nPlotting…", t0)
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
axes = axes.reshape(-1)
colors = {N: c for N, c in zip(CONTEXT_SIZES, ['#FFAA55', '#3388DD', '#003366'])}

for i, qi in enumerate(query_idx):
    ax = axes[i]
    true_t = true_tau_all[int(qi)]
    for N in CONTEXT_SIZES:
        p_tau = results[N][i]
        E_t   = E_tau_per_N[N][i]
        ax.plot(tau, p_tau, color=colors[N], lw=1.8, label=f'N={N}  E[τ]={E_t:+.2f}')
    ax.axvline(true_t, color='red', ls='--', lw=1.2, alpha=0.7)
    ax.plot(true_t, 0, 'o', color='red', markersize=10, zorder=5, clip_on=False)
    ax.set_title(f"query {int(qi)}  true τ = {true_t:+.3f}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    if i % 3 == 0:
        ax.set_ylabel(r'$p(\tau)$')
    if i // 3 == 1:
        ax.set_xlabel(r'$\tau = Y_{do(1)} - Y_{do(0)}$  (scaled)')
    # Zoom to region of interest
    lo = min(true_t, min(E_tau_per_N[N][i] for N in CONTEXT_SIZES)) - 1.0
    hi = max(true_t, max(E_tau_per_N[N][i] for N in CONTEXT_SIZES)) + 1.0
    ax.set_xlim(lo, hi)

fig.suptitle(
    f"Per-query TE distribution at increasing context size  "
    f"(SCM seed={SCM_SEED}, true τ mean={true_tau_all.mean():+.2f}, std={true_tau_all.std():.2f})",
    y=1.00,
)
fig.tight_layout()
out_png = f'{OUT_DIR}/te_by_context.png'
fig.savefig(out_png, dpi=140, bbox_inches='tight')
_log(f"Saved: {out_png}", t0)

# save arrays
np.save(f'{OUT_DIR}/tau_grid.npy', tau)
for N in CONTEXT_SIZES:
    np.save(f'{OUT_DIR}/p_tau_N{N}.npy', np.stack(results[N]))
    np.save(f'{OUT_DIR}/E_tau_N{N}.npy', np.array(E_tau_per_N[N]))
np.save(f'{OUT_DIR}/true_TE_selected.npy', true_tau_all[query_idx])
np.save(f'{OUT_DIR}/query_idx.npy', query_idx)
_log("Saved arrays to eval_context_sweep/", t0)
