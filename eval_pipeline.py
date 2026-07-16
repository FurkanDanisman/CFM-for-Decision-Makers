"""
End-to-end evaluation pipeline: 1 SCM, N queries → per-query TE distribution.

For each of --n-queries query points:
  1. Model forward pass (batched over all queries)  → J*J+9+4 raw logits
  2. Unpack                                          → p_mat (J,J), w_region (9), tails (4)
  3. MALC_2D fit                                     → smooth log-concave density on inner region
  4. Evaluate on 200×200 grid                        → f(y0,y1)
  5. Diagonal integration                            → p(τ) = ∫ f(y0, y0+τ) dy0

Plots a grid of TE densities with the true τ per query marked in red.

Env overrides:
  SCM_SEED        (default 2)      — deterministic SCM
  N_QUERIES       (default 6)      — number of queries to plot
  QUERY_SELECT    (default spread) — one of: spread | first | random | extreme
  CHECKPOINT      (default checkpoints/step_50000_final.pt)
  UWYK_SRC        (default /tmp/g4cfm_uwyk/src)
  OUT_DIR         (default eval_pipeline)
  MALC_B          (default 100)    — MALC EM synthetic points
  N_EVAL          (default 200)    — density grid resolution per axis
  N_TAU           (default 401)    — τ grid resolution
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
UWYK_SRC     = os.environ.get('UWYK_SRC', '/tmp/g4cfm_uwyk/src')
CHECKPOINT   = os.environ.get('CHECKPOINT', 'checkpoints/step_50000_final.pt')
OUT_DIR      = os.environ.get('OUT_DIR', 'eval_pipeline')
SCM_SEED     = int(os.environ.get('SCM_SEED', 2))
N_QUERIES    = int(os.environ.get('N_QUERIES', 6))
QUERY_SELECT = os.environ.get('QUERY_SELECT', 'spread')
MALC_B       = int(os.environ.get('MALC_B', 100))
N_EVAL       = int(os.environ.get('N_EVAL', 200))
N_TAU        = int(os.environ.get('N_TAU', 401))
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# path priority: repo root first so our models.InterventionalPFN wins over UWYK's
_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, UWYK_SRC)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'MALC'))

from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import unpack_pred, fit_malc_inner
from malc_2d import dmalc_2d
from eval_scm_gen import generate_paired_sample_with_raw

os.makedirs(OUT_DIR, exist_ok=True)


def _log(msg: str, t0: float | None = None) -> None:
    prefix = f"[{time.time() - t0:6.1f}s]" if t0 is not None else "[  0.0s]"
    print(f"{prefix} {msg}", flush=True)


# ── 1. Generate SCM (with raw + affine) ───────────────────────────────────────
t0 = time.time()
_log(f"Generating SCM (seed_base={SCM_SEED}, n_train=1000, n_test=500)…")
sample = generate_paired_sample_with_raw(
    scm_seed=SCM_SEED, idx=0, n_train=1000, n_test=500, max_features=50,
)
n_test_total = sample['X_intv'].shape[0]
ymin, ymax = sample['ymin'], sample['ymax']
y_range = (ymax - ymin)
half_range = y_range / 2.0
_log(f"Got {n_test_total} query points.  affine: ymin={ymin:.4f} ymax={ymax:.4f} "
     f"range={y_range:.4f}", t0)

# ── 2. Pick query indices ─────────────────────────────────────────────────────
# True TE in RAW units (SCM-native).
true_TE_raw_all    = (sample['Y_do1_raw'] - sample['Y_do0_raw']).reshape(-1).numpy()
# True TE in SCALED units (model space, for consistency with model output).
true_TE_scaled_all = (sample['Y_do1']     - sample['Y_do0']    ).reshape(-1).numpy()

def pick_indices(mode: str, n: int, total: int, te: np.ndarray) -> np.ndarray:
    if mode == 'first':
        return np.arange(n)
    if mode == 'random':
        rng = np.random.default_rng(SCM_SEED)
        return rng.choice(total, size=n, replace=False)
    if mode == 'spread':
        # even spread across the sorted true-TE range (informative diverse plot)
        order = np.argsort(te)
        quantiles = np.linspace(0.05, 0.95, n)
        return order[(quantiles * (total - 1)).astype(int)]
    if mode == 'extreme':
        order = np.argsort(np.abs(te))[::-1]
        return order[:n]
    raise ValueError(f"unknown QUERY_SELECT={mode!r}")

query_idx = pick_indices(QUERY_SELECT, N_QUERIES, n_test_total, true_TE_scaled_all)
_log(f"Selected query indices ({QUERY_SELECT}): {list(map(int, query_idx))}", t0)
_log(f"  true τ per selected query (raw):    "
     f"{true_TE_raw_all[query_idx].round(3).tolist()}", t0)
_log(f"  true τ per selected query (scaled): "
     f"{true_TE_scaled_all[query_idx].round(3).tolist()}", t0)

# ── 3. Load model ─────────────────────────────────────────────────────────────
_log(f"Loading checkpoint {CHECKPOINT}…", t0)
ckpt   = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
config = ckpt['config']
J      = config['J']
edges  = ckpt['edges'].to(DEVICE)
edges_np  = edges.detach().cpu().numpy()
bin_width = float(edges[1] - edges[0])

model = InterventionalPFN(
    num_features=config['num_features'],
    d_model=config['d_model'],
    depth=config['depth'],
    heads_feat=config['heads'],
    heads_samp=config['heads'],
    dropout=0.0,
    output_dim=J * J + 9 + 4,
    hidden_mult=config['hidden_mult'],
    normalize_features=True,
    normalize_treatment=False,
    use_treatment_in_query=False,
    use_checkpoint=False,
).to(DEVICE).eval()
model.load_state_dict(ckpt['model_state_dict'])
_log(f"Model loaded. J={J} bin_width={bin_width:.4f} device={DEVICE}", t0)

# ── 4. Batched forward pass over selected queries ────────────────────────────
_log(f"Forward pass over {len(query_idx)} queries…", t0)
X_obs  = sample['X_obs'].unsqueeze(0).to(DEVICE)                # (1, 1000, 50)
T_obs  = sample['T_obs'].unsqueeze(0).to(DEVICE)                # (1, 1000, 1)
Y_obs  = sample['Y_obs'].unsqueeze(0).to(DEVICE)                # (1, 1000, 1)
X_intv = sample['X_intv'][query_idx].unsqueeze(0).to(DEVICE)    # (1, Q, 50)

with torch.no_grad():
    pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]  # (Q, J*J+9+4)

# ── 5. Per-query: unpack, MALC fit, evaluate density, integrate along diagonal
_log(f"Per-query MALC fit (B={MALC_B}) + TE derivation…", t0)

xs = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
ys = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
XX, YY = np.meshgrid(xs, ys, indexing='xy')
eval_pts = np.column_stack([XX.ravel(), YY.ravel()])
dy0 = xs[1] - xs[0]
dy1 = ys[1] - ys[0]

tau_min = ys[0] - xs[-1]
tau_max = ys[-1] - xs[0]
tau = np.linspace(tau_min, tau_max, N_TAU)
dtau = tau[1] - tau[0]

def te_density_from_grid(density: np.ndarray) -> np.ndarray:
    """p(τ) = ∫ f(y0, y0+τ) dy0 via bilinear interpolation along diagonals."""
    out = np.zeros_like(tau)
    for k, t in enumerate(tau):
        y1_target = xs + t
        valid = (y1_target >= ys[0]) & (y1_target <= ys[-1])
        if not np.any(valid):
            continue
        col_idx = np.searchsorted(xs, xs[valid]) - 1
        col_idx = np.clip(col_idx, 0, len(xs) - 1)
        row_f  = (y1_target[valid] - ys[0]) / dy1
        row_lo = np.clip(np.floor(row_f).astype(int), 0, len(ys) - 2)
        row_hi = row_lo + 1
        w_hi   = row_f - row_lo
        w_lo   = 1.0 - w_hi
        f_diag = w_lo * density[row_lo, col_idx] + w_hi * density[row_hi, col_idx]
        out[k] = f_diag.sum() * dy0
    return out

per_query = []
for i, qi in enumerate(query_idx):
    q_t0 = time.time()
    p_mat, w_region, sL0, sR0, sL1, sR1 = unpack_pred(pred[i], J, bin_width)
    # Convention: our p_mat[a, b] = p(Y_do0=a, Y_do1=b).
    # MALC treats its input p_mat[i, j] as p(y=grid_y[i], x=grid_x[j]).
    # Transpose so MALC's x-axis is Y_do0 and y-axis is Y_do1 → downstream
    # naming (xs↔Y_do0, ys↔Y_do1, τ = y1 - y0) is consistent.
    p_mat_np    = p_mat.detach().cpu().numpy().T
    w_region_np = w_region.detach().cpu().numpy()

    fit = fit_malc_inner(
        p_mat_np, edges_np, edges_np,
        B_select=MALC_B, B_fit=MALC_B, seed=SCM_SEED + int(qi),
        parallel=False,
    )
    density = dmalc_2d(fit, eval_pts).reshape(N_EVAL, N_EVAL)
    # After transpose + this eval, density[i, j] = f(y_do0 = xs[j], y_do1 = ys[i]).
    p_tau = te_density_from_grid(density)

    # τ distribution in scaled space (native output of MALC diagonal integration)
    E_tau_s   = float((tau * p_tau).sum() * dtau)
    mass_s    = float(p_tau.sum() * dtau)
    true_ts   = float(true_TE_scaled_all[int(qi)])
    p_true_s  = float(np.interp(true_ts, tau, p_tau))

    # Convert τ distribution to RAW units. Only the outcome differences carry
    # units, so τ_raw = τ_scaled * (ymax-ymin)/2. Change of variables also
    # rescales the density: p_raw(τ_raw) = p_scaled(τ_scaled) / half_range.
    tau_raw   = tau * half_range
    p_tau_raw = p_tau / half_range
    E_tau_r   = float((tau_raw * p_tau_raw).sum() * (tau_raw[1] - tau_raw[0]))
    mass_r    = float(p_tau_raw.sum() * (tau_raw[1] - tau_raw[0]))
    true_tr   = float(true_TE_raw_all[int(qi)])
    p_true_r  = float(np.interp(true_tr, tau_raw, p_tau_raw))

    per_query.append({
        'query_idx':     int(qi),
        # scaled
        'true_TE_s':     true_ts,   'E_tau_s':  E_tau_s,   'p_true_s':  p_true_s,
        'p_tau_s':       p_tau,     'mass_s':   mass_s,
        # raw
        'true_TE_r':     true_tr,   'E_tau_r':  E_tau_r,   'p_true_r':  p_true_r,
        'p_tau_r':       p_tau_raw, 'mass_r':   mass_r,
        # MALC bookkeeping
        'K':             int(fit.K),
        'pi':            np.array(fit.pi).round(3).tolist(),
    })
    _log(f"  q{i+1}/{len(query_idx)}  idx={int(qi):>3}  "
         f"τ_true_raw={true_tr:+.3f}  E[τ]_raw={E_tau_r:+.3f}  "
         f"(scaled: τ_true={true_ts:+.3f}  E[τ]={E_tau_s:+.3f})  "
         f"K={fit.K}  ∫p={mass_r:.3f}  {time.time() - q_t0:.1f}s",
         t0)

# Save arrays — both scaled and raw
tau_raw_grid = tau * half_range
np.save(f'{OUT_DIR}/tau_grid_scaled.npy',     tau)
np.save(f'{OUT_DIR}/tau_grid_raw.npy',        tau_raw_grid)
np.save(f'{OUT_DIR}/te_densities_scaled.npy', np.stack([q['p_tau_s'] for q in per_query]))
np.save(f'{OUT_DIR}/te_densities_raw.npy',    np.stack([q['p_tau_r'] for q in per_query]))
np.save(f'{OUT_DIR}/true_TE_scaled.npy',      np.array([q['true_TE_s'] for q in per_query]))
np.save(f'{OUT_DIR}/true_TE_raw.npy',         np.array([q['true_TE_r'] for q in per_query]))
np.save(f'{OUT_DIR}/query_idx.npy',           np.array([q['query_idx'] for q in per_query]))
np.save(f'{OUT_DIR}/y_affine.npy',            np.array([ymin, ymax]))

# ── 6. Plot 2×3 (or whatever fits N_QUERIES) ──────────────────────────────────
n_row, n_col = 2, 3
if N_QUERIES != 6:
    n_col = int(np.ceil(np.sqrt(N_QUERIES)))
    n_row = int(np.ceil(N_QUERIES / n_col))

def _plot_grid(units: str, tau_axis: np.ndarray, key_p: str,
               key_true: str, key_E: str, key_p_true: str,
               out_path: str) -> None:
    fig, axes = plt.subplots(n_row, n_col, figsize=(4.5 * n_col, 3.5 * n_row))
    axes = np.array(axes).reshape(-1)
    for i, q in enumerate(per_query):
        ax = axes[i]
        ax.plot(tau_axis, q[key_p], color='steelblue', lw=1.7)
        ax.fill_between(tau_axis, 0, q[key_p], color='steelblue', alpha=0.15)
        ax.axvline(q[key_true], color='red', ls='--', lw=1, alpha=0.55)
        ax.plot(q[key_true], q[key_p_true], 'o', color='red',
                markersize=9, zorder=5)
        ax.axvline(q[key_E], color='navy', ls=':', lw=1, alpha=0.7)
        ax.set_title(
            f"query {q['query_idx']}  τ_true={q[key_true]:+.2f}  "
            f"E[τ]={q[key_E]:+.2f}  K={q['K']}"
        )
        ax.grid(alpha=0.3)
        if i % n_col == 0:
            ax.set_ylabel(r'$p(\tau)$')
        if i // n_col == n_row - 1:
            ax.set_xlabel(fr'$\tau$  ({units})')
    for j in range(len(per_query), len(axes)):
        axes[j].axis('off')
    fig.suptitle(
        f"MALC-derived TE distribution ({units})  "
        f"seed={SCM_SEED}, select={QUERY_SELECT}",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    _log(f"Saved: {out_path}", t0)


_plot_grid(
    'scaled [-1,1]', tau, 'p_tau_s',
    'true_TE_s', 'E_tau_s', 'p_true_s',
    f'{OUT_DIR}/te_grid_scaled.png',
)
_plot_grid(
    f'raw units  (ymin={ymin:.2f}, ymax={ymax:.2f})', tau_raw_grid, 'p_tau_r',
    'true_TE_r', 'E_tau_r', 'p_true_r',
    f'{OUT_DIR}/te_grid_raw.png',
)

# Also save individual context
torch.save({
    'X_obs':      sample['X_obs'],
    'T_obs':      sample['T_obs'],
    'Y_obs':      sample['Y_obs'],
    'X_intv':     sample['X_intv'][query_idx],
    'query_idx':  query_idx,
    'scm_seed':   SCM_SEED,
}, f'{OUT_DIR}/context.pt')
_log(f"Saved context: {OUT_DIR}/context.pt", t0)
