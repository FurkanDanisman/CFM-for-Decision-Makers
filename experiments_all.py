"""
Unified experiments script — MALC B=1000, max_K=3, EVERY plot saved to
   /Users/furkandanisman/R-PFN/experiments/

Runs three experiments in sequence:

  (E1) CONTEXT SWEEP at N = 50, 250, 1000, 5000
       For 6 spread queries from one SCM, plot p(τ) at each context size.
       Output: experiments/E1_context_sweep.png

  (E2) MULTIMODAL SEARCH across 12 SCMs × 500 queries (fast raw-p_mat scan),
       then detailed MALC fit on top-12 most multimodal.
       Output: experiments/E2_top_multimodal.png
               experiments/E2_prevalence.png

  (E3) MALC B COMPARISON — for the top-6 multimodal (query, SCM) pairs,
       compare MALC B=100 vs B=1000 vs raw p_mat.
       Output: experiments/E3_B_comparison.png

All MALC fits use B_select=B_fit=1000, max_K=3.
"""
from __future__ import annotations
import os, sys, time
from dataclasses import dataclass
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

UWYK_SRC   = os.environ.get('UWYK_SRC', '/tmp/g4cfm_uwyk/src')
CHECKPOINT = os.environ.get('CHECKPOINT', 'checkpoints/step_50000_final.pt')
OUT_DIR    = '/Users/furkandanisman/R-PFN/experiments'
SCM_SEED_A = 5    # E1 anchor SCM (this one had multimodal queries)
CONTEXT_SIZES = [50, 250, 1000]  # dropped 5000 (out-of-distribution + kept killing)
N_SCM_SCAN = int(os.environ.get('N_SCM_SCAN', 12))
N_QUERIES_SCAN = int(os.environ.get('N_QUERIES_SCAN', 500))
N_TOP_MULTIMODAL = 12
MALC_B = 1000
MALC_MAX_K = 3
N_EVAL = 200
N_TAU  = 401
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, UWYK_SRC); sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, 'MALC'))

from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import unpack_pred, fit_malc_inner
from malc_2d import dmalc_2d
from eval_scm_gen import generate_paired_sample_with_raw

os.makedirs(OUT_DIR, exist_ok=True)
CACHE_DIR = os.path.join(OUT_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)


def cached_malc(p_mat_np, seed_hint, B, max_K):
    """Cache each MALC fit's p_tau on disk keyed by content hash + params.
    Kills the pain of restarting a 40-min run from scratch."""
    import hashlib
    h = hashlib.sha1(np.ascontiguousarray(p_mat_np).tobytes()).hexdigest()[:16]
    key = f"{h}_B{B}_maxK{max_K}_s{seed_hint}"
    cache_p = os.path.join(CACHE_DIR, f"{key}.npz")
    if os.path.exists(cache_p):
        d = np.load(cache_p)
        return d['density'], int(d['K'])
    fit = fit_malc_inner(
        p_mat_np.T, edges_np, edges_np,
        B_select=B, B_fit=B, max_K=max_K,
        seed=seed_hint, parallel=False,
    )
    density = dmalc_2d(fit, eval_pts).reshape(N_EVAL, N_EVAL)
    np.savez(cache_p, density=density, K=int(fit.K))
    return density, int(fit.K)


def _log(msg, t0=None):
    t = f"[{time.time() - t0:6.1f}s]" if t0 is not None else "[  0.0s]"
    print(f"{t} {msg}", flush=True)


# ── Model ────────────────────────────────────────────────────────────────────
t0 = time.time()
_log(f"Loading model, MALC B={MALC_B}, max_K={MALC_MAX_K}")
ckpt   = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
config = ckpt['config']
J      = config['J']
edges  = ckpt['edges'].to(DEVICE)
edges_np = edges.detach().cpu().numpy()
bin_width = float(edges[1] - edges[0])
centers = 0.5 * (edges_np[:-1] + edges_np[1:])
k_range = np.arange(-J + 1, J)
tau_raw = k_range * bin_width

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


# ── Shared grids for MALC ─────────────────────────────────────────────────────
xs = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
ys = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
XX, YY = np.meshgrid(xs, ys, indexing='xy')
eval_pts = np.column_stack([XX.ravel(), YY.ravel()])
dy0 = xs[1] - xs[0]; dy1 = ys[1] - ys[0]
tau_smooth = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], N_TAU)


def p_tau_from_malc_grid(density):
    out = np.zeros_like(tau_smooth)
    for k, t in enumerate(tau_smooth):
        y1_target = xs + t
        valid = (y1_target >= ys[0]) & (y1_target <= ys[-1])
        if not np.any(valid): continue
        col_idx = np.clip(np.searchsorted(xs, xs[valid]) - 1, 0, len(xs) - 1)
        row_f  = (y1_target[valid] - ys[0]) / dy1
        row_lo = np.clip(np.floor(row_f).astype(int), 0, len(ys) - 2)
        row_hi = row_lo + 1
        w_hi   = row_f - row_lo; w_lo = 1.0 - w_hi
        f_diag = w_lo * density[row_lo, col_idx] + w_hi * density[row_hi, col_idx]
        out[k] = f_diag.sum() * dy0
    return out


def p_tau_from_raw_p_mat(p_mat_np):
    P = np.array([np.trace(p_mat_np, offset=k) for k in k_range])
    return P / bin_width


def score_multimodal(p_tau, tau, prominence_frac=0.10, min_sep_bins=8):
    if p_tau.max() <= 0: return 0, np.array([]), 0.0
    prom = float(p_tau.max() * prominence_frac)
    peaks, _ = find_peaks(p_tau, prominence=prom, distance=min_sep_bins)
    p_norm = p_tau / (p_tau.sum() + 1e-12)
    mean = float((tau * p_norm).sum())
    spread = float(np.sqrt(((tau - mean) ** 2 * p_norm).sum()))
    return len(peaks), tau[peaks], spread


def malc_fit(p_mat_np, seed_hint):
    return fit_malc_inner(
        p_mat_np.T, edges_np, edges_np,
        B_select=MALC_B, B_fit=MALC_B, max_K=MALC_MAX_K,
        seed=seed_hint, parallel=False,
    )


# ============================================================================
# E1: CONTEXT SWEEP
# ============================================================================
_log("\n### E1: CONTEXT SWEEP ###", t0)
n_ctx_max = max(CONTEXT_SIZES)
_log(f"Generating SCM seed={SCM_SEED_A} with n_train={n_ctx_max}, n_test=500…", t0)
sample = generate_paired_sample_with_raw(
    scm_seed=SCM_SEED_A, idx=0, n_train=n_ctx_max, n_test=500,
)
true_tau_all = (sample['Y_do1'] - sample['Y_do0']).reshape(-1).numpy()
_log(f"  true τ mean={true_tau_all.mean():+.3f} std={true_tau_all.std():.3f}", t0)

# E1 queries: mix of known-multimodal (from earlier scan on SCM seed=5) and
# spread by true τ, so every subplot has a chance to show shape changes.
# Multimodal queries in seed=5 (from previous scan): 447, 178, 129, 480, 92, 137
# Spread queries by true τ percentile.
order = np.argsort(true_tau_all)
spread_queries = order[(np.array([0.05, 0.5, 0.95]) * (len(true_tau_all) - 1)).astype(int)].tolist()
known_multimodal = [447, 178, 137]  # from earlier find_multimodal scan on SCM 5
# Take multimodal queries first, then fill with spread; unique.
query_idx = np.array(list(dict.fromkeys(known_multimodal + spread_queries))[:6])
_log(f"Queries: {list(map(int, query_idx))}  "
     f"(first 3 are known multimodal, last 3 are true-τ percentiles)", t0)
_log(f"  true τ: {true_tau_all[query_idx].round(3).tolist()}", t0)

X_obs_full, T_obs_full, Y_obs_full = sample['X_obs'], sample['T_obs'], sample['Y_obs']
X_intv_sel = sample['X_intv'][query_idx]

p_tau_E1 = {N: [] for N in CONTEXT_SIZES}
E_tau_E1 = {N: [] for N in CONTEXT_SIZES}
K_E1     = {N: [] for N in CONTEXT_SIZES}

for N in CONTEXT_SIZES:
    _log(f"  N={N}", t0)
    X_obs  = X_obs_full[:N].unsqueeze(0).to(DEVICE)
    T_obs  = T_obs_full[:N].unsqueeze(0).to(DEVICE)
    Y_obs  = Y_obs_full[:N].unsqueeze(0).to(DEVICE)
    X_intv = X_intv_sel.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]
    for i, qi in enumerate(query_idx):
        q_t0 = time.time()
        p_mat, *_ = unpack_pred(pred[i], J, bin_width)
        p_mat_np  = p_mat.detach().cpu().numpy()
        density, K = cached_malc(p_mat_np, SCM_SEED_A + int(qi) + 100000 * N, MALC_B, MALC_MAX_K)
        p_t = p_tau_from_malc_grid(density)
        p_tau_E1[N].append(p_t)
        E_tau_E1[N].append(float((tau_smooth * p_t).sum() * (tau_smooth[1] - tau_smooth[0])))
        K_E1[N].append(K)
        _log(f"    q{i+1}/6 idx={int(qi):3d}  K={K}  {time.time() - q_t0:.1f}s", t0)

# Plot E1
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.reshape(-1)
cmap = plt.get_cmap('viridis')
colors = {N: cmap(k / max(len(CONTEXT_SIZES) - 1, 1)) for k, N in enumerate(CONTEXT_SIZES)}
for i, qi in enumerate(query_idx):
    ax = axes[i]
    true_t = true_tau_all[int(qi)]
    for N in CONTEXT_SIZES:
        ax.plot(tau_smooth, p_tau_E1[N][i], color=colors[N], lw=1.6,
                label=f'N={N}  E[τ]={E_tau_E1[N][i]:+.2f}  K={K_E1[N][i]}')
    ax.axvline(true_t, color='red', ls='--', lw=1.2, alpha=0.7)
    ax.plot(true_t, 0, 'o', color='red', markersize=10, zorder=5, clip_on=False)
    ax.set_title(f"query {int(qi)}  true τ = {true_t:+.3f}")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper right')
    if i % 3 == 0: ax.set_ylabel(r'$p(\tau)$')
    if i // 3 == 1: ax.set_xlabel(r'$\tau$ (scaled)')

fig.suptitle(f"(E1) TE distribution across context sizes  "
             f"MALC B={MALC_B}, max_K={MALC_MAX_K}, SCM seed={SCM_SEED_A}", y=1.00)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/E1_context_sweep.png', dpi=140, bbox_inches='tight')
plt.close(fig)
_log(f"Saved: {OUT_DIR}/E1_context_sweep.png", t0)


# ============================================================================
# E2: MULTIMODAL SEARCH
# ============================================================================
_log("\n### E2: MULTIMODAL SEARCH ###", t0)

@dataclass
class Row:
    scm_seed: int
    query_idx: int
    true_tau: float
    n_peaks: int
    peak_positions: np.ndarray
    spread: float
    p_tau_raw: np.ndarray

rows: list[Row] = []
scm_seeds_used = []
for scm_seed in range(N_SCM_SCAN):
    try:
        s = generate_paired_sample_with_raw(
            scm_seed=scm_seed, idx=0, n_train=1000, n_test=N_QUERIES_SCAN,
        )
    except Exception as e:
        _log(f"  seed={scm_seed}: SKIP ({type(e).__name__})", t0)
        continue
    scm_seeds_used.append(scm_seed)
    true_tau = (s['Y_do1'] - s['Y_do0']).reshape(-1).numpy()
    X_obs  = s['X_obs'].unsqueeze(0).to(DEVICE)
    T_obs  = s['T_obs'].unsqueeze(0).to(DEVICE)
    Y_obs  = s['Y_obs'].unsqueeze(0).to(DEVICE)
    X_intv = s['X_intv'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]
    for i in range(pred.shape[0]):
        p_mat, *_ = unpack_pred(pred[i], J, bin_width)
        p_mat_np = p_mat.detach().cpu().numpy()
        p_t = p_tau_from_raw_p_mat(p_mat_np)
        n_peaks, peak_pos, spread = score_multimodal(p_t, tau_raw)
        rows.append(Row(scm_seed, i, float(true_tau[i]), n_peaks, peak_pos, spread, p_t))
    _log(f"  seed={scm_seed}: scanned {pred.shape[0]}", t0)

peaks_all = np.array([r.n_peaks for r in rows])
u, c = np.unique(peaks_all, return_counts=True)
_log(f"Total scanned {len(rows)} queries. n_peaks: {dict(zip(u, c))}", t0)

# Prevalence bar chart
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.bar(u, c, color='steelblue', edgecolor='black')
ax.set_xlabel('# peaks in raw p(τ) (prominence ≥ 10% of max)')
ax.set_ylabel(f'# queries (of {len(peaks_all)})')
ax.set_title(f'Multimodality prevalence: {len(scm_seeds_used)} SCMs × {N_QUERIES_SCAN} queries')
frac_multi = (peaks_all >= 2).mean()
ax.text(0.98, 0.95, f'multimodal fraction: {frac_multi*100:.1f}%',
        transform=ax.transAxes, ha='right', va='top',
        bbox=dict(facecolor='white', edgecolor='gray'))
for xi, ci in zip(u, c):
    ax.text(xi, ci, f'{ci}', ha='center', va='bottom')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/E2_prevalence.png', dpi=140, bbox_inches='tight')
plt.close(fig)
_log(f"Saved: {OUT_DIR}/E2_prevalence.png", t0)

# Top-N most multimodal — detailed MALC B=1000 fit
rows.sort(key=lambda r: (-r.n_peaks, -r.spread))
top_rows = rows[:N_TOP_MULTIMODAL]
_log(f"Detailed MALC B={MALC_B} fits on top {len(top_rows)} multimodal:", t0)

top_by_scm: dict[int, list[Row]] = {}
for r in top_rows:
    top_by_scm.setdefault(r.scm_seed, []).append(r)

malc_top = {}
for scm_seed, seed_rows in top_by_scm.items():
    s = generate_paired_sample_with_raw(
        scm_seed=scm_seed, idx=0, n_train=1000, n_test=N_QUERIES_SCAN,
    )
    X_obs  = s['X_obs'].unsqueeze(0).to(DEVICE)
    T_obs  = s['T_obs'].unsqueeze(0).to(DEVICE)
    Y_obs  = s['Y_obs'].unsqueeze(0).to(DEVICE)
    X_intv = s['X_intv'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]
    for r in seed_rows:
        q_t0 = time.time()
        p_mat, *_ = unpack_pred(pred[r.query_idx], J, bin_width)
        p_mat_np = p_mat.detach().cpu().numpy()
        density, K = cached_malc(p_mat_np, scm_seed + r.query_idx, MALC_B, MALC_MAX_K)
        p_t_smooth = p_tau_from_malc_grid(density)
        malc_top[(scm_seed, r.query_idx)] = {
            'K': K, 'p_tau_smooth': p_t_smooth,
        }
        _log(f"  seed={scm_seed} q={r.query_idx:3d}  K={K}  {time.time() - q_t0:.1f}s", t0)

n_col = 4
n_row = int(np.ceil(N_TOP_MULTIMODAL / n_col))
fig, axes = plt.subplots(n_row, n_col, figsize=(4.5 * n_col, 3.2 * n_row))
axes = np.array(axes).reshape(-1)
for i, r in enumerate(top_rows):
    ax = axes[i]
    m = malc_top[(r.scm_seed, r.query_idx)]
    ax.plot(tau_raw, r.p_tau_raw, color='black', lw=1.4,
            label=f'raw p_mat ({r.n_peaks} peaks)', drawstyle='steps-mid')
    ax.plot(tau_smooth, m['p_tau_smooth'], color='steelblue', lw=1.6,
            label=f"MALC B={MALC_B}  K={m['K']}")
    ax.axvline(r.true_tau, color='red', ls='--', lw=1.2, alpha=0.7)
    ax.plot(r.true_tau, 0, 'o', color='red', markersize=9, zorder=5, clip_on=False)
    ax.set_title(f"seed={r.scm_seed} q={r.query_idx}  τ_true={r.true_tau:+.3f}", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper right')
    if i % n_col == 0: ax.set_ylabel(r'$p(\tau)$')
    if i // n_col == n_row - 1: ax.set_xlabel(r'$\tau$ (scaled)')
for j in range(N_TOP_MULTIMODAL, len(axes)):
    axes[j].axis('off')
fig.suptitle(
    f"(E2) Top {N_TOP_MULTIMODAL} multimodal — MALC B={MALC_B}, max_K={MALC_MAX_K}", y=1.00)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/E2_top_multimodal.png', dpi=140, bbox_inches='tight')
plt.close(fig)
_log(f"Saved: {OUT_DIR}/E2_top_multimodal.png", t0)


# ============================================================================
# E3: MALC B COMPARISON (B=100 vs B=1000) on top-6 multimodal
# ============================================================================
_log("\n### E3: MALC B=100 vs B=1000 comparison ###", t0)

top6 = top_rows[:6]
top6_by_scm = {}
for r in top6:
    top6_by_scm.setdefault(r.scm_seed, []).append(r)

E3_data = {}
for scm_seed, seed_rows in top6_by_scm.items():
    s = generate_paired_sample_with_raw(
        scm_seed=scm_seed, idx=0, n_train=1000, n_test=N_QUERIES_SCAN,
    )
    X_obs  = s['X_obs'].unsqueeze(0).to(DEVICE)
    T_obs  = s['T_obs'].unsqueeze(0).to(DEVICE)
    Y_obs  = s['Y_obs'].unsqueeze(0).to(DEVICE)
    X_intv = s['X_intv'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]
    for r in seed_rows:
        p_mat, *_ = unpack_pred(pred[r.query_idx], J, bin_width)
        p_mat_np = p_mat.detach().cpu().numpy()

        entry = {'raw': r.p_tau_raw, 'true_tau': r.true_tau}
        for B_ in (100, MALC_B):
            q_t0 = time.time()
            density, K = cached_malc(p_mat_np, scm_seed + r.query_idx, B_, MALC_MAX_K)
            entry[f'B{B_}_p_tau'] = p_tau_from_malc_grid(density)
            entry[f'B{B_}_K']     = K
            _log(f"  seed={scm_seed} q={r.query_idx:3d}  B={B_}  K={K}  "
                 f"{time.time() - q_t0:.1f}s", t0)
        E3_data[(scm_seed, r.query_idx)] = entry

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.reshape(-1)
for i, r in enumerate(top6):
    ax = axes[i]
    d = E3_data[(r.scm_seed, r.query_idx)]
    ax.plot(tau_raw, d['raw'], color='black', lw=1.4,
            label=f'raw p_mat ({r.n_peaks} peaks)', drawstyle='steps-mid')
    ax.plot(tau_smooth, d['B100_p_tau'], color='#FF7F0E', lw=1.6,
            label=f"MALC B=100  K={d['B100_K']}")
    ax.plot(tau_smooth, d[f'B{MALC_B}_p_tau'], color='#1F77B4', lw=1.8,
            label=f"MALC B={MALC_B}  K={d[f'B{MALC_B}_K']}")
    ax.axvline(d['true_tau'], color='red', ls='--', lw=1.2, alpha=0.7)
    ax.plot(d['true_tau'], 0, 'o', color='red', markersize=10, zorder=5, clip_on=False)
    ax.set_title(f"seed={r.scm_seed} q={r.query_idx}  τ_true={d['true_tau']:+.3f}", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper right')
    if i % 3 == 0: ax.set_ylabel(r'$p(\tau)$')
    if i // 3 == 1: ax.set_xlabel(r'$\tau$ (scaled)')

fig.suptitle(f"(E3) MALC B=100 vs B={MALC_B}, max_K={MALC_MAX_K}  — top-6 multimodal", y=1.00)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/E3_B_comparison.png', dpi=140, bbox_inches='tight')
plt.close(fig)
_log(f"Saved: {OUT_DIR}/E3_B_comparison.png", t0)

_log(f"\n\nALL PLOTS SAVED TO: {OUT_DIR}/", t0)
_log(f"  E1_context_sweep.png     — TE distribution across N=50/250/1000/5000", t0)
_log(f"  E2_prevalence.png        — multimodality histogram across 6000 queries", t0)
_log(f"  E2_top_multimodal.png    — top-12 most multimodal queries with MALC B={MALC_B}", t0)
_log(f"  E3_B_comparison.png      — MALC B=100 vs B=1000 on top-6 multimodal", t0)
_log(f"Total wall: {time.time() - t0:.1f}s", t0)
