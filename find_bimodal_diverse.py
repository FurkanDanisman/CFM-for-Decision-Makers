"""
Find diverse bimodal TE distributions and plot them with forced MALC K=1/2/3.

Selection criteria (much stricter than before):
  - n_peaks >= 2 in raw p(τ)
  - |τ_true| > 0.15                          (skip near-zero-effect queries)
  - peak balance ratio > 0.30                (2nd peak ≥ 30% of 1st peak height)
  - peak separation > 0.20 in τ              (distinct modes, not one wide bump)

For the top hits:
  - fit MALC at K=1, K=2, K=3 explicitly (each forced, no BIC)
  - overlay all three on the raw p(τ)
  - red dot = true τ

Runs 3 stages, each independently checkpointed.
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
N_SCM      = int(os.environ.get('N_SCM', 12))
N_QUERIES  = int(os.environ.get('N_QUERIES', 500))
N_TOP      = int(os.environ.get('N_TOP', 9))
MALC_B     = int(os.environ.get('MALC_B', 1000))
KS         = [1, 2, 3]
N_EVAL     = 200
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, UWYK_SRC); sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, 'MALC'))

from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import unpack_pred, fit_malc_inner
from malc_2d import dmalc_2d
from eval_scm_gen import generate_paired_sample_with_raw

os.makedirs(OUT_DIR, exist_ok=True)
CACHE_DIR = os.path.join(OUT_DIR, 'cache_bimodal')
os.makedirs(CACHE_DIR, exist_ok=True)


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

xs = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
ys = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
XX, YY = np.meshgrid(xs, ys, indexing='xy')
eval_pts = np.column_stack([XX.ravel(), YY.ravel()])
tau_smooth = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], 401)
dy0 = xs[1] - xs[0]; dy1 = ys[1] - ys[0]


def p_tau_from_raw_p_mat(p_mat_np):
    P = np.array([np.trace(p_mat_np, offset=k) for k in k_range])
    return P / bin_width


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


def score_multimodal(p_tau, tau, prominence_frac=0.10, min_sep_bins=8):
    if p_tau.max() <= 0:
        return 0, np.array([]), np.array([]), 0.0
    prom = float(p_tau.max() * prominence_frac)
    peaks, _ = find_peaks(p_tau, prominence=prom, distance=min_sep_bins)
    heights = p_tau[peaks]
    positions = tau[peaks]
    if len(peaks) >= 2:
        top2 = np.argsort(heights)[-2:]
        pos_top2 = positions[top2]
        sep = float(abs(pos_top2[1] - pos_top2[0]))
    else:
        sep = 0.0
    return len(peaks), positions, heights, sep


def cached_malc_forced_K(p_mat_np, seed_hint, K, B):
    """MALC fit with FORCED K (no BIC), cached."""
    import hashlib
    h = hashlib.sha1(np.ascontiguousarray(p_mat_np).tobytes()).hexdigest()[:16]
    key = f"forced_{h}_B{B}_K{K}_s{seed_hint}"
    cache_p = os.path.join(CACHE_DIR, f"{key}.npz")
    if os.path.exists(cache_p):
        d = np.load(cache_p)
        return d['density']
    fit = fit_malc_inner(
        p_mat_np.T, edges_np, edges_np,
        K=K, B_fit=B, B_select=B,       # K set → BIC skipped
        seed=seed_hint, parallel=False,
    )
    density = dmalc_2d(fit, eval_pts).reshape(N_EVAL, N_EVAL)
    np.savez(cache_p, density=density, K=int(fit.K))
    return density


# ── (1) SCAN & FILTER ────────────────────────────────────────────────────────
_log(f"\n### SCAN across {N_SCM} SCMs × {N_QUERIES} queries ###", t0)

@dataclass
class Row:
    scm_seed: int
    query_idx: int
    true_tau: float
    n_peaks: int
    peak_positions: np.ndarray
    peak_heights: np.ndarray
    sep: float
    balance: float
    p_tau_raw: np.ndarray

rows = []
for scm_seed in range(N_SCM):
    try:
        s = generate_paired_sample_with_raw(
            scm_seed=scm_seed, idx=0, n_train=1000, n_test=N_QUERIES,
        )
    except Exception as e:
        _log(f"  seed={scm_seed}: SKIP", t0)
        continue
    true_tau_arr = (s['Y_do1'] - s['Y_do0']).reshape(-1).numpy()
    X_obs  = s['X_obs'].unsqueeze(0).to(DEVICE)
    T_obs  = s['T_obs'].unsqueeze(0).to(DEVICE)
    Y_obs  = s['Y_obs'].unsqueeze(0).to(DEVICE)
    X_intv = s['X_intv'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]
    n_bi = 0
    for i in range(pred.shape[0]):
        p_mat, *_ = unpack_pred(pred[i], J, bin_width)
        p_mat_np = p_mat.detach().cpu().numpy()
        p_t = p_tau_from_raw_p_mat(p_mat_np)
        n_peaks, pos, heights, sep = score_multimodal(p_t, tau_raw)
        balance = float(sorted(heights)[-2] / heights.max()) if len(heights) >= 2 else 0.0
        rows.append(Row(scm_seed, i, float(true_tau_arr[i]),
                        n_peaks, pos, heights, sep, balance, p_t))
        if n_peaks >= 2 and balance >= 0.3 and sep >= 0.2 and abs(true_tau_arr[i]) > 0.15:
            n_bi += 1
    _log(f"  seed={scm_seed}: {pred.shape[0]} queries, {n_bi} pass bimodal filter", t0)

# Apply filter and rank
candidates = [
    r for r in rows
    if r.n_peaks >= 2 and r.balance >= 0.3 and r.sep >= 0.2 and abs(r.true_tau) > 0.15
]
_log(f"\nTotal {len(candidates)} queries pass bimodal filter", t0)

# Sort so we get diverse τ_true — pick candidates spread across τ range
if len(candidates) == 0:
    # Relax filter progressively if strict criteria yield nothing
    _log("No candidates — relaxing balance to 0.2, |τ_true| to 0.10", t0)
    candidates = [
        r for r in rows
        if r.n_peaks >= 2 and r.balance >= 0.2 and r.sep >= 0.15 and abs(r.true_tau) > 0.10
    ]
    _log(f"After relaxation: {len(candidates)} candidates", t0)

if len(candidates) == 0:
    _log("Still no candidates. Falling back to top-|τ| bimodal (any balance).", t0)
    bi = [r for r in rows if r.n_peaks >= 2]
    bi.sort(key=lambda r: -abs(r.true_tau))
    candidates = bi[: max(1, N_TOP)]

# Pick N_TOP with maximum diversity in τ_true
if len(candidates) > N_TOP:
    candidates.sort(key=lambda r: r.true_tau)
    idx = np.linspace(0, len(candidates) - 1, N_TOP).astype(int)
    top = [candidates[i] for i in idx]
else:
    top = candidates

_log(f"Selected {len(top)} diverse bimodal examples:", t0)
for r in top:
    _log(f"  seed={r.scm_seed:2d} q={r.query_idx:3d}  τ_true={r.true_tau:+.3f}  "
         f"peaks@{r.peak_positions.round(2).tolist()}  balance={r.balance:.2f}  sep={r.sep:.2f}",
         t0)


# ── (2) FIT MALC K=1, K=2, K=3 (forced) FOR TOP N ─────────────────────────────
_log(f"\n### FORCED-K MALC FITS on {len(top)} top examples (K∈{KS}, B={MALC_B}) ###", t0)

top_by_scm = {}
for r in top:
    top_by_scm.setdefault(r.scm_seed, []).append(r)

results = {}
for scm_seed, seed_rows in top_by_scm.items():
    s = generate_paired_sample_with_raw(
        scm_seed=scm_seed, idx=0, n_train=1000, n_test=N_QUERIES,
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
        entry = {'raw': r.p_tau_raw}
        for K in KS:
            q_t0 = time.time()
            density = cached_malc_forced_K(p_mat_np, scm_seed + r.query_idx, K, MALC_B)
            entry[f'K{K}'] = p_tau_from_malc_grid(density)
            _log(f"  seed={r.scm_seed} q={r.query_idx:3d}  K={K}  {time.time() - q_t0:.1f}s", t0)
        results[(r.scm_seed, r.query_idx)] = entry


# ── (3) PLOT ─────────────────────────────────────────────────────────────────
n_col = 3
n_row = int(np.ceil(len(top) / n_col))
fig, axes = plt.subplots(n_row, n_col, figsize=(6 * n_col, 3.6 * n_row))
axes = np.array(axes).reshape(-1)
colors_K = {1: '#FFAA00', 2: '#009944', 3: '#0033CC'}
for i, r in enumerate(top):
    ax = axes[i]
    e = results[(r.scm_seed, r.query_idx)]
    ax.plot(tau_raw, e['raw'], color='black', lw=1.4,
            label=f'raw p_mat ({r.n_peaks} peaks)', drawstyle='steps-mid')
    for K in KS:
        ax.plot(tau_smooth, e[f'K{K}'], color=colors_K[K], lw=1.7,
                label=f'MALC K={K}', alpha=0.85)
    ax.axvline(r.true_tau, color='red', ls='--', lw=1.2, alpha=0.7)
    ax.plot(r.true_tau, 0, 'o', color='red', markersize=10, zorder=5, clip_on=False)
    ax.set_title(f"seed={r.scm_seed} q={r.query_idx}  τ_true={r.true_tau:+.3f}  "
                 f"balance={r.balance:.2f}", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper right')
    if i % n_col == 0: ax.set_ylabel(r'$p(\tau)$')
    if i // n_col == n_row - 1: ax.set_xlabel(r'$\tau$ (scaled)')

for j in range(len(top), len(axes)):
    axes[j].axis('off')
fig.suptitle(
    f"Diverse bimodal TE distributions with forced MALC K=1/2/3 "
    f"(B={MALC_B}, filter: |τ_true|>0.15, balance>0.3, sep>0.2)",
    y=1.00,
)
fig.tight_layout()
out_png = f'{OUT_DIR}/E4_diverse_bimodal_Kforced.png'
fig.savefig(out_png, dpi=140, bbox_inches='tight')
_log(f"\nSaved: {out_png}", t0)
_log(f"Total wall: {time.time() - t0:.1f}s", t0)
