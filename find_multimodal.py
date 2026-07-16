"""
Scan many (SCM, query) pairs to find TE distributions with clear multi-modal
structure — i.e. ones like query 447 where the model expresses distinct SCM
hypotheses as separate modes.

Two-stage:

  (1) FAST SCAN — for each (SCM, query), compute p(τ) from the RAW p_mat
      (no MALC, just discrete diagonal sum on the 100×100 grid). Score
      multimodality by counting well-separated peaks in the density and by
      the effective spread. Fast: ~ms per query.

  (2) DETAILED FIT — for the top-N most multimodal (query, SCM) pairs, do
      the full MALC B=100 fit + smooth density + integrate.

Plot the top 12 detailed cases in a 3×4 grid with the raw p(τ) overlaid.
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
OUT_DIR    = os.environ.get('OUT_DIR', 'find_multimodal')
N_SCM      = int(os.environ.get('N_SCM', 12))            # scms to scan
N_QUERIES  = int(os.environ.get('N_QUERIES', 500))       # queries per scm
N_TOP      = int(os.environ.get('N_TOP', 12))            # top multimodal to plot
MALC_B     = int(os.environ.get('MALC_B', 100))
N_EVAL     = int(os.environ.get('N_EVAL', 200))
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


def p_tau_from_raw_p_mat(p_mat_np):
    """Discrete diagonal sum. Returns density on tau_raw grid (length 2J-1)."""
    P = np.array([np.trace(p_mat_np, offset=k) for k in k_range])
    return P / bin_width


def score_multimodal(p_tau, tau, prominence_frac=0.10, min_sep_bins=8):
    """Count well-separated significant peaks. Prominence relative to max height.
    Return (n_peaks, peak_positions, peak_heights, spread)."""
    if p_tau.max() <= 0:
        return 0, np.array([]), np.array([]), 0.0
    prom = float(p_tau.max() * prominence_frac)
    peaks, props = find_peaks(p_tau, prominence=prom, distance=min_sep_bins)
    heights = p_tau[peaks]
    # Also compute spread (std)
    p_norm = p_tau / (p_tau.sum() + 1e-12)
    mean = float((tau * p_norm).sum())
    var  = float(((tau - mean) ** 2 * p_norm).sum())
    return len(peaks), tau[peaks], heights, float(np.sqrt(var))


# ── (1) FAST SCAN ─────────────────────────────────────────────────────────────
_log(f"\n=== FAST SCAN across {N_SCM} SCMs × up to {N_QUERIES} queries ===", t0)

@dataclass
class Row:
    scm_seed: int
    query_idx: int
    true_tau: float
    n_peaks: int
    peak_positions: np.ndarray
    peak_heights: np.ndarray
    spread: float
    p_tau_raw: np.ndarray

rows: list[Row] = []
scm_seeds_used = []
for scm_seed in range(N_SCM):
    try:
        sample = generate_paired_sample_with_raw(
            scm_seed=scm_seed, idx=0, n_train=1000, n_test=N_QUERIES,
        )
    except Exception as e:
        _log(f"  seed={scm_seed}: SKIP ({type(e).__name__})", t0)
        continue
    scm_seeds_used.append(scm_seed)
    true_tau = (sample['Y_do1'] - sample['Y_do0']).reshape(-1).numpy()

    X_obs  = sample['X_obs'].unsqueeze(0).to(DEVICE)
    T_obs  = sample['T_obs'].unsqueeze(0).to(DEVICE)
    Y_obs  = sample['Y_obs'].unsqueeze(0).to(DEVICE)
    X_intv = sample['X_intv'].unsqueeze(0).to(DEVICE)   # all N_QUERIES

    with torch.no_grad():
        pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]  # (Q, ...)

    Q = pred.shape[0]
    for i in range(Q):
        p_mat, *_ = unpack_pred(pred[i], J, bin_width)
        p_mat_np = p_mat.detach().cpu().numpy()
        p_t = p_tau_from_raw_p_mat(p_mat_np)
        n_peaks, peak_pos, peak_h, spread = score_multimodal(p_t, tau_raw)
        rows.append(Row(
            scm_seed=scm_seed, query_idx=i, true_tau=float(true_tau[i]),
            n_peaks=n_peaks, peak_positions=peak_pos,
            peak_heights=peak_h, spread=spread, p_tau_raw=p_t,
        ))
    _log(f"  seed={scm_seed}: {Q} queries scanned; "
         f"n_peaks histogram: {dict(zip(*np.unique([r.n_peaks for r in rows[-Q:]], return_counts=True)))}",
         t0)

_log(f"\nTotal scanned: {len(rows)} (query, SCM) pairs", t0)
peaks_all = np.array([r.n_peaks for r in rows])
_log(f"n_peaks distribution: {dict(zip(*np.unique(peaks_all, return_counts=True)))}", t0)

# Sort by n_peaks descending, then by spread descending → most multimodal first
rows.sort(key=lambda r: (-r.n_peaks, -r.spread))
top_rows = rows[:N_TOP]

_log(f"\nTop {N_TOP} multimodal (query, SCM) pairs:", t0)
for r in top_rows:
    _log(f"  seed={r.scm_seed:2d}  q={r.query_idx:3d}  true_τ={r.true_tau:+.3f}  "
         f"n_peaks={r.n_peaks}  spread={r.spread:.3f}  peak_pos={r.peak_positions.round(2).tolist()}",
         t0)


# ── (2) DETAILED MALC FIT for top-N ───────────────────────────────────────────
_log(f"\n=== DETAILED MALC FIT for top {N_TOP} ===", t0)

xs = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
ys = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
XX, YY = np.meshgrid(xs, ys, indexing='xy')
eval_pts = np.column_stack([XX.ravel(), YY.ravel()])
dy0 = xs[1] - xs[0]; dy1 = ys[1] - ys[0]
tau_smooth = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], 401)

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

# Re-generate SCMs and grab the specific query for each top row.
# Batch by SCM to save on generation cost.
top_by_scm: dict[int, list[Row]] = {}
for r in top_rows:
    top_by_scm.setdefault(r.scm_seed, []).append(r)

malc_results = {}
for scm_seed, seed_rows in top_by_scm.items():
    _log(f"  regenerating seed={scm_seed}…", t0)
    sample = generate_paired_sample_with_raw(
        scm_seed=scm_seed, idx=0, n_train=1000, n_test=N_QUERIES,
    )
    X_obs  = sample['X_obs'].unsqueeze(0).to(DEVICE)
    T_obs  = sample['T_obs'].unsqueeze(0).to(DEVICE)
    Y_obs  = sample['Y_obs'].unsqueeze(0).to(DEVICE)
    X_intv = sample['X_intv'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]

    for r in seed_rows:
        q_t0 = time.time()
        p_mat, *_ = unpack_pred(pred[r.query_idx], J, bin_width)
        p_mat_np = p_mat.detach().cpu().numpy()
        fit = fit_malc_inner(
            p_mat_np.T, edges_np, edges_np,
            B_select=MALC_B, B_fit=MALC_B, seed=scm_seed + r.query_idx,
            parallel=False,
        )
        density = dmalc_2d(fit, eval_pts).reshape(N_EVAL, N_EVAL)
        p_t_smooth = p_tau_from_malc_grid(density)
        malc_results[(scm_seed, r.query_idx)] = {
            'K': int(fit.K),
            'pi': np.array(fit.pi).round(3).tolist(),
            'p_tau_smooth': p_t_smooth,
            'p_mat_np': p_mat_np,
        }
        _log(f"    seed={scm_seed} q={r.query_idx:3d}  K={fit.K}  {time.time() - q_t0:.1f}s", t0)


# ── Plot top-N ────────────────────────────────────────────────────────────────
n_col = 4
n_row = int(np.ceil(N_TOP / n_col))
fig, axes = plt.subplots(n_row, n_col, figsize=(4.2 * n_col, 3.2 * n_row))
axes = np.array(axes).reshape(-1)
for i, r in enumerate(top_rows):
    ax = axes[i]
    m = malc_results[(r.scm_seed, r.query_idx)]
    ax.plot(tau_raw, r.p_tau_raw, color='black', lw=1.5,
            label=f'raw p_mat  ({r.n_peaks} peaks)', drawstyle='steps-mid')
    ax.plot(tau_smooth, m['p_tau_smooth'], color='steelblue', lw=1.6,
            label=f"MALC B={MALC_B}  K={m['K']}")
    ax.axvline(r.true_tau, color='red', ls='--', lw=1.2, alpha=0.7)
    ax.plot(r.true_tau, 0, 'o', color='red', markersize=9, zorder=5, clip_on=False)
    ax.set_title(f"seed={r.scm_seed} q={r.query_idx}  τ_true={r.true_tau:+.3f}", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper right')
    if i % n_col == 0: ax.set_ylabel(r'$p(\tau)$')
    if i // n_col == n_row - 1: ax.set_xlabel(r'$\tau$ (scaled)')

for j in range(N_TOP, len(axes)):
    axes[j].axis('off')

fig.suptitle(
    f"Top {N_TOP} most-multimodal TE distributions found in {len(rows)} scanned queries "
    f"across {len(scm_seeds_used)} SCMs", y=1.00,
)
fig.tight_layout()
out_png = f'{OUT_DIR}/top_multimodal.png'
fig.savefig(out_png, dpi=140, bbox_inches='tight')
_log(f"Saved: {out_png}", t0)

# Save the aggregate stats
np.save(f'{OUT_DIR}/n_peaks_all.npy', peaks_all)
np.save(f'{OUT_DIR}/spreads_all.npy', np.array([r.spread for r in rows]))
np.save(f'{OUT_DIR}/true_tau_all.npy', np.array([r.true_tau for r in rows]))
_log("Saved aggregates.", t0)


# ── Multimodality histogram plot ──────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
u, c = np.unique(peaks_all, return_counts=True)
ax.bar(u, c, color='steelblue', edgecolor='black')
ax.set_xlabel('# peaks in raw p(τ) (prominence ≥ 10% of max)')
ax.set_ylabel(f'# queries (of {len(peaks_all)})')
ax.set_title(f'Multimodality prevalence across {len(scm_seeds_used)} SCMs × {N_QUERIES} queries each')
for xi, ci in zip(u, c):
    ax.text(xi, ci, f'{ci}', ha='center', va='bottom')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/multimodal_prevalence.png', dpi=140, bbox_inches='tight')
_log(f"Saved: {OUT_DIR}/multimodal_prevalence.png", t0)
_log(f"\nDone.  total {time.time() - t0:.1f}s", t0)
