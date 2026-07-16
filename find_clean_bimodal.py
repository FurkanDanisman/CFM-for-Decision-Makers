"""
Find genuinely clean bimodal TE cases and show MALC actually working.

Approach:
  1. CACHE the SCM samples on disk so re-runs use the SAME data
     (fixes the SCM-non-determinism problem that has been biting us).
  2. Scan cached SCMs for queries with GENUINELY bimodal raw p(τ):
       - ≥ 2 peaks with balance ≥ 0.30 and separation ≥ 0.25
       - |τ_true| > 0.05
       - true τ lands within 0.25 of one of the two peaks
     (last criterion picks cases where the paper's claim "true τ falls in
      one hypothesis mode" is testable).
  3. Rank by cleanness = balance × separation.
  4. For top-4:
       - 2D heatmap: raw p_mat, MALC K=1, MALC K=2 (forced), MALC BIC-selected
       - 1D p(τ) overlay: raw, MALC(K=BIC), red dot at true τ
  5. Pick the #1 cleanest, run context sweep at N = 100 / 500 / 1000 / 2000
     showing MALC(K=BIC) evolves with more data.

Everything saved to experiments/CLEAN_*.png
"""
from __future__ import annotations
import os, sys, time, pickle, hashlib
from dataclasses import dataclass
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

UWYK_SRC   = os.environ.get('UWYK_SRC', '/tmp/g4cfm_uwyk/src')
CHECKPOINT = 'checkpoints/step_50000_final.pt'
OUT_DIR    = '/Users/furkandanisman/R-PFN/experiments'
SCM_CACHE_DIR = os.path.join(OUT_DIR, 'scm_cache')
MALC_CACHE_DIR = os.path.join(OUT_DIR, 'malc_cache_clean')
N_SCM      = int(os.environ.get('N_SCM', 20))
N_QUERIES  = 500
N_TRAIN    = 2000              # generate max needed for context sweep
N_TOP      = 4
CONTEXT_SIZES_SWEEP = [100, 500, 1000, 2000]
MALC_B     = 1000
MALC_MAX_K = 3
N_EVAL     = 200
DEVICE     = torch.device('cpu')

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, UWYK_SRC); sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, 'MALC'))

from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import unpack_pred, fit_malc_inner
from malc_2d import dmalc_2d
from eval_scm_gen import generate_paired_sample_with_raw

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SCM_CACHE_DIR, exist_ok=True)
os.makedirs(MALC_CACHE_DIR, exist_ok=True)


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
dy0 = xs[1] - xs[0]; dy1 = ys[1] - ys[0]
tau_smooth = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], 401)


# ── SCM sample caching (fixes non-determinism) ────────────────────────────────
def get_scm_sample(scm_seed):
    cache_p = os.path.join(SCM_CACHE_DIR, f'scm_seed{scm_seed}_n{N_TRAIN}.pt')
    if os.path.exists(cache_p):
        return torch.load(cache_p, weights_only=False)
    s = generate_paired_sample_with_raw(
        scm_seed=scm_seed, idx=0, n_train=N_TRAIN, n_test=N_QUERIES,
    )
    torch.save(s, cache_p)
    return s


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


def cached_malc(p_mat_np, seed_hint, K=None, B=MALC_B, max_K=MALC_MAX_K):
    h = hashlib.sha1(np.ascontiguousarray(p_mat_np).tobytes()).hexdigest()[:16]
    key = f"{h}_B{B}_K{K if K is not None else f'bicMK{max_K}'}_s{seed_hint}"
    cache_p = os.path.join(MALC_CACHE_DIR, f"{key}.npz")
    if os.path.exists(cache_p):
        d = np.load(cache_p)
        return d['density'], int(d['K'])
    kw = dict(B_fit=B, B_select=B, seed=seed_hint, parallel=False)
    if K is not None:
        fit = fit_malc_inner(p_mat_np.T, edges_np, edges_np, K=K, **kw)
    else:
        fit = fit_malc_inner(p_mat_np.T, edges_np, edges_np, max_K=max_K, **kw)
    density = dmalc_2d(fit, eval_pts).reshape(N_EVAL, N_EVAL)
    np.savez(cache_p, density=density, K=int(fit.K))
    return density, int(fit.K)


def score_peaks(p_tau, tau, prominence_frac=0.10, min_sep_bins=8):
    if p_tau.max() <= 0: return 0, np.array([]), np.array([]), 0.0
    prom = float(p_tau.max() * prominence_frac)
    peaks, _ = find_peaks(p_tau, prominence=prom, distance=min_sep_bins)
    heights = p_tau[peaks]
    positions = tau[peaks]
    if len(peaks) >= 2:
        top2 = np.argsort(heights)[-2:]
        sep = float(abs(positions[top2[1]] - positions[top2[0]]))
    else:
        sep = 0.0
    return len(peaks), positions, heights, sep


# ── SCAN cached SCMs at N_train context ───────────────────────────────────────
_log(f"\n### SCAN across {N_SCM} SCMs × {N_QUERIES} queries ###", t0)

@dataclass
class Row:
    scm_seed: int
    q: int
    true_tau: float
    n_peaks: int
    peak_positions: np.ndarray
    peak_heights: np.ndarray
    sep: float
    balance: float
    dist_to_mode: float
    p_tau_raw: np.ndarray
    p_mat_np: np.ndarray

rows = []
for scm_seed in range(N_SCM):
    try:
        s = get_scm_sample(scm_seed)
    except Exception as e:
        _log(f"  seed={scm_seed}: SKIP ({type(e).__name__}: {str(e)[:60]})", t0)
        continue
    true_tau_arr = (s['Y_do1'] - s['Y_do0']).reshape(-1).numpy()
    # Use N_TRAIN=2000 for the scan (out-of-distribution but consistent)
    X_obs  = s['X_obs'][:1000].unsqueeze(0).to(DEVICE)      # scan at N=1000
    T_obs  = s['T_obs'][:1000].unsqueeze(0).to(DEVICE)
    Y_obs  = s['Y_obs'][:1000].unsqueeze(0).to(DEVICE)
    X_intv = s['X_intv'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]
    n_bi = 0
    for i in range(pred.shape[0]):
        p_mat, *_ = unpack_pred(pred[i], J, bin_width)
        p_mat_np = p_mat.detach().cpu().numpy()
        p_t = p_tau_from_raw_p_mat(p_mat_np)
        n_peaks, pos, heights, sep = score_peaks(p_t, tau_raw)
        balance = float(sorted(heights)[-2] / heights.max()) if len(heights) >= 2 else 0.0
        # distance from true τ to nearest top-2 peak
        if len(pos) >= 2:
            top2 = np.argsort(heights)[-2:]
            dist_mode = float(min(abs(float(true_tau_arr[i]) - pos[top2[0]]),
                                  abs(float(true_tau_arr[i]) - pos[top2[1]])))
        elif len(pos) == 1:
            dist_mode = float(abs(float(true_tau_arr[i]) - pos[0]))
        else:
            dist_mode = np.inf
        rows.append(Row(scm_seed, i, float(true_tau_arr[i]),
                        n_peaks, pos, heights, sep, balance, dist_mode,
                        p_t, p_mat_np))
        if (n_peaks >= 2 and balance >= 0.30 and sep >= 0.25
            and abs(true_tau_arr[i]) > 0.05 and dist_mode <= 0.25):
            n_bi += 1
    _log(f"  seed={scm_seed:2d}: {pred.shape[0]} q → {n_bi} pass strict bimodal filter", t0)

candidates = [
    r for r in rows
    if r.n_peaks >= 2 and r.balance >= 0.30 and r.sep >= 0.25
    and abs(r.true_tau) > 0.05 and r.dist_to_mode <= 0.25
]
_log(f"\nTotal bimodal candidates: {len(candidates)}", t0)

if len(candidates) < 2:
    _log("Relaxing filter (balance ≥ 0.25, sep ≥ 0.20, dist ≤ 0.30)", t0)
    candidates = [
        r for r in rows
        if r.n_peaks >= 2 and r.balance >= 0.25 and r.sep >= 0.20
        and abs(r.true_tau) > 0.05 and r.dist_to_mode <= 0.30
    ]
    _log(f"After relaxation: {len(candidates)}", t0)

# Rank by cleanness = balance × sep × 1/(1+dist_to_mode)
candidates.sort(key=lambda r: -(r.balance * r.sep / (1 + r.dist_to_mode)))
top = candidates[:N_TOP]
_log(f"\nTop {len(top)} cleanest bimodal cases:", t0)
for r in top:
    _log(f"  seed={r.scm_seed:2d} q={r.q:3d}  τ_true={r.true_tau:+.3f}  "
         f"peaks@{r.peak_positions.round(2).tolist()}  balance={r.balance:.2f}  "
         f"sep={r.sep:.2f}  dist_true_to_mode={r.dist_to_mode:.2f}", t0)


# ── For top-N: raw p_mat + MALC (K=1, K=2, K=BIC) 2D + p(τ) overlay ──────────
_log(f"\n### DETAILED FITS on top {len(top)} ###", t0)

detailed = {}
for r in top:
    _log(f"  seed={r.scm_seed} q={r.q}: fitting K=1, K=2, BIC(max_K=3)…", t0)
    K1_dens, _ = cached_malc(r.p_mat_np, r.scm_seed + r.q, K=1)
    K2_dens, _ = cached_malc(r.p_mat_np, r.scm_seed + r.q, K=2)
    Kb_dens, Kb = cached_malc(r.p_mat_np, r.scm_seed + r.q)
    detailed[(r.scm_seed, r.q)] = {
        'K1': K1_dens, 'K2': K2_dens, 'Kbic': Kb_dens, 'Kb': Kb,
    }
    _log(f"    BIC picked K={Kb}", t0)

# Plot: 4 rows (top-4 examples) × 4 cols (raw, K=1, K=2, K=BIC) heatmaps + p(τ) row
for i, r in enumerate(top):
    d = detailed[(r.scm_seed, r.q)]
    # Build raw density heatmap
    raw_2d = np.zeros((N_EVAL, N_EVAL))
    for ii in range(N_EVAL):
        for jj in range(N_EVAL):
            b0 = int(np.clip((xs[jj] - edges_np[0]) / bin_width, 0, J - 1))
            b1 = int(np.clip((ys[ii] - edges_np[0]) / bin_width, 0, J - 1))
            raw_2d[ii, jj] = r.p_mat_np[b0, b1] / (bin_width ** 2)

    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
    vmax = max(raw_2d.max(), d['K1'].max(), d['K2'].max(), d['Kbic'].max())
    for ax, arr, ttl in zip(
        axes[:4],
        [raw_2d, d['K1'], d['K2'], d['Kbic']],
        ['RAW p_mat (density)', 'MALC K=1', 'MALC K=2',
         f"MALC K={d['Kb']} (BIC)"],
    ):
        im = ax.imshow(arr, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin='lower',
                       cmap='viridis', vmin=0, vmax=vmax, aspect='equal')
        ax.plot([xs[0], xs[-1]], [ys[0], ys[-1]], 'w--', lw=0.7, alpha=0.4)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel('Y_do0'); ax.set_ylabel('Y_do1')
        plt.colorbar(im, ax=ax, shrink=0.7)

    # p(τ) overlay
    ax = axes[4]
    ax.plot(tau_raw, r.p_tau_raw, color='black', lw=1.4,
            label=f'raw p_mat ({r.n_peaks} peaks)', drawstyle='steps-mid')
    ax.plot(tau_smooth, p_tau_from_malc_grid(d['K1']), color='#FFAA00', lw=1.5,
            label='MALC K=1', alpha=0.7)
    ax.plot(tau_smooth, p_tau_from_malc_grid(d['K2']), color='#009944', lw=1.7,
            label='MALC K=2')
    ax.plot(tau_smooth, p_tau_from_malc_grid(d['Kbic']), color='#0033CC', lw=1.8,
            label=f"MALC K={d['Kb']} (BIC)", alpha=0.9)
    ax.axvline(r.true_tau, color='red', ls='--', lw=1.2, alpha=0.7)
    ax.plot(r.true_tau, 0, 'o', color='red', markersize=10, zorder=5, clip_on=False)
    ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel('τ (scaled)'); ax.set_ylabel('p(τ)')
    ax.set_title(f'τ marginal  τ_true={r.true_tau:+.3f}', fontsize=10)
    ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3)

    fig.suptitle(
        f'CLEAN bimodal example #{i+1}:  seed={r.scm_seed} q={r.q}  '
        f'balance={r.balance:.2f}  sep={r.sep:.2f}  dist_true_to_mode={r.dist_to_mode:.2f}',
        y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'CLEAN_bimodal_{i+1}_seed{r.scm_seed}_q{r.q}.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    _log(f"Saved: {out}", t0)


# ── Context sweep on the #1 example ──────────────────────────────────────────
if len(top) > 0:
    best = top[0]
    _log(f"\n### CONTEXT SWEEP on best example: seed={best.scm_seed} q={best.q} ###", t0)
    s = get_scm_sample(best.scm_seed)
    p_taus_by_N = {}
    Ks_by_N = {}
    for N in CONTEXT_SIZES_SWEEP:
        if N > s['X_obs'].shape[0]:
            _log(f"  N={N} skipped (only {s['X_obs'].shape[0]} rows)", t0)
            continue
        X_obs  = s['X_obs'][:N].unsqueeze(0).to(DEVICE)
        T_obs  = s['T_obs'][:N].unsqueeze(0).to(DEVICE)
        Y_obs  = s['Y_obs'][:N].unsqueeze(0).to(DEVICE)
        X_intv = s['X_intv'][best.q:best.q+1].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]
        p_mat, *_ = unpack_pred(pred[0], J, bin_width)
        p_mat_np = p_mat.detach().cpu().numpy()
        density, K = cached_malc(p_mat_np, best.scm_seed + best.q + 1000 * N)
        p_taus_by_N[N] = (p_tau_from_raw_p_mat(p_mat_np),
                          p_tau_from_malc_grid(density), K)
        _log(f"  N={N}: MALC K={K}", t0)

    fig, axes = plt.subplots(1, len(p_taus_by_N), figsize=(5 * len(p_taus_by_N), 4.5),
                              sharey=True, sharex=True)
    if len(p_taus_by_N) == 1: axes = [axes]
    for ax, (N, (p_r, p_m, K)) in zip(axes, sorted(p_taus_by_N.items())):
        ax.plot(tau_raw, p_r, color='black', lw=1.3,
                label='raw p_mat', drawstyle='steps-mid')
        ax.plot(tau_smooth, p_m, color='#0033CC', lw=1.8,
                label=f'MALC K={K} (BIC)')
        ax.axvline(best.true_tau, color='red', ls='--', lw=1.2, alpha=0.7)
        ax.plot(best.true_tau, 0, 'o', color='red', markersize=10, zorder=5, clip_on=False)
        ax.set_xlim(-1.5, 1.5)
        ax.set_title(f'N_context = {N}   MALC K={K}', fontsize=10)
        ax.set_xlabel('τ (scaled)')
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='upper right')
    axes[0].set_ylabel('p(τ)')
    fig.suptitle(
        f'CLEAN bimodal example — context sweep  seed={best.scm_seed} q={best.q}  '
        f'τ_true={best.true_tau:+.3f}',
        y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'CLEAN_context_sweep.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    _log(f"Saved: {out}", t0)

_log(f"\nDone. Total {time.time() - t0:.1f}s", t0)
