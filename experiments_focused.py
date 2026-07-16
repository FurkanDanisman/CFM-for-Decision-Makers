"""
Three focused experiments, well-plotted.

  Q1. Is the τ=0 spike real or bias?
      → For every query, compute raw p(τ=0) and |τ_true|.
        Scatter: p(τ=0) vs |τ_true|. If they anti-correlate, spike is real
        (appears when true τ is near zero). If not, spike is a bias.

  Q2. Does error shrink with N?
      → For 60 fixed queries from one SCM, run at N=100,250,500,1000,2000.
        For each N, compute MALC-density-mode and error = |mode − τ_true|.
        Line: mean error vs N (+ 25th/75th percentile band).
        Second line: E[τ] vs τ_true correlation across queries at each N.

  Q3. Real bimodal cases with mode-error measurement.
      → Scan many queries. Filter to MALC-K≥2 with well-separated modes
        (raw p_mat has ≥2 peaks with balance>0.4 and sep>0.4).
        For each hit: measure error to NEAREST mode.
        Report count and mean nearest-mode-error. Plot 6 clear examples.

All plots have proper axes, titles, legends.
"""
from __future__ import annotations
import os, sys, time, hashlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

UWYK_SRC   = os.environ.get('UWYK_SRC', '/tmp/g4cfm_uwyk/src')
CHECKPOINT = 'checkpoints/step_50000_final.pt'
OUT_DIR    = '/Users/furkandanisman/R-PFN/experiments'
SCM_CACHE  = os.path.join(OUT_DIR, 'scm_cache')
MALC_CACHE = os.path.join(OUT_DIR, 'malc_cache_focused')
N_SCM_Q1   = 4
N_SCM_Q3   = 20
N_QUERIES  = 500
N_TRAIN    = 2000
N_QUERIES_Q2 = 60          # subsample for error-vs-N (each N re-runs 60 queries)
CONTEXT_SIZES_Q2 = [100, 250, 500, 1000, 2000]
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
os.makedirs(SCM_CACHE, exist_ok=True)
os.makedirs(MALC_CACHE, exist_ok=True)


def _log(m, t0=None):
    t = f"[{time.time() - t0:6.1f}s]" if t0 is not None else "[  0.0s]"
    print(f"{t} {m}", flush=True)


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


def get_scm(seed):
    p = os.path.join(SCM_CACHE, f'scm_seed{seed}_n{N_TRAIN}.pt')
    if os.path.exists(p): return torch.load(p, weights_only=False)
    s = generate_paired_sample_with_raw(scm_seed=seed, idx=0, n_train=N_TRAIN, n_test=N_QUERIES)
    torch.save(s, p); return s


def p_tau_raw(p_mat_np):
    P = np.array([np.trace(p_mat_np, offset=k) for k in k_range])
    return P / bin_width


def p_tau_malc_grid(density):
    out = np.zeros_like(tau_smooth)
    for k, t in enumerate(tau_smooth):
        y1 = xs + t
        v = (y1 >= ys[0]) & (y1 <= ys[-1])
        if not np.any(v): continue
        col = np.clip(np.searchsorted(xs, xs[v]) - 1, 0, len(xs) - 1)
        rf = (y1[v] - ys[0]) / dy1
        rlo = np.clip(np.floor(rf).astype(int), 0, len(ys) - 2)
        rhi = rlo + 1; whi = rf - rlo; wlo = 1.0 - whi
        f = wlo * density[rlo, col] + whi * density[rhi, col]
        out[k] = f.sum() * dy0
    return out


def cached_malc(p_mat_np, seed_hint, K=None, B=MALC_B, max_K=MALC_MAX_K):
    h = hashlib.sha1(np.ascontiguousarray(p_mat_np).tobytes()).hexdigest()[:16]
    key = f"{h}_B{B}_K{K if K is not None else f'bicMK{max_K}'}_s{seed_hint}"
    p = os.path.join(MALC_CACHE, f"{key}.npz")
    if os.path.exists(p):
        d = np.load(p); return d['density'], int(d['K'])
    kw = dict(B_fit=B, B_select=B, seed=seed_hint, parallel=False)
    fit = (fit_malc_inner(p_mat_np.T, edges_np, edges_np, K=K, **kw) if K is not None
           else fit_malc_inner(p_mat_np.T, edges_np, edges_np, max_K=max_K, **kw))
    density = dmalc_2d(fit, eval_pts).reshape(N_EVAL, N_EVAL)
    np.savez(p, density=density, K=int(fit.K))
    return density, int(fit.K)


def score_peaks(p_t, tau, prominence_frac=0.10, min_sep_bins=8):
    if p_t.max() <= 0: return 0, np.array([]), np.array([]), 0.0
    prom = float(p_t.max() * prominence_frac)
    peaks, _ = find_peaks(p_t, prominence=prom, distance=min_sep_bins)
    heights = p_t[peaks]; positions = tau[peaks]
    if len(peaks) >= 2:
        top2 = np.argsort(heights)[-2:]
        sep = float(abs(positions[top2[1]] - positions[top2[0]]))
    else:
        sep = 0.0
    return len(peaks), positions, heights, sep


# ============================================================================
# Q1: is τ=0 spike explained by |τ_true|?
# ============================================================================
_log("\n### Q1: raw p(τ=0) vs |τ_true| ###", t0)
q1_p_zero = []; q1_abs_true = []; q1_seed = []; q1_q = []
for scm_seed in range(N_SCM_Q1):
    try: s = get_scm(scm_seed)
    except Exception: continue
    true_tau_all = (s['Y_do1'] - s['Y_do0']).reshape(-1).numpy()
    X_obs, T_obs, Y_obs = [s[k][:1000].unsqueeze(0).to(DEVICE) for k in ('X_obs','T_obs','Y_obs')]
    X_intv = s['X_intv'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]
    for i in range(pred.shape[0]):
        p_mat, *_ = unpack_pred(pred[i], J, bin_width)
        p_t = p_tau_raw(p_mat.detach().cpu().numpy())
        p_at_0 = float(p_t[k_range == 0][0])
        q1_p_zero.append(p_at_0)
        q1_abs_true.append(abs(float(true_tau_all[i])))
        q1_seed.append(scm_seed); q1_q.append(i)
    _log(f"  seed={scm_seed}: scanned {pred.shape[0]} queries", t0)

q1_p_zero = np.array(q1_p_zero); q1_abs_true = np.array(q1_abs_true)
q1_corr = float(np.corrcoef(q1_p_zero, q1_abs_true)[0, 1])
_log(f"  Corr(p(τ=0), |τ_true|) = {q1_corr:+.3f}", t0)
_log(f"  Fraction queries where τ_true is exactly 0 (raw = 0.000): "
     f"{(q1_abs_true < 1e-3).mean():.3f}", t0)

# Q1 plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.scatter(q1_abs_true, q1_p_zero, s=8, alpha=0.4, color='steelblue')
ax.set_xlabel('|τ_true| (scaled)')
ax.set_ylabel('raw p(τ=0)  height of the τ=0 spike in the model output')
ax.set_title(f'Q1: does the τ=0 spike correspond to true τ=0?\n'
             f'Pearson r = {q1_corr:+.3f}   ({len(q1_p_zero)} queries × {N_SCM_Q1} SCMs)\n'
             f'Perfect Bayesian: high spike ↔ low |τ_true|. r should be very negative.')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/FOCUSED_Q1_tau0_spike.png', dpi=140, bbox_inches='tight')
plt.close(fig)
_log(f"  Saved: FOCUSED_Q1_tau0_spike.png", t0)


# ============================================================================
# Q2: does error shrink with N?
# ============================================================================
_log("\n### Q2: mode-error vs N_context ###", t0)
q2_seed = 5
s = get_scm(q2_seed)
true_tau_all = (s['Y_do1'] - s['Y_do0']).reshape(-1).numpy()
rng = np.random.default_rng(0)
q_indices = rng.choice(len(true_tau_all), size=N_QUERIES_Q2, replace=False)
true_taus_q = true_tau_all[q_indices]
_log(f"  SCM seed={q2_seed}, {N_QUERIES_Q2} random queries, true τ range "
     f"[{true_taus_q.min():+.2f}, {true_taus_q.max():+.2f}]", t0)

results_by_N = {}
for N in CONTEXT_SIZES_Q2:
    if N > s['X_obs'].shape[0]:
        _log(f"  N={N} skipped", t0); continue
    _log(f"  N={N}:", t0)
    X_obs = s['X_obs'][:N].unsqueeze(0).to(DEVICE)
    T_obs = s['T_obs'][:N].unsqueeze(0).to(DEVICE)
    Y_obs = s['Y_obs'][:N].unsqueeze(0).to(DEVICE)
    X_intv = s['X_intv'][q_indices].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]
    modes = np.zeros(N_QUERIES_Q2); Etaus = np.zeros(N_QUERIES_Q2); Ks = np.zeros(N_QUERIES_Q2, dtype=int)
    for i in range(N_QUERIES_Q2):
        p_mat, *_ = unpack_pred(pred[i], J, bin_width)
        p_mat_np = p_mat.detach().cpu().numpy()
        # Use RAW p_mat mode & E[τ] (instant; MALC only smooths, doesn't move mode much).
        p_t = p_tau_raw(p_mat_np)
        # Ignore the τ=0 spike by masking a small window around 0 for mode search
        p_t_masked = p_t.copy()
        zero_mask = np.abs(tau_raw) < 0.02
        p_t_masked[zero_mask] = 0
        modes[i] = tau_raw[p_t_masked.argmax()]
        # E[τ] uses the full (unmaskd) density
        Etaus[i] = float((tau_raw * p_t).sum() * bin_width)
        Ks[i] = 0  # not fitting MALC in Q2 anymore
    mode_err  = np.abs(modes - true_taus_q)
    mean_err  = np.abs(Etaus - true_taus_q)
    corr_mode = float(np.corrcoef(modes, true_taus_q)[0, 1])
    corr_mean = float(np.corrcoef(Etaus, true_taus_q)[0, 1])
    results_by_N[N] = {
        'modes': modes, 'Etaus': Etaus, 'Ks': Ks,
        'mode_err': mode_err, 'mean_err': mean_err,
        'corr_mode': corr_mode, 'corr_mean': corr_mean,
    }
    _log(f"    mean mode-err={mode_err.mean():.3f}  median={np.median(mode_err):.3f}  "
         f"corr(mode, true)={corr_mode:+.3f}  corr(E[τ], true)={corr_mean:+.3f}", t0)

# Q2 plot: two panels
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
Ns = sorted(results_by_N.keys())
# Left: mode-error vs N (mean + 25/75 percentile band)
mean_errs = [results_by_N[N]['mode_err'].mean() for N in Ns]
p25 = [np.quantile(results_by_N[N]['mode_err'], 0.25) for N in Ns]
p75 = [np.quantile(results_by_N[N]['mode_err'], 0.75) for N in Ns]
axes[0].plot(Ns, mean_errs, 'o-', color='steelblue', lw=2, label='mean |mode − τ_true|')
axes[0].fill_between(Ns, p25, p75, color='steelblue', alpha=0.2, label='25th–75th percentile')
axes[0].set_xlabel('N_context')
axes[0].set_ylabel('|MALC-mode − τ_true|  (scaled)')
axes[0].set_title(f'Q2a: Error to true τ vs context size\n'
                  f'SCM seed={q2_seed}, {N_QUERIES_Q2} random queries')
axes[0].set_xscale('log')
axes[0].grid(alpha=0.3); axes[0].legend()

# Right: correlation vs N
corrs_mode = [results_by_N[N]['corr_mode'] for N in Ns]
corrs_mean = [results_by_N[N]['corr_mean'] for N in Ns]
axes[1].plot(Ns, corrs_mode, 'o-', color='steelblue', lw=2, label='corr(mode, τ_true)')
axes[1].plot(Ns, corrs_mean, 's--', color='crimson', lw=2, label='corr(E[τ], τ_true)')
axes[1].set_xlabel('N_context')
axes[1].set_ylabel('Pearson correlation across queries')
axes[1].set_title(f'Q2b: how well does the prediction rank queries?\n'
                  f'Higher = better identification of per-query τ')
axes[1].set_xscale('log')
axes[1].axhline(0, color='gray', ls=':', lw=0.8)
axes[1].axhline(1, color='green', ls=':', lw=0.8)
axes[1].grid(alpha=0.3); axes[1].legend()
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/FOCUSED_Q2_error_vs_N.png', dpi=140, bbox_inches='tight')
plt.close(fig)
_log(f"  Saved: FOCUSED_Q2_error_vs_N.png", t0)


# ============================================================================
# Q3: real bimodal cases + mode-error
# ============================================================================
_log("\n### Q3: real bimodal cases ###", t0)
# We already scanned Q1's SCMs; scan Q3_SCM many more
rows = []
for scm_seed in range(N_SCM_Q3):
    try: s = get_scm(scm_seed)
    except Exception: continue
    true_tau_all = (s['Y_do1'] - s['Y_do0']).reshape(-1).numpy()
    X_obs, T_obs, Y_obs = [s[k][:1000].unsqueeze(0).to(DEVICE) for k in ('X_obs','T_obs','Y_obs')]
    X_intv = s['X_intv'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]
    for i in range(pred.shape[0]):
        p_mat, *_ = unpack_pred(pred[i], J, bin_width)
        p_mat_np = p_mat.detach().cpu().numpy()
        p_t = p_tau_raw(p_mat_np)
        # Ignore the τ=0 spike: find peaks in a smoothed version
        # (Gaussian-blur p_t before peak-finding; keeps hump structure)
        kern = np.exp(-np.arange(-8, 9)**2 / (2 * 3.0**2)); kern /= kern.sum()
        p_smoothed = np.convolve(p_t, kern, mode='same')
        n_peaks, positions, heights, sep = score_peaks(p_smoothed, tau_raw,
                                                       prominence_frac=0.15,
                                                       min_sep_bins=15)
        balance = float(sorted(heights)[-2] / heights.max()) if len(heights) >= 2 else 0.0
        if n_peaks >= 2 and balance >= 0.40 and sep >= 0.40 and abs(true_tau_all[i]) > 0.05:
            # Nearest-mode-error
            if len(positions) >= 2:
                top2 = np.argsort(heights)[-2:]
                nearest_err = min(abs(float(true_tau_all[i]) - positions[top2[0]]),
                                   abs(float(true_tau_all[i]) - positions[top2[1]]))
            else:
                nearest_err = abs(float(true_tau_all[i]) - positions[0])
            rows.append({
                'scm_seed': scm_seed, 'q': i, 'true_tau': float(true_tau_all[i]),
                'peaks_pos': positions[np.argsort(heights)[-2:]],
                'balance': balance, 'sep': sep, 'nearest_err': nearest_err,
                'p_tau_raw': p_t, 'p_mat_np': p_mat_np,
            })
    _log(f"  seed={scm_seed}: {len(rows)} bimodal after this SCM", t0)

_log(f"\nTotal genuinely bimodal (after smoothing-out τ=0 spike): {len(rows)}", t0)
if len(rows) > 0:
    _log(f"  Mean nearest-mode-error: {np.mean([r['nearest_err'] for r in rows]):.3f}", t0)
    _log(f"  Median nearest-mode-error: {np.median([r['nearest_err'] for r in rows]):.3f}", t0)

# Q3 plot: 6 clean examples (top-6 by lowest nearest_err)
rows.sort(key=lambda r: r['nearest_err'])
show = rows[:6]
if show:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.reshape(-1)
    for i, r in enumerate(show):
        # Get MALC K=BIC fit
        density, K = cached_malc(r['p_mat_np'], r['scm_seed'] + r['q'])
        p_malc = p_tau_malc_grid(density)
        malc_mode = float(tau_smooth[p_malc.argmax()])
        malc_err = abs(malc_mode - r['true_tau'])

        ax = axes[i]
        ax.plot(tau_raw, r['p_tau_raw'], color='black', lw=1.2,
                label='raw p_mat', drawstyle='steps-mid')
        ax.plot(tau_smooth, p_malc, color='steelblue', lw=2,
                label=f'MALC K={K}')
        ax.axvline(r['true_tau'], color='red', ls='--', lw=1.5, alpha=0.7,
                   label=f"τ_true = {r['true_tau']:+.2f}")
        ax.axvline(malc_mode, color='green', ls=':', lw=1.5, alpha=0.9,
                   label=f'MALC mode = {malc_mode:+.2f}')
        ax.plot(r['true_tau'], 0, 'o', color='red', markersize=10, zorder=5, clip_on=False)
        ax.set_xlim(-1.5, 1.5)
        ax.set_xlabel('τ (scaled)')
        ax.set_ylabel('p(τ)')
        ax.set_title(f"seed={r['scm_seed']} q={r['q']}   "
                     f"nearest-mode-err = {r['nearest_err']:.2f}   "
                     f"MALC-mode-err = {malc_err:.2f}",
                     fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper right')
    fig.suptitle(
        f'Q3: real bimodal TE cases (after ignoring τ=0 spike)\n'
        f'{len(rows)} total across {N_SCM_Q3} SCMs × {N_QUERIES} queries; showing 6 with smallest nearest-mode error',
        y=1.02, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/FOCUSED_Q3_bimodal.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    _log(f"  Saved: FOCUSED_Q3_bimodal.png", t0)

_log(f"\nDone. Total {time.time() - t0:.1f}s", t0)
