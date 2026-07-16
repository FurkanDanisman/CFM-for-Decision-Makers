"""
Three tight diagnostics, no time-wasting MALC loops.

  D1. τ=0 MASK test.
      For 6 queries where raw p_mat has a real bimodal hump-plus-spike
      structure, run MALC:
        - on raw p_mat (baseline, spike included)
        - on p_mat with the τ=0 spike zeroed and renormalized
      Compare: does masking flip MALC's chosen mode toward the true τ?
      Save plot showing before/after side by side.

  D2. Model vs "predict-zero" baseline (across 500 queries × 4 SCMs).
      For every query compute:
        model_err   = |E[τ]_model − τ_true|
        zero_err    = |0 − τ_true|
        mean_err    = |μ_τ_train − τ_true|    (μ_τ_train = -0.06 from earlier)
      Report medians. If model_err ≥ zero_err, model has learned nothing about
      per-query τ beyond the marginal — publishability question.

  D3. Y_do0 == Y_do1 exactly in training data?
      For 500 fresh training samples, check how many queries have
      Y_do0_raw exactly == Y_do1_raw (bit-exact). If > 0 → data bug.
      Also count "close" cases (|Y_do0 - Y_do1| < 1e-6).
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
MALC_CACHE = os.path.join(OUT_DIR, 'malc_cache_diag')
N_SCM_D2   = 4
N_TRAIN    = 2000
N_QUERIES  = 500
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

os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(SCM_CACHE, exist_ok=True); os.makedirs(MALC_CACHE, exist_ok=True)


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


def p_tau_raw_fn(p_mat_np):
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
        rlo = np.clip(np.floor(rf).astype(int), 0, len(ys) - 2); rhi = rlo + 1
        whi = rf - rlo; wlo = 1.0 - whi
        f = wlo * density[rlo, col] + whi * density[rhi, col]
        out[k] = f.sum() * dy0
    return out


def cached_malc(p_mat_np, seed_hint, K=None):
    h = hashlib.sha1(np.ascontiguousarray(p_mat_np).tobytes()).hexdigest()[:16]
    key = f"{h}_B{MALC_B}_K{K if K is not None else f'bicMK{MALC_MAX_K}'}_s{seed_hint}"
    p = os.path.join(MALC_CACHE, f"{key}.npz")
    if os.path.exists(p): d = np.load(p); return d['density'], int(d['K'])
    kw = dict(B_fit=MALC_B, B_select=MALC_B, seed=seed_hint, parallel=False)
    fit = (fit_malc_inner(p_mat_np.T, edges_np, edges_np, K=K, **kw) if K is not None
           else fit_malc_inner(p_mat_np.T, edges_np, edges_np, max_K=MALC_MAX_K, **kw))
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
    else: sep = 0.0
    return len(peaks), positions, heights, sep


# ============================================================================
# D3 first (fastest) — is there a bit-exact Y_do0 == Y_do1 bug?
# ============================================================================
_log("\n### D3: Y_do0 == Y_do1 exactly in training data? ###", t0)
n_exact = 0; n_close = 0; n_total = 0
tau_bins = []
for scm_seed in range(4):
    try: s = get_scm(scm_seed)
    except Exception: continue
    y0 = s['Y_do0_raw'].reshape(-1).numpy()
    y1 = s['Y_do1_raw'].reshape(-1).numpy()
    n_exact += int((y0 == y1).sum())
    n_close += int((np.abs(y0 - y1) < 1e-6).sum())
    n_total += len(y0)
    tau_bins.append(y1 - y0)
_log(f"  Across {n_total} training queries (4 SCMs):", t0)
_log(f"    Y_do0 == Y_do1 bit-exact: {n_exact} ({100*n_exact/n_total:.2f}%)", t0)
_log(f"    |Y_do0 - Y_do1| < 1e-6  : {n_close} ({100*n_close/n_total:.2f}%)", t0)
all_tau = np.concatenate(tau_bins)
_log(f"    |τ_raw| < 0.001: {(np.abs(all_tau) < 0.001).sum()}", t0)
_log(f"    |τ_raw| < 0.01 : {(np.abs(all_tau) < 0.01).sum()}", t0)
_log(f"    |τ_raw| < 0.05 : {(np.abs(all_tau) < 0.05).sum()}", t0)
_log(f"    |τ_raw| > 0.10 : {(np.abs(all_tau) > 0.10).sum()}", t0)


# ============================================================================
# D2 — model vs zero baseline vs training-marginal baseline
# ============================================================================
_log("\n### D2: model E[τ] vs baselines ###", t0)
all_true = []; all_Etau = []; all_mode = []
for scm_seed in range(N_SCM_D2):
    try: s = get_scm(scm_seed)
    except Exception: continue
    true_tau_all = (s['Y_do1'] - s['Y_do0']).reshape(-1).numpy()
    X_obs = s['X_obs'][:1000].unsqueeze(0).to(DEVICE)
    T_obs = s['T_obs'][:1000].unsqueeze(0).to(DEVICE)
    Y_obs = s['Y_obs'][:1000].unsqueeze(0).to(DEVICE)
    X_intv = s['X_intv'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]
    for i in range(pred.shape[0]):
        p_mat, *_ = unpack_pred(pred[i], J, bin_width)
        p_t = p_tau_raw_fn(p_mat.detach().cpu().numpy())
        E_tau = float((tau_raw * p_t).sum() * bin_width)
        # Mode masking τ=0
        p_t_m = p_t.copy(); p_t_m[np.abs(tau_raw) < 0.02] = 0
        mode  = float(tau_raw[p_t_m.argmax()])
        all_true.append(float(true_tau_all[i])); all_Etau.append(E_tau); all_mode.append(mode)
    _log(f"  seed={scm_seed}: {pred.shape[0]} queries", t0)

all_true = np.array(all_true); all_Etau = np.array(all_Etau); all_mode = np.array(all_mode)
err_model_E   = np.abs(all_Etau - all_true)
err_model_mode = np.abs(all_mode - all_true)
err_zero     = np.abs(0 - all_true)
err_marginal = np.abs(all_true.mean() - all_true)   # oracle: use the OBS train marginal

_log(f"\nErrors (median across {len(all_true)} queries × {N_SCM_D2} SCMs):", t0)
_log(f"  model E[τ]     median-err = {np.median(err_model_E):.4f}  mean = {err_model_E.mean():.4f}", t0)
_log(f"  model mode(0-masked) median-err = {np.median(err_model_mode):.4f}  mean = {err_model_mode.mean():.4f}", t0)
_log(f"  zero baseline  median-err = {np.median(err_zero):.4f}  mean = {err_zero.mean():.4f}", t0)
_log(f"  marginal-mean baseline median-err = {np.median(err_marginal):.4f}  mean = {err_marginal.mean():.4f}", t0)
_log(f"  Fraction queries where model E[τ] beats zero baseline: "
     f"{(err_model_E < err_zero).mean():.3f}", t0)

# Plot D2: three-histogram overlay + scatter
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bins = np.linspace(0, max(err_zero.max(), err_model_E.max()), 60)
axes[0].hist(err_zero,      bins=bins, alpha=0.5, color='crimson',  label=f'zero baseline (median={np.median(err_zero):.3f})')
axes[0].hist(err_marginal,  bins=bins, alpha=0.5, color='orange',   label=f'training-mean baseline (median={np.median(err_marginal):.3f})')
axes[0].hist(err_model_E,   bins=bins, alpha=0.5, color='steelblue', label=f'model E[τ] (median={np.median(err_model_E):.3f})')
axes[0].hist(err_model_mode, bins=bins, alpha=0.5, color='green',   label=f'model mode τ≠0 (median={np.median(err_model_mode):.3f})')
axes[0].set_xlabel('|prediction − τ_true|')
axes[0].set_ylabel(f'# queries (of {len(all_true)})')
axes[0].set_title(f'D2a: prediction error distributions\n'
                  f'{len(all_true)} queries × {N_SCM_D2} SCMs')
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

axes[1].scatter(all_true, all_Etau, s=6, alpha=0.35, color='steelblue', label='model E[τ]')
axes[1].plot([-1, 1], [-1, 1], 'k--', lw=1, label='y=x (perfect)')
axes[1].axhline(0, color='crimson', ls=':', lw=1, label='zero baseline')
axes[1].axhline(all_true.mean(), color='orange', ls=':', lw=1,
                label=f'training mean = {all_true.mean():+.3f}')
axes[1].set_xlim(all_true.min() - 0.1, all_true.max() + 0.1)
axes[1].set_ylim(-1.5, 1.5)
axes[1].set_xlabel('τ_true'); axes[1].set_ylabel('model E[τ]')
axes[1].set_title(f'D2b: predicted vs true τ\ncorr={np.corrcoef(all_Etau, all_true)[0,1]:+.3f}')
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

fig.suptitle('D2: does the model beat "predict zero"?', y=1.02)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/DIAG_D2_vs_baseline.png', dpi=140, bbox_inches='tight')
plt.close(fig)
_log(f"  Saved: DIAG_D2_vs_baseline.png", t0)


# ============================================================================
# D1 — τ=0 mask test on 6 known-bimodal queries
# ============================================================================
_log("\n### D1: τ=0 mask helps MALC? ###", t0)
# Find bimodal queries (raw + smoothed spike removed) in seed=13 (had lots of these)
mask_probes = []
for scm_seed in (13, 15, 12):
    try: s = get_scm(scm_seed)
    except Exception: continue
    true_tau_all = (s['Y_do1'] - s['Y_do0']).reshape(-1).numpy()
    X_obs = s['X_obs'][:1000].unsqueeze(0).to(DEVICE)
    T_obs = s['T_obs'][:1000].unsqueeze(0).to(DEVICE)
    Y_obs = s['Y_obs'][:1000].unsqueeze(0).to(DEVICE)
    X_intv = s['X_intv'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]
    for i in range(pred.shape[0]):
        p_mat, *_ = unpack_pred(pred[i], J, bin_width)
        p_mat_np = p_mat.detach().cpu().numpy()
        p_t = p_tau_raw_fn(p_mat_np)
        # Smooth + remove τ=0 spike
        kern = np.exp(-np.arange(-8, 9)**2 / (2 * 3.0**2)); kern /= kern.sum()
        p_smooth = np.convolve(p_t, kern, mode='same')
        n, pos, heights, sep = score_peaks(p_smooth, tau_raw, prominence_frac=0.15, min_sep_bins=15)
        bal = float(sorted(heights)[-2] / heights.max()) if len(heights) >= 2 else 0.0
        if n >= 2 and bal >= 0.4 and sep >= 0.4 and abs(true_tau_all[i]) > 0.1:
            mask_probes.append((scm_seed, i, float(true_tau_all[i]), p_mat_np, p_t))
        if len(mask_probes) >= 6: break
    if len(mask_probes) >= 6: break

_log(f"  Found {len(mask_probes)} bimodal probe queries", t0)

# Apply τ=0 mask: zero out a window around Y_do0=Y_do1 diagonal (specifically
# the near-diagonal bins where |j0-j1| <= 1 in p_mat). Renormalize.
def mask_diagonal_p_mat(p_mat_np, band=1):
    p = p_mat_np.copy()
    for j0 in range(J):
        for j1 in range(max(0, j0-band), min(J, j0+band+1)):
            p[j0, j1] = 0.0
    p = p / max(p.sum(), 1e-12)
    return p

results_D1 = []
for (scm_seed, q, true_t, p_mat_np, p_t_orig) in mask_probes:
    p_mat_masked = mask_diagonal_p_mat(p_mat_np, band=1)
    p_t_masked   = p_tau_raw_fn(p_mat_masked)

    _log(f"  seed={scm_seed} q={q} τ_true={true_t:+.3f}  fitting MALC on original & masked…", t0)
    dens_orig, K_orig = cached_malc(p_mat_np, scm_seed + q + 111)
    dens_mask, K_mask = cached_malc(p_mat_masked, scm_seed + q + 222)
    p_malc_orig = p_tau_malc_grid(dens_orig)
    p_malc_mask = p_tau_malc_grid(dens_mask)

    mode_orig = float(tau_smooth[p_malc_orig.argmax()])
    mode_mask = float(tau_smooth[p_malc_mask.argmax()])
    err_orig  = abs(mode_orig - true_t)
    err_mask  = abs(mode_mask - true_t)
    _log(f"    K_orig={K_orig}  mode={mode_orig:+.3f}  err={err_orig:.3f}", t0)
    _log(f"    K_mask={K_mask}  mode={mode_mask:+.3f}  err={err_mask:.3f}", t0)
    results_D1.append({
        'scm_seed': scm_seed, 'q': q, 'true_tau': true_t,
        'p_raw_orig': p_t_orig, 'p_raw_mask': p_t_masked,
        'p_malc_orig': p_malc_orig, 'p_malc_mask': p_malc_mask,
        'K_orig': K_orig, 'K_mask': K_mask,
        'mode_orig': mode_orig, 'mode_mask': mode_mask,
        'err_orig': err_orig, 'err_mask': err_mask,
    })

if results_D1:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.reshape(-1)
    for i, r in enumerate(results_D1[:6]):
        ax = axes[i]
        ax.plot(tau_raw, r['p_raw_orig'], color='black', lw=1.2,
                label='raw p_mat (orig)', drawstyle='steps-mid', alpha=0.6)
        ax.plot(tau_smooth, r['p_malc_orig'], color='#FF7F0E', lw=1.6,
                label=f"MALC on orig K={r['K_orig']}  err={r['err_orig']:.2f}")
        ax.plot(tau_smooth, r['p_malc_mask'], color='steelblue', lw=2,
                label=f"MALC on masked K={r['K_mask']}  err={r['err_mask']:.2f}")
        ax.axvline(r['true_tau'], color='red', ls='--', lw=1.5, alpha=0.7)
        ax.plot(r['true_tau'], 0, 'o', color='red', markersize=10, zorder=5, clip_on=False)
        ax.set_xlim(-1.5, 1.5)
        ax.set_title(f"seed={r['scm_seed']} q={r['q']}  τ_true={r['true_tau']:+.2f}", fontsize=10)
        ax.set_xlabel('τ (scaled)'); ax.set_ylabel('p(τ)')
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper right')
    fig.suptitle('D1: does zeroing the τ=0 spike in p_mat help MALC pick the correct mode?', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/DIAG_D1_mask_test.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    _log(f"  Saved: DIAG_D1_mask_test.png", t0)

_log(f"\nDone. Total {time.time() - t0:.1f}s", t0)
