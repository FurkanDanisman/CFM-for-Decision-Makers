"""Rank queries by triple multimodality: joint, marginals, AND treatment effect.

For each (SCM_SEED, N) we run inference on every test query and score how
"cleanly multimodal" the query looks across three views:

  * joint    p(Y_do0, Y_do1)   — 2D peak count via local-max filter
  * marginal p(Y_do0)          — 1D peak count via scipy.find_peaks
  * marginal p(Y_do1)          — 1D peak count via scipy.find_peaks
  * TE      p(τ = Y_do1 - Y_do0) — 1D peak count on the diagonal-integrated
                                    density (no MALC — raw p_mat sums)

Only queries where ALL FOUR views show ≥ 2 peaks are kept. The remaining
queries are ranked by a peak-balance × peak-count score so the top ones
are the most visually striking multimodal examples (tri- or bi-modal
across the board, not just noisy bumps).

The script prints a QUERY_IDXS=... line you can copy-paste directly into
plot_joint_marginals.py, so the two scripts compose:

    python benchmarks/plots/find_multimodal_queries.py SCM_SEED=7 TOP_K=6
    # produces:  QUERY_IDXS=12,88,143,...
    QUERY_IDXS=12,88,143,... SCM_SEED=7 OUT_PREFIX=scm7_mm \
        python benchmarks/plots/plot_joint_marginals.py

Environment knobs
-----------------
  SCM_SEED       (default 2)         which SCM instance to search
  N              (default 500)       context size at which to score
  N_TRAIN        (default 2000)      context pool size sampled once
  N_TEST         (default 500)       test queries per SCM
  TOP_K          (default 6)         how many top candidates to print
  MIN_PEAKS      (default 2)         require ≥ this many peaks in every view
  PROMINENCE_RATIO (default 0.10)    peak prominence as fraction of dist max
  SMOOTH_SIGMA   (default 1.5)       gaussian sigma applied before peak finding
  CHECKPOINT     (default checkpoints/step_50000_final.pt)
  UWYK_SRC       (default /tmp/g4cfm_uwyk/src)
  PYTHONHASHSEED (default 0)         must stay 0 for reproducibility
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
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_filter

_HERE  = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
_REPO  = os.path.dirname(_BENCH)
_CSWEEP = os.path.join(_BENCH, 'context_sweep')

CKPT       = os.environ.get('CHECKPOINT', os.path.join(_REPO, 'checkpoints', 'step_50000_final.pt'))
UWYK_SRC   = os.environ.get('UWYK_SRC',   '/tmp/g4cfm_uwyk/src')
DATA_SOURCE = os.environ.get('DATA_SOURCE', 'prior')  # 'prior' | 'poly'
CAUSALPFN_ROOT = os.environ.get('CAUSALPFN_ROOT', '')
# SCM_SEEDS: comma list OR "0-49" range. Default: single seed=2 (back-compat).
SCM_SEED_ENV = os.environ.get('SCM_SEEDS', os.environ.get('SCM_SEED', '2'))
if '-' in SCM_SEED_ENV and ',' not in SCM_SEED_ENV:
    a, b = SCM_SEED_ENV.split('-'); SCM_SEEDS = list(range(int(a), int(b) + 1))
else:
    SCM_SEEDS = [int(x) for x in SCM_SEED_ENV.split(',')]
N          = int(os.environ.get('N', 500))
N_TRAIN    = int(os.environ.get('N_TRAIN', max(2000, N)))
N_TEST     = int(os.environ.get('N_TEST',  500))
TOP_K      = int(os.environ.get('TOP_K', 6))
MIN_PEAKS  = int(os.environ.get('MIN_PEAKS', 2))
PROMINENCE_RATIO = float(os.environ.get('PROMINENCE_RATIO', 0.10))
SMOOTH_SIGMA = float(os.environ.get('SMOOTH_SIGMA', 1.5))

sys.path.insert(0, UWYK_SRC)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'MALC'))
sys.path.insert(0, _CSWEEP)

from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import unpack_pred
from scm_prior import generate_paired_sample_with_raw
if DATA_SOURCE == 'poly':
    # scm_polynomial's shim reimplements CausalPFN's PolynomialDataset — no
    # external deps needed. Uses polynomial mechanisms + Laplace-noise treatment,
    # which produces more heterogeneous joints than the default SCMSampler.
    from scm_polynomial_shim import PolynomialDataset  # local fallback (no faiss)

DEVICE = torch.device('cpu')


# ── 1D peak scoring ─────────────────────────────────────────────────────────
def score_1d(density: np.ndarray):
    """Return (n_peaks, peak_balance) where peak_balance = min/max peak height."""
    d = gaussian_filter1d(density, sigma=SMOOTH_SIGMA, mode='nearest')
    if d.max() <= 0:
        return 0, 0.0
    peaks, props = find_peaks(d, prominence=d.max() * PROMINENCE_RATIO,
                                distance=max(3, len(d) // 30))
    n = len(peaks)
    if n < 2:
        return n, 0.0
    heights = d[peaks]
    balance = float(heights.min() / heights.max())
    return n, balance


# ── 2D peak scoring ─────────────────────────────────────────────────────────
def score_2d(pm: np.ndarray, centers_arr: np.ndarray | None = None,
              tau_tol: float = 0.10):
    """Return (n_peaks, peak_balance, n_distinct_tau).

    n_distinct_tau clusters the 2D peaks by their τ = Y_do1 - Y_do0 offset
    with tolerance `tau_tol` and counts how many distinct τ-bands they fall
    into. This is the direct predictor of τ-marginal multimodality: peaks
    that share a τ get integrated onto the same diagonal.
    """
    d = gaussian_filter(pm, sigma=SMOOTH_SIGMA, mode='nearest')
    neighborhood = int(max(5, round(4 * SMOOTH_SIGMA + 1)))
    if neighborhood % 2 == 0: neighborhood += 1
    local_max = (d == maximum_filter(d, size=neighborhood))
    thresh = d.max() * PROMINENCE_RATIO
    peaks_mask = local_max & (d > thresh)
    n = int(peaks_mask.sum())
    if n < 2:
        return n, 0.0, 1
    heights = d[peaks_mask]
    balance = float(heights.min() / heights.max())

    n_distinct_tau = 1
    if centers_arr is not None:
        # peaks_mask indexed as (y0_idx, y1_idx) — pm rows = Y_do0, cols = Y_do1
        y0_idx, y1_idx = np.where(peaks_mask)
        taus = centers_arr[y1_idx] - centers_arr[y0_idx]
        # Keep peaks in descending height order, then cluster on τ
        order = np.argsort(-d[y0_idx, y1_idx])
        kept_taus = []
        for j in order:
            t = taus[j]
            if all(abs(t - kt) > tau_tol for kt in kept_taus):
                kept_taus.append(t)
        n_distinct_tau = len(kept_taus)
    return n, balance, n_distinct_tau


# ── p(τ) via raw diagonal integration (fast — no MALC) ──────────────────────
def diagonal_p_tau(pm: np.ndarray, centers: np.ndarray):
    """Integrate the joint along diagonals to get p(τ = Y_do1 - Y_do0).

    pm is (J, J) with pm[i, j] ≈ P(Y_do0 in bin i, Y_do1 in bin j).
    """
    J = pm.shape[0]
    # τ index k = j - i ∈ [-(J-1), J-1]; length 2J-1
    tau_grid = centers[:, None] - centers[None, :]  # (J, J) — Y_do1 - Y_do0? careful
    # Convention: rows = Y_do0 index (i), cols = Y_do1 index (j). τ = c[j] - c[i].
    # We use `np.add.reduceat`-style diagonal sums via arange offsets.
    n_tau = 2 * J - 1
    tau_dens = np.zeros(n_tau, dtype=np.float64)
    for k, off in enumerate(range(-(J - 1), J)):
        tau_dens[k] = np.trace(pm, offset=off)
    # Grid of τ values for peak-finding — centers spacing = bin_width, so τ_step = bin_width
    return tau_dens


# ── Load model ──────────────────────────────────────────────────────────────
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
cfg = ckpt['config']; J = cfg['J']
edges_np = ckpt['edges'].cpu().numpy()
bin_width = float(edges_np[1] - edges_np[0])
centers = 0.5 * (edges_np[:-1] + edges_np[1:])
model = InterventionalPFN(
    num_features=cfg['num_features'], d_model=cfg['d_model'], depth=cfg['depth'],
    heads_feat=cfg['heads'], heads_samp=cfg['heads'], dropout=0.0,
    output_dim=J*J + 9 + 4, hidden_mult=cfg['hidden_mult'],
    normalize_features=True, normalize_treatment=False,
    use_treatment_in_query=False, use_checkpoint=False,
).to(DEVICE).eval()
model.load_state_dict(ckpt['model_state_dict'])
print(f'[load] model J={J}', flush=True)


all_records = []          # per-(seed, q) records across all seeds
per_seed_summary = []     # (seed, joint≥2, marg≥2, tau≥2, all4)

for SCM_SEED in SCM_SEEDS:
    # Re-seed globals so the SCM sampler is deterministic per SCM_SEED
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    if DATA_SOURCE == 'prior':
        sample = generate_paired_sample_with_raw(
            scm_seed=SCM_SEED, idx=0, n_train=N_TRAIN, n_test=N_TEST,
        )
        X_obs_full = sample['X_obs']; T_obs_full = sample['T_obs']; Y_obs_full = sample['Y_obs']
        X_intv     = sample['X_intv']
        Y_do0      = sample['Y_do0'].numpy().reshape(-1)
        Y_do1      = sample['Y_do1'].numpy().reshape(-1)
    else:  # poly
        n_samples = N_TRAIN + N_TEST
        ds = PolynomialDataset(n_tables=SCM_SEED + 1, n_samples=n_samples,
                                test_ratio=N_TEST / n_samples, seed=42 + SCM_SEED)
        cd, _ad = ds[SCM_SEED]
        X_all = torch.cat([cd.X_train, cd.X_test], dim=0)
        # Scale to [-1, 1] using train's range so distributions match model's grid
        x_min, x_max = X_all.min(0).values, X_all.max(0).values
        rng = (x_max - x_min).clamp(min=1e-6)
        X_all_s = 2 * (X_all - x_min) / rng - 1
        # Only train rows have observed T/Y; fabricate synthetic potential outcomes
        # via CATE-Dataset's true_cate + a mid-point Y_do0 estimate.
        n_train_actual = cd.X_train.shape[0]
        # Pad to model's num_features by appending NaN cols — the model handles
        # NaN cols as unused features (same convention as the prior sampler).
        num_feat_model = cfg['num_features']
        d = X_all_s.shape[1]
        if d < num_feat_model:
            pad = torch.full((X_all_s.shape[0], num_feat_model - d),
                              float('nan'), dtype=X_all_s.dtype)
            X_all_s = torch.cat([X_all_s, pad], dim=1)
        else:
            X_all_s = X_all_s[:, :num_feat_model]

        X_obs_full = X_all_s[:n_train_actual]
        T_obs_full = cd.t_train.reshape(-1, 1)
        Y_obs_full = cd.y_train.reshape(-1, 1)
        # Scale Y with same trick — use train's range for stability
        y_min = float(Y_obs_full.min()); y_max = float(Y_obs_full.max())
        y_rng = max(y_max - y_min, 1e-6)
        Y_obs_full = (2 * (Y_obs_full - y_min) / y_rng - 1).to(torch.float32)
        X_intv = X_all_s[n_train_actual:]
        # We don't have Y_do0/Y_do1 pointwise (poly dataset doesn't return them),
        # so approximate: Y_do0 = 0 for all, Y_do1 = true_cate (in Y units → rescale)
        tc = cd.true_cate.numpy().reshape(-1)
        # Rescale τ to same [-1,1] Y-space
        tc_scaled = 2.0 * tc / y_rng
        Y_do0 = np.zeros_like(tc_scaled, dtype=np.float32)
        Y_do1 = tc_scaled.astype(np.float32)

    true_tau_scaled = Y_do1 - Y_do0

    Xc = X_obs_full[:N].unsqueeze(0)
    Tc = T_obs_full[:N].unsqueeze(0)
    Yc = Y_obs_full[:N].unsqueeze(0)
    Xq = X_intv.unsqueeze(0)
    with torch.no_grad():
        pred = model(Xc, Tc, Yc, Xq)['predictions'][0]

    n_j_all = n_m_all = n_t_all = n_all = 0
    for q in range(pred.shape[0]):
        p_mat, *_ = unpack_pred(pred[q], J, bin_width)
        pm = p_mat.detach().cpu().numpy()
        p_y0 = pm.sum(axis=1)
        p_y1 = pm.sum(axis=0)
        p_tau_raw = diagonal_p_tau(pm, centers)

        n_j, b_j, n_dt = score_2d(pm, centers, tau_tol=0.15)
        n_y0, b_y0 = score_1d(p_y0)
        n_y1, b_y1 = score_1d(p_y1)
        n_t, b_t   = score_1d(p_tau_raw)

        # `n_dt` = distinct-τ clusters among 2D peaks (predicts τ multimodality
        # even when joint has many peaks along a fixed y0 or y1 stripe).
        passes = (n_dt >= MIN_PEAKS and
                  n_y0 >= MIN_PEAKS and n_y1 >= MIN_PEAKS and
                  n_t  >= MIN_PEAKS)
        min_peaks = min(n_dt, n_y0, n_y1, n_t)
        balance_prod = (b_j * b_y0 * b_y1 * b_t) ** 0.25
        all_records.append({
            'seed': SCM_SEED, 'q': q,
            'true_tau': float(true_tau_scaled[q]),
            'peaks': (n_j, n_y0, n_y1, n_t),
            'n_distinct_tau': n_dt,
            'balances': (b_j, b_y0, b_y1, b_t),
            'min_peaks': min_peaks,
            'balance': balance_prod,
            'passes': passes,
        })
        n_j_all += (n_dt >= 2)   # now: n_distinct_tau, not raw 2D peaks
        n_m_all += (n_y0 >= 2 and n_y1 >= 2)
        n_t_all += (n_t >= 2)
        n_all   += passes

    per_seed_summary.append((SCM_SEED, n_j_all, n_m_all, n_t_all, n_all,
                             float(true_tau_scaled.mean()), float(true_tau_scaled.std())))
    print(f'[seed {SCM_SEED:>3d}] τ~N({true_tau_scaled.mean():+.2f},{true_tau_scaled.std():.2f}) '
          f'distinct-τ≥2: {n_j_all:>3d}  both-marg≥2: {n_m_all:>3d}  '
          f'τ-peaks≥2: {n_t_all:>3d}  all-four: {n_all:>3d}', flush=True)

records = all_records

# ── Debug: peak-count histograms across all queries (per view) ──────────────
def _hist(vals, name):
    c = np.bincount(np.array(vals, dtype=int), minlength=5)
    print(f'  {name:>7}: peaks=1,2,3,4+  → {c[1]:>3d}  {c[2]:>3d}  {c[3]:>3d}  {c[4:].sum():>3d}',
          flush=True)

print('[debug] peak-count distribution across all queries:', flush=True)
_hist([r['peaks'][0] for r in records], 'joint')
_hist([r['peaks'][1] for r in records], 'p(Y0)')
_hist([r['peaks'][2] for r in records], 'p(Y1)')
_hist([r['peaks'][3] for r in records], 'p(τ)')

passing = [r for r in records if r['passes']]
passing.sort(key=lambda r: (-r['min_peaks'], -r['balance']))
print(f'[score] {len(passing)}/{len(records)} queries pass min-peaks ≥ {MIN_PEAKS} across all four views',
      flush=True)


# ── Report top-K ────────────────────────────────────────────────────────────
top = passing[:TOP_K]
if not top:
    print('[warn] no queries pass — try a different SCM_SEED, lower MIN_PEAKS, or bump PROMINENCE_RATIO down')
    sys.exit(0)

print()
print(f'{"rank":>4} {"seed":>4} {"q":>4} {"τ_true":>7}  '
      f'{"joint":>5} {"dτ":>3} {"y0":>3} {"y1":>3} {"tau":>4}  {"balance":>7}')
for i, r in enumerate(top, 1):
    pk = r['peaks']
    print(f'{i:>4} {r["seed"]:>4} {r["q"]:>4} {r["true_tau"]:+7.3f}   '
          f'{pk[0]:>4} {r["n_distinct_tau"]:>3} {pk[1]:>3} {pk[2]:>3} {pk[3]:>4}   '
          f'{r["balance"]:>6.2f}')

# Group top by SCM seed so the user can plot per-seed.
by_seed = {}
for r in top:
    by_seed.setdefault(r['seed'], []).append(r['q'])
print()
for s, qs in sorted(by_seed.items()):
    qs_str = ','.join(map(str, qs))
    print(f'SCM_SEED={s}  QUERY_IDXS={qs_str}   # {len(qs)} candidates from seed {s}')
