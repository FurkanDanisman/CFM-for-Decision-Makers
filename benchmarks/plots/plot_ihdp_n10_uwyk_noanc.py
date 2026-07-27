"""IHDP demonstration — UWYK No-Ancestral pipeline (marginals + naive TE).

Companion to plot_ihdp_n10.py. Uses the SAME IHDP realisation, SAME
first-N training observations, and SAME 10 test queries so the resulting
plots are directly comparable to those in
``benchmarks/plots/ihdp_n10/UWYK-2DMALC``.

Since UWYK No-Ancestral outputs only 1D BarDistribution marginals per
treatment arm — not a joint — the treatment-effect distribution can only
be derived under an independence assumption. This script therefore
produces:

  UWYK-NOANC/
    ihdp_n10_marginals.png    p(Y_do0), p(Y_do1) per query
    ihdp_n10_te.png           naive independence p(τ) per query
    ihdp_n10_ot.png           W_2 barycenter of the naive per-query p(τ)

No joint plot exists for this pipeline. Runs only where the UWYK
checkpoint is accessible (typically killarney).

Environment
-----------
  REPO, CAUSALPFN, DOPFN   as in plot_ihdp_n10.py
  UWYK_SRC                 path to Graphs4CausalFoundationModels/src
  UWYK_CKPT_DIR            (default: ancestral checkpoint dir derived from UWYK_SRC)
  REALIZATION, N_CONTEXT,  match the defaults of plot_ihdp_n10.py so the
  N_QUERIES                same queries and context are used.
  UWYK_N_SAMPLES           number of samples per query per treatment for
                            histogram estimation of the marginals (default 1024)
"""
from __future__ import annotations
import os, sys, importlib, types
import numpy as np
import torch
import matplotlib.pyplot as plt

_HERE   = os.path.dirname(os.path.abspath(__file__))
_BENCH  = os.path.dirname(_HERE)
_REPO   = os.environ.get('REPO', os.path.dirname(_BENCH))
_OUTDIR = os.path.join(_HERE, 'ihdp_n10', 'UWYK-NOANC')
os.makedirs(_OUTDIR, exist_ok=True)

CAUSALPFN     = os.environ.get('CAUSALPFN', '')
DOPFN         = os.environ.get('DOPFN', '')
UWYK_SRC      = os.environ.get('UWYK_SRC', '')
REALIZATION   = int(os.environ.get('REALIZATION', 0))
N_CONTEXT     = int(os.environ.get('N_CONTEXT', 200))
N_QUERIES     = int(os.environ.get('N_QUERIES', 10))
N_SAMPLES     = int(os.environ.get('UWYK_N_SAMPLES', 1024))
CHECKPOINT    = os.environ.get('CHECKPOINT',
                                os.path.join(_REPO, 'checkpoints', 'step_50000_final.pt'))
_UWYK_ROOT    = os.path.dirname(os.path.abspath(UWYK_SRC.rstrip('/'))) if UWYK_SRC else ''
UWYK_CKPT_DIR = os.environ.get(
    'UWYK_CKPT_DIR',
    os.path.join(_UWYK_ROOT,
                  'experiments/checkpoints/full_conditioned_model/'
                  'final_earlytest_full_conditioning_16773252.0')
    if _UWYK_ROOT else '')

if not (os.path.isdir(UWYK_SRC) and os.path.isdir(UWYK_CKPT_DIR)
        and os.path.isfile(os.path.join(UWYK_CKPT_DIR, 'best_model.pt'))
        and os.path.isdir(CAUSALPFN) and os.path.isdir(DOPFN)):
    print('[skip] UWYK checkpoint / source paths not set correctly:')
    print(f'   UWYK_SRC       = {UWYK_SRC}')
    print(f'   UWYK_CKPT_DIR  = {UWYK_CKPT_DIR}')
    print(f'   CAUSALPFN      = {CAUSALPFN}')
    print(f'   DOPFN          = {DOPFN}')
    sys.exit(0)


# ── Load UWYK No-Ancestral checkpoint ────────────────────────────────────
_saved = {}
for name in list(sys.modules):
    if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
        _saved[name] = sys.modules.pop(name)
sys.path.insert(0, UWYK_SRC)
UWYK_pre_mod = importlib.import_module('models.PreprocessingGraphConditionedPFN')
sys.path.remove(UWYK_SRC)
for name in list(sys.modules):
    if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
        del sys.modules[name]
sys.modules.update(_saved)

_orig_load = torch.load
def _p_load(*a, **kw):
    kw.setdefault('weights_only', False); return _orig_load(*a, **kw)
torch.load = _p_load

uwyk_model = UWYK_pre_mod.PreprocessingGraphConditionedPFN(
    config_path=os.path.join(UWYK_CKPT_DIR, 'best_model_config.yaml'),
    checkpoint_path=os.path.join(UWYK_CKPT_DIR, 'best_model.pt'),
    device='cpu', verbose=False,
).load()
torch.load = _orig_load
NUM_FEATURES = uwyk_model.model.num_features
print(f'[load] UWYK No-Ancestral  num_features={NUM_FEATURES}', flush=True)


# ── Load IHDP realisation + reproduce plot_ihdp_n10's context/queries ────
sys.path.insert(0, DOPFN)
ds_mod = types.ModuleType('datasets')
with open(os.path.join(DOPFN, 'datasets', '__init__.py')) as fp:
    _src = fp.read().split('def load_semi_real')[0]
exec(_src, ds_mod.__dict__)
sys.modules['datasets'] = ds_mod
sys.path.insert(0, CAUSALPFN)

from benchmarks import IHDPDataset
cd, ad = IHDPDataset()[REALIZATION]
X_train_full = cd.X_train.numpy() if hasattr(cd.X_train, 'numpy') else np.asarray(cd.X_train)
t_train_full = cd.t_train.numpy() if hasattr(cd.t_train, 'numpy') else np.asarray(cd.t_train)
y_train_full = cd.y_train.numpy() if hasattr(cd.y_train, 'numpy') else np.asarray(cd.y_train)
X_test = cd.X_test.numpy() if hasattr(cd.X_test, 'numpy') else np.asarray(cd.X_test)
true_cate = cd.true_cate.numpy() if hasattr(cd.true_cate, 'numpy') else np.asarray(cd.true_cate)
true_cate = true_cate.reshape(-1)

X_context = X_train_full[:N_CONTEXT].astype(np.float32)
T_context = t_train_full[:N_CONTEXT].astype(np.float32).reshape(-1, 1)
Y_context = y_train_full[:N_CONTEXT].astype(np.float32).reshape(-1, 1)

# Same Y scaling as plot_ihdp_n10.py: min/max across training Y_train_full
y_min = float(y_train_full.min()); y_max = float(y_train_full.max())
y_rng = max(y_max - y_min, 1e-6)
true_cate_scaled = true_cate * (2.0 / y_rng)

# Query selection by true-τ percentile (identical to plot_ihdp_n10.py)
order = np.argsort(true_cate_scaled)
qs = np.linspace(0.05, 0.95, N_QUERIES)
QUERY_IDXS = order[(qs * (len(true_cate_scaled) - 1)).astype(int)].tolist()
print(f'[data] IHDP r={REALIZATION}  N_context={N_CONTEXT}  '
      f'queries={QUERY_IDXS}', flush=True)


# ── UWYK target encoding + fit ───────────────────────────────────────────
# Real IHDP has X.shape[1] < UWYK's expected NUM_FEATURES → pad with NaN.
def _pad_to_features(X):
    d = X.shape[1]
    if d < NUM_FEATURES:
        pad = np.full((X.shape[0], NUM_FEATURES - d), np.nan, dtype=X.dtype)
        return np.concatenate([X.astype(np.float32), pad], axis=1)
    return X.astype(np.float32)[:, :NUM_FEATURES]


X_train_p = _pad_to_features(X_context)
X_test_p  = _pad_to_features(X_test.astype(np.float32))

t_train_orig = T_context.reshape(-1, 1)
y_train      = Y_context.reshape(-1, 1)
mean_y_t0 = float(y_train[t_train_orig == 0].mean())
mean_y_t1 = float(y_train[t_train_orig == 1].mean())
t_train_enc = np.where(t_train_orig == 0, mean_y_t0, mean_y_t1).astype(np.float32)
uwyk_model.fit(X_train_p, t_train_enc, y_train)

# No-Ancestral adjacency: zero for real features, -1 for padded (absent)
n_real_features = X_context.shape[1]
adj = np.zeros((NUM_FEATURES + 2, NUM_FEATURES + 2), dtype=np.float32)
for i in range(n_real_features, NUM_FEATURES):
    fi = 2 + i
    adj[fi, :] = -1.0; adj[:, fi] = -1.0; adj[fi, fi] = -1.0


# ── Sample per-query outcomes at T=0 and T=1 ─────────────────────────────
X_intv = X_test_p[QUERY_IDXS]
T_intv_0 = np.full((len(QUERY_IDXS), 1), mean_y_t0, dtype=np.float32)
T_intv_1 = np.full((len(QUERY_IDXS), 1), mean_y_t1, dtype=np.float32)


def _predict_samples(T_intv):
    n_have = 0; chunks = []
    while n_have < N_SAMPLES:
        r = uwyk_model.predict(
            X_obs=X_train_p, T_obs=t_train_enc, Y_obs=y_train,
            X_intv=X_intv, T_intv=T_intv,
            adjacency_matrix=adj,
            prediction_type='sample', inverse_transform=True,
        )
        arr = np.asarray(r).reshape(len(QUERY_IDXS), -1)
        chunks.append(arr); n_have += arr.shape[1]
    return np.concatenate(chunks, axis=1)[:, :N_SAMPLES]


print(f'[predict] sampling {N_SAMPLES} draws per query per treatment', flush=True)
Y0_samples = _predict_samples(T_intv_0)     # (N_QUERIES, N_SAMPLES) — in raw Y units
Y1_samples = _predict_samples(T_intv_1)

# Scale to the same [-1, 1] range as OURS
Y0_scaled = (Y0_samples - y_min) / y_rng * 2.0 - 1.0
Y1_scaled = (Y1_samples - y_min) / y_rng * 2.0 - 1.0


# ── Histogram → marginals p(Y_do0), p(Y_do1) on a shared centers grid ────
Jn = 100
edges = np.linspace(-1.5, 1.5, Jn + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
bin_width = float(centers[1] - centers[0])

p_y0 = np.zeros((N_QUERIES, Jn), dtype=np.float64)
p_y1 = np.zeros((N_QUERIES, Jn), dtype=np.float64)
for k in range(N_QUERIES):
    h0, _ = np.histogram(Y0_scaled[k], bins=edges, density=True)
    h1, _ = np.histogram(Y1_scaled[k], bins=edges, density=True)
    p_y0[k] = h0
    p_y1[k] = h1


# ── Naive p(τ) via convolution / diagonal-sum ────────────────────────────
tau_edges   = np.linspace(-3.0, 3.0, 601)
tau_centers = 0.5 * (tau_edges[:-1] + tau_edges[1:])


def _naive_p_tau(p_y0_row, p_y1_row):
    p_y0_n = p_y0_row / max(p_y0_row.sum(), 1e-12)
    p_y1_n = p_y1_row / max(p_y1_row.sum(), 1e-12)
    outer = np.outer(p_y1_n, p_y0_n)                # (Jn, Jn)
    diag_sums = np.array([np.trace(outer, offset=off)
                           for off in range(-(Jn - 1), Jn)])
    tau_naive_grid = np.arange(-(Jn - 1), Jn) * bin_width
    density = diag_sums / bin_width
    return np.interp(tau_centers, tau_naive_grid, density,
                      left=0.0, right=0.0)


p_taus_naive = np.zeros((N_QUERIES, len(tau_centers)))
for k in range(N_QUERIES):
    p_taus_naive[k] = _naive_p_tau(p_y0[k], p_y1[k])


# ── Layout ───────────────────────────────────────────────────────────────
n_cols = 5 if N_QUERIES == 10 else 3
n_rows = (N_QUERIES + n_cols - 1) // n_cols
palette = {'do0': '#2E7DAF', 'do1': '#7B3E9E'}


# ── Figure: marginals ────────────────────────────────────────────────────
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.6 * n_rows),
                          squeeze=False)
for k, q in enumerate(QUERY_IDXS):
    ax = axes[k // n_cols][k % n_cols]
    ax.plot(centers, p_y0[k], color=palette['do0'], lw=1.9, label=r'$p(Y_{do0})$')
    ax.plot(centers, p_y1[k], color=palette['do1'], lw=1.9, label=r'$p(Y_{do1})$')
    E_y0 = float((centers * p_y0[k]).sum() * bin_width)
    E_y1 = float((centers * p_y1[k]).sum() * bin_width)
    ax.plot(E_y0, float(np.interp(E_y0, centers, p_y0[k])), 'o',
             color=palette['do0'], markersize=9, markeredgecolor='white',
             markeredgewidth=1.0, zorder=5)
    ax.plot(E_y1, float(np.interp(E_y1, centers, p_y1[k])), 'o',
             color=palette['do1'], markersize=9, markeredgecolor='white',
             markeredgewidth=1.0, zorder=5)
    ax.set_title(f'query {q}   $\\tau_{{true}}$={true_cate_scaled[q]:+.2f}',
                  fontsize=10)
    if k // n_cols == n_rows - 1: ax.set_xlabel(r'$Y$  (scaled)')
    if k %  n_cols == 0:          ax.set_ylabel('density')
    ax.grid(alpha=0.25)
    if k == 0: ax.legend(fontsize=9, loc='upper right')
for k in range(N_QUERIES, n_rows * n_cols):
    axes[k // n_cols][k % n_cols].set_visible(False)
fig.suptitle(f'IHDP r={REALIZATION}   UWYK No-Ancestral marginal densities at N={N_CONTEXT}',
              fontsize=12, y=0.999)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out = os.path.join(_OUTDIR, 'ihdp_n10_marginals.png')
fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
print(f'[save] {out}')


# ── Figure: naive TE per query ───────────────────────────────────────────
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.6 * n_rows),
                          squeeze=False)
tau_step = tau_centers[1] - tau_centers[0]
for k, q in enumerate(QUERY_IDXS):
    ax = axes[k // n_cols][k % n_cols]
    E_tau_naive = float((tau_centers * p_taus_naive[k]).sum() * tau_step)
    ax.fill_between(tau_centers, p_taus_naive[k], alpha=0.25, color='#C1420F')
    ax.plot(tau_centers, p_taus_naive[k], color='#C1420F', lw=2.0,
             label=f'naive $p(\\tau)$  E={E_tau_naive:+.2f}')
    ax.axvline(E_tau_naive, color='#C1420F', ls='--', lw=1.4, alpha=0.85)
    ax.axvline(true_cate_scaled[q], color='red', ls='--', lw=1.4,
                label=f'true $\\tau$={true_cate_scaled[q]:+.2f}')
    ax.plot(true_cate_scaled[q], 0, 'o', color='red', markersize=9,
             clip_on=False, zorder=6)
    ax.set_xlim(-1.5, 1.5)
    ax.set_title(f'query {q}', fontsize=10)
    if k // n_cols == n_rows - 1: ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$  (scaled)')
    if k %  n_cols == 0:          ax.set_ylabel(r'$p(\tau)$')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='upper right')
for k in range(N_QUERIES, n_rows * n_cols):
    axes[k // n_cols][k % n_cols].set_visible(False)
fig.suptitle(f'IHDP r={REALIZATION}   UWYK No-Ancestral naive TE (independence) at N={N_CONTEXT}',
              fontsize=12, y=0.999)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out = os.path.join(_OUTDIR, 'ihdp_n10_te.png')
fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
print(f'[save] {out}')


# ── Figure: OT barycenter of the naive per-query TEs ─────────────────────
sys.path.insert(0, os.path.join(_REPO, 'MALC', 'Optimal_Transport'))
from ot_barycenter import wasserstein_barycenter_1d

bary = wasserstein_barycenter_1d(p_taus_naive, tau_centers)
bary_norm = bary / max(bary.sum() * tau_step, 1e-12)
bary_mode = float(tau_centers[int(np.argmax(bary_norm))])
bary_mean = float((tau_centers * bary_norm).sum() * tau_step)
per_q_means = (tau_centers[None, :] * p_taus_naive).sum(axis=1) * tau_step
mean_of_means = float(per_q_means.mean())
true_ate_local = float(true_cate_scaled.mean())

fig, ax = plt.subplots(figsize=(9, 4.6))
palette_Q = plt.cm.tab10(np.linspace(0, 0.9, N_QUERIES))
for k in range(N_QUERIES):
    ax.plot(tau_centers, p_taus_naive[k], color=palette_Q[k], lw=1.1, alpha=0.35)
ax.fill_between(tau_centers, bary_norm, alpha=0.20, color='#0F8A3C')
ax.plot(tau_centers, bary_norm, color='#0F8A3C', lw=2.6, label='W₂ barycenter (OT)')
ax.axvline(true_ate_local, color='red', ls='--', lw=1.6,
            label=f'true population ATE = {true_ate_local:+.2f}')
ax.axvline(bary_mode, color='#0F8A3C', ls='--', lw=1.6,
            label=f'OT-mode = {bary_mode:+.2f}')
ax.axvline(bary_mean, color='#0F8A3C', ls=':', lw=1.9,
            label=f'OT-mean = {bary_mean:+.2f}')
ax.axvline(mean_of_means, color='#C1420F', ls=':', lw=1.9,
            label=f'mean-of-means = {mean_of_means:+.2f}')
ax.set_xlim(-1.5, 1.5)
ax.set_xlabel(r'$\tau = Y_{do1} - Y_{do0}$  (scaled)')
ax.set_ylabel('density')
ax.set_title(f'IHDP r={REALIZATION}   UWYK No-Ancestral OT aggregation (naive TE) at N={N_CONTEXT}',
              fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.25)
fig.tight_layout()
out = os.path.join(_OUTDIR, 'ihdp_n10_ot.png')
fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
print(f'[save] {out}')

print(f'\n[summary — UWYK-NOANC   (naive independence-derived p(τ))]')
print(f'  true population ATE (scaled) = {true_ate_local:+.3f}')
print(f'  mean-of-per-query-means      = {mean_of_means:+.3f}')
print(f'  W2 barycenter mode           = {bary_mode:+.3f}')
print(f'  W2 barycenter mean           = {bary_mean:+.3f}')
