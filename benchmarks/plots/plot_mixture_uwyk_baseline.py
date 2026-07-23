"""UWYK-Baseline marginals on the same mixture SCM used in plot_mixture_scm.py.

We instantiate the same 3-cluster mixture SCM (see plot_mixture_scm.py for the
generative process), then have UWYK-Baseline (the separately-trained
unconditional checkpoint from Graphs4CausalFoundationModels) predict the
outcome distribution p(Y | X_i, do(t)) for each of the six diagnostic queries
under both t=0 and t=1. The predicted marginals are extracted from UWYK's
1D BarDistribution head and plotted in the same 2×3 layout as
plot_mixture_scm.py — so the two figures can sit side-by-side and make the
"joint vs marginal-only" argument visually.

Environment knobs mirror plot_mixture_scm.py, plus:
  UWYK_SRC         path to Graphs4CausalFoundationModels/src (has models/, priors/)
  UWYK_CKPT_DIR    path to no_graph_conditioning/unconditional/ (has best_model.pt
                    and best_model_config.yaml). If missing, the script prints a
                    skip message and exits cleanly.
"""
from __future__ import annotations
import os, sys, importlib
os.environ.setdefault('PYTHONHASHSEED', '0')
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import matplotlib.pyplot as plt

_HERE  = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
_REPO  = os.path.dirname(_BENCH)
_OUTDIR = os.path.join(_HERE, 'mixture_scm')
os.makedirs(_OUTDIR, exist_ok=True)

# Same mixture-SCM knobs as plot_mixture_scm.py — defaults reproduce that fig
UWYK_SRC       = os.environ.get('UWYK_SRC',
                                  '/Users/furkandanisman/.claude/jobs/7758df90/tmp/uwyk_upstream/src')
UWYK_CKPT_DIR  = os.environ.get('UWYK_CKPT_DIR',
                                  '/Users/furkandanisman/.claude/jobs/7758df90/tmp/uwyk_upstream/experiments/checkpoints/no_graph_conditioning/unconditional')
N_CONTEXT      = int(os.environ.get('N_CONTEXT', 1000))
N_CONTEXT_MAX  = int(os.environ.get('N_CONTEXT_MAX', max(2000, N_CONTEXT + 100)))
R              = float(os.environ.get('R', 0.7))
SIGMA_X        = float(os.environ.get('SIGMA_X', 0.35))
A_VEC = [float(x) for x in os.environ.get('A', '0.4,-0.4,-0.4').split(',')]
B_VEC = [float(x) for x in os.environ.get('B', '-0.4,-0.4,0.4').split(',')]
SIGMA_Y = float(os.environ.get('SIGMA_Y', 0.05))
SEED    = int(os.environ.get('SEED', 0))
OUT_PREFIX = os.environ.get('OUT_PREFIX', 'mixture3_uwyk_baseline')
K = len(A_VEC)
_ANGLES_DEG = [90.0, 210.0, 330.0][:K]
MU_2D = np.array([[R * np.cos(np.deg2rad(a)), R * np.sin(np.deg2rad(a))]
                   for a in _ANGLES_DEG], dtype=np.float32)
TAU = [B_VEC[z] - A_VEC[z] for z in range(K)]


# ── Skip early if the UWYK checkpoint isn't available ───────────────────────
ckpt_pt   = os.path.join(UWYK_CKPT_DIR, 'best_model.pt')
ckpt_yaml = os.path.join(UWYK_CKPT_DIR, 'best_model_config.yaml')
if not (os.path.isfile(ckpt_pt) and os.path.isfile(ckpt_yaml) and os.path.isdir(UWYK_SRC)):
    print(f'[skip] UWYK checkpoint/source not accessible at:')
    print(f'   UWYK_SRC       = {UWYK_SRC}')
    print(f'   UWYK_CKPT_DIR  = {UWYK_CKPT_DIR}')
    print(f'Set UWYK_SRC and UWYK_CKPT_DIR to run this script (or invoke on '
          f'the machine where the git-lfs blob was pulled).')
    sys.exit(0)


# ── Load UWYK Baseline ──────────────────────────────────────────────────────
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
def _patched_load(*a, **kw):
    kw.setdefault('weights_only', False); return _orig_load(*a, **kw)
torch.load = _patched_load

uwyk_model = UWYK_pre_mod.PreprocessingGraphConditionedPFN(
    config_path=ckpt_yaml,
    checkpoint_path=ckpt_pt,
    device='cpu', verbose=False,
).load()
torch.load = _orig_load
NUM_FEATURES = uwyk_model.model.num_features
print(f'[load] UWYK Baseline  num_features={NUM_FEATURES}', flush=True)


# ── Sample mixture SCM (same seed → same data as plot_mixture_scm.py) ───────
rng = np.random.default_rng(SEED)
Z_ctx = rng.integers(0, K, size=N_CONTEXT_MAX)
X_ctx = np.zeros((N_CONTEXT_MAX, NUM_FEATURES), dtype=np.float32)
noise2d = rng.normal(0, SIGMA_X, size=(N_CONTEXT_MAX, 2)).astype(np.float32)
X_ctx[:, :2] = MU_2D[Z_ctx] + noise2d
X_ctx[:, 2:] = np.nan
T_ctx = rng.binomial(1, 0.5, size=N_CONTEXT_MAX).astype(np.float32)
Y0_ctx = np.array(A_VEC)[Z_ctx] + rng.normal(0, SIGMA_Y, size=N_CONTEXT_MAX)
Y1_ctx = np.array(B_VEC)[Z_ctx] + rng.normal(0, SIGMA_Y, size=N_CONTEXT_MAX)
Y_ctx  = (T_ctx * Y1_ctx + (1 - T_ctx) * Y0_ctx).astype(np.float32)

# The 6 queries (same X positions as plot_mixture_scm.py)
QUERY_LABELS_XY = [
    ('q0  cluster-0 centre',        MU_2D[0]),
    ('q1  cluster-1 centre',        MU_2D[1]),
    ('q2  cluster-2 centre',        MU_2D[2]),
    ('q3  midpoint of 0↔1',         0.5 * (MU_2D[0] + MU_2D[1])),
    ('q4  midpoint of 1↔2',         0.5 * (MU_2D[1] + MU_2D[2])),
    ('q5  triangle centroid',       MU_2D.mean(axis=0)),
]
Q = len(QUERY_LABELS_XY)
X_intv = np.zeros((Q, NUM_FEATURES), dtype=np.float32)
for k, (_, xy) in enumerate(QUERY_LABELS_XY):
    X_intv[k, 0] = xy[0]; X_intv[k, 1] = xy[1]; X_intv[k, 2:] = np.nan

# UWYK protocol: target-encoded T, zero adjacency (Baseline: uses no graph)
X_train = X_ctx[:N_CONTEXT]
t_train_orig = T_ctx[:N_CONTEXT].reshape(-1, 1)
y_train      = Y_ctx[:N_CONTEXT].reshape(-1, 1)
mean_y_t0 = float(y_train[t_train_orig == 0].mean())
mean_y_t1 = float(y_train[t_train_orig == 1].mean())
t_train   = np.where(t_train_orig == 0, mean_y_t0, mean_y_t1).astype(np.float32)
uwyk_model.fit(X_train, t_train, y_train)

# Baseline uses zero-adjacency (no graph info)
adj = np.zeros((NUM_FEATURES + 2, NUM_FEATURES + 2), dtype=np.float32)
n_real_features = 2   # only first two features are real; rest are NaN
for i in range(n_real_features, NUM_FEATURES):
    fi = 2 + i
    adj[fi, :] = -1.0; adj[:, fi] = -1.0; adj[fi, fi] = -1.0

T_intv_1 = np.full((Q, 1), mean_y_t1, dtype=np.float32)
T_intv_0 = np.full((Q, 1), mean_y_t0, dtype=np.float32)

# Predict distributions (not just means). If UWYK's predict doesn't support
# "distribution", we fall back to sampling.
try:
    y1_dist = uwyk_model.predict(X_obs=X_train, T_obs=t_train, Y_obs=y_train,
                                   X_intv=X_intv, T_intv=T_intv_1,
                                   adjacency_matrix=adj,
                                   prediction_type='distribution',
                                   inverse_transform=True)
    y0_dist = uwyk_model.predict(X_obs=X_train, T_obs=t_train, Y_obs=y_train,
                                   X_intv=X_intv, T_intv=T_intv_0,
                                   adjacency_matrix=adj,
                                   prediction_type='distribution',
                                   inverse_transform=True)
    # y*_dist may be a tuple (bin_centers, probs) or an array — normalise to (Q, nbins)
    if isinstance(y1_dist, tuple):
        y_centers = np.asarray(y1_dist[0]).reshape(-1)
        p_y1_per_q = np.asarray(y1_dist[1])
        p_y0_per_q = np.asarray(y0_dist[1])
    else:
        p_y1_per_q = np.asarray(y1_dist)
        p_y0_per_q = np.asarray(y0_dist)
        y_centers = np.linspace(-1.0, 1.0, p_y1_per_q.shape[-1])
    p_y1_per_q = p_y1_per_q.reshape(Q, -1)
    p_y0_per_q = p_y0_per_q.reshape(Q, -1)
    use_samples = False
    print('[predict] using prediction_type="distribution"', flush=True)
except Exception as exc:
    print(f'[predict] "distribution" not supported ({exc}); falling back to samples',
          flush=True)
    use_samples = True
    # Fallback: draw N samples per query via prediction_type="samples"
    N_SAMPLES = 512
    y1_samples = uwyk_model.predict(X_obs=X_train, T_obs=t_train, Y_obs=y_train,
                                      X_intv=X_intv, T_intv=T_intv_1,
                                      adjacency_matrix=adj,
                                      prediction_type='samples',
                                      inverse_transform=True, n_samples=N_SAMPLES)
    y0_samples = uwyk_model.predict(X_obs=X_train, T_obs=t_train, Y_obs=y_train,
                                      X_intv=X_intv, T_intv=T_intv_0,
                                      adjacency_matrix=adj,
                                      prediction_type='samples',
                                      inverse_transform=True, n_samples=N_SAMPLES)
    # Estimate density via histogram
    y_centers = np.linspace(-1.0, 1.0, 100)
    bw = y_centers[1] - y_centers[0]
    p_y0_per_q = np.zeros((Q, len(y_centers)))
    p_y1_per_q = np.zeros((Q, len(y_centers)))
    for k in range(Q):
        h0, _ = np.histogram(y0_samples[k].ravel(), bins=len(y_centers) + 1,
                              range=(y_centers[0] - bw / 2, y_centers[-1] + bw / 2), density=True)
        h1, _ = np.histogram(y1_samples[k].ravel(), bins=len(y_centers) + 1,
                              range=(y_centers[0] - bw / 2, y_centers[-1] + bw / 2), density=True)
        p_y0_per_q[k] = h0[:len(y_centers)]
        p_y1_per_q[k] = h1[:len(y_centers)]

# ── Also compute posteriors so we can annotate cluster-truth as in plot_mixture_scm
def posterior_z(xy):
    diffs = MU_2D - np.asarray(xy)[None, :]
    log_p = -0.5 * (diffs ** 2).sum(axis=1) / (SIGMA_X ** 2)
    log_p -= log_p.max()
    p = np.exp(log_p); p /= p.sum()
    return p


# ── Plot — same 2×3 layout as plot_mixture_scm's marginals ─────────────────
n_cols = 3
n_rows = (Q + n_cols - 1) // n_cols
palette = {'do0': '#2E7DAF', 'do1': '#7B3E9E'}
bw = float(y_centers[1] - y_centers[0])
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.6 * n_rows),
                          squeeze=False)
for k in range(Q):
    ax = axes[k // n_cols][k % n_cols]
    label, xy = QUERY_LABELS_XY[k]
    post = posterior_z(xy)
    p_y0 = p_y0_per_q[k] / max(p_y0_per_q[k].sum() * bw, 1e-12)
    p_y1 = p_y1_per_q[k] / max(p_y1_per_q[k].sum() * bw, 1e-12)
    ax.plot(y_centers, p_y0, color=palette['do0'], lw=1.8, label=r'$p(Y_{do0})$')
    ax.plot(y_centers, p_y1, color=palette['do1'], lw=1.8, label=r'$p(Y_{do1})$')
    E_y0 = float((y_centers * p_y0).sum() * bw)
    E_y1 = float((y_centers * p_y1).sum() * bw)
    y_at_Ey0 = float(np.interp(E_y0, y_centers, p_y0))
    y_at_Ey1 = float(np.interp(E_y1, y_centers, p_y1))
    ax.plot(E_y0, y_at_Ey0, 'o', color='red', markersize=8, zorder=5,
             label=r'$\mathbb{E}[Y_{do0}]$, $\mathbb{E}[Y_{do1}]$' if k == 0 else None)
    ax.plot(E_y1, y_at_Ey1, 'o', color='red', markersize=8, zorder=5)
    for z in range(K):
        ax.axvline(A_VEC[z], color=palette['do0'], ls=':', lw=1.0, alpha=0.3 + 0.6 * post[z])
        ax.axvline(B_VEC[z], color=palette['do1'], ls=':', lw=1.0, alpha=0.3 + 0.6 * post[z])
    ax.set_title(label, fontsize=10)
    ax.grid(alpha=0.3)
    if k == 0: ax.legend(fontsize=8, loc='upper right')
    if k // n_cols == n_rows - 1: ax.set_xlabel(r'$Y$ (scaled)')
    if k % n_cols == 0:            ax.set_ylabel(r'density')

for k in range(Q, n_rows * n_cols):
    axes[k // n_cols][k % n_cols].set_visible(False)
fig.suptitle(f'UWYK-Baseline marginal potential-outcome densities at '
              f'N={N_CONTEXT} — same mixture SCM (K={K})', fontsize=12, y=0.999)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out_path = os.path.join(_OUTDIR, f'{OUT_PREFIX}_marginals.png')
fig.savefig(out_path, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'[save] {out_path}')
