"""IHDP demonstration — marginals comparison between Ours (2DMALC) and UWYK-NoAnc.

Same IHDP realisation, same first-N training observations, same 10 test
queries as sibling scripts. Overlays the two pipelines' marginals on the
same axes per query. Uses a single palette per outcome so the two
p(Y_do0) curves share a colour and the two p(Y_do1) curves share a
colour; the two pipelines are distinguished by line style.

  * Ours (UWYK-2DMALC)  solid line, filled dot for E[Y]
  * UWYK No-Ancestral   dashed line, open dot for E[Y]

Output:
  MARGINALS-COMPARE/
    ihdp_n10_marginals_compare.png
"""
from __future__ import annotations
import os, sys, importlib, types
import numpy as np
import torch
import matplotlib.pyplot as plt

_HERE   = os.path.dirname(os.path.abspath(__file__))
_BENCH  = os.path.dirname(_HERE)
_REPO   = os.environ.get('REPO', os.path.dirname(_BENCH))
_OUTDIR = os.path.join(_HERE, 'ihdp_n10', 'MARGINALS-COMPARE')
os.makedirs(_OUTDIR, exist_ok=True)

CAUSALPFN     = os.environ.get('CAUSALPFN', '')
DOPFN         = os.environ.get('DOPFN', '')
UWYK_SRC      = os.environ.get('UWYK_SRC', '')
CHECKPOINT    = os.environ.get('CHECKPOINT',
                                os.path.join(_REPO, 'checkpoints', 'step_50000_final.pt'))
REALIZATION   = int(os.environ.get('REALIZATION', 0))
N_CONTEXT     = int(os.environ.get('N_CONTEXT', 200))
N_QUERIES     = int(os.environ.get('N_QUERIES', 10))
N_SAMPLES     = int(os.environ.get('UWYK_N_SAMPLES', 1024))
_UWYK_ROOT    = os.path.dirname(os.path.abspath(UWYK_SRC.rstrip('/'))) if UWYK_SRC else ''
UWYK_CKPT_DIR = os.environ.get(
    'UWYK_CKPT_DIR',
    os.path.join(_UWYK_ROOT,
                  'experiments/checkpoints/full_conditioned_model/'
                  'final_earlytest_full_conditioning_16773252.0')
    if _UWYK_ROOT else '')

if not (os.path.isfile(CHECKPOINT) and os.path.isdir(CAUSALPFN)
        and os.path.isdir(DOPFN) and os.path.isdir(UWYK_SRC)
        and os.path.isdir(UWYK_CKPT_DIR)):
    print('[skip] paths missing:')
    for k, v in (('CHECKPOINT', CHECKPOINT), ('CAUSALPFN', CAUSALPFN),
                  ('DOPFN', DOPFN), ('UWYK_SRC', UWYK_SRC),
                  ('UWYK_CKPT_DIR', UWYK_CKPT_DIR)):
        print(f'   {k:<14} = {v!r}')
    sys.exit(0)


# ── Load OURS (InterventionalPFN) ────────────────────────────────────────
sys.path.insert(0, _REPO)
from models.InterventionalPFN import InterventionalPFN
from losses.BarDistribution2D import unpack_pred

DEVICE = torch.device('cpu')
_orig_load = torch.load
def _p_load(*a, **kw):
    kw.setdefault('weights_only', False); return _orig_load(*a, **kw)
torch.load = _p_load

ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
cfg = ckpt['config']; J = cfg['J']
edges_np = ckpt['edges'].cpu().numpy()
bin_width = float(edges_np[1] - edges_np[0])
centers = 0.5 * (edges_np[:-1] + edges_np[1:])
NUM_FEATURES_OURS = cfg['num_features']
ours_model = InterventionalPFN(
    num_features=NUM_FEATURES_OURS, d_model=cfg['d_model'], depth=cfg['depth'],
    heads_feat=cfg['heads'], heads_samp=cfg['heads'], dropout=0.0,
    output_dim=J*J + 9 + 4, hidden_mult=cfg['hidden_mult'],
    normalize_features=True, normalize_treatment=False,
    use_treatment_in_query=False, use_checkpoint=False,
).to(DEVICE).eval()
ours_model.load_state_dict(ckpt['model_state_dict'])
print(f'[load] OURS  J={J}', flush=True)


# ── Load IHDP realisation ───────────────────────────────────────────────
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
Y_context_raw = y_train_full[:N_CONTEXT].astype(np.float32).reshape(-1, 1)

y_min = float(y_train_full.min()); y_max = float(y_train_full.max())
y_rng = max(y_max - y_min, 1e-6)
Y_context = ((Y_context_raw - y_min) / y_rng * 2.0 - 1.0).astype(np.float32)
true_cate_scaled = true_cate * (2.0 / y_rng)

order = np.argsort(true_cate_scaled)
qs = np.linspace(0.05, 0.95, N_QUERIES)
QUERY_IDXS = order[(qs * (len(true_cate_scaled) - 1)).astype(int)].tolist()
print(f'[data] IHDP r={REALIZATION}  N_context={N_CONTEXT}  '
      f'queries={QUERY_IDXS}', flush=True)


# ── OURS inference (produces the 2D joint we marginalise) ──────────────
def _pad(X, n_feat):
    d = X.shape[1]
    if d < n_feat:
        pad = np.full((X.shape[0], n_feat - d), np.nan, dtype=X.dtype)
        return np.concatenate([X.astype(np.float32), pad], axis=1)
    return X.astype(np.float32)[:, :n_feat]


X_ctx_ours = torch.from_numpy(_pad(X_context, NUM_FEATURES_OURS)).unsqueeze(0)
T_ctx_ours = torch.from_numpy(T_context).unsqueeze(0)
Y_ctx_ours = torch.from_numpy(Y_context).unsqueeze(0)
X_qry_ours = torch.from_numpy(_pad(X_test.astype(np.float32), NUM_FEATURES_OURS)).unsqueeze(0)
with torch.no_grad():
    pred = ours_model(X_ctx_ours, T_ctx_ours, Y_ctx_ours, X_qry_ours)['predictions'][0]

p_y0_ours = np.zeros((N_QUERIES, len(centers)), dtype=np.float64)
p_y1_ours = np.zeros((N_QUERIES, len(centers)), dtype=np.float64)
for k, q in enumerate(QUERY_IDXS):
    p_mat, *_ = unpack_pred(pred[q], J, bin_width)
    pm = p_mat.detach().cpu().numpy()
    m0 = pm.sum(axis=1); m1 = pm.sum(axis=0)
    p_y0_ours[k] = m0 / max(m0.sum() * bin_width, 1e-12)
    p_y1_ours[k] = m1 / max(m1.sum() * bin_width, 1e-12)


# ── Load UWYK NoAnc + sample marginals ─────────────────────────────────
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

uwyk_model = UWYK_pre_mod.PreprocessingGraphConditionedPFN(
    config_path=os.path.join(UWYK_CKPT_DIR, 'best_model_config.yaml'),
    checkpoint_path=os.path.join(UWYK_CKPT_DIR, 'best_model.pt'),
    device='cpu', verbose=False,
).load()
NUM_FEATURES_UWYK = uwyk_model.model.num_features
print(f'[load] UWYK NoAnc  num_features={NUM_FEATURES_UWYK}', flush=True)

X_train_uwyk = _pad(X_context, NUM_FEATURES_UWYK)
X_test_uwyk  = _pad(X_test.astype(np.float32), NUM_FEATURES_UWYK)

t_train_orig = T_context.reshape(-1, 1)
y_train      = Y_context_raw.reshape(-1, 1)
mean_y_t0 = float(y_train[t_train_orig == 0].mean())
mean_y_t1 = float(y_train[t_train_orig == 1].mean())
t_train_enc = np.where(t_train_orig == 0, mean_y_t0, mean_y_t1).astype(np.float32)
uwyk_model.fit(X_train_uwyk, t_train_enc, y_train)

n_real_features = X_context.shape[1]
adj = np.zeros((NUM_FEATURES_UWYK + 2, NUM_FEATURES_UWYK + 2), dtype=np.float32)
for i in range(n_real_features, NUM_FEATURES_UWYK):
    fi = 2 + i
    adj[fi, :] = -1.0; adj[:, fi] = -1.0; adj[fi, fi] = -1.0

X_intv = X_test_uwyk[QUERY_IDXS]
T_intv_0 = np.full((len(QUERY_IDXS), 1), mean_y_t0, dtype=np.float32)
T_intv_1 = np.full((len(QUERY_IDXS), 1), mean_y_t1, dtype=np.float32)


def _predict_samples(T_intv):
    n_have = 0; chunks = []
    while n_have < N_SAMPLES:
        r = uwyk_model.predict(
            X_obs=X_train_uwyk, T_obs=t_train_enc, Y_obs=y_train,
            X_intv=X_intv, T_intv=T_intv,
            adjacency_matrix=adj,
            prediction_type='sample', inverse_transform=True,
        )
        arr = np.asarray(r).reshape(len(QUERY_IDXS), -1)
        chunks.append(arr); n_have += arr.shape[1]
    return np.concatenate(chunks, axis=1)[:, :N_SAMPLES]


print(f'[predict] sampling {N_SAMPLES} draws per query per treatment', flush=True)
Y0_samples = _predict_samples(T_intv_0)
Y1_samples = _predict_samples(T_intv_1)

Y0_scaled = (Y0_samples - y_min) / y_rng * 2.0 - 1.0
Y1_scaled = (Y1_samples - y_min) / y_rng * 2.0 - 1.0

# Histogram on the same centers grid Ours uses
edges_shared = np.concatenate([[centers[0] - bin_width / 2],
                                centers + bin_width / 2])
p_y0_uwyk = np.zeros_like(p_y0_ours)
p_y1_uwyk = np.zeros_like(p_y1_ours)
for k in range(N_QUERIES):
    h0, _ = np.histogram(Y0_scaled[k], bins=edges_shared, density=True)
    h1, _ = np.histogram(Y1_scaled[k], bins=edges_shared, density=True)
    p_y0_uwyk[k] = h0
    p_y1_uwyk[k] = h1


# ── Plot overlay ────────────────────────────────────────────────────────
n_cols = 5 if N_QUERIES == 10 else 3
n_rows = (N_QUERIES + n_cols - 1) // n_cols

COLOR_DO0 = '#2E7DAF'
COLOR_DO1 = '#7B3E9E'

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.6 * n_cols, 3.7 * n_rows),
                          squeeze=False)
for k, q in enumerate(QUERY_IDXS):
    ax = axes[k // n_cols][k % n_cols]
    # OURS — solid lines + filled dots
    ax.plot(centers, p_y0_ours[k], color=COLOR_DO0, lw=1.9, ls='-',
             label=r'$p(Y_{do0})$  Ours' if k == 0 else None)
    ax.plot(centers, p_y1_ours[k], color=COLOR_DO1, lw=1.9, ls='-',
             label=r'$p(Y_{do1})$  Ours' if k == 0 else None)
    E_y0_ours = float((centers * p_y0_ours[k]).sum() * bin_width)
    E_y1_ours = float((centers * p_y1_ours[k]).sum() * bin_width)
    ax.plot(E_y0_ours, float(np.interp(E_y0_ours, centers, p_y0_ours[k])),
             'o', color=COLOR_DO0, markersize=9, markeredgecolor='white',
             markeredgewidth=1.0, zorder=6)
    ax.plot(E_y1_ours, float(np.interp(E_y1_ours, centers, p_y1_ours[k])),
             'o', color=COLOR_DO1, markersize=9, markeredgecolor='white',
             markeredgewidth=1.0, zorder=6)
    # UWYK No-Ancestral — dashed lines + open dots
    ax.plot(centers, p_y0_uwyk[k], color=COLOR_DO0, lw=1.9, ls='--',
             label=r'$p(Y_{do0})$  UWYK-NoAnc' if k == 0 else None)
    ax.plot(centers, p_y1_uwyk[k], color=COLOR_DO1, lw=1.9, ls='--',
             label=r'$p(Y_{do1})$  UWYK-NoAnc' if k == 0 else None)
    E_y0_uwyk = float((centers * p_y0_uwyk[k]).sum() * bin_width)
    E_y1_uwyk = float((centers * p_y1_uwyk[k]).sum() * bin_width)
    ax.plot(E_y0_uwyk, float(np.interp(E_y0_uwyk, centers, p_y0_uwyk[k])),
             'o', markerfacecolor='none', markeredgecolor=COLOR_DO0,
             markersize=10, markeredgewidth=1.8, zorder=6)
    ax.plot(E_y1_uwyk, float(np.interp(E_y1_uwyk, centers, p_y1_uwyk[k])),
             'o', markerfacecolor='none', markeredgecolor=COLOR_DO1,
             markersize=10, markeredgewidth=1.8, zorder=6)
    ax.set_title(f'query {q}   $\\tau_{{true}}$={true_cate_scaled[q]:+.2f}',
                  fontsize=10)
    if k // n_cols == n_rows - 1: ax.set_xlabel(r'$Y$  (scaled)')
    if k %  n_cols == 0:          ax.set_ylabel('density')
    ax.grid(alpha=0.25)
    if k == 0: ax.legend(fontsize=8, loc='upper right')
for k in range(N_QUERIES, n_rows * n_cols):
    axes[k // n_cols][k % n_cols].set_visible(False)
fig.suptitle(f'IHDP r={REALIZATION}   marginals comparison '
              f'(Ours vs UWYK No-Ancestral) at N={N_CONTEXT}',
              fontsize=12, y=0.999)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out = os.path.join(_OUTDIR, 'ihdp_n10_marginals_compare.png')
fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
print(f'[save] {out}')
