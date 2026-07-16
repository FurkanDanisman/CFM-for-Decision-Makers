"""
3-way comparison on Do-PFN's synthetic benchmarks: Do-PFN vs UWYK vs OURS.

For each of Do-PFN's synthetic case studies (Observed_Confounder,
Backdoor_Criterion, Frontdoor_Criterion, Observed_Mediator, ...), load N
benchmark instances, 80/20 split, fit each model, predict CATE on test, and
compute PEHE.

Reports mean ± std PEHE per model per case study. This is exactly Do-PFN's
paper's evaluation protocol, so if we don't match their published numbers
for Do-PFN, we're doing something wrong upstream.
"""
from __future__ import annotations
import os, sys, time, pickle, warnings, types
import numpy as np
import torch

warnings.filterwarnings('ignore')

DOPFN_ROOT = '/tmp/dopfn'
UWYK_SRC   = '/tmp/g4cfm_uwyk/src'
UWYK_CKPT  = '/tmp/g4cfm_uwyk/experiments/checkpoints/full_conditioned_model'
OUR_CKPT   = '/Users/furkandanisman/R-PFN/checkpoints/step_50000_final.pt'
OUT_DIR    = '/Users/furkandanisman/R-PFN/experiments'
_REPO      = '/Users/furkandanisman/R-PFN'
DEVICE     = torch.device('cpu')
N_BENCH_PER_CASE = int(os.environ.get('N_BENCH', 10))
CASE_STUDIES = [
    'Observed_Confounder', 'Backdoor_Criterion', 'Frontdoor_Criterion',
    'Observed_Mediator', 'Observed_Mediator_and_Confounder',
    'Unobserved_Confounder', 'Common_Effect', 'Observed_Confounder_Small',
]

# ── Path hackery: Do-PFN, UWYK, and our repo have colliding `models` package ─
# We import each carefully.
def _log(m, t0=None):
    t = f"[{time.time() - t0:6.1f}s]" if t0 is not None else "[  0.0s]"
    print(f"{t} {m}", flush=True)

t0 = time.time()

# ── Patch torch.load globally (both UWYK and Do-PFN need weights_only=False) ─
_orig_torch_load = torch.load
def _p_load(*a, **kw):
    kw.setdefault('weights_only', False); return _orig_torch_load(*a, **kw)
torch.load = _p_load

# ── Register Do-PFN's `datasets` module for pickle deserialization ────────────
sys.path.insert(0, DOPFN_ROOT)
ds_mod = types.ModuleType('datasets')
_src = open(os.path.join(DOPFN_ROOT, 'datasets/__init__.py')).read().split('def load_semi_real')[0]
exec(_src, ds_mod.__dict__)
sys.modules['datasets'] = ds_mod


def load_case_studies(case, n_max=N_BENCH_PER_CASE):
    """Load n_max benchmark instances for a given case study."""
    d = os.path.join(DOPFN_ROOT, 'data/prior_sampling', case)
    out = []
    for i in range(1, n_max + 1):
        fp = os.path.join(d, f'{case}_{i}.pkl')
        if not os.path.exists(fp): continue
        with open(fp, 'rb') as f:
            out.append(pickle.load(f))
    return out


# ── Do-PFN loader ────────────────────────────────────────────────────────────
_log("Loading Do-PFN…")
from scripts.transformer_prediction_interface.base import DoPFNRegressor
from scripts.tabular_metrics.regression import root_mean_squared_error_metric
dopfn = DoPFNRegressor()
_log("Do-PFN ready", t0)


# ── Our model ────────────────────────────────────────────────────────────────
_log("Loading OUR model…", t0)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'MALC'))
from models.InterventionalPFN import InterventionalPFN as OurModel
from losses.BarDistribution2D import unpack_pred
ckpt = torch.load(OUR_CKPT, map_location=DEVICE)
cfg = ckpt['config']; J = cfg['J']
edges_np = ckpt['edges'].cpu().numpy()
bin_width = float(edges_np[1] - edges_np[0])
centers = 0.5 * (edges_np[:-1] + edges_np[1:])
NUM_FEATURES = cfg['num_features']  # 50 padded
our_model = OurModel(
    num_features=NUM_FEATURES, d_model=cfg['d_model'], depth=cfg['depth'],
    heads_feat=cfg['heads'], heads_samp=cfg['heads'], dropout=0.0,
    output_dim=J*J + 9 + 4, hidden_mult=cfg['hidden_mult'],
    normalize_features=True, normalize_treatment=False,
    use_treatment_in_query=False, use_checkpoint=False,
).to(DEVICE).eval()
our_model.load_state_dict(ckpt['model_state_dict'])
_log("OUR model ready", t0)


# ── UWYK — imported by file path to bypass package name collision ────────────
_log("Loading UWYK…", t0)
import importlib.util
_uwyk_mod_path = os.path.join(UWYK_SRC, 'models', 'GraphConditionedInterventionalPFN_sklearn.py')
_spec = importlib.util.spec_from_file_location('_uwyk_gcip', _uwyk_mod_path)
_uwyk_mod = importlib.util.module_from_spec(_spec)
# temporarily prepend UWYK's src so its intra-package imports resolve
sys.path.insert(0, UWYK_SRC)
_spec.loader.exec_module(_uwyk_mod)
sys.path.remove(UWYK_SRC)
UWYK_Model = _uwyk_mod.GraphConditionedInterventionalPFNSklearn
uwyk = UWYK_Model(
    config_path=os.path.join(UWYK_CKPT, 'config.yaml'),
    checkpoint_path=os.path.join(UWYK_CKPT, 'model.pt'),
    device='cpu', verbose=False,
).load()
_log("UWYK ready", t0)


# ── Helper: pad features to NUM_FEATURES with zeros ─────────────────────────
def pad_features(x, target_L=NUM_FEATURES):
    """x: (N, L) tensor. Pad columns with zeros to target_L."""
    L = x.shape[1]
    if L >= target_L: return x[:, :target_L]
    pad = torch.zeros(x.shape[0], target_L - L, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)


# ── Prediction wrappers ──────────────────────────────────────────────────────
def predict_dopfn(train_x, train_y, test_x):
    """Do-PFN: fit + predict_cate. Returns numpy (M,) predicted CATE."""
    r = DoPFNRegressor()
    r.fit(train_x, train_y)
    return np.asarray(r.predict_cate(test_x))


def predict_uwyk(train_x, train_y, test_x):
    """UWYK: predict E[Y|do(T=1), X, D] - E[Y|do(T=0), X, D].
    Input format: x[:,0] = T (binary), x[:,1:] = covariates.
    UWYK expects a fixed L=50 features (its trained max_features)."""
    x_np = train_x.numpy() if isinstance(train_x, torch.Tensor) else np.asarray(train_x)
    tx = test_x.numpy() if isinstance(test_x, torch.Tensor) else np.asarray(test_x)
    y_np = train_y.numpy() if isinstance(train_y, torch.Tensor) else np.asarray(train_y)

    UWYK_L = 50   # UWYK was trained at 50 features
    T_obs = x_np[:, 0].astype(np.float32)
    X_obs_real = x_np[:, 1:].astype(np.float32)
    Y_obs = y_np.astype(np.float32)
    X_intv_real = tx[:, 1:].astype(np.float32)

    # Pad X to UWYK_L with zeros (same trick our model uses)
    def _pad(arr, L):
        if arr.shape[1] >= L: return arr[:, :L]
        z = np.zeros((arr.shape[0], L - arr.shape[1]), dtype=np.float32)
        return np.concatenate([arr, z], axis=1)
    X_obs = _pad(X_obs_real, UWYK_L)
    X_intv = _pad(X_intv_real, UWYK_L)

    M = X_intv.shape[0]
    T1 = np.ones(M, dtype=np.float32)
    T0 = np.zeros(M, dtype=np.float32)
    # Adjacency is (L+2, L+2) with all zeros = no known edges
    adj = np.zeros((UWYK_L + 2, UWYK_L + 2), dtype=np.float32)
    y1 = uwyk.predict(X_obs, T_obs, Y_obs, X_intv, T1, adj,
                      prediction_type='mean', batched=False)
    y0 = uwyk.predict(X_obs, T_obs, Y_obs, X_intv, T0, adj,
                      prediction_type='mean', batched=False)
    return np.asarray(y1) - np.asarray(y0)


def predict_ours(train_x, train_y, test_x):
    """OURS: joint 2D BarDistribution → E[τ] = E[Y_do1] - E[Y_do0]."""
    x_np = train_x if isinstance(train_x, torch.Tensor) else torch.as_tensor(train_x)
    y_np = train_y if isinstance(train_y, torch.Tensor) else torch.as_tensor(train_y)
    tx  = test_x if isinstance(test_x, torch.Tensor) else torch.as_tensor(test_x)

    T_obs = x_np[:, 0:1].float()
    X_obs = pad_features(x_np[:, 1:].float())      # (N, 50)
    Y_obs = y_np.reshape(-1, 1).float()
    X_intv = pad_features(tx[:, 1:].float())       # (M, 50)

    X_obs_b  = X_obs.unsqueeze(0).to(DEVICE)       # (1, N, 50)
    T_obs_b  = T_obs.unsqueeze(0).to(DEVICE)       # (1, N, 1)
    Y_obs_b  = Y_obs.unsqueeze(0).to(DEVICE)       # (1, N, 1)
    X_intv_b = X_intv.unsqueeze(0).to(DEVICE)      # (1, M, 50)

    with torch.no_grad():
        pred = our_model(X_obs_b, T_obs_b, Y_obs_b, X_intv_b)['predictions'][0]  # (M, ...)

    M = pred.shape[0]
    taus = np.zeros(M)
    for i in range(M):
        p_mat, *_ = unpack_pred(pred[i], J, bin_width)
        p = p_mat.detach().cpu().numpy()
        E_y0 = (centers[:, None] * p).sum()
        E_y1 = (centers[None, :] * p).sum()
        taus[i] = float(E_y1 - E_y0)
    return taus


def pehe(y_pred, y_true):
    y_true = y_true.numpy() if isinstance(y_true, torch.Tensor) else np.asarray(y_true)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# ── Run comparison ──────────────────────────────────────────────────────────
results = {}   # results[case] = {'dopfn': [...], 'uwyk': [...], 'ours': [...]}
for case in CASE_STUDIES:
    _log(f"\n=== {case} ===", t0)
    benches = load_case_studies(case, n_max=N_BENCH_PER_CASE)
    if not benches:
        _log(f"  no data; skipping", t0); continue
    results[case] = {'dopfn': [], 'uwyk': [], 'ours': []}
    for k, ds in enumerate(benches):
        n = ds.x.shape[0]
        n_train = int(0.8 * n)
        train_x, test_x = ds.x[:n_train], ds.x[n_train:]
        train_y, test_y = ds.y[:n_train], ds.y[n_train:]
        cate_true = ds.cate[n_train:]

        try:
            p_dopfn = predict_dopfn(train_x, train_y, test_x)
            e_dopfn = pehe(p_dopfn, cate_true)
        except Exception as e:
            _log(f"  bench {k+1}: DoPFN FAIL ({type(e).__name__}: {str(e)[:80]})", t0)
            e_dopfn = np.nan

        try:
            p_uwyk = predict_uwyk(train_x, train_y, test_x)
            e_uwyk = pehe(p_uwyk, cate_true)
        except Exception as e:
            _log(f"  bench {k+1}: UWYK FAIL ({type(e).__name__}: {str(e)[:80]})", t0)
            e_uwyk = np.nan

        try:
            p_ours = predict_ours(train_x, train_y, test_x)
            e_ours = pehe(p_ours, cate_true)
        except Exception as e:
            _log(f"  bench {k+1}: OURS FAIL ({type(e).__name__}: {str(e)[:80]})", t0)
            e_ours = np.nan

        results[case]['dopfn'].append(e_dopfn)
        results[case]['uwyk'].append(e_uwyk)
        results[case]['ours'].append(e_ours)
        _log(f"  bench {k+1}/{len(benches)} n={n}  "
             f"DoPFN PEHE={e_dopfn:.4f}  UWYK PEHE={e_uwyk:.4f}  OURS PEHE={e_ours:.4f}", t0)


# ── Aggregate + print table ──────────────────────────────────────────────────
_log("\n" + "="*90, t0)
_log(f"{'Case study':<40} {'Do-PFN':>15} {'UWYK':>15} {'OURS':>15}", t0)
_log("-"*90, t0)
for case, r in results.items():
    d = np.array(r['dopfn']); u = np.array(r['uwyk']); o = np.array(r['ours'])
    d = d[~np.isnan(d)]; u = u[~np.isnan(u)]; o = o[~np.isnan(o)]
    dstr = f"{d.mean():.4f}±{d.std():.4f}" if len(d) else "n/a"
    ustr = f"{u.mean():.4f}±{u.std():.4f}" if len(u) else "n/a"
    ostr = f"{o.mean():.4f}±{o.std():.4f}" if len(o) else "n/a"
    _log(f"{case:<40} {dstr:>15} {ustr:>15} {ostr:>15}", t0)


# ── Save arrays ──────────────────────────────────────────────────────────────
np.savez(os.path.join(OUT_DIR, 'DOPFN_BENCH_results.npz'),
         **{f'{case}_{model}': np.array(r[model]) for case, r in results.items() for model in r})
_log(f"\nDone. Total {time.time() - t0:.1f}s", t0)
