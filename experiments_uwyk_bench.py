"""
3-way comparison on UWYK's paper benchmarks: IHDP, ACIC, Lalonde-CPS, Lalonde-PSID.

For each benchmark, iterate over realizations. Standard PEHE / ATE-relative-error
metrics per Sec. of UWYK & Do-PFN papers.

Per-dataset methodology:
  - Take (X_train, t_train, y_train, X_test, true_cate)
  - Do-PFN: fit(x=[t|X_train], y_train), predict_cate on [t|X_test]
  - UWYK:   predict E[Y|do(T=1), X, D] and T=0, take diff. Adjacency all-zeros.
  - OURS:   feed (X_obs, T_obs, Y_obs, X_intv), extract E[τ] from joint p_mat.

For each realization: PEHE = sqrt(mean((cate_pred - true_cate)²)),
                      ATE_rel_err = |mean(cate_pred) - true_ATE| / |true_ATE|

Aggregate: mean ± std across realizations per (dataset, model).
"""
from __future__ import annotations
import os, sys, time, types, warnings, importlib.util
import numpy as np
import torch
warnings.filterwarnings('ignore')

DOPFN_ROOT   = '/tmp/dopfn'
CAUSALPFN    = '/tmp/causalpfn_full'
UWYK_SRC     = '/tmp/g4cfm_uwyk/src'
UWYK_CKPT    = '/tmp/g4cfm_uwyk/experiments/checkpoints/full_conditioned_model'
OUR_CKPT     = '/Users/furkandanisman/R-PFN/checkpoints/step_50000_final.pt'
OUT_DIR      = '/Users/furkandanisman/R-PFN/experiments'
_REPO        = '/Users/furkandanisman/R-PFN'
DEVICE       = torch.device('cpu')
N_REAL       = int(os.environ.get('N_REAL', 20))  # realizations per dataset
DATASETS     = os.environ.get('DATASETS', 'IHDP,ACIC').split(',')
MAX_CONTEXT  = int(os.environ.get('MAX_CONTEXT', 1000))  # subsample large train sets
MAX_QUERIES  = int(os.environ.get('MAX_QUERIES', 150))   # subsample large test sets


def _log(m, t0=None):
    t = f"[{time.time() - t0:6.1f}s]" if t0 is not None else "[  0.0s]"
    print(f"{t} {m}", flush=True)


t0 = time.time()

# torch.load compatibility (UWYK / Do-PFN)
_orig_torch_load = torch.load
def _p_load(*a, **kw):
    kw.setdefault('weights_only', False); return _orig_torch_load(*a, **kw)
torch.load = _p_load

# ── Set up Do-PFN's `datasets` module stub (for pickle deserialization) ────
sys.path.insert(0, DOPFN_ROOT)
ds_mod = types.ModuleType('datasets')
_src = open(os.path.join(DOPFN_ROOT, 'datasets/__init__.py')).read().split('def load_semi_real')[0]
exec(_src, ds_mod.__dict__)
sys.modules['datasets'] = ds_mod

# ── Load CausalPFN's benchmark suite ─────────────────────────────────────────
sys.path.insert(0, CAUSALPFN)
from benchmarks import (
    IHDPDataset, ACIC2016Dataset,
    RealCauseLalondeCPSDataset, RealCauseLalondePSIDDataset,
)
DS_CATALOG = {
    'IHDP': IHDPDataset(),
    'ACIC': ACIC2016Dataset(),
    'CPS':  RealCauseLalondeCPSDataset(),
    'PSID': RealCauseLalondePSIDDataset(),
}
for name in DATASETS:
    d = DS_CATALOG[name]
    _log(f"  {name}: n_tables={d.n_tables}")

# ── Do-PFN ────────────────────────────────────────────────────────────────
_log("Loading Do-PFN…", t0)
from scripts.transformer_prediction_interface.base import DoPFNRegressor
_log("Do-PFN ready", t0)

# ── UWYK — MUST be loaded BEFORE our repo's `models` package pollutes sys.path
_log("Loading UWYK (with preprocessing wrapper)…", t0)
_saved_modules = {}
for name in list(sys.modules):
    if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
        _saved_modules[name] = sys.modules.pop(name)
_dopfn_removed = False
if DOPFN_ROOT in sys.path:
    sys.path.remove(DOPFN_ROOT); _dopfn_removed = True
sys.path.insert(0, UWYK_SRC)
import importlib
UWYK_pre_mod = importlib.import_module('models.PreprocessingGraphConditionedPFN')
sys.path.remove(UWYK_SRC)
if _dopfn_removed: sys.path.insert(0, DOPFN_ROOT)
for name in list(sys.modules):
    if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
        del sys.modules[name]
sys.modules.update(_saved_modules)
UWYK_Model = UWYK_pre_mod.PreprocessingGraphConditionedPFN
uwyk = UWYK_Model(
    config_path=os.path.join(UWYK_CKPT, 'config.yaml'),
    checkpoint_path=os.path.join(UWYK_CKPT, 'model.pt'),
    device='cpu', verbose=False,
).load()
_log("UWYK ready (Preprocessing wrapper)", t0)

# ── OUR model ─────────────────────────────────────────────────────────────
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
NUM_FEATURES = cfg['num_features']
our_model = OurModel(
    num_features=NUM_FEATURES, d_model=cfg['d_model'], depth=cfg['depth'],
    heads_feat=cfg['heads'], heads_samp=cfg['heads'], dropout=0.0,
    output_dim=J*J + 9 + 4, hidden_mult=cfg['hidden_mult'],
    normalize_features=True, normalize_treatment=False,
    use_treatment_in_query=False, use_checkpoint=False,
).to(DEVICE).eval()
our_model.load_state_dict(ckpt['model_state_dict'])
_log("OUR model ready", t0)


# ── Prediction wrappers ─────────────────────────────────────────────────────
def _to_np(a):
    if isinstance(a, torch.Tensor): return a.numpy()
    return np.asarray(a)


def _pad(arr, L):
    """Pad columns to L with zeros; truncate if too wide."""
    if arr.shape[1] >= L: return arr[:, :L]
    z = np.zeros((arr.shape[0], L - arr.shape[1]), dtype=np.float32)
    return np.concatenate([arr, z], axis=1)


def _subsample_context(X, t, y, max_n, seed):
    """Deterministic random subsample of the training context to max_n rows."""
    n = X.shape[0]
    if n <= max_n: return X, t, y, np.arange(n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    idx.sort()
    return X[idx], t[idx], y[idx], idx


def _subsample_test(X, cate, max_m, seed):
    m = X.shape[0]
    if m <= max_m: return X, cate, np.arange(m)
    rng = np.random.default_rng(seed + 999)
    idx = rng.choice(m, size=max_m, replace=False)
    idx.sort()
    return X[idx], cate[idx], idx


def predict_dopfn(cd):
    """Do-PFN expects x = [T | X] then intervenes internally.
    Do-PFN input: x[:,0] is T, x[:,1:] is X."""
    X_train = _to_np(cd.X_train).astype(np.float32)
    t_train = _to_np(cd.t_train).astype(np.float32).reshape(-1)
    y_train = _to_np(cd.y_train).astype(np.float32).reshape(-1)
    X_test  = _to_np(cd.X_test).astype(np.float32)
    t_test  = _to_np(cd.t_test).astype(np.float32).reshape(-1) \
              if hasattr(cd, 't_test') else np.zeros(X_test.shape[0], dtype=np.float32)
    x_train_full = np.concatenate([t_train[:, None], X_train], axis=1)
    x_test_full  = np.concatenate([t_test[:, None], X_test], axis=1)
    r = DoPFNRegressor()
    r.fit(torch.tensor(x_train_full), torch.tensor(y_train))
    return np.asarray(r.predict_cate(torch.tensor(x_test_full)))


def predict_uwyk(cd):
    """UWYK PreprocessingGraphConditionedPFN.predict_cate: returns CATE in
    ORIGINAL y-scale (handles feature standardization + Y scaling internally)."""
    X_train = _to_np(cd.X_train).astype(np.float32)
    t_train = _to_np(cd.t_train).astype(np.float32).reshape(-1)
    y_train = _to_np(cd.y_train).astype(np.float32).reshape(-1)
    X_test  = _to_np(cd.X_test).astype(np.float32)
    uwyk.fit(X_train, t_train, y_train)
    cate = uwyk.predict_cate(X_train, t_train, y_train, X_test,
                              adjacency_matrix=None, prediction_type='mean')
    return np.asarray(cate).reshape(-1)


def predict_ours(cd, use_mode=True):
    """OURS with proper preprocessing:
    1. Standardize X (mean=0, std=1) using X_train stats
    2. Scale Y to [-1, 1] using Y_train min/max
    3. Feed to model
    4. Get τ_scaled (mode or mean of p(τ))
    5. Inverse-scale: τ_raw = τ_scaled * (y_max - y_min) / 2
    """
    X_train = _to_np(cd.X_train).astype(np.float32)
    t_train = _to_np(cd.t_train).astype(np.float32).reshape(-1, 1)
    y_train_raw = _to_np(cd.y_train).astype(np.float32).reshape(-1, 1)
    X_test  = _to_np(cd.X_test).astype(np.float32)

    # ── Preprocess X: standardize using X_train stats ─────────────────────
    Xtr_pad = _pad(X_train, NUM_FEATURES)
    Xte_pad = _pad(X_test,  NUM_FEATURES)
    X_mean = Xtr_pad.mean(axis=0, keepdims=True)
    X_std  = Xtr_pad.std(axis=0, keepdims=True); X_std[X_std < 1e-6] = 1.0
    Xtr_s  = (Xtr_pad - X_mean) / X_std
    Xte_s  = (Xte_pad - X_mean) / X_std

    # ── Preprocess Y: scale to [-1, 1] using Y_train min/max ─────────────
    y_min = float(y_train_raw.min()); y_max = float(y_train_raw.max())
    y_range = max(y_max - y_min, 1e-8)
    y_train_s = 2 * (y_train_raw - y_min) / y_range - 1.0

    X_obs  = torch.from_numpy(Xtr_s).unsqueeze(0)
    T_obs  = torch.from_numpy(t_train).unsqueeze(0)
    Y_obs  = torch.from_numpy(y_train_s).unsqueeze(0)
    X_intv = torch.from_numpy(Xte_s).unsqueeze(0)

    with torch.no_grad():
        pred = our_model(X_obs, T_obs, Y_obs, X_intv)['predictions'][0]

    k_range = np.arange(-J + 1, J)
    tau_axis = k_range * bin_width       # scaled τ ∈ [-2, +2]

    M = pred.shape[0]
    taus_scaled = np.zeros(M)
    for i in range(M):
        p_mat, *_ = unpack_pred(pred[i], J, bin_width)
        p = p_mat.detach().cpu().numpy()
        if use_mode:
            p_tau = np.array([np.trace(p, offset=k) for k in k_range])
            taus_scaled[i] = float(tau_axis[p_tau.argmax()])
        else:
            E_y0 = (centers[:, None] * p).sum()
            E_y1 = (centers[None, :] * p).sum()
            taus_scaled[i] = float(E_y1 - E_y0)

    # ── Inverse-scale τ back to original y-space ─────────────────────────
    # y_scaled = 2*(y - y_min)/y_range - 1  →  τ_raw = τ_scaled * y_range / 2
    taus_raw = taus_scaled * y_range / 2.0
    return taus_raw


def pehe(pred, true):
    true = _to_np(true).reshape(-1); pred = np.asarray(pred).reshape(-1)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def ate_rel_err(pred, true):
    true = _to_np(true).reshape(-1); pred = np.asarray(pred).reshape(-1)
    tp, pp = float(true.mean()), float(pred.mean())
    if abs(tp) < 1e-9: return float('nan')
    return abs(tp - pp) / abs(tp)


# ── Run comparison ──────────────────────────────────────────────────────────
results = {}
for dname in DATASETS:
    _log(f"\n=== {dname} ===", t0)
    ds = DS_CATALOG[dname]
    n_max = min(N_REAL, ds.n_tables)
    results[dname] = {
        'dopfn_pehe':     [], 'dopfn_ate':     [],
        'uwyk_pehe':      [], 'uwyk_ate':      [],
        'oursmean_pehe':  [], 'oursmean_ate':  [],
        'oursmode_pehe':  [], 'oursmode_ate':  [],
    }
    for r in range(n_max):
        try:
            cd = ds[r][0]
        except Exception as e:
            _log(f"  real {r+1}: load FAIL {type(e).__name__}: {str(e)[:80]}", t0); continue
        # ── Subsample large train/test for tractability ────────────────────
        X_train_np = _to_np(cd.X_train).astype(np.float32)
        t_train_np = _to_np(cd.t_train).astype(np.float32).reshape(-1)
        y_train_np = _to_np(cd.y_train).astype(np.float32).reshape(-1)
        X_test_np  = _to_np(cd.X_test).astype(np.float32)
        cate_np    = _to_np(cd.true_cate).reshape(-1)

        X_train_np, t_train_np, y_train_np, _ = _subsample_context(
            X_train_np, t_train_np, y_train_np, MAX_CONTEXT, seed=r)
        X_test_np, cate_np, _ = _subsample_test(X_test_np, cate_np, MAX_QUERIES, seed=r)

        # Overwrite cd with subsampled views
        class _CD: pass
        cd = _CD()
        cd.X_train = torch.from_numpy(X_train_np)
        cd.t_train = torch.from_numpy(t_train_np)
        cd.y_train = torch.from_numpy(y_train_np)
        cd.X_test  = torch.from_numpy(X_test_np)
        cd.t_test  = torch.from_numpy(np.zeros(X_test_np.shape[0], dtype=np.float32))
        cd.true_cate = torch.from_numpy(cate_np)

        true_cate = cate_np
        _log(f"  real {r+1}: sizes  train={X_train_np.shape}  test={X_test_np.shape}", t0)
        try:
            p_dopfn = predict_dopfn(cd); e_dopfn_p = pehe(p_dopfn, true_cate); e_dopfn_a = ate_rel_err(p_dopfn, true_cate)
        except Exception as e:
            _log(f"  real {r+1}: Do-PFN FAIL {type(e).__name__}: {str(e)[:120]}", t0)
            e_dopfn_p = e_dopfn_a = float('nan')
        try:
            p_uwyk = predict_uwyk(cd); e_uwyk_p = pehe(p_uwyk, true_cate); e_uwyk_a = ate_rel_err(p_uwyk, true_cate)
        except Exception as e:
            _log(f"  real {r+1}: UWYK FAIL {type(e).__name__}: {str(e)[:120]}", t0)
            e_uwyk_p = e_uwyk_a = float('nan')
        try:
            p_ours_mean = predict_ours(cd, use_mode=False)
            p_ours_mode = predict_ours(cd, use_mode=True)
            e_om_p = pehe(p_ours_mean, true_cate); e_om_a = ate_rel_err(p_ours_mean, true_cate)
            e_od_p = pehe(p_ours_mode, true_cate); e_od_a = ate_rel_err(p_ours_mode, true_cate)
        except Exception as e:
            _log(f"  real {r+1}: OURS FAIL {type(e).__name__}: {str(e)[:120]}", t0)
            e_om_p = e_om_a = e_od_p = e_od_a = float('nan')
        results[dname]['dopfn_pehe'].append(e_dopfn_p);      results[dname]['dopfn_ate'].append(e_dopfn_a)
        results[dname]['uwyk_pehe'].append(e_uwyk_p);        results[dname]['uwyk_ate'].append(e_uwyk_a)
        results[dname]['oursmean_pehe'].append(e_om_p);      results[dname]['oursmean_ate'].append(e_om_a)
        results[dname]['oursmode_pehe'].append(e_od_p);      results[dname]['oursmode_ate'].append(e_od_a)
        _log(f"  real {r+1}/{n_max}  DoPFN {e_dopfn_p:.2f}/{e_dopfn_a:.2f}  "
             f"UWYK {e_uwyk_p:.2f}/{e_uwyk_a:.2f}  "
             f"OURS_mean {e_om_p:.2f}/{e_om_a:.2f}  OURS_mode {e_od_p:.2f}/{e_od_a:.2f}", t0)


# ── Aggregate + print ────────────────────────────────────────────────────────
_log("\n" + "="*115, t0)
_log(f"{'Dataset':<10} {'metric':<8} {'Do-PFN':>16} {'UWYK':>16} {'OURS-mean':>16} {'OURS-mode':>16}", t0)
_log("-"*115, t0)
for dname, r in results.items():
    for metric in ('pehe', 'ate'):
        d = np.array(r[f'dopfn_{metric}']); u = np.array(r[f'uwyk_{metric}'])
        om = np.array(r[f'oursmean_{metric}']); od = np.array(r[f'oursmode_{metric}'])
        d = d[~np.isnan(d)]; u = u[~np.isnan(u)]
        om = om[~np.isnan(om)]; od = od[~np.isnan(od)]
        s = lambda a: f"{a.mean():.3f}±{a.std():.3f}" if len(a) else "n/a"
        _log(f"{dname:<10} {metric:<8} {s(d):>16} {s(u):>16} {s(om):>16} {s(od):>16}", t0)

# save
np.savez(os.path.join(OUT_DIR, 'UWYK_BENCH_results.npz'),
         **{f'{d}_{k}': np.array(v) for d, r in results.items() for k, v in r.items()})
_log(f"\nDone. Total {time.time() - t0:.1f}s", t0)
