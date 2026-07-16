"""
CPS ATE-relerr comparison:  DoPFN | UWYK | MALC-mean | MALC-mode | MALC-mode-msk |
                             MALC-mean-msk | MALC-OT-Mode (NEW).

MALC-OT-Mode = mode of the OT (Wasserstein-2 barycenter) aggregation of per-query
CATE densities. All other MALC-* methods aggregate by mean of per-query point
estimates (each method's per-query CATE).

CPS is ATE-only (no per-sample CATE), so only ATE-relerr is reported.
Train subsampled to 1000, test to 100 (per-query MALC is the bottleneck).
"""
from __future__ import annotations
import os, sys, time, types, warnings, importlib
import numpy as np
import torch
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

DOPFN_ROOT   = '/tmp/dopfn'
CAUSALPFN    = '/tmp/causalpfn_full'
UWYK_SRC     = '/tmp/g4cfm_uwyk/src'
UWYK_CKPT    = '/tmp/g4cfm_uwyk/experiments/checkpoints/full_conditioned_model'
OUR_CKPT     = '/Users/furkandanisman/R-PFN/checkpoints/step_50000_final.pt'
OT_DIR       = '/Users/furkandanisman/R-PFN/MALC/Optimal_Transport'
OUT_DIR      = '/Users/furkandanisman/R-PFN/experiments'
_REPO        = '/Users/furkandanisman/R-PFN'
DEVICE       = torch.device('cpu')
N_REAL       = 5
MAX_CONTEXT  = 1000
MAX_QUERIES  = 100
MALC_B       = 100
MALC_MAX_K   = 3
N_EVAL       = 200


def _log(m, t0=None):
    t = f"[{time.time()-t0:6.1f}s]" if t0 is not None else "[  0.0s]"
    print(f"{t} {m}", flush=True)


t0 = time.time()

_orig_torch_load = torch.load
def _p_load(*a, **kw):
    kw.setdefault('weights_only', False); return _orig_torch_load(*a, **kw)
torch.load = _p_load

sys.path.insert(0, DOPFN_ROOT)
ds_mod = types.ModuleType('datasets')
_src = open(os.path.join(DOPFN_ROOT, 'datasets/__init__.py')).read().split('def load_semi_real')[0]
exec(_src, ds_mod.__dict__)
sys.modules['datasets'] = ds_mod

sys.path.insert(0, CAUSALPFN)
from benchmarks import RealCauseLalondeCPSDataset
CPS = RealCauseLalondeCPSDataset()
_log(f"CPS ready ({CPS.n_tables} tables)", t0)

# UWYK
_log("Loading UWYK…", t0)
_saved_modules = {}
for name in list(sys.modules):
    if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
        _saved_modules[name] = sys.modules.pop(name)
_dopfn_removed = False
if DOPFN_ROOT in sys.path: sys.path.remove(DOPFN_ROOT); _dopfn_removed = True
sys.path.insert(0, UWYK_SRC)
UWYK_pre_mod = importlib.import_module('models.PreprocessingGraphConditionedPFN')
sys.path.remove(UWYK_SRC)
if _dopfn_removed: sys.path.insert(0, DOPFN_ROOT)
for name in list(sys.modules):
    if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
        del sys.modules[name]
sys.modules.update(_saved_modules)
uwyk = UWYK_pre_mod.PreprocessingGraphConditionedPFN(
    config_path=os.path.join(UWYK_CKPT, 'config.yaml'),
    checkpoint_path=os.path.join(UWYK_CKPT, 'model.pt'),
    device='cpu', verbose=False,
).load()
_log("UWYK ready", t0)

_log("Loading Do-PFN…", t0)
from scripts.transformer_prediction_interface.base import DoPFNRegressor
_log("Do-PFN ready", t0)

sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, 'MALC'))
from models.InterventionalPFN import InterventionalPFN as OurModel
from losses.BarDistribution2D import unpack_pred, fit_malc_inner
from malc_2d import dmalc_2d
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

sys.path.insert(0, OT_DIR)
from ot_barycenter import wasserstein_barycenter_1d, linear_mixture
_log("OT ready", t0)

xs = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
ys = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
XX, YY = np.meshgrid(xs, ys, indexing='xy')
eval_pts = np.column_stack([XX.ravel(), YY.ravel()])
dy0 = xs[1] - xs[0]; dy1 = ys[1] - ys[0]
tau_smooth = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], 401)
dtau = tau_smooth[1] - tau_smooth[0]
k_range = np.arange(-J + 1, J); tau_raw_axis = k_range * bin_width


def _to_np(a):
    if isinstance(a, torch.Tensor): return a.numpy()
    return np.asarray(a)


def _pad(arr, L):
    if arr.shape[1] >= L: return arr[:, :L]
    z = np.zeros((arr.shape[0], L - arr.shape[1]), dtype=np.float32)
    return np.concatenate([arr, z], axis=1)


def _subsample(cd, max_ctx, max_qry, seed):
    Xt = _to_np(cd.X_train).astype(np.float32)
    tt = _to_np(cd.t_train).astype(np.float32).reshape(-1)
    yt = _to_np(cd.y_train).astype(np.float32).reshape(-1)
    Xte = _to_np(cd.X_test).astype(np.float32)
    # CPS may or may not have true_cate — but always has true_ate
    rng = np.random.default_rng(seed)
    if Xt.shape[0] > max_ctx:
        idx = np.sort(rng.choice(Xt.shape[0], max_ctx, replace=False))
        Xt, tt, yt = Xt[idx], tt[idx], yt[idx]
    if Xte.shape[0] > max_qry:
        idx = np.sort(rng.choice(Xte.shape[0], max_qry, replace=False))
        Xte = Xte[idx]
    class _CD: pass
    cd2 = _CD()
    cd2.X_train = torch.from_numpy(Xt); cd2.t_train = torch.from_numpy(tt)
    cd2.y_train = torch.from_numpy(yt); cd2.X_test  = torch.from_numpy(Xte)
    return cd2


def _mask_diag(p_mat_np, band=1):
    p = p_mat_np.copy()
    for j0 in range(J):
        for j1 in range(max(0, j0-band), min(J, j0+band+1)):
            p[j0, j1] = 0.0
    p /= max(p.sum(), 1e-12)
    return p


def _fit_malc(p_mat_np, seed_hint):
    fit = fit_malc_inner(p_mat_np.T, edges_np, edges_np,
                          B_fit=MALC_B, B_select=MALC_B, max_K=MALC_MAX_K,
                          seed=seed_hint, parallel=False)
    return dmalc_2d(fit, eval_pts).reshape(N_EVAL, N_EVAL)


def _p_tau_from_malc(density):
    out = np.zeros_like(tau_smooth)
    for k, t in enumerate(tau_smooth):
        y1 = xs + t; v = (y1 >= ys[0]) & (y1 <= ys[-1])
        if not np.any(v): continue
        col = np.clip(np.searchsorted(xs, xs[v]) - 1, 0, len(xs) - 1)
        rf = (y1[v] - ys[0]) / dy1
        rlo = np.clip(np.floor(rf).astype(int), 0, len(ys) - 2)
        rhi = rlo + 1; whi = rf - rlo; wlo = 1.0 - whi
        f = wlo * density[rlo, col] + whi * density[rhi, col]
        out[k] = f.sum() * dy0
    s = out.sum() * dtau
    if s > 0: out /= s
    return out


def our_predict_all(cd):
    """Return per-query estimates for all OURS variants + per-query densities."""
    Xtr = _to_np(cd.X_train).astype(np.float32)
    tt  = _to_np(cd.t_train).astype(np.float32).reshape(-1, 1)
    yt  = _to_np(cd.y_train).astype(np.float32).reshape(-1, 1)
    Xte = _to_np(cd.X_test).astype(np.float32)

    Xtr_p = _pad(Xtr, NUM_FEATURES); Xte_p = _pad(Xte, NUM_FEATURES)
    mu = Xtr_p.mean(0, keepdims=True); sd = Xtr_p.std(0, keepdims=True); sd[sd < 1e-6] = 1.0
    Xtr_s = (Xtr_p - mu) / sd; Xte_s = (Xte_p - mu) / sd
    y_min = float(yt.min()); y_max = float(yt.max()); y_rng = max(y_max - y_min, 1e-8)
    yt_s = 2 * (yt - y_min) / y_rng - 1.0

    with torch.no_grad():
        pred = our_model(
            torch.from_numpy(Xtr_s).unsqueeze(0),
            torch.from_numpy(tt).unsqueeze(0),
            torch.from_numpy(yt_s).unsqueeze(0),
            torch.from_numpy(Xte_s).unsqueeze(0),
        )['predictions'][0]

    M = pred.shape[0]
    est_mean = np.zeros(M); est_malc_mode = np.zeros(M); est_malc_mode_msk = np.zeros(M)
    est_malc_mean = np.zeros(M); est_malc_mean_msk = np.zeros(M)
    p_taus_msk = np.zeros((M, len(tau_smooth)))   # for OT aggregation

    for i in range(M):
        p_mat, *_ = unpack_pred(pred[i], J, bin_width)
        p_np      = p_mat.detach().cpu().numpy()
        p_np_msk  = _mask_diag(p_np, band=1)

        # E[τ] from raw marginals (no MALC)
        E_y0 = (centers[:, None] * p_np).sum()
        E_y1 = (centers[None, :] * p_np).sum()
        est_mean[i] = float(E_y1 - E_y0)

        dens_raw = _fit_malc(p_np,     seed_hint=hash((i, 0)) % (10**8))
        dens_msk = _fit_malc(p_np_msk, seed_hint=hash((i, 1)) % (10**8))
        pt_raw = _p_tau_from_malc(dens_raw)
        pt_msk = _p_tau_from_malc(dens_msk)
        est_malc_mode[i]     = float(tau_smooth[pt_raw.argmax()])
        est_malc_mode_msk[i] = float(tau_smooth[pt_msk.argmax()])
        est_malc_mean[i]     = float((tau_smooth * pt_raw).sum() * dtau)
        est_malc_mean_msk[i] = float((tau_smooth * pt_msk).sum() * dtau)
        p_taus_msk[i] = pt_msk

    scale = y_rng / 2.0
    return {
        'ours_mean':          est_mean          * scale,
        'ours_malc_mode':     est_malc_mode     * scale,
        'ours_malc_mode_msk': est_malc_mode_msk * scale,
        'ours_malc_mean':     est_malc_mean     * scale,
        'ours_malc_mean_msk': est_malc_mean_msk * scale,
        'p_taus_msk_scaled':  p_taus_msk,
        'scale':              scale,
    }


def _ate_from_ot(p_taus_scaled, scale):
    """Aggregate per-query masked MALC densities via W2 barycenter,
    return mode of resulting density in RAW y-units."""
    ate_bary_scaled = wasserstein_barycenter_1d(p_taus_scaled, tau_smooth)
    tau_raw = tau_smooth * scale
    ate_bary_raw = ate_bary_scaled / scale
    return float(tau_raw[ate_bary_raw.argmax()]), tau_raw, ate_bary_raw


def dopfn_ate(cd):
    Xtr = _to_np(cd.X_train).astype(np.float32); tt = _to_np(cd.t_train).astype(np.float32).reshape(-1)
    yt  = _to_np(cd.y_train).astype(np.float32).reshape(-1); Xte = _to_np(cd.X_test).astype(np.float32)
    x_tr = np.concatenate([tt[:, None], Xtr], axis=1)
    x_te = np.concatenate([np.zeros((Xte.shape[0], 1), dtype=np.float32), Xte], axis=1)
    r = DoPFNRegressor(); r.fit(torch.tensor(x_tr), torch.tensor(yt))
    return float(np.mean(np.asarray(r.predict_cate(torch.tensor(x_te)))))


def uwyk_ate(cd):
    Xtr = _to_np(cd.X_train).astype(np.float32); tt = _to_np(cd.t_train).astype(np.float32).reshape(-1)
    yt  = _to_np(cd.y_train).astype(np.float32).reshape(-1); Xte = _to_np(cd.X_test).astype(np.float32)
    uwyk.fit(Xtr, tt, yt)
    return float(np.mean(np.asarray(uwyk.predict_cate(Xtr, tt, yt, Xte,
                                                       adjacency_matrix=None,
                                                       prediction_type='mean'))))


# ── Run ─────────────────────────────────────────────────────────────────────
_log(f"\n=== CPS === ({N_REAL} realizations)", t0)
methods = ['dopfn', 'uwyk', 'ours_mean', 'ours_malc_mode',
           'ours_malc_mode_msk', 'ours_malc_mean', 'ours_malc_mean_msk',
           'ours_ot_mode']
errs   = {m: [] for m in methods}
ates   = {m: [] for m in methods}
trues  = []

for r in range(N_REAL):
    try:
        cd_raw = CPS[r][0]
    except Exception as e:
        _log(f"  real {r+1}: load FAIL {type(e).__name__}: {str(e)[:80]}", t0); continue
    cd = _subsample(cd_raw, MAX_CONTEXT, MAX_QUERIES, seed=r)
    # CPS true ATE: from the raw dataset object
    true_ate = float(_to_np(cd_raw.true_ate)) if hasattr(cd_raw, 'true_ate') else \
               float(np.mean(_to_np(cd_raw.true_cate).reshape(-1))) if hasattr(cd_raw, 'true_cate') else \
               np.nan
    trues.append(true_ate)

    try:
        ate_d = dopfn_ate(cd)
    except Exception as e:
        _log(f"  real {r+1}: DoPFN FAIL {type(e).__name__}: {str(e)[:80]}", t0); continue
    try:
        ate_u = uwyk_ate(cd)
    except Exception as e:
        _log(f"  real {r+1}: UWYK FAIL {type(e).__name__}: {str(e)[:80]}", t0); continue

    ours = our_predict_all(cd)
    ate_ot_mode, tau_raw, ate_bary_raw = _ate_from_ot(ours['p_taus_msk_scaled'], ours['scale'])

    ates_r = {
        'dopfn':               ate_d,
        'uwyk':                ate_u,
        'ours_mean':           float(np.mean(ours['ours_mean'])),
        'ours_malc_mode':      float(np.mean(ours['ours_malc_mode'])),
        'ours_malc_mode_msk':  float(np.mean(ours['ours_malc_mode_msk'])),
        'ours_malc_mean':      float(np.mean(ours['ours_malc_mean'])),
        'ours_malc_mean_msk':  float(np.mean(ours['ours_malc_mean_msk'])),
        'ours_ot_mode':        ate_ot_mode,
    }
    def _rel(x): return abs(x - true_ate) / max(abs(true_ate), 1e-9)
    for m in methods:
        errs[m].append(_rel(ates_r[m])); ates[m].append(ates_r[m])
    _log(f"  real {r+1}  true ATE={true_ate:+.3f}  "
         f"DoPFN {ates_r['dopfn']:+.2f}  UWYK {ates_r['uwyk']:+.2f}  "
         f"OURS(mean/mode/mode-msk/OT) "
         f"{ates_r['ours_mean']:+.2f}/{ates_r['ours_malc_mode_msk']:+.2f}/"
         f"{ates_r['ours_ot_mode']:+.2f}", t0)

    # Plot the OT ATE density
    if r < 3:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(tau_raw, ate_bary_raw, color='steelblue', lw=2,
                label=f'OT ATE density (mode={ate_ot_mode:+.2f})')
        ax.plot(tau_raw, np.mean(ours['p_taus_msk_scaled'], axis=0) / ours['scale'],
                color='orange', lw=1.5, ls='--',
                label='linear mixture')
        ax.axvline(true_ate, color='red', ls='--', lw=1.5,
                   label=f'true ATE = {true_ate:+.2f}')
        ax.axvline(ates_r['dopfn'], color='green', ls=':', lw=1.5,
                   label=f"Do-PFN = {ates_r['dopfn']:+.2f}")
        ax.axvline(ates_r['uwyk'], color='purple', ls=':', lw=1.5,
                   label=f"UWYK = {ates_r['uwyk']:+.2f}")
        ax.set_xlabel('τ_ATE (original y units)'); ax.set_ylabel('p(τ_ATE)')
        ax.set_title(f'CPS real {r+1}: ATE density (W2 barycenter of per-query CATE)')
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f'CPS_ATE_OT_real{r+1}.png')
        fig.savefig(out, dpi=140, bbox_inches='tight')
        plt.close(fig)
        _log(f"    Saved: {out}", t0)


# ── Aggregate ────────────────────────────────────────────────────────────
_log("\n" + "="*90, t0)
_log(f"{'method':<25} {'ATE-relerr (mean±std)':>25}", t0)
_log("-"*90, t0)
for m in methods:
    a = np.array(errs[m])
    _log(f"{m:<25} {a.mean():.4f} ± {a.std():.4f}", t0)

np.savez(os.path.join(OUT_DIR, 'CPS_results.npz'), **{m: np.array(errs[m]) for m in methods},
         true_ates=np.array(trues))
_log(f"\nDone. Total {time.time() - t0:.1f}s", t0)
