"""
End-to-end ATE distribution via W2 barycenter of per-query CATE densities.

For each benchmark realization:
  1. Get per-query CATE density p_i(τ) from model
       - MALC on masked p_mat (best variant from previous experiments)
       - Put on common τ grid
  2. Aggregate to get ATE density:
       - OT: Wasserstein-2 barycenter (shared-rank / average quantile function)
       - Linear: (1/N) Σᵢ p_i (naive average of densities)
  3. ATE point estimate: mode of each aggregated density
  4. True ATE = mean(true_CATE across queries)
  5. Report |ATE_estimate - true_ATE|

Plots ATE density (OT vs linear) with true ATE marked, for 3 realizations
per dataset.
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
DATASETS     = ['IHDP', 'ACIC']
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
from benchmarks import IHDPDataset, ACIC2016Dataset
DS_CATALOG = {'IHDP': IHDPDataset(), 'ACIC': ACIC2016Dataset()}
_log("Datasets ready", t0)

# UWYK preprocessing wrapper
_log("Loading UWYK…", t0)
_saved_modules = {}
for name in list(sys.modules):
    if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
        _saved_modules[name] = sys.modules.pop(name)
if DOPFN_ROOT in sys.path: sys.path.remove(DOPFN_ROOT); _dopfn_removed = True
else: _dopfn_removed = False
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

# Import OT barycenter
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
    ca = _to_np(cd.true_cate).reshape(-1)
    rng = np.random.default_rng(seed)
    if Xt.shape[0] > max_ctx:
        idx = np.sort(rng.choice(Xt.shape[0], max_ctx, replace=False))
        Xt, tt, yt = Xt[idx], tt[idx], yt[idx]
    if Xte.shape[0] > max_qry:
        idx = np.sort(rng.choice(Xte.shape[0], max_qry, replace=False))
        Xte, ca = Xte[idx], ca[idx]
    class _CD: pass
    cd2 = _CD()
    cd2.X_train = torch.from_numpy(Xt); cd2.t_train = torch.from_numpy(tt)
    cd2.y_train = torch.from_numpy(yt); cd2.X_test  = torch.from_numpy(Xte)
    cd2.true_cate = torch.from_numpy(ca)
    return cd2


def _mask_diagonal(p_mat_np, band=1):
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
    density = dmalc_2d(fit, eval_pts).reshape(N_EVAL, N_EVAL)
    return density


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
    # Normalize
    s = out.sum() * dtau
    if s > 0: out /= s
    return out


def get_per_query_ptau(cd):
    """Return per-query MALC-smoothed p(τ) on common grid tau_smooth (scaled space)
    plus the y_range scale factor to invert to original units."""
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
    p_taus_masked = np.zeros((M, len(tau_smooth)))
    for i in range(M):
        p_mat, *_ = unpack_pred(pred[i], J, bin_width)
        p_np = p_mat.detach().cpu().numpy()
        p_np_masked = _mask_diagonal(p_np, band=1)
        density = _fit_malc(p_np_masked, seed_hint=hash((i, 0)) % (10**8))
        p_taus_masked[i] = _p_tau_from_malc(density)

    return p_taus_masked, y_rng


def ate_estimates(p_taus_scaled, y_rng, true_cates_raw):
    """Compute ATE density via OT and linear mixture, extract modes, compare to
    true ATE."""
    # OT and linear on scaled grid
    ate_ot_scaled  = wasserstein_barycenter_1d(p_taus_scaled, tau_smooth)
    ate_lin_scaled = linear_mixture(p_taus_scaled)

    # Inverse-scale grid to raw τ (original outcome units)
    scale = y_rng / 2.0
    tau_raw = tau_smooth * scale
    # Density transforms: p_raw(τ_raw) = p_scaled(τ_scaled) / scale
    ate_ot_raw  = ate_ot_scaled  / scale
    ate_lin_raw = ate_lin_scaled / scale

    # Mode (argmax) — the ATE point estimate
    mode_ot  = float(tau_raw[ate_ot_raw.argmax()])
    mode_lin = float(tau_raw[ate_lin_raw.argmax()])

    # Also compute means (E[τ] under the aggregated densities)
    dtau_raw = tau_raw[1] - tau_raw[0]
    mean_ot  = float((tau_raw * ate_ot_raw).sum()  * dtau_raw)
    mean_lin = float((tau_raw * ate_lin_raw).sum() * dtau_raw)

    true_ate = float(np.mean(true_cates_raw))
    return {
        'tau_raw':      tau_raw,
        'ate_ot_raw':   ate_ot_raw,
        'ate_lin_raw':  ate_lin_raw,
        'true_ate':     true_ate,
        'mode_ot':      mode_ot,
        'mode_lin':     mode_lin,
        'mean_ot':      mean_ot,
        'mean_lin':     mean_lin,
        'err_mode_ot':  abs(mode_ot  - true_ate),
        'err_mode_lin': abs(mode_lin - true_ate),
        'err_mean_ot':  abs(mean_ot  - true_ate),
        'err_mean_lin': abs(mean_lin - true_ate),
    }


def predict_dopfn_ate(cd):
    Xtr = _to_np(cd.X_train).astype(np.float32); tt = _to_np(cd.t_train).astype(np.float32).reshape(-1)
    yt  = _to_np(cd.y_train).astype(np.float32).reshape(-1); Xte = _to_np(cd.X_test).astype(np.float32)
    x_tr = np.concatenate([tt[:, None], Xtr], axis=1)
    x_te = np.concatenate([np.zeros((Xte.shape[0], 1), dtype=np.float32), Xte], axis=1)
    r = DoPFNRegressor(); r.fit(torch.tensor(x_tr), torch.tensor(yt))
    return float(np.mean(np.asarray(r.predict_cate(torch.tensor(x_te)))))


def predict_uwyk_ate(cd):
    Xtr = _to_np(cd.X_train).astype(np.float32); tt = _to_np(cd.t_train).astype(np.float32).reshape(-1)
    yt  = _to_np(cd.y_train).astype(np.float32).reshape(-1); Xte = _to_np(cd.X_test).astype(np.float32)
    uwyk.fit(Xtr, tt, yt)
    return float(np.mean(np.asarray(uwyk.predict_cate(Xtr, tt, yt, Xte,
                                                       adjacency_matrix=None,
                                                       prediction_type='mean'))))


# ── Run ─────────────────────────────────────────────────────────────────────
results = {d: [] for d in DATASETS}

for dname in DATASETS:
    _log(f"\n=== {dname} ===", t0)
    ds = DS_CATALOG[dname]
    for r in range(min(N_REAL, ds.n_tables)):
        try:
            cd_raw = ds[r][0]
        except Exception:
            _log(f"  real {r+1}: load FAIL", t0); continue
        cd = _subsample(cd_raw, MAX_CONTEXT, MAX_QUERIES, seed=r)
        true_cate = _to_np(cd.true_cate).reshape(-1)
        true_ate = float(true_cate.mean())

        try:
            ate_dopfn = predict_dopfn_ate(cd)
        except Exception as e:
            _log(f"  real {r+1}: DoPFN FAIL {type(e).__name__}: {str(e)[:60]}", t0); continue
        try:
            ate_uwyk = predict_uwyk_ate(cd)
        except Exception as e:
            _log(f"  real {r+1}: UWYK FAIL {type(e).__name__}: {str(e)[:60]}", t0); continue

        p_taus, y_rng = get_per_query_ptau(cd)
        est = ate_estimates(p_taus, y_rng, true_cate)

        # Do-PFN / UWYK just give ATE point estimates (no distribution)
        err_dopfn = abs(ate_dopfn - true_ate)
        err_uwyk  = abs(ate_uwyk  - true_ate)

        results[dname].append({
            'real':       r, 'true_ate': true_ate,
            'ate_dopfn':  ate_dopfn, 'err_dopfn': err_dopfn,
            'ate_uwyk':   ate_uwyk,  'err_uwyk':  err_uwyk,
            'mode_ot':    est['mode_ot'],  'err_mode_ot':  est['err_mode_ot'],
            'mode_lin':   est['mode_lin'], 'err_mode_lin': est['err_mode_lin'],
            'mean_ot':    est['mean_ot'],  'err_mean_ot':  est['err_mean_ot'],
            'mean_lin':   est['mean_lin'], 'err_mean_lin': est['err_mean_lin'],
            'est':        est,
        })
        _log(f"  real {r+1}  true ATE={true_ate:+.3f}  "
             f"DoPFN {ate_dopfn:+.3f} (err {err_dopfn:.3f})  "
             f"UWYK {ate_uwyk:+.3f} (err {err_uwyk:.3f})  "
             f"OURS OT-mode {est['mode_ot']:+.3f} (err {est['err_mode_ot']:.3f})  "
             f"LIN-mode {est['mode_lin']:+.3f} (err {est['err_mode_lin']:.3f})", t0)

        # Plot ATE density for the first 3 realizations per dataset
        if r < 3:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(est['tau_raw'], est['ate_lin_raw'], color='orange', lw=1.5,
                    label=f"linear mixture  mode={est['mode_lin']:+.2f}", ls='--')
            ax.plot(est['tau_raw'], est['ate_ot_raw'], color='steelblue', lw=2,
                    label=f"OT W2 barycenter  mode={est['mode_ot']:+.2f}")
            ax.axvline(true_ate, color='red', ls='--', lw=1.5, alpha=0.7,
                       label=f'true ATE = {true_ate:+.3f}')
            ax.plot(true_ate, 0, 'o', color='red', markersize=12, zorder=5, clip_on=False)
            ax.axvline(ate_dopfn, color='green', ls=':', lw=1.5, alpha=0.7,
                       label=f'Do-PFN ATE = {ate_dopfn:+.3f}')
            ax.axvline(ate_uwyk, color='purple', ls=':', lw=1.5, alpha=0.7,
                       label=f'UWYK ATE = {ate_uwyk:+.3f}')
            ax.set_xlabel('τ (original y units)'); ax.set_ylabel('p(τ_ATE)')
            ax.set_title(f'{dname} real {r+1} — ATE density from {p_taus.shape[0]} per-query CATE distributions')
            ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper right')
            fig.tight_layout()
            out = os.path.join(OUT_DIR, f'ATE_OT_{dname}_real{r+1}.png')
            fig.savefig(out, dpi=140, bbox_inches='tight')
            plt.close(fig)
            _log(f"    Saved: {out}", t0)

# ── Aggregate ───────────────────────────────────────────────────────────────
_log("\n" + "="*100, t0)
_log(f"{'Dataset':<8} {'method':<20} {'|est − true|':>15}", t0)
_log("-"*100, t0)
for dname in DATASETS:
    R = results[dname]
    if not R: continue
    for method in ['dopfn', 'uwyk', 'mode_ot', 'mode_lin', 'mean_ot', 'mean_lin']:
        errs = np.array([r[f'err_{method}'] for r in R])
        _log(f"{dname:<8} {method:<20} {errs.mean():.4f} ± {errs.std():.4f}", t0)

np.savez(os.path.join(OUT_DIR, 'ATE_OT_results.npz'),
         **{f'{d}_{k}': np.array([rr[k] for rr in results[d]])
            for d in DATASETS for k in ('true_ate','err_dopfn','err_uwyk',
                                          'err_mode_ot','err_mode_lin',
                                          'err_mean_ot','err_mean_lin')})
_log(f"\nDone. Total {time.time() - t0:.1f}s", t0)
