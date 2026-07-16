"""
Test OURS with:
  (1) E[τ] from raw p_mat                              — baseline mean estimator
  (2) mode of p(τ) from raw p_mat                      — mode, spike-biased
  (3) mask τ=0 spike (zero out diagonal) then mode     — spike-suppressed mode
  (4) MALC fit + mean of MALC density                  — MALC-smoothed mean
  (5) MALC fit on masked p_mat + mean                  — MALC-smoothed + spike-suppressed

Compare all five to Do-PFN + UWYK on PEHE, ATE-relerr, and Wasserstein-1
distance between predicted τ distribution and empirical true-CATE distribution.

For 3 IHDP realizations × 6 queries each, plot p(τ) with all variants + true CATE.
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
OUT_DIR      = '/Users/furkandanisman/R-PFN/experiments'
_REPO        = '/Users/furkandanisman/R-PFN'
DEVICE       = torch.device('cpu')
N_REAL       = 5   # per dataset (fewer since we now do MALC for every query)
DATASETS     = ['IHDP', 'ACIC']
MAX_CONTEXT  = 1000
MAX_QUERIES  = 100 # smaller test set — MALC per query is expensive
MALC_B       = 100 # small for speed; we're doing 2 fits × 100 queries × 5 real × 2 ds
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

_log("Loading Do-PFN…", t0)
from scripts.transformer_prediction_interface.base import DoPFNRegressor
_log("Do-PFN ready", t0)

# ── UWYK Preprocessing wrapper ─────────────────────────────────────────────
_log("Loading UWYK (with preprocessing)…", t0)
_saved_modules = {}
for name in list(sys.modules):
    if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
        _saved_modules[name] = sys.modules.pop(name)
_dopfn_removed = False
if DOPFN_ROOT in sys.path:
    sys.path.remove(DOPFN_ROOT); _dopfn_removed = True
sys.path.insert(0, UWYK_SRC)
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
_log("UWYK ready", t0)

# ── OUR model ──────────────────────────────────────────────────────────────
_log("Loading OUR model…", t0)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'MALC'))
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

# Grids
xs = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
ys = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
XX, YY = np.meshgrid(xs, ys, indexing='xy')
eval_pts = np.column_stack([XX.ravel(), YY.ravel()])
dy0 = xs[1] - xs[0]; dy1 = ys[1] - ys[0]
tau_smooth = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], 401)
k_range = np.arange(-J + 1, J)
tau_raw_axis = k_range * bin_width


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


def _p_tau_from_pmat(p_mat_np):
    """Discrete p(τ) from joint p_mat via diagonal sum."""
    P = np.array([np.trace(p_mat_np, offset=k) for k in k_range])
    return P / bin_width  # density


def _p_tau_from_malc(density):
    """MALC-smoothed p(τ) via diagonal integration on eval grid."""
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
    return out


def _mask_diagonal(p_mat_np, band=1):
    """Zero out cells within ±band of the diagonal, renormalize."""
    p = p_mat_np.copy()
    for j0 in range(J):
        for j1 in range(max(0, j0 - band), min(J, j0 + band + 1)):
            p[j0, j1] = 0.0
    p /= max(p.sum(), 1e-12)
    return p


def _fit_malc(p_mat_np, seed_hint):
    fit = fit_malc_inner(p_mat_np.T, edges_np, edges_np,
                          B_fit=MALC_B, B_select=MALC_B, max_K=MALC_MAX_K,
                          seed=seed_hint, parallel=False)
    density = dmalc_2d(fit, eval_pts).reshape(N_EVAL, N_EVAL)
    return density, int(fit.K)


def _wasserstein1(a, b):
    """W-1 distance between two 1D empirical distributions (sample arrays)."""
    a = np.sort(np.asarray(a).reshape(-1))
    b = np.sort(np.asarray(b).reshape(-1))
    # Interp CDFs at joint quantile grid
    n = max(len(a), len(b))
    q = (np.arange(n) + 0.5) / n
    qa = np.quantile(a, q); qb = np.quantile(b, q)
    return float(np.mean(np.abs(qa - qb)))


# ── Do-PFN predict ─────────────────────────────────────────────────────────
def predict_dopfn(cd):
    Xtr = _to_np(cd.X_train).astype(np.float32); tt = _to_np(cd.t_train).astype(np.float32).reshape(-1)
    yt  = _to_np(cd.y_train).astype(np.float32).reshape(-1); Xte = _to_np(cd.X_test).astype(np.float32)
    x_tr = np.concatenate([tt[:, None], Xtr], axis=1)
    x_te = np.concatenate([np.zeros((Xte.shape[0], 1), dtype=np.float32), Xte], axis=1)
    r = DoPFNRegressor(); r.fit(torch.tensor(x_tr), torch.tensor(yt))
    return np.asarray(r.predict_cate(torch.tensor(x_te)))


def predict_uwyk(cd):
    Xtr = _to_np(cd.X_train).astype(np.float32); tt = _to_np(cd.t_train).astype(np.float32).reshape(-1)
    yt  = _to_np(cd.y_train).astype(np.float32).reshape(-1); Xte = _to_np(cd.X_test).astype(np.float32)
    uwyk.fit(Xtr, tt, yt)
    return np.asarray(uwyk.predict_cate(Xtr, tt, yt, Xte, adjacency_matrix=None,
                                          prediction_type='mean')).reshape(-1)


def predict_ours_variants(cd, malc_query_idx=None, save_probes=False):
    """Return dict of 5 CATE predictions per test query + optionally the p_mat
    arrays for a subset of queries (for plotting)."""
    Xtr = _to_np(cd.X_train).astype(np.float32); tt = _to_np(cd.t_train).astype(np.float32).reshape(-1, 1)
    yt  = _to_np(cd.y_train).astype(np.float32).reshape(-1, 1); Xte = _to_np(cd.X_test).astype(np.float32)

    # Preprocess
    Xtr_p = _pad(Xtr, NUM_FEATURES); Xte_p = _pad(Xte, NUM_FEATURES)
    mu = Xtr_p.mean(0, keepdims=True); sd = Xtr_p.std(0, keepdims=True); sd[sd < 1e-6] = 1.0
    Xtr_s = (Xtr_p - mu) / sd; Xte_s = (Xte_p - mu) / sd
    y_min = float(yt.min()); y_max = float(yt.max()); y_rng = max(y_max - y_min, 1e-8)
    yt_s  = 2 * (yt - y_min) / y_rng - 1.0

    with torch.no_grad():
        pred = our_model(
            torch.from_numpy(Xtr_s).unsqueeze(0),
            torch.from_numpy(tt).unsqueeze(0),
            torch.from_numpy(yt_s).unsqueeze(0),
            torch.from_numpy(Xte_s).unsqueeze(0),
        )['predictions'][0]

    M = pred.shape[0]
    est_mean          = np.zeros(M)   # E[τ] from raw p_mat marginals
    est_mode_malc     = np.zeros(M)   # argmax of MALC-smoothed p(τ) on raw p_mat
    est_mode_malc_msk = np.zeros(M)   # argmax of MALC-smoothed p(τ) on masked p_mat
    est_malc_mean     = np.zeros(M)   # E[τ] of MALC-smoothed p(τ) on raw p_mat
    est_malc_mean_msk = np.zeros(M)   # E[τ] of MALC-smoothed p(τ) on masked p_mat
    dtau = tau_smooth[1] - tau_smooth[0]
    probes = []

    for i in range(M):
        p_mat, *_ = unpack_pred(pred[i], J, bin_width)
        p_np      = p_mat.detach().cpu().numpy()
        p_np_mask = _mask_diagonal(p_np, band=1)

        # (1) E[τ] from raw marginals (no MALC)
        E_y0 = (centers[:, None] * p_np).sum()
        E_y1 = (centers[None, :] * p_np).sum()
        est_mean[i] = float(E_y1 - E_y0)

        # MALC fit — raw and masked — for EVERY query now (mode uses MALC)
        dens_raw, K_raw = _fit_malc(p_np,      seed_hint=hash((i, 0)) % (10**8))
        dens_msk, K_msk = _fit_malc(p_np_mask, seed_hint=hash((i, 1)) % (10**8))
        pt_malc      = _p_tau_from_malc(dens_raw)
        pt_malc_msk  = _p_tau_from_malc(dens_msk)

        est_mode_malc[i]     = float(tau_smooth[pt_malc.argmax()])
        est_mode_malc_msk[i] = float(tau_smooth[pt_malc_msk.argmax()])
        est_malc_mean[i]     = float((tau_smooth * pt_malc).sum()     * dtau)
        est_malc_mean_msk[i] = float((tau_smooth * pt_malc_msk).sum() * dtau)

        if save_probes and malc_query_idx is not None and i in malc_query_idx:
            # Discrete versions for the probe plot only
            pt_raw    = _p_tau_from_pmat(p_np)
            pt_raw_m  = _p_tau_from_pmat(p_np_mask)
            probes.append({
                'query_idx':          i,
                'p_tau_raw':          pt_raw,
                'p_tau_raw_masked':   pt_raw_m,
                'p_tau_malc':         pt_malc,
                'p_tau_malc_masked':  pt_malc_msk,
                'K_raw_malc':         K_raw,
                'K_masked_malc':      K_msk,
            })

    # Inverse-scale everything back to original y-scale
    scale = y_rng / 2.0
    out = {
        'ours_mean':          est_mean          * scale,   # E[τ], no MALC
        'ours_mode':          est_mode_malc     * scale,   # mode(MALC(p_mat))
        'ours_mode_masked':   est_mode_malc_msk * scale,   # mode(MALC(p_mat_masked))
        'ours_malc_mean':     est_malc_mean     * scale,   # E[τ] of MALC(p_mat)
        'ours_malc_mean_msk': est_malc_mean_msk * scale,   # E[τ] of MALC(p_mat_masked)
    }
    if save_probes:
        return out, probes, scale
    return out


def pehe(pred, true):
    true = _to_np(true).reshape(-1); pred = np.asarray(pred).reshape(-1)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def ate_rel(pred, true):
    true = _to_np(true).reshape(-1); pred = np.asarray(pred).reshape(-1)
    tp, pp = float(true.mean()), float(pred.mean())
    return abs(tp - pp) / max(abs(tp), 1e-9)


# ── Run comparison + probe plots ────────────────────────────────────────────
results = {d: {} for d in DATASETS}

for dname in DATASETS:
    _log(f"\n=== {dname} ===", t0)
    ds = DS_CATALOG[dname]
    n_max = min(N_REAL, ds.n_tables)
    # metrics buckets
    for method in ['dopfn', 'uwyk', 'ours_mean', 'ours_mode', 'ours_mode_masked',
                    'ours_malc_mean', 'ours_malc_mean_msk']:
        results[dname][f'{method}_pehe'] = []
        results[dname][f'{method}_ate']  = []
        results[dname][f'{method}_w1']   = []

    for r in range(n_max):
        try:
            cd_raw = ds[r][0]
        except Exception as e:
            _log(f"  real {r+1}: load FAIL", t0); continue
        cd = _subsample(cd_raw, MAX_CONTEXT, MAX_QUERIES, seed=r)
        true_cate = _to_np(cd.true_cate)

        # Probe plotting for both datasets, first 3 realizations each (6 queries)
        is_probe = r < 3
        malc_query_idx = list(range(6)) if is_probe else None

        try:
            p_dopfn = predict_dopfn(cd)
        except Exception as e:
            _log(f"  real {r+1}: DoPFN FAIL {type(e).__name__}: {str(e)[:80]}", t0); continue
        try:
            p_uwyk = predict_uwyk(cd)
        except Exception as e:
            _log(f"  real {r+1}: UWYK FAIL {type(e).__name__}: {str(e)[:80]}", t0); continue
        if is_probe:
            p_ours, probes, scale = predict_ours_variants(cd, malc_query_idx, save_probes=True)
        else:
            p_ours = predict_ours_variants(cd)

        # Aggregate metrics
        p_all = {'dopfn': p_dopfn, 'uwyk': p_uwyk}
        p_all.update(p_ours)
        for method, pred in p_all.items():
            results[dname][f'{method}_pehe'].append(pehe(pred, true_cate))
            results[dname][f'{method}_ate'].append(ate_rel(pred, true_cate))
            results[dname][f'{method}_w1'].append(_wasserstein1(pred, true_cate))
        _log(f"  real {r+1}/{n_max}: "
             f"DoPFN PEHE={pehe(p_dopfn,true_cate):.2f}  "
             f"UWYK={pehe(p_uwyk,true_cate):.2f}  "
             f"OURS_mean={pehe(p_ours['ours_mean'],true_cate):.2f}  "
             f"MODE_mask={pehe(p_ours['ours_mode_masked'],true_cate):.2f}  "
             f"MALC_mean_msk={pehe(p_ours['ours_malc_mean_msk'],true_cate):.2f}", t0)

        # Save probe plot
        if is_probe:
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            axes = axes.reshape(-1)
            for ax, pr in zip(axes, probes):
                qi = pr['query_idx']
                true_tau_scaled = float(true_cate[qi]) / scale  # scaled space
                ax.plot(tau_raw_axis, pr['p_tau_raw'], color='black', lw=1.2,
                        label='raw p(τ)', drawstyle='steps-mid')
                ax.plot(tau_raw_axis, pr['p_tau_raw_masked'], color='gray', lw=1.2, ls='--',
                        label='raw p(τ) masked')
                ax.plot(tau_smooth, pr['p_tau_malc'], color='steelblue', lw=1.6,
                        label=f"MALC K={pr['K_raw_malc']}")
                ax.plot(tau_smooth, pr['p_tau_malc_masked'], color='green', lw=1.7,
                        label=f"MALC masked K={pr['K_masked_malc']}")
                ax.axvline(true_tau_scaled, color='red', ls='--', lw=1.5, alpha=0.7,
                           label=f'true CATE (scaled)={true_tau_scaled:+.2f}')
                ax.plot(true_tau_scaled, 0, 'o', color='red', markersize=10, zorder=5, clip_on=False)
                ax.set_xlim(-1.5, 1.5)
                ax.set_xlabel('τ (scaled)'); ax.set_ylabel('p(τ)')
                ax.set_title(f'{dname} real {r} query {qi}  CATE_true={true_cate[qi]:+.2f}',
                             fontsize=10)
                ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper right')
            fig.suptitle(f'p(τ) for {dname} real {r+1} — 6 queries', y=1.02)
            fig.tight_layout()
            out = os.path.join(OUT_DIR, f'MASK_MALC_{dname}_real{r+1}.png')
            fig.savefig(out, dpi=140, bbox_inches='tight')
            plt.close(fig)
            _log(f"  Saved: {out}", t0)


# ── Print aggregated table ─────────────────────────────────────────────────
_log("\n" + "="*140, t0)
header = f"{'Dataset':<8} {'Metric':<6} {'DoPFN':>10} {'UWYK':>10} {'M-mean':>10} {'M-mode':>10} {'M-mode-msk':>11} {'MALCmean':>10} {'MALCmn-msk':>11}"
_log(header, t0)
_log("-"*140, t0)
for dname in DATASETS:
    R = results[dname]
    for metric in ('pehe', 'ate', 'w1'):
        def _s(k):
            a = np.array(R[f'{k}_{metric}']); a = a[~np.isnan(a)]
            return f"{a.mean():.3f}" if len(a) else "n/a"
        _log(f"{dname:<8} {metric:<6} {_s('dopfn'):>10} {_s('uwyk'):>10} "
             f"{_s('ours_mean'):>10} {_s('ours_mode'):>10} {_s('ours_mode_masked'):>11} "
             f"{_s('ours_malc_mean'):>10} {_s('ours_malc_mean_msk'):>11}", t0)

_log(f"\nDone. Total {time.time() - t0:.1f}s", t0)
