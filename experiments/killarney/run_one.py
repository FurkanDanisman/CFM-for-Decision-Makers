"""
Per-(dataset, realization) benchmark job — uses UWYK's own eval pipelines
for Do-PFN / UWYK rows, computes our method variants alongside.

Protocol matches UWYK's Table 3 exactly:
  - UWYK Ancestral   : dofm_full_conditioning.py pipeline (target-encoded T,
                       full-graph adjacency, calls model.predict directly)
  - Do-PFN           : DoPFNRegressor.fit(x, y).predict_cate(x_test) matching
                       Do-PFN's own inference_example.py
  - OURS             : our 6 variants (mean, MALC-{mean,mean_msk,mode,mode_msk},
                       OT-mode) — unchanged
Metrics use UWYK's eval.py definitions:
  PEHE       = sqrt(mean_squared_error(true_cate, pred_cate))
  ATE_relerr = |true_cate.mean() - pred_cate.mean()| / |true_cate.mean()|
"""
from __future__ import annotations
import argparse, os, sys, time, types, warnings, importlib, hashlib, traceback
from multiprocessing import Pool, get_context
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

DEVICE = torch.device('cpu')
DATASETS_ALL = ['IHDP', 'ACIC', 'CPS', 'PSID', 'PSIDbal']
DATASET_N_TABLES = {'IHDP': 100, 'ACIC': 10, 'CPS': 100, 'PSID': 100, 'PSIDbal': 100}


def _log(m, t0):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def _to_np(a):
    if isinstance(a, torch.Tensor): return a.numpy()
    return np.asarray(a)


def _pehe(true_cate, pred_cate):
    return float(np.sqrt(mean_squared_error(true_cate, pred_cate)))


def _ate_relerr(true_cate, pred_cate):
    true_ate = float(np.mean(true_cate)); pred_ate = float(np.mean(pred_cate))
    if abs(true_ate) < 1e-12:
        return 0.0 if abs(pred_ate) < 1e-12 else float('inf')
    return abs(true_ate - pred_ate) / abs(true_ate)


# ────────── UWYK Ancestral pipeline (mirrors dofm_full_conditioning.py) ──────
def build_ancestral_adjacency(model_n_features, n_real_features):
    """Full-graph adjacency: T→Y=1, X→T=1, X→Y=1 for real features; padded=-1."""
    adj = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
    T_idx = 0; Y_idx = 1; feature_offset = 2
    adj[T_idx, Y_idx] = 1.0
    for i in range(n_real_features):
        adj[feature_offset + i, T_idx] = 1.0
        adj[feature_offset + i, Y_idx] = 1.0
    for i in range(n_real_features, model_n_features):
        fi = feature_offset + i
        adj[fi, :] = -1.0; adj[:, fi] = -1.0; adj[fi, fi] = -1.0
    return adj


def uwyk_ancestral_pipeline(uwyk_model, cate_dataset):
    """Reproduces dofm_full_conditioning.py's dofm_full_conditioning_pipeline exactly."""
    X_train = _to_np(cate_dataset.X_train)
    t_train_orig = _to_np(cate_dataset.t_train)
    t_train_orig = t_train_orig.reshape(-1, 1) if t_train_orig.ndim == 1 else t_train_orig
    y_train_orig = _to_np(cate_dataset.y_train)
    y_train_orig = y_train_orig.reshape(-1, 1) if y_train_orig.ndim == 1 else y_train_orig
    X_test = _to_np(cate_dataset.X_test)
    y_train = y_train_orig

    n_test = X_test.shape[0]
    n_features_orig = X_train.shape[1]
    model_n_features = uwyk_model.model.num_features

    # UWYK's target encoding: T ← mean(Y | T)
    t_flat = t_train_orig.flatten()
    y_flat = y_train.flatten()
    mean_y_t0 = float(y_flat[t_flat == 0].mean())
    mean_y_t1 = float(y_flat[t_flat == 1].mean())
    t_train = np.where(t_train_orig == 0, mean_y_t0, mean_y_t1).astype(np.float32)

    uwyk_model.fit(X_train, t_train, y_train)

    n_real_features = min(n_features_orig, model_n_features)
    adjacency_matrix = build_ancestral_adjacency(model_n_features, n_real_features)

    T_intv_1 = np.full((n_test, 1), mean_y_t1, dtype=np.float32)
    y_pred_1 = uwyk_model.predict(
        X_obs=X_train, T_obs=t_train, Y_obs=y_train,
        X_intv=X_test, T_intv=T_intv_1,
        adjacency_matrix=adjacency_matrix,
        prediction_type="mean", inverse_transform=True,
    )

    T_intv_0 = np.full((n_test, 1), mean_y_t0, dtype=np.float32)
    y_pred_0 = uwyk_model.predict(
        X_obs=X_train, T_obs=t_train, Y_obs=y_train,
        X_intv=X_test, T_intv=T_intv_0,
        adjacency_matrix=adjacency_matrix,
        prediction_type="mean", inverse_transform=True,
    )
    return np.asarray(y_pred_1 - y_pred_0).reshape(-1)


# ────────── Do-PFN pipeline (matches inference_example.py) ───────────────────
def dopfn_pipeline(cate_dataset, DoPFNRegressor):
    X_train = _to_np(cate_dataset.X_train).astype(np.float32)
    t_train = _to_np(cate_dataset.t_train).astype(np.float32).reshape(-1)
    y_train = _to_np(cate_dataset.y_train).astype(np.float32).reshape(-1)
    X_test  = _to_np(cate_dataset.X_test).astype(np.float32)

    # Do-PFN convention: first column is treatment
    x_tr = np.concatenate([t_train[:, None], X_train], axis=1)
    x_te = np.concatenate([np.zeros((X_test.shape[0], 1), dtype=np.float32), X_test], axis=1)

    reg = DoPFNRegressor()
    reg.fit(torch.tensor(x_tr), torch.tensor(y_train))
    cate = reg.predict_cate(torch.tensor(x_te))
    return np.asarray(cate).reshape(-1)


# ────────── OURS pipeline (unchanged inference) ──────────────────────────────
def _pad(arr, L):
    if arr.shape[1] >= L: return arr[:, :L]
    z = np.zeros((arr.shape[0], L - arr.shape[1]), dtype=np.float32)
    return np.concatenate([arr, z], axis=1)


def _mask_diag(p_mat_np, J, band=1):
    p = p_mat_np.copy()
    for j0 in range(J):
        for j1 in range(max(0, j0 - band), min(J, j0 + band + 1)):
            p[j0, j1] = 0.0
    p /= max(p.sum(), 1e-12)
    return p


_GLOBAL = {}


def _init_worker(edges_np, J, bin_width, N_EVAL, MALC_B, MALC_MAX_K, repo, malc_dir):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    if repo not in sys.path: sys.path.insert(0, repo)
    if malc_dir not in sys.path: sys.path.insert(0, malc_dir)
    from losses.BarDistribution2D import fit_malc_inner
    from malc_2d import dmalc_2d
    _GLOBAL['fit'] = fit_malc_inner
    _GLOBAL['dmalc'] = dmalc_2d
    _GLOBAL['edges'] = edges_np
    _GLOBAL['J'] = J
    _GLOBAL['bw'] = bin_width
    _GLOBAL['MALC_B'] = MALC_B
    _GLOBAL['MALC_MAX_K'] = MALC_MAX_K
    xs = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
    ys = np.linspace(edges_np[0], edges_np[-1], N_EVAL)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    _GLOBAL['xs'] = xs; _GLOBAL['ys'] = ys
    _GLOBAL['eval_pts'] = np.column_stack([XX.ravel(), YY.ravel()])
    _GLOBAL['dy0'] = xs[1] - xs[0]; _GLOBAL['dy1'] = ys[1] - ys[0]
    _GLOBAL['tau'] = np.linspace(ys[0] - xs[-1], ys[-1] - xs[0], 401)
    _GLOBAL['dtau'] = _GLOBAL['tau'][1] - _GLOBAL['tau'][0]


def _fit_and_marginalize(p_mat_np, seed):
    fit = _GLOBAL['fit'](p_mat_np.T, _GLOBAL['edges'], _GLOBAL['edges'],
                          B_fit=_GLOBAL['MALC_B'], B_select=_GLOBAL['MALC_B'],
                          max_K=_GLOBAL['MALC_MAX_K'], seed=seed, parallel=False)
    density = _GLOBAL['dmalc'](fit, _GLOBAL['eval_pts']).reshape(len(_GLOBAL['xs']), len(_GLOBAL['ys']))
    tau = _GLOBAL['tau']
    xs = _GLOBAL['xs']; ys = _GLOBAL['ys']; dy0 = _GLOBAL['dy0']; dy1 = _GLOBAL['dy1']
    out = np.zeros_like(tau)
    for k, t in enumerate(tau):
        y1 = xs + t; v = (y1 >= ys[0]) & (y1 <= ys[-1])
        if not np.any(v): continue
        col = np.clip(np.searchsorted(xs, xs[v]) - 1, 0, len(xs) - 1)
        rf = (y1[v] - ys[0]) / dy1
        rlo = np.clip(np.floor(rf).astype(int), 0, len(ys) - 2)
        rhi = rlo + 1; whi = rf - rlo; wlo = 1.0 - whi
        f = wlo * density[rlo, col] + whi * density[rhi, col]
        out[k] = f.sum() * dy0
    s = out.sum() * _GLOBAL['dtau']
    if s > 0: out /= s
    return out


def _worker_one_query(args):
    i, p_mat_np = args
    from losses.BarDistribution2D import fit_malc_inner  # noqa
    p_masked = _mask_diag(p_mat_np, _GLOBAL['J'], band=1)
    seed_raw = int(hashlib.md5(f"q{i}r".encode()).hexdigest()[:8], 16) % (10**8)
    seed_msk = int(hashlib.md5(f"q{i}m".encode()).hexdigest()[:8], 16) % (10**8)
    p_raw = _fit_and_marginalize(p_mat_np, seed_raw)
    p_msk = _fit_and_marginalize(p_masked, seed_msk)
    return i, p_raw, p_msk


def ours_pipeline(cate_dataset, our_model, edges_np, J, bin_width, NUM_FEATURES,
                   centers, args, wasserstein_barycenter_1d):
    """Returns dict of all OURS variants' cate predictions."""
    from losses.BarDistribution2D import unpack_pred

    Xtr = _to_np(cate_dataset.X_train).astype(np.float32)
    tt  = _to_np(cate_dataset.t_train).astype(np.float32).reshape(-1, 1)
    yt  = _to_np(cate_dataset.y_train).astype(np.float32).reshape(-1, 1)
    Xte = _to_np(cate_dataset.X_test).astype(np.float32)

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
    est_mean = np.zeros(M)
    p_mats = np.zeros((M, J, J), dtype=np.float32)
    for i in range(M):
        p_mat, *_ = unpack_pred(pred[i], J, bin_width)
        p_np = p_mat.detach().cpu().numpy().astype(np.float32)
        p_mats[i] = p_np
        E_y0 = (centers[:, None] * p_np).sum()
        E_y1 = (centers[None, :] * p_np).sum()
        est_mean[i] = float(E_y1 - E_y0)

    worker_args = [(i, p_mats[i]) for i in range(M)]
    p_taus_raw = np.zeros((M, 401)); p_taus_msk = np.zeros((M, 401))
    if args.workers > 1:
        ctx = get_context('spawn')
        with ctx.Pool(processes=args.workers, initializer=_init_worker,
                      initargs=(edges_np, J, bin_width, args.n_eval,
                                args.malc_B, args.malc_max_K,
                                args.repo, os.path.join(args.repo, 'MALC'))) as pool:
            for (i, pr, pm) in pool.imap_unordered(_worker_one_query, worker_args, chunksize=1):
                p_taus_raw[i] = pr; p_taus_msk[i] = pm
    else:
        _init_worker(edges_np, J, bin_width, args.n_eval, args.malc_B, args.malc_max_K,
                      args.repo, os.path.join(args.repo, 'MALC'))
        for a in worker_args:
            i, pr, pm = _worker_one_query(a); p_taus_raw[i] = pr; p_taus_msk[i] = pm

    tau = np.linspace(edges_np[0] - edges_np[-1], edges_np[-1] - edges_np[0], 401)
    dtau_ = tau[1] - tau[0]
    est_malc_mode     = tau[p_taus_raw.argmax(axis=1)]
    est_malc_mode_msk = tau[p_taus_msk.argmax(axis=1)]
    est_malc_mean     = (tau[None, :] * p_taus_raw).sum(axis=1) * dtau_
    est_malc_mean_msk = (tau[None, :] * p_taus_msk).sum(axis=1) * dtau_

    scale = y_rng / 2.0
    ours_mean          = est_mean          * scale
    ours_malc_mean     = est_malc_mean     * scale
    ours_malc_mean_msk = est_malc_mean_msk * scale
    ours_malc_mode     = est_malc_mode     * scale
    ours_malc_mode_msk = est_malc_mode_msk * scale

    # OT-mode: population-level ATE from W2 barycenter of masked per-query densities
    ate_bary_scaled = wasserstein_barycenter_1d(p_taus_msk, tau)
    tau_raw = tau * scale
    ate_bary_raw = ate_bary_scaled / scale
    ate_ot_mode_scalar = float(tau_raw[ate_bary_raw.argmax()])

    return dict(
        ours_mean          = ours_mean,
        ours_malc_mean     = ours_malc_mean,
        ours_malc_mean_msk = ours_malc_mean_msk,
        ours_malc_mode     = ours_malc_mode,
        ours_malc_mode_msk = ours_malc_mode_msk,
        ours_ot_mode_ate   = ate_ot_mode_scalar,  # NOTE: population ATE, not per-query
    )


# ────────── main ─────────────────────────────────────────────────────────────
def load_realization(dname, r):
    if dname == 'IHDP':
        from benchmarks import IHDPDataset
        cd, ad = IHDPDataset()[r]
    elif dname == 'ACIC':
        from benchmarks import ACIC2016Dataset
        cd, ad = ACIC2016Dataset()[r]
    elif dname == 'CPS':
        from benchmarks import RealCauseLalondeCPSDataset
        cd, ad = RealCauseLalondeCPSDataset()[r]
    elif dname == 'PSID':
        from benchmarks import RealCauseLalondePSIDDataset
        cd, ad = RealCauseLalondePSIDDataset()[r]
    elif dname == 'PSIDbal':
        from benchmarks import RealCauseLalondePSIDDataset
        cd, ad = RealCauseLalondePSIDDataset()[r]
    else:
        raise ValueError(dname)
    return cd, ad


def apply_balanced(cd, max_control=500, seed=0):
    """PSID-balanced: all T=1 + up to max_control T=0."""
    Xt = _to_np(cd.X_train).astype(np.float32)
    tt = _to_np(cd.t_train).astype(np.float32).reshape(-1)
    yt = _to_np(cd.y_train).astype(np.float32).reshape(-1)
    rng = np.random.default_rng(seed)
    idx_t = np.where(tt > 0.5)[0]; idx_c = np.where(tt < 0.5)[0]
    if idx_c.size > max_control:
        idx_c = np.sort(rng.choice(idx_c, max_control, replace=False))
    keep = np.sort(np.concatenate([idx_t, idx_c]))
    class _CD: pass
    cd2 = _CD()
    cd2.X_train = torch.from_numpy(Xt[keep])
    cd2.t_train = torch.from_numpy(tt[keep])
    cd2.y_train = torch.from_numpy(yt[keep])
    cd2.X_test  = cd.X_test
    cd2.true_cate = cd.true_cate
    return cd2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',      required=True, choices=DATASETS_ALL)
    ap.add_argument('--realization',  type=int, required=True)
    ap.add_argument('--outdir',       required=True)
    ap.add_argument('--repo',         required=True)
    ap.add_argument('--dopfn',        required=True)
    ap.add_argument('--uwyk',         required=True)
    ap.add_argument('--causalpfn',    required=True)
    ap.add_argument('--checkpoint',   required=True)
    ap.add_argument('--uwyk-ckpt-dir',required=True)
    ap.add_argument('--malc-B',       type=int, default=100)
    ap.add_argument('--malc-max-K',   type=int, default=3)
    ap.add_argument('--n-eval',       type=int, default=200)
    ap.add_argument('--workers',      type=int, default=1)
    args = ap.parse_args()

    n_avail = DATASET_N_TABLES.get(args.dataset, 100)
    if args.realization >= n_avail:
        print(f'[SKIP] {args.dataset} only has {n_avail} tables; r={args.realization} out of range.', flush=True)
        return

    os.makedirs(args.outdir, exist_ok=True)
    out_file = os.path.join(args.outdir, f'{args.dataset}_r{args.realization:03d}.npz')
    if os.path.exists(out_file):
        print(f'[SKIP] {out_file} exists.', flush=True); return

    t0 = time.time()
    _orig_torch_load = torch.load
    def _p_load(*a, **kw):
        kw.setdefault('weights_only', False); return _orig_torch_load(*a, **kw)
    torch.load = _p_load

    sys.path.insert(0, args.dopfn)
    ds_mod = types.ModuleType('datasets')
    _src = open(os.path.join(args.dopfn, 'datasets/__init__.py')).read().split('def load_semi_real')[0]
    exec(_src, ds_mod.__dict__)
    sys.modules['datasets'] = ds_mod
    sys.path.insert(0, args.causalpfn)

    _log(f"loading {args.dataset} r{args.realization}", t0)
    cd_raw, ad = load_realization(args.dataset, args.realization)
    if args.dataset == 'PSIDbal':
        cd_raw = apply_balanced(cd_raw, max_control=500, seed=args.realization)

    # UWYK model (Ancestral)
    _log("loading UWYK model", t0)
    _saved = {}
    for name in list(sys.modules):
        if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
            _saved[name] = sys.modules.pop(name)
    _dopfn_removed = False
    if args.dopfn in sys.path: sys.path.remove(args.dopfn); _dopfn_removed = True
    uwyk_src = os.path.join(args.uwyk, 'src')
    sys.path.insert(0, uwyk_src)
    UWYK_pre_mod = importlib.import_module('models.PreprocessingGraphConditionedPFN')
    sys.path.remove(uwyk_src)
    if _dopfn_removed: sys.path.insert(0, args.dopfn)
    for name in list(sys.modules):
        if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
            del sys.modules[name]
    sys.modules.update(_saved)

    uwyk_model = UWYK_pre_mod.PreprocessingGraphConditionedPFN(
        config_path=os.path.join(args.uwyk_ckpt_dir, 'config.yaml'),
        checkpoint_path=os.path.join(args.uwyk_ckpt_dir, 'model.pt'),
        device='cpu', verbose=False,
    ).load()

    _log("loading Do-PFN", t0)
    from scripts.transformer_prediction_interface.base import DoPFNRegressor

    _log("loading OURS", t0)
    sys.path.insert(0, args.repo); sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    from models.InterventionalPFN import InterventionalPFN as OurModel
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
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

    ot_dir = os.path.join(args.repo, 'MALC', 'Optimal_Transport')
    if ot_dir not in sys.path: sys.path.insert(0, ot_dir)
    from ot_barycenter import wasserstein_barycenter_1d

    true_cate = _to_np(cd_raw.true_cate).reshape(-1)
    true_ate  = float(getattr(ad, 'true_ate', np.mean(true_cate)))

    # ── run each pipeline (UWYK's protocol for UWYK/Do-PFN rows) ────────────
    _log("Do-PFN inference", t0)
    dopfn_cate = dopfn_pipeline(cd_raw, DoPFNRegressor)

    _log("UWYK-Ancestral inference (target-encoded T, full-graph adj)", t0)
    uwyk_anc_cate = uwyk_ancestral_pipeline(uwyk_model, cd_raw)

    _log("OURS inference (all variants)", t0)
    ours = ours_pipeline(cd_raw, our_model, edges_np, J, bin_width, NUM_FEATURES,
                          centers, args, wasserstein_barycenter_1d)

    # ── metrics per UWYK's eval.py definitions ──────────────────────────────
    out = dict(dataset=args.dataset, realization=args.realization,
                true_ate=true_ate,
                n_queries=int(len(true_cate)), n_context=int(cd_raw.X_train.shape[0]),
                runtime_s=time.time() - t0)

    def _record(name, cate_pred):
        out[f'pehe_{name}'] = _pehe(true_cate, cate_pred)
        out[f'err_{name}']  = _ate_relerr(true_cate, cate_pred)
        out[f'ate_{name}']  = float(np.mean(cate_pred))

    _record('dopfn',              dopfn_cate)
    _record('uwyk_anc',           uwyk_anc_cate)
    _record('ours_mean',          ours['ours_mean'])
    _record('ours_malc_mean',     ours['ours_malc_mean'])
    _record('ours_malc_mean_msk', ours['ours_malc_mean_msk'])
    _record('ours_malc_mode',     ours['ours_malc_mode'])
    _record('ours_malc_mode_msk', ours['ours_malc_mode_msk'])

    # OT-mode gives population ATE only (not per-query cate)
    ot_ate = ours['ours_ot_mode_ate']
    out['ate_ours_ot_mode'] = ot_ate
    out['err_ours_ot_mode'] = abs(ot_ate - true_ate) / max(abs(true_ate), 1e-9)

    np.savez(out_file, **{k: np.array(v) for k, v in out.items()})
    _log(f"saved {out_file}  ({out['runtime_s']:.1f}s)", t0)
    print(f"SUMMARY dopfn PEHE={out['pehe_dopfn']:.3f} ATE={out['err_dopfn']:.3f} | "
          f"uwyk_anc PEHE={out['pehe_uwyk_anc']:.3f} ATE={out['err_uwyk_anc']:.3f} | "
          f"ours_mode_msk PEHE={out['pehe_ours_malc_mode_msk']:.3f} ATE={out['err_ours_malc_mode_msk']:.3f}",
          flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
