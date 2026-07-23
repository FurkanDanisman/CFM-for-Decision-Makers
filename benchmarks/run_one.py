"""Per-(dataset, realization) benchmark job.

For one dataset/realization pair, runs three method pipelines and saves the
result to `<outdir>/<dataset>_r<realization>.npz`. The SLURM array in
`cluster/submit.sbatch` schedules 500 tasks (5 datasets × 100 realizations)
and each task invokes this script.

Metrics follow UWYK's `RealCauseEval/run_baselines/eval.py`:
  PEHE       = sqrt(mean_squared_error(true_cate, pred_cate))
  ATE_relerr = |mean(true_cate) - mean(pred_cate)| / |mean(true_cate)|

Methods live in `benchmarks/methods/` — one file each for Do-PFN, UWYK, ours.
"""
from __future__ import annotations
import argparse, os, sys, time, types, warnings, importlib, traceback
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

# benchmarks/methods/ is a sibling of this file
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from methods.dopfn import dopfn_pipeline
from methods.uwyk  import uwyk_ancestral_pipeline, uwyk_no_ancestral_pipeline
from methods.ours  import ours_pipeline


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


# ── dataset loaders (CausalPFN's benchmarks package) ────────────────────────
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
    """PSID-balanced training set: all T=1 + up to `max_control` T=0."""
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


# ── UWYK model loader (isolates the models/ namespace collision with Do-PFN) ─
def _load_uwyk_model(args, ckpt_dir):
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
    return UWYK_pre_mod.PreprocessingGraphConditionedPFN(
        config_path=os.path.join(ckpt_dir, 'best_model_config.yaml'),
        checkpoint_path=os.path.join(ckpt_dir, 'best_model.pt'),
        device='cpu', verbose=False,
    ).load()


# ── OURS model loader ───────────────────────────────────────────────────────
def _load_our_model(args):
    sys.path.insert(0, args.repo); sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    from models.InterventionalPFN import InterventionalPFN
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    cfg = ckpt['config']; J = cfg['J']
    edges_np = ckpt['edges'].cpu().numpy()
    bin_width = float(edges_np[1] - edges_np[0])
    centers = 0.5 * (edges_np[:-1] + edges_np[1:])
    NUM_FEATURES = cfg['num_features']
    m = InterventionalPFN(
        num_features=NUM_FEATURES, d_model=cfg['d_model'], depth=cfg['depth'],
        heads_feat=cfg['heads'], heads_samp=cfg['heads'], dropout=0.0,
        output_dim=J*J + 9 + 4, hidden_mult=cfg['hidden_mult'],
        normalize_features=True, normalize_treatment=False,
        use_treatment_in_query=False, use_checkpoint=False,
    ).to(DEVICE).eval()
    m.load_state_dict(ckpt['model_state_dict'])

    ot_dir = os.path.join(args.repo, 'MALC', 'Optimal_Transport')
    if ot_dir not in sys.path: sys.path.insert(0, ot_dir)
    from ot_barycenter import wasserstein_barycenter_1d
    return m, edges_np, J, bin_width, centers, NUM_FEATURES, wasserstein_barycenter_1d


# ── UWYK-BASELINE backfill: add pehe_uwyk_baseline etc. to an existing npz ──
def _run_baseline_backfill(args, out_file):
    """Load dataset + UWYK unconditional ckpt, compute CATE, merge into npz."""
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

    _log(f"[baseline-only] loading {args.dataset} r{args.realization}", t0)
    cd_raw, ad = load_realization(args.dataset, args.realization)
    if args.dataset == 'PSIDbal':
        cd_raw = apply_balanced(cd_raw, max_control=500, seed=args.realization)

    _log("[baseline-only] loading UWYK-BASELINE (unconditional ckpt)", t0)
    uwyk_baseline_model = _load_uwyk_model(args, args.uwyk_baseline_ckpt_dir)

    _log("[baseline-only] UWYK-BASELINE inference (target-encoded T, zero adjacency)", t0)
    uwyk_baseline_cate = uwyk_no_ancestral_pipeline(uwyk_baseline_model, cd_raw)

    true_cate = _to_np(cd_raw.true_cate).reshape(-1)
    extras = {
        'pehe_uwyk_baseline': np.array(_pehe(true_cate, uwyk_baseline_cate), dtype=np.float64),
        'err_uwyk_baseline':  np.array(_ate_relerr(true_cate, uwyk_baseline_cate), dtype=np.float64),
        'ate_uwyk_baseline':  np.array(float(np.mean(uwyk_baseline_cate)), dtype=np.float64),
    }
    with np.load(out_file, allow_pickle=True) as f:
        payload = {k: f[k] for k in f.files}
    payload.update(extras)
    tmp_path = out_file + '.tmp.npz'
    np.savez(tmp_path, **payload)
    os.replace(tmp_path, out_file)
    _log(f"[baseline-only] merged into {out_file}  "
         f"(PEHE={float(extras['pehe_uwyk_baseline']):.3f}  "
         f"ATE={float(extras['err_uwyk_baseline']):.3f})", t0)


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',       required=True, choices=DATASETS_ALL)
    ap.add_argument('--realization',   type=int, required=True)
    ap.add_argument('--outdir',        required=True)
    ap.add_argument('--repo',          required=True,
                     help='Path to R-PFN repo root (contains models/, losses/, MALC/)')
    ap.add_argument('--dopfn',         required=True, help='Path to patched Do-PFN repo')
    ap.add_argument('--uwyk',          required=True, help='Path to patched UWYK repo')
    ap.add_argument('--causalpfn',     required=True, help='Path to CausalPFN repo (has benchmarks/)')
    ap.add_argument('--checkpoint',    required=True, help='Path to our .pt checkpoint')
    ap.add_argument('--uwyk-ckpt-dir', required=True,
                     help='Path to UWYK Ancestral checkpoint dir '
                          '(e.g. .../full_conditioned_model/final_earlytest_full_conditioning_16773252.0)')
    ap.add_argument('--uwyk-baseline-ckpt-dir', default=None,
                     help='Path to UWYK separately-trained baseline (unconditional) '
                          'checkpoint dir (e.g. .../no_graph_conditioning/unconditional). '
                          'If set, adds pehe_uwyk_baseline / err_uwyk_baseline / '
                          'ate_uwyk_baseline fields to the output npz.')
    ap.add_argument('--malc-B',        type=int, default=100)
    ap.add_argument('--malc-max-K',    type=int, default=3)
    ap.add_argument('--n-eval',        type=int, default=200)
    ap.add_argument('--workers',       type=int, default=1)
    ap.add_argument('--only-uwyk-baseline', action='store_true',
                     help='Backfill mode: compute ONLY UWYK-BASELINE and merge into the '
                          'existing npz. Skip Do-PFN, UWYK-Ancestral, OURS. Requires '
                          '--uwyk-baseline-ckpt-dir. If the target npz already carries '
                          'pehe_uwyk_baseline, this task is a no-op.')
    args = ap.parse_args()

    n_avail = DATASET_N_TABLES.get(args.dataset, 100)
    if args.realization >= n_avail:
        print(f'[SKIP] {args.dataset} only has {n_avail} tables; r={args.realization} out of range.', flush=True)
        return

    os.makedirs(args.outdir, exist_ok=True)
    out_file = os.path.join(args.outdir, f'{args.dataset}_r{args.realization:03d}.npz')

    if args.only_uwyk_baseline:
        if not args.uwyk_baseline_ckpt_dir:
            print('[ERR] --only-uwyk-baseline requires --uwyk-baseline-ckpt-dir', flush=True)
            sys.exit(2)
        if not os.path.exists(out_file):
            print(f'[SKIP] {out_file} does not exist yet — run the full pipeline first.', flush=True)
            return
        with np.load(out_file, allow_pickle=True) as _f:
            if {'pehe_uwyk_baseline', 'err_uwyk_baseline', 'ate_uwyk_baseline'} <= set(_f.files):
                print(f'[SKIP] {out_file} already has uwyk_baseline fields.', flush=True)
                return
        _run_baseline_backfill(args, out_file); return

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

    _log("loading UWYK Ancestral model", t0)
    uwyk_model = _load_uwyk_model(args, args.uwyk_ckpt_dir)

    _log("loading Do-PFN", t0)
    from scripts.transformer_prediction_interface.base import DoPFNRegressor

    _log("loading OURS", t0)
    (our_model, edges_np, J, bin_width, centers, NUM_FEATURES,
     wasserstein_barycenter_1d) = _load_our_model(args)

    true_cate = _to_np(cd_raw.true_cate).reshape(-1)
    true_ate  = float(getattr(ad, 'true_ate', np.mean(true_cate)))

    _log("Do-PFN inference", t0)
    dopfn_cate = dopfn_pipeline(cd_raw, DoPFNRegressor)

    _log("UWYK-Ancestral inference (target-encoded T, full-graph adj)", t0)
    uwyk_anc_cate = uwyk_ancestral_pipeline(uwyk_model, cd_raw)

    # Free the two transformers before spawning MALC workers — on CPS the
    # combined Do-PFN + UWYK footprint plus 16-worker MALC pool exceeds 64 GB.
    import gc
    del uwyk_model, DoPFNRegressor
    gc.collect()

    uwyk_baseline_cate = None
    if args.uwyk_baseline_ckpt_dir:
        _log("loading UWYK-BASELINE (unconditional ckpt)", t0)
        uwyk_baseline_model = _load_uwyk_model(args, args.uwyk_baseline_ckpt_dir)
        _log("UWYK-BASELINE inference (target-encoded T, zero adjacency)", t0)
        uwyk_baseline_cate = uwyk_no_ancestral_pipeline(uwyk_baseline_model, cd_raw)
        del uwyk_baseline_model
        gc.collect()

    _log("Do-PFN + UWYK released, running OURS", t0)

    # OURS: hierarchical clustering matches UWYK's `PreprocessingGraphConditioned
    # PFN.predict()` — when train > MAX_N_TRAIN (1000, the model's training-time
    # cap) we k-means-cluster the covariates into k = ceil(N/1000) groups, assign
    # each test query to its nearest cluster, and run one forward pass per
    # cluster. Sample-attention memory becomes O(1000²) per cluster instead of
    # O(N²) globally.
    _log("OURS inference (all variants)", t0)
    ours = ours_pipeline(cd_raw, our_model, edges_np, J, bin_width, NUM_FEATURES,
                          centers, args, wasserstein_barycenter_1d)

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
    if uwyk_baseline_cate is not None:
        _record('uwyk_baseline',  uwyk_baseline_cate)
    _record('ours_mean',          ours['ours_mean'])
    _record('ours_malc_mean',     ours['ours_malc_mean'])
    _record('ours_malc_mean_msk', ours['ours_malc_mean_msk'])
    _record('ours_malc_mode',     ours['ours_malc_mode'])
    _record('ours_malc_mode_msk', ours['ours_malc_mode_msk'])

    # OT-mode + OT-mean: single population ATE each, not per-query cate
    ot_mode_ate = ours['ours_ot_mode_ate']
    ot_mean_ate = ours['ours_ot_mean_ate']
    out['ate_ours_ot_mode'] = ot_mode_ate
    out['err_ours_ot_mode'] = abs(ot_mode_ate - true_ate) / max(abs(true_ate), 1e-9)
    out['ate_ours_ot_mean'] = ot_mean_ate
    out['err_ours_ot_mean'] = abs(ot_mean_ate - true_ate) / max(abs(true_ate), 1e-9)

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
