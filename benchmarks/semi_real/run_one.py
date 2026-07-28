"""Per-(dataset, seed, checkpoint) job for the Do-PFN semi-real benchmark.

Datasets (shipped inside the Do-PFN repo under `data/semi_real/`):
  - sales       (Amazon Sales; Blöbaum et al. 2024)
  - law_race    (Law School Admissions; Kusner et al. 2017)

Methods:
  - Do-PFN           (baseline)
  - UWYK Ancestral   (full DAG hint at inference)
  - UWYK NoAncestral (zero DAG hint, same checkpoint)
  - OURS             (7 variants — see benchmarks/methods/ours.py)

Ground-truth CATE is read from `test_ds.cate`, which Do-PFN's dataset
generator precomputes on the paper's agreed graph. (Their
`reproduce.ipynb` recomputes it via DoWhy for demonstration purposes;
we just use the cached tensor.) Output npz per job:

    <outdir>/<dataset>_seed<seed>.npz

fields: dataset, seed, n_train, n_test, true_ate, runtime_s,
         pehe_/err_/ate_ for every method (Do-PFN gets pehe/err/ate;
         Ours OT-{mode,mean} give only ate + err).
"""
from __future__ import annotations
import argparse, gc, importlib, os, sys, time, traceback, warnings
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

_HERE  = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
sys.path.insert(0, _BENCH)
from methods.ours import ours_pipeline
from methods.uwyk import uwyk_no_ancestral_pipeline, uwyk_ancestral_pipeline
from methods.dopfn import dopfn_pipeline

DEVICE = torch.device('cpu')


def _pehe(true_cate, pred_cate):
    return float(np.sqrt(mean_squared_error(true_cate, pred_cate)))


def _ate_relerr(true_cate, pred_cate):
    ta = float(np.mean(true_cate)); pa = float(np.mean(pred_cate))
    if abs(ta) < 1e-12:
        return 0.0 if abs(pa) < 1e-12 else float('inf')
    return abs(ta - pa) / abs(ta)


def _nmse(true_cate, pred_cate):
    """Do-PFN's normalized MSE (reproduce.ipynb cell 4):
        n_mse = mean(((pred - true) / (true.max() - true.min())) ** 2).
    Scale-invariant across datasets."""
    t = np.asarray(true_cate, dtype=float).reshape(-1)
    p = np.asarray(pred_cate, dtype=float).reshape(-1)
    span = float(t.max() - t.min())
    if span < 1e-12:
        return float('nan')
    return float(np.mean(((p - t) / span) ** 2))


def _load_ours(args):
    sys.path.insert(0, args.repo); sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    from models.InterventionalPFN import InterventionalPFN
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
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


def _load_uwyk(uwyk_src, uwyk_ckpt_dir, dopfn_path):
    """Isolate the models/ namespace clash with Do-PFN."""
    _saved = {}
    for name in list(sys.modules):
        if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
            _saved[name] = sys.modules.pop(name)
    _removed = False
    if dopfn_path in sys.path: sys.path.remove(dopfn_path); _removed = True
    sys.path.insert(0, uwyk_src)
    UWYK_pre_mod = importlib.import_module('models.PreprocessingGraphConditionedPFN')
    sys.path.remove(uwyk_src)
    if _removed: sys.path.insert(0, dopfn_path)
    for name in list(sys.modules):
        if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
            del sys.modules[name]
    sys.modules.update(_saved)
    return UWYK_pre_mod.PreprocessingGraphConditionedPFN(
        config_path=os.path.join(uwyk_ckpt_dir, 'best_model_config.yaml'),
        checkpoint_path=os.path.join(uwyk_ckpt_dir, 'best_model.pt'),
        device='cpu', verbose=False,
    ).load()


def _to_np(a):
    if isinstance(a, torch.Tensor): return a.cpu().numpy()
    return np.asarray(a)


def _build_cate_dataset(train_ds, test_ds):
    """Split Do-PFN's (x_obs with treatment in col 0, y_obs) into a
    CATE_Dataset view compatible with our pipelines."""
    x_tr = _to_np(train_ds.x_obs)
    y_tr = _to_np(train_ds.y_obs).reshape(-1)
    x_te = _to_np(test_ds.x_obs)
    t_tr = x_tr[:, 0].astype(np.float32).reshape(-1)
    X_tr = x_tr[:, 1:].astype(np.float32)
    X_te = x_te[:, 1:].astype(np.float32)
    class _CD: pass
    cd = _CD()
    cd.X_train = torch.from_numpy(X_tr)
    cd.t_train = torch.from_numpy(t_tr)
    cd.y_train = torch.from_numpy(y_tr.astype(np.float32))
    cd.X_test  = torch.from_numpy(X_te)
    return cd


def _compute_true_cate(test_ds):
    """Do-PFN's semi-real datasets ship with a precomputed ground-truth CATE
    on the test split (`test_ds.cate`), obviating the DoWhy counterfactual
    fitting from reproduce.ipynb cell 3. If `cate` is missing (defensive),
    fall back to y_int - y_obs which encodes the interventional contrast on
    the outcome directly."""
    if hasattr(test_ds, 'cate') and test_ds.cate is not None:
        return _to_np(test_ds.cate).astype(np.float32).reshape(-1)
    y_int = _to_np(test_ds.y_int).astype(np.float32).reshape(-1)
    y_obs = _to_np(test_ds.y_obs).astype(np.float32).reshape(-1)
    return (y_int - y_obs).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',       required=True, choices=['sales', 'law_race'])
    ap.add_argument('--seed',          type=int, required=True)
    ap.add_argument('--outdir',        required=True)
    ap.add_argument('--repo',          required=True)
    ap.add_argument('--dopfn',         required=True, help='Path to Do-PFN repo')
    ap.add_argument('--uwyk-src',      required=True, help='e.g. .../external/uwyk/src')
    ap.add_argument('--uwyk-ckpt-dir', required=True)
    ap.add_argument('--checkpoint',    required=True)
    ap.add_argument('--n-splits',      type=int, default=5)
    ap.add_argument('--malc-B',        type=int, default=100)
    ap.add_argument('--malc-max-K',    type=int, default=3)
    ap.add_argument('--n-eval',        type=int, default=200)
    ap.add_argument('--workers',       type=int, default=1)
    ap.add_argument('--ours-only',     action='store_true',
                    help='Skip Do-PFN + UWYK — used for the second-checkpoint pass.')
    args = ap.parse_args()

    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path: sys.path.insert(0, _here)

    os.makedirs(args.outdir, exist_ok=True)
    out_file = os.path.join(args.outdir, f'{args.dataset}_seed{args.seed:04d}.npz')
    if os.path.exists(out_file):
        print(f'[SKIP] {out_file} exists.', flush=True); return

    t0 = time.time()
    _orig_torch_load = torch.load
    def _p_load(*a, **kw):
        kw.setdefault('weights_only', False); return _orig_torch_load(*a, **kw)
    torch.load = _p_load

    # Import Do-PFN's datasets — they use relative paths for their pkls, and
    # DoPFNRegressor also opens artifacts/dopfn_config.pkl relatively. Stay in
    # $DOPFN throughout the run; every output path we write is absolute, so
    # leaving cwd there is harmless.
    sys.path.insert(0, args.dopfn)
    os.chdir(args.dopfn)
    from datasets import load_dataset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # generate_valid_split's split seed is (split_number // n_splits) + 1, so
    # to actually vary the split across our seed iterations we index into the
    # 5-fold CV by passing split_number = seed + 1 (their API uses 1-based
    # indexing). With n_splits=5 and split_number in {1..5} we get the five
    # disjoint folds of a 5-fold CV; anything above n_splits kicks the seed
    # forward (splits_seed = split_number // n_splits + 1) and gives fresh
    # random splits, so seed=5 → seed 2, seed=10 → seed 3, etc.
    split_number = int(args.seed) + 1
    print(f"[{time.time()-t0:6.1f}s] load_dataset({args.dataset!r})  seed={args.seed}  split_number={split_number}",
          flush=True)
    dataset = load_dataset(ds_name=args.dataset)
    train_ds, test_ds = dataset.generate_valid_split(
        n_splits=args.n_splits, split_number=split_number,
    )

    print(f"[{time.time()-t0:6.1f}s] building CATE_Dataset view", flush=True)
    cd = _build_cate_dataset(train_ds, test_ds)

    print(f"[{time.time()-t0:6.1f}s] reading precomputed true CATE from test_ds.cate", flush=True)
    true_cate = _compute_true_cate(test_ds)
    cd.true_cate = torch.from_numpy(true_cate)
    true_ate = float(np.mean(true_cate))
    print(f"[{time.time()-t0:6.1f}s] n_train={cd.t_train.shape[0]}  "
          f"n_test={true_cate.shape[0]}  true_ate={true_ate:+.4f}", flush=True)

    out = dict(dataset=args.dataset, seed=args.seed,
                n_train=int(cd.t_train.shape[0]), n_test=int(true_cate.shape[0]),
                true_ate=true_ate, runtime_s=0.0)

    def _record(name, cate_pred):
        out[f'pehe_{name}'] = _pehe(true_cate, cate_pred)
        out[f'nmse_{name}'] = _nmse(true_cate, cate_pred)
        out[f'err_{name}']  = _ate_relerr(true_cate, cate_pred)
        out[f'ate_{name}']  = float(np.mean(cate_pred))

    # ── Baselines (skipped on --ours-only) ──────────────────────────────────
    if not args.ours_only:
        print(f"[{time.time()-t0:6.1f}s] Do-PFN", flush=True)
        from scripts.transformer_prediction_interface.base import DoPFNRegressor
        _record('dopfn', dopfn_pipeline(cd, DoPFNRegressor))
        del DoPFNRegressor; gc.collect()

        print(f"[{time.time()-t0:6.1f}s] loading UWYK", flush=True)
        uwyk_model = _load_uwyk(args.uwyk_src, args.uwyk_ckpt_dir, args.dopfn)

        print(f"[{time.time()-t0:6.1f}s] UWYK-Ancestral", flush=True)
        _record('uwyk_anc',   uwyk_ancestral_pipeline(uwyk_model, cd))
        print(f"[{time.time()-t0:6.1f}s] UWYK-NoAncestral", flush=True)
        _record('uwyk_noanc', uwyk_no_ancestral_pipeline(uwyk_model, cd))

        del uwyk_model; gc.collect()

    # ── OURS ────────────────────────────────────────────────────────────────
    print(f"[{time.time()-t0:6.1f}s] loading OURS ({os.path.basename(args.checkpoint)})", flush=True)
    (our_model, edges_np, J, bin_width, centers, NUM_FEATURES,
     wasserstein_barycenter_1d) = _load_ours(args)

    print(f"[{time.time()-t0:6.1f}s] OURS inference", flush=True)
    ours = ours_pipeline(cd, our_model, edges_np, J, bin_width, NUM_FEATURES,
                          centers, args, wasserstein_barycenter_1d)

    _record('ours_mean',          ours['ours_mean'])
    _record('ours_malc_mean',     ours['ours_malc_mean'])
    _record('ours_malc_mean_msk', ours['ours_malc_mean_msk'])
    _record('ours_malc_mode',     ours['ours_malc_mode'])
    _record('ours_malc_mode_msk', ours['ours_malc_mode_msk'])

    ot_mode_ate = ours['ours_ot_mode_ate']
    out['ate_ours_ot_mode'] = ot_mode_ate
    out['err_ours_ot_mode'] = abs(ot_mode_ate - true_ate) / max(abs(true_ate), 1e-9)
    ot_mean_ate = ours['ours_ot_mean_ate']
    out['ate_ours_ot_mean'] = ot_mean_ate
    out['err_ours_ot_mean'] = abs(ot_mean_ate - true_ate) / max(abs(true_ate), 1e-9)

    out['runtime_s'] = time.time() - t0
    np.savez(out_file, **{k: np.array(v) for k, v in out.items()})
    print(f"[{time.time()-t0:6.1f}s] saved {out_file}", flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
