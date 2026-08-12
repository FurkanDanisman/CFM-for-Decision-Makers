"""Per-(source, seed, N_context) job for the context-size sweep.

Computes OURS CATE variants + UWYK No-Ancestral baseline on one SCM instance
at one context size. UWYK No-Ancestral is the apples-to-apples comparison:
both methods see the training data with no DAG hint (paper convention: the
adjacency matrix passes zeros for real features, meaning "edge status unknown").

Output npz per job: `<outdir>/<source>_seed<seed>_N<N>.npz`.
"""
from __future__ import annotations
import argparse, importlib, os, sys, time, types, warnings, traceback
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


def _load_ours(args):
    """Load OURS model. Supports both InterventionalPFN and DoPFN-backbone
    checkpoints via --backbone flag (default: ipfn). See benchmarks/run_one.py
    for the same loader shape.
    """
    sys.path.insert(0, args.repo); sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    cfg = ckpt['config']; J = cfg['J']
    edges_np = ckpt['edges'].cpu().numpy()
    bin_width = float(edges_np[1] - edges_np[0])
    centers = 0.5 * (edges_np[:-1] + edges_np[1:])

    if getattr(args, 'backbone', 'ipfn') == 'dopfn_bb':
        sys.path.insert(0, os.path.join(args.repo, 'training_dopfn_base'))
        from dopfn_backbone_head import DoPFNBackboneWith2DHead
        m = DoPFNBackboneWith2DHead(dopfn_root=args.dopfn, K=J).to(DEVICE).eval()
        m.load_state_dict(ckpt['model_state_dict'])
        NUM_FEATURES = -1
    else:
        from models.InterventionalPFN import InterventionalPFN
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


def _sample_scm(source, seed, N, n_test, uwyk_src, causalpfn_root):
    if source == 'prior':
        if uwyk_src not in sys.path: sys.path.insert(0, uwyk_src)
        from scm_prior import sample_as_cate_dataset
        cd, _ = sample_as_cate_dataset(scm_seed=seed, n_context=N, n_test=n_test)
        return cd
    elif source == 'poly':
        # scm_polynomial loads causalpfn's PolynomialDataset by file path
        # to sidestep sys.path collisions with our own R-PFN benchmarks/.
        from scm_polynomial import sample_as_cate_dataset
        return sample_as_cate_dataset(scm_seed=seed, n_context=N, n_test=n_test,
                                       causalpfn_root=causalpfn_root)
    else:
        raise ValueError(source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source',       required=True, choices=['prior', 'poly'])
    ap.add_argument('--seed',         type=int, required=True)
    ap.add_argument('--n-context',    type=int, required=True)
    ap.add_argument('--n-test',       type=int, default=50)
    ap.add_argument('--outdir',       required=True)
    ap.add_argument('--repo',         required=True)
    ap.add_argument('--uwyk-src',     required=True, help='e.g. .../external/uwyk/src')
    ap.add_argument('--causalpfn',    required=True, help='e.g. .../external/causalpfn')
    ap.add_argument('--checkpoint',   required=True)
    ap.add_argument('--uwyk-ckpt-dir', required=True,
                    help='Path to UWYK Ancestral checkpoint dir '
                         '(e.g. .../full_conditioned_model/final_earlytest_full_conditioning_16773252.0)')
    ap.add_argument('--dopfn',        default=None,
                    help='Path to Do-PFN repo (required when --source=poly). '
                         'Used to import DoPFNRegressor for the poly-only baseline.')
    ap.add_argument('--malc-B',       type=int, default=100)
    ap.add_argument('--malc-max-K',   type=int, default=3)
    ap.add_argument('--n-eval',       type=int, default=200)
    ap.add_argument('--workers',      type=int, default=1)
    ap.add_argument('--backbone',     choices=['ipfn', 'dopfn_bb'], default='ipfn',
                     help='ipfn = InterventionalPFN checkpoint (default); '
                          'dopfn_bb = DoPFN-backbone-with-2D-head checkpoint.')
    ap.add_argument('--y-log-transform', action='store_true',
                     help='Apply log(y - y_min + 1) transform at inference '
                          '(NaNs MALC/EM fields; raw-mean only).')
    ap.add_argument('--y-power-transform', action='store_true',
                     help='Apply Yeo-Johnson PowerTransformer on Y at inference '
                          '(NaNs MALC/EM fields; raw-mean only).')
    ap.add_argument('--y-winsorize', type=float, default=0.0,
                     help='Winsorize Y at (P, 100-P) percentiles.')
    ap.add_argument('--skip-malc', action='store_true',
                     help='Skip MALC pool entirely (raw-mean only, ~10x faster).')
    ap.add_argument('--ours-only',    action='store_true',
                    help='Skip UWYK-NoAnc + UWYK-Ancestral + Do-PFN. Use when '
                         'baselines are already populated in a sibling corpus '
                         '(they are checkpoint-independent).')
    args = ap.parse_args()

    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path: sys.path.insert(0, _here)

    os.makedirs(args.outdir, exist_ok=True)
    out_file = os.path.join(args.outdir,
                            f'{args.source}_seed{args.seed:04d}_N{args.n_context:06d}.npz')
    if os.path.exists(out_file):
        print(f'[SKIP] {out_file} exists.', flush=True); return

    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] sampling {args.source} scm seed={args.seed} N={args.n_context}",
          flush=True)
    cd = _sample_scm(args.source, args.seed, args.n_context, args.n_test,
                      args.uwyk_src, args.causalpfn)
    true_cate = cd.true_cate.numpy().reshape(-1) if hasattr(cd.true_cate, 'numpy') \
                 else np.asarray(cd.true_cate).reshape(-1)

    import gc
    uwyk_noanc_cate = uwyk_anc_cate = dopfn_cate = None

    if args.ours_only:
        print(f"[{time.time()-t0:6.1f}s] --ours-only: skipping UWYK + Do-PFN", flush=True)
    else:
        # ── Load UWYK model (Ancestral ckpt; zero adjacency = No-Ancestral) ──
        print(f"[{time.time()-t0:6.1f}s] loading UWYK", flush=True)
        _saved = {}
        for name in list(sys.modules):
            if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
                _saved[name] = sys.modules.pop(name)
        sys.path.insert(0, args.uwyk_src)
        UWYK_pre_mod = importlib.import_module('models.PreprocessingGraphConditionedPFN')
        sys.path.remove(args.uwyk_src)
        for name in list(sys.modules):
            if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
                del sys.modules[name]
        sys.modules.update(_saved)
        _orig_torch_load = torch.load
        def _p_load(*a, **kw):
            kw.setdefault('weights_only', False); return _orig_torch_load(*a, **kw)
        torch.load = _p_load
        uwyk_model = UWYK_pre_mod.PreprocessingGraphConditionedPFN(
            config_path=os.path.join(args.uwyk_ckpt_dir, 'best_model_config.yaml'),
            checkpoint_path=os.path.join(args.uwyk_ckpt_dir, 'best_model.pt'),
            device='cpu', verbose=False,
        ).load()

        print(f"[{time.time()-t0:6.1f}s] UWYK-NoAncestral inference", flush=True)
        uwyk_noanc_cate = uwyk_no_ancestral_pipeline(uwyk_model, cd)

        print(f"[{time.time()-t0:6.1f}s] UWYK-Ancestral inference", flush=True)
        uwyk_anc_cate = uwyk_ancestral_pipeline(uwyk_model, cd)

        del uwyk_model
        gc.collect()

        # Do-PFN baseline — poly source only (PEHE reported).
        if args.source == 'poly':
            if not args.dopfn:
                raise SystemExit('--dopfn is required when --source=poly')
            print(f"[{time.time()-t0:6.1f}s] loading Do-PFN", flush=True)
            sys.path.insert(0, args.dopfn)
            _cwd_dopfn_prev = os.getcwd()
            os.chdir(args.dopfn)
            from scripts.transformer_prediction_interface.base import DoPFNRegressor
            print(f"[{time.time()-t0:6.1f}s] Do-PFN inference", flush=True)
            dopfn_cate = dopfn_pipeline(cd, DoPFNRegressor)
            os.chdir(_cwd_dopfn_prev)
            del DoPFNRegressor
            gc.collect()

    print(f"[{time.time()-t0:6.1f}s] loading OURS", flush=True)
    (our_model, edges_np, J, bin_width, centers, NUM_FEATURES,
     wasserstein_barycenter_1d) = _load_ours(args)

    print(f"[{time.time()-t0:6.1f}s] OURS inference", flush=True)
    ours = ours_pipeline(cd, our_model, edges_np, J, bin_width, NUM_FEATURES,
                          centers, args, wasserstein_barycenter_1d)

    out = dict(source=args.source, seed=args.seed, n_context=args.n_context,
                n_test=int(len(true_cate)), true_ate=float(np.mean(true_cate)),
                runtime_s=time.time() - t0)

    def _record(name, cate_pred):
        arr = np.asarray(cate_pred, dtype=float)
        valid = np.isfinite(arr)
        if not np.any(valid):
            return                                      # all-NaN → skip (e.g. skip_malc)
        cate_v = arr[valid]
        tc_v = np.asarray(true_cate, dtype=float)[valid]
        out[f'pehe_{name}'] = _pehe(tc_v, cate_v)
        out[f'err_{name}']  = _ate_relerr(tc_v, cate_v)
        out[f'ate_{name}']  = float(np.mean(cate_v))

    if uwyk_noanc_cate is not None: _record('uwyk_noanc',     uwyk_noanc_cate)
    if uwyk_anc_cate   is not None: _record('uwyk_ancestral', uwyk_anc_cate)
    if dopfn_cate      is not None:
        # Only PEHE is written for Do-PFN in the sweep (paper convention).
        out['pehe_dopfn'] = _pehe(true_cate, dopfn_cate)
    _record('ours_mean',          ours['ours_mean'])
    _record('ours_malc_mean',     ours['ours_malc_mean'])
    _record('ours_malc_mean_msk', ours['ours_malc_mean_msk'])
    _record('ours_malc_mode',     ours['ours_malc_mode'])
    _record('ours_malc_mode_msk', ours['ours_malc_mode_msk'])

    # EM-adjusted MALC means (per-query CATE; NaN entries → per-query fit failed)
    _true_ate = float(np.mean(true_cate))
    def _record_nan_safe(name, cate_pred):
        arr = np.asarray(cate_pred, dtype=float)
        valid = np.isfinite(arr)
        if not np.any(valid):
            return
        cate_v = arr[valid]; tc_v = true_cate[valid]
        ate_pred = float(np.mean(cate_v))
        out[f'pehe_{name}'] = float(np.sqrt(np.mean((cate_v - tc_v) ** 2)))
        out[f'err_{name}']  = 0.0 if abs(_true_ate) < 1e-12 else \
                              abs(_true_ate - ate_pred) / abs(_true_ate)
        out[f'ate_{name}']  = ate_pred
        out[f'n_{name}']    = int(valid.sum())
    for name, key in [('ours_em_mix_mean', 'ours_em_mix_mean'),
                       ('ours_em_k1_mean',  'ours_em_k1_mean')]:
        if key in ours:
            _record_nan_safe(name, ours[key])

    true_ate = out['true_ate']
    ot_mode_ate = ours['ours_ot_mode_ate']
    out['ate_ours_ot_mode'] = ot_mode_ate
    out['err_ours_ot_mode'] = abs(ot_mode_ate - true_ate) / max(abs(true_ate), 1e-9)
    ot_mean_ate = ours['ours_ot_mean_ate']
    out['ate_ours_ot_mean'] = ot_mean_ate
    out['err_ours_ot_mean'] = abs(ot_mean_ate - true_ate) / max(abs(true_ate), 1e-9)

    np.savez(out_file, **{k: np.array(v) for k, v in out.items()})
    print(f"[{time.time()-t0:6.1f}s] saved {out_file}", flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
