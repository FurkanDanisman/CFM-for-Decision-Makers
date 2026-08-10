"""Per-realization L2 evaluation on IHDP.

For one IHDP realization r, computes:
  1. Analytical true densities per query (marginals, CATE) and the true
     population ATE via the 2-Wasserstein barycenter of per-query CATEs.
  2. Predicted densities from each of the 4 methods:
       ours_fn50, ours_fn10, uwyk_noanc, dopfn
     on the common Y_CENTERS / TAU_CENTERS grids.
  3. Per-query L2 distances between predicted and true for p_y0, p_y1, p_tau.
  4. Per-realization L2 for the ATE density (predicted-vs-true barycenter).

Output: one .npz shard per realization at $OUT.r{r:03d}.npz.

Environment
-----------
  REPO             R-PFN repo root                   (auto-detected)
  CAUSALPFN        path to CausalPFN source repo
  DOPFN            path to Do-PFN source repo
  UWYK_SRC         path to Graphs4CausalFoundationModels/src
  UWYK_CKPT_DIR    UWYK No-Ancestral checkpoint dir
  CHECKPOINT50     Ours(fn=50) .pt file
  CHECKPOINT10     Ours(fn=10) .pt file
  OUT              output NPZ path prefix   (default: $REPO/l2_ihdp/out)
  REALIZATION      int, 0..99                 (default: 0)
  N_CONTEXT        max training-context size  (default: full training set)
  MALC_B           MALC bootstrap size        (default: 60)
  MALC_MAX_K       MALC mixture order cap     (default: 3)
  N_EVAL           MALC evaluation grid size  (default: 200)
  UWYK_N_SAMPLES   samples per query per arm  (default: 1024)
  METHODS          comma list, subset of {ours_fn50, ours_fn10, uwyk_noanc, dopfn}
                     default: all four
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
import types

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--realization', type=int,
                     default=int(os.environ.get('REALIZATION', 0)))
    ap.add_argument('--out', default=os.environ.get('OUT', 'l2_ihdp/out'))
    ap.add_argument('--n-context', type=int,
                     default=int(os.environ.get('N_CONTEXT', 0)))         # 0 = full
    ap.add_argument('--malc-B', type=int,
                     default=int(os.environ.get('MALC_B', 60)))
    ap.add_argument('--malc-max-K', type=int,
                     default=int(os.environ.get('MALC_MAX_K', 3)))
    ap.add_argument('--n-eval', type=int,
                     default=int(os.environ.get('N_EVAL', 200)))
    ap.add_argument('--uwyk-n-samples', type=int,
                     default=int(os.environ.get('UWYK_N_SAMPLES', 1024)))
    ap.add_argument('--methods', default=os.environ.get(
        'METHODS', 'ours_fn50,ours_fn10,uwyk_noanc,dopfn'))
    ap.add_argument('--repo', default=os.environ.get('REPO', ''))
    ap.add_argument('--causalpfn', default=os.environ.get('CAUSALPFN', ''))
    ap.add_argument('--dopfn', default=os.environ.get('DOPFN', ''))
    ap.add_argument('--uwyk-src', default=os.environ.get('UWYK_SRC', ''))
    ap.add_argument('--uwyk-ckpt-dir', default=os.environ.get('UWYK_CKPT_DIR', ''))
    ap.add_argument('--checkpoint50', default=os.environ.get('CHECKPOINT50', ''))
    ap.add_argument('--checkpoint10', default=os.environ.get('CHECKPOINT10', ''))
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    assert all(m in {'ours_fn50', 'ours_fn10', 'uwyk_noanc', 'dopfn'}
                for m in methods), f'bad methods: {methods}'

    if not args.repo:
        args.repo = _guess_repo()

    _require_paths(args, methods)

    # Path plumbing — mirror the sibling plot_ihdp_n10*.py convention.
    # Our own package name (`benchmarks`) collides with CausalPFN's `benchmarks`,
    # so we insert `l2_ihdp/` directly and import our modules unqualified.
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC', 'Optimal_Transport'))
    sys.path.insert(0, _here)

    from l2 import l2_distance
    from true_ihdp import (
        TAU_CENTERS, Y_CENTERS, load_ihdp_truth,
        true_ate_barycenter, true_cate_per_query, true_marginals_per_query,
    )
    from ot_barycenter import wasserstein_barycenter_1d
    from methods_densities import (
        dopfn_densities, ours_densities, uwyk_noanc_densities,
    )

    # ── Load IHDP realization ───────────────────────────────────────────
    print(f'[start] realization {args.realization}  methods={methods}', flush=True)
    _install_dopfn_datasets_shim(args.dopfn)
    sys.path.insert(0, args.causalpfn)
    from benchmarks import IHDPDataset

    cd, _ = IHDPDataset()[args.realization]
    y_train_full = _np(cd.y_train)
    n_ctx = args.n_context if args.n_context > 0 else y_train_full.shape[0]

    # ── Truth ───────────────────────────────────────────────────────────
    truth = load_ihdp_truth(args.realization, args.causalpfn, y_train_full)
    n_queries = truth.mu0_test_scaled.shape[0]
    print(f'[truth] r={args.realization}  n_queries={n_queries}  '
          f'sigma_scaled={truth.sigma_scaled:.4f}', flush=True)
    p_y0_true, p_y1_true = true_marginals_per_query(truth)
    p_tau_true = true_cate_per_query(truth)
    p_ate_true = true_ate_barycenter(p_tau_true, wasserstein_barycenter_1d)

    # ── Method densities ────────────────────────────────────────────────
    # Order matters: UWYK's constructor loads its own `utils` and `models`
    # modules and pollutes sys.path/modules in ways that break subsequent
    # imports of Do-PFN (which has its own `utils` package with different
    # symbols). Run UWYK last so downstream imports never encounter its
    # cruft.
    RUN_ORDER = ['ours_fn50', 'ours_fn10', 'dopfn', 'uwyk_noanc']
    method_out: dict[str, dict[str, np.ndarray]] = {}
    for m in RUN_ORDER:
        if m not in methods:
            continue
        if m == 'ours_fn50':
            method_out[m] = _run_ours(cd, args.checkpoint50, truth, args, n_ctx)
        elif m == 'ours_fn10':
            method_out[m] = _run_ours(cd, args.checkpoint10, truth, args, n_ctx)
        elif m == 'dopfn':
            method_out[m] = _run_dopfn(cd, truth, args, n_ctx)
        elif m == 'uwyk_noanc':
            method_out[m] = _run_uwyk_noanc(cd, truth, args, n_ctx)

    # ── L2 distances ────────────────────────────────────────────────────
    out: dict[str, np.ndarray] = dict(
        r=np.int32(args.realization),
        n_queries=np.int32(n_queries),
        p_y0_true=p_y0_true, p_y1_true=p_y1_true,
        p_tau_true=p_tau_true, p_ate_true=p_ate_true,
    )
    for name, d in method_out.items():
        p_ate = wasserstein_barycenter_1d(d['p_tau'], TAU_CENTERS)
        s = p_ate.sum() * float(TAU_CENTERS[1] - TAU_CENTERS[0])
        if s > 0: p_ate = p_ate / s

        l2_y0 = np.array([l2_distance(d['p_y0'][q], p_y0_true[q], Y_CENTERS)
                          for q in range(n_queries)])
        l2_y1 = np.array([l2_distance(d['p_y1'][q], p_y1_true[q], Y_CENTERS)
                          for q in range(n_queries)])
        l2_tau = np.array([l2_distance(d['p_tau'][q], p_tau_true[q], TAU_CENTERS)
                           for q in range(n_queries)])
        l2_ate = l2_distance(p_ate, p_ate_true, TAU_CENTERS)

        out[f'{name}__p_y0']  = d['p_y0']
        out[f'{name}__p_y1']  = d['p_y1']
        out[f'{name}__p_tau'] = d['p_tau']
        out[f'{name}__p_ate'] = p_ate
        out[f'{name}__l2_y0']  = l2_y0.astype(np.float32)
        out[f'{name}__l2_y1']  = l2_y1.astype(np.float32)
        out[f'{name}__l2_tau'] = l2_tau.astype(np.float32)
        out[f'{name}__l2_ate'] = np.float32(l2_ate)
        print(f'[l2 ] {name:12s}  y0={l2_y0.mean():.4f}  y1={l2_y1.mean():.4f}  '
              f'tau={l2_tau.mean():.4f}  ate={l2_ate:.4f}', flush=True)

    # ── Save shard ──────────────────────────────────────────────────────
    shard = f'{args.out}.r{args.realization:03d}.npz'
    os.makedirs(os.path.dirname(shard) or '.', exist_ok=True)
    np.savez_compressed(shard, **out)
    print(f'[save] {shard}', flush=True)
    return 0


# ── Method drivers ──────────────────────────────────────────────────────
def _run_ours(cd, ckpt_path, truth, args, n_ctx):
    from models.InterventionalPFN import InterventionalPFN
    from losses.BarDistribution2D import fit_malc_inner
    from malc_2d import dmalc_2d
    from methods_densities import ours_densities

    print(f'[ours] loading {ckpt_path}', flush=True)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']; J = cfg['J']
    edges_np = ckpt['edges'].cpu().numpy()
    bin_width = float(edges_np[1] - edges_np[0])
    num_features = cfg['num_features']
    model = InterventionalPFN(
        num_features=num_features, d_model=cfg['d_model'], depth=cfg['depth'],
        heads_feat=cfg['heads'], heads_samp=cfg['heads'], dropout=0.0,
        output_dim=J*J + 9 + 4, hidden_mult=cfg['hidden_mult'],
        normalize_features=True, normalize_treatment=False,
        use_treatment_in_query=False, use_checkpoint=False,
    ).eval()
    model.load_state_dict(ckpt['model_state_dict'])

    t0 = time.time()
    d = ours_densities(
        cd, model, edges_np, J, bin_width, num_features,
        y_min=truth.y_min, y_rng=truth.y_rng,
        malc_B=args.malc_B, malc_max_K=args.malc_max_K, n_eval=args.n_eval,
        n_context=n_ctx,
        fit_malc_inner=fit_malc_inner, dmalc_2d=dmalc_2d,
    )
    print(f'[ours] done in {time.time() - t0:.1f}s', flush=True)
    return d


def _run_uwyk_noanc(cd, truth, args, n_ctx):
    from methods_densities import uwyk_noanc_densities

    # UWYK's imports collide with local models/utils — isolate as
    # plot_ihdp_n10_uwyk_noanc.py does.
    _saved = {}
    for name in list(sys.modules):
        if (name == 'models' or name.startswith('models.') or
                name == 'utils' or name.startswith('utils.')):
            _saved[name] = sys.modules.pop(name)
    sys.path.insert(0, args.uwyk_src)
    pre_mod = importlib.import_module('models.PreprocessingGraphConditionedPFN')
    sys.path.remove(args.uwyk_src)
    for name in list(sys.modules):
        if (name == 'models' or name.startswith('models.') or
                name == 'utils' or name.startswith('utils.')):
            del sys.modules[name]
    sys.modules.update(_saved)

    print('[uwyk] loading checkpoint', flush=True)
    _orig_load = torch.load
    def _p_load(*a, **kw):
        kw.setdefault('weights_only', False); return _orig_load(*a, **kw)
    torch.load = _p_load
    uwyk_model = pre_mod.PreprocessingGraphConditionedPFN(
        config_path=os.path.join(args.uwyk_ckpt_dir, 'best_model_config.yaml'),
        checkpoint_path=os.path.join(args.uwyk_ckpt_dir, 'best_model.pt'),
        device='cpu', verbose=False,
    ).load()
    torch.load = _orig_load
    num_features = uwyk_model.model.num_features

    t0 = time.time()
    d = uwyk_noanc_densities(
        cd, uwyk_model, num_features,
        y_min=truth.y_min, y_rng=truth.y_rng,
        n_context=n_ctx, n_samples=args.uwyk_n_samples,
    )
    print(f'[uwyk] done in {time.time() - t0:.1f}s', flush=True)
    return d


def _run_dopfn(cd, truth, args, n_ctx):
    # dopfn.py installs an sklearn check_array shim on import — pull it in
    # under a temporary sys.path since we can't `from benchmarks.methods...`
    # (the `benchmarks` package name is taken by CausalPFN).
    _bench_methods = os.path.join(args.repo, 'benchmarks', 'methods')
    if _bench_methods not in sys.path:
        sys.path.insert(0, _bench_methods)
    import dopfn as _dopfn_shim  # noqa: F401  — imports install the shim
    from scripts.transformer_prediction_interface.base import DoPFNRegressor
    from methods_densities import dopfn_densities

    t0 = time.time()
    d = dopfn_densities(
        cd, DoPFNRegressor,
        y_min=truth.y_min, y_rng=truth.y_rng,
        dopfn_root=args.dopfn,
        n_context=n_ctx,
    )
    print(f'[dopfn] done in {time.time() - t0:.1f}s', flush=True)
    return d


# ── Path plumbing ───────────────────────────────────────────────────────
def _install_dopfn_datasets_shim(dopfn_dir: str) -> None:
    """CausalPFN's IHDPDataset imports a bare `datasets` module — shim it in."""
    if 'datasets' in sys.modules:
        return
    sys.path.insert(0, dopfn_dir)
    ds_mod = types.ModuleType('datasets')
    with open(os.path.join(dopfn_dir, 'datasets/__init__.py')) as fp:
        src = fp.read().split('def load_semi_real')[0]
    exec(src, ds_mod.__dict__)
    sys.modules['datasets'] = ds_mod


def _require_paths(args, methods) -> None:
    need_ours = any(m.startswith('ours_') for m in methods)
    need_uwyk = 'uwyk_noanc' in methods
    need_dopfn = 'dopfn' in methods

    for label, path, cond in [
        ('CAUSALPFN', args.causalpfn, True),
        ('DOPFN',     args.dopfn,     True),
        ('CHECKPOINT50', args.checkpoint50, 'ours_fn50' in methods),
        ('CHECKPOINT10', args.checkpoint10, 'ours_fn10' in methods),
        ('UWYK_SRC',      args.uwyk_src,      need_uwyk),
        ('UWYK_CKPT_DIR', args.uwyk_ckpt_dir, need_uwyk),
    ]:
        if not cond:
            continue
        exists = os.path.isdir(path) or os.path.isfile(path)
        if not exists:
            print(f'[fatal] {label} not found at {path!r}')
            sys.exit(2)


def _guess_repo() -> str:
    here = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def _np(a):
    if isinstance(a, torch.Tensor): return a.numpy()
    return np.asarray(a)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc(); sys.exit(1)
