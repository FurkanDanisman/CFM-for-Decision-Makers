"""Per-seed density L2 evaluation on linear-Gaussian synthetic SCM.

Fixed N_TRAIN=500, D=5 (override with N_TRAIN / SYN_D env vars). Truth is
closed-form Gaussian per query — see l2_syn/true_syn.py.

Env vars:
  REPO, DOPFN, CAUSALPFN  (paths, for reusing l2_ihdp runners)
  CHECKPOINT_DOPFN_BB     (Ours DoPFN-backbone checkpoint)
  METHODS  (comma list; subset of {dopfn, ours_dopfn_bb})
  OUT      (output prefix, one shard per seed)
  SEED, N_TRAIN, N_TEST, SYN_D, MALC_B, MALC_MAX_K, N_EVAL
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch


def _np(a):
    if isinstance(a, torch.Tensor): return a.numpy()
    return np.asarray(a)


def _guess_repo():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, '..', '..'))


def _install_dopfn_datasets_shim(dopfn_dir):
    import types
    if 'datasets' in sys.modules: return
    sys.path.insert(0, dopfn_dir)
    ds_mod = types.ModuleType('datasets')
    with open(os.path.join(dopfn_dir, 'datasets/__init__.py')) as fp:
        src = fp.read().split('def load_semi_real')[0]
    exec(src, ds_mod.__dict__)
    sys.modules['datasets'] = ds_mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int,
                     default=int(os.environ.get('SEED', 0)))
    ap.add_argument('--out', default=os.environ.get('OUT', 'l2_syn/out'))
    ap.add_argument('--n-train', type=int,
                     default=int(os.environ.get('N_TRAIN', 500)))
    ap.add_argument('--n-test', type=int,
                     default=int(os.environ.get('N_TEST', 200)))
    ap.add_argument('--syn-d', type=int,
                     default=int(os.environ.get('SYN_D', 5)))
    ap.add_argument('--malc-B', type=int,
                     default=int(os.environ.get('MALC_B', 500)))
    ap.add_argument('--malc-max-K', type=int,
                     default=int(os.environ.get('MALC_MAX_K', 3)))
    ap.add_argument('--n-eval', type=int,
                     default=int(os.environ.get('N_EVAL', 200)))
    ap.add_argument('--methods', default=os.environ.get(
        'METHODS', 'dopfn,ours_dopfn_bb'))
    ap.add_argument('--repo', default=os.environ.get('REPO', ''))
    ap.add_argument('--dopfn', default=os.environ.get('DOPFN', ''))
    ap.add_argument('--causalpfn', default=os.environ.get('CAUSALPFN', ''))
    ap.add_argument('--checkpoint50', default=os.environ.get('CHECKPOINT50', ''))
    ap.add_argument('--checkpoint-dopfn-bb',
                     default=os.environ.get('CHECKPOINT_DOPFN_BB', ''))
    ap.add_argument('--uwyk-src', default=os.environ.get('UWYK_SRC', ''))
    ap.add_argument('--uwyk-ckpt-dir', default=os.environ.get('UWYK_CKPT_DIR', ''))
    ap.add_argument('--uwyk-n-samples', type=int, default=1024)
    ap.add_argument('--n-context', type=int, default=0)
    ap.add_argument('--restrict-features', type=int, default=0)   # unused
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    allowed = {'dopfn', 'ours_dopfn_bb', 'ours_fn50'}
    assert all(m in allowed for m in methods), f'bad methods: {methods}'

    if not args.repo:
        args.repo = _guess_repo()

    _here = os.path.dirname(os.path.abspath(__file__))
    _ihdp = os.path.join(args.repo, 'benchmarks', 'l2_ihdp')
    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    sys.path.insert(0, os.path.join(args.repo, 'MALC', 'Optimal_Transport'))
    sys.path.insert(0, _ihdp)                  # methods_densities.py + runners
    sys.path.insert(0, _here)

    from l2 import l2_distance
    from ot_barycenter import wasserstein_barycenter_1d
    from syn_dgp import sample_realization
    from true_syn import (
        Y_CENTERS, TAU_CENTERS, build_syn_truth,
        true_marginals_per_query, true_cate_per_query, true_ate_barycenter,
    )
    # Load l2_ihdp/eval_realization.py by absolute path to avoid name
    # collision with THIS file (both are eval_realization.py).
    import importlib.util as _iu
    _ihdp_ev_path = os.path.join(_ihdp, 'eval_realization.py')
    _spec = _iu.spec_from_file_location('_l2_ihdp_eval_realization', _ihdp_ev_path)
    ihdp_ev = _iu.module_from_spec(_spec)
    _spec.loader.exec_module(ihdp_ev)

    print(f'[start] seed={args.seed} d={args.syn_d} N={args.n_train} '
          f'methods={methods}', flush=True)

    # Sample synthetic dataset + build closed-form truth
    cd, truth_raw = sample_realization(
        args.seed, d=args.syn_d, n_train=args.n_train, n_test=args.n_test)
    y_train_full = _np(cd.y_train)
    truth = build_syn_truth(truth_raw, y_train_full)
    n_queries = truth.mu0_test_scaled.shape[0]
    print(f'[truth] seed={args.seed}  n_queries={n_queries}  '
          f'sigma_scaled={truth.sigma_scaled:.4f}  y_rng={truth.y_rng:.3f}',
          flush=True)
    p_y0_true, p_y1_true = true_marginals_per_query(truth)
    p_tau_true = true_cate_per_query(truth)
    p_ate_true = true_ate_barycenter(p_tau_true, wasserstein_barycenter_1d)

    _install_dopfn_datasets_shim(args.dopfn)
    n_ctx = args.n_context if args.n_context > 0 else args.n_train

    RUN_ORDER = ['ours_fn50', 'ours_dopfn_bb', 'dopfn']
    method_out = {}
    for m in RUN_ORDER:
        if m not in methods: continue
        if m == 'ours_fn50':
            method_out[m] = ihdp_ev._run_ours(cd, args.checkpoint50, truth, args, n_ctx)
        elif m == 'ours_dopfn_bb':
            method_out[m] = ihdp_ev._run_ours_dopfn_bb(
                cd, args.checkpoint_dopfn_bb, truth, args, n_ctx)
        elif m == 'dopfn':
            method_out[m] = ihdp_ev._run_dopfn(cd, truth, args, n_ctx)

    # Per-realization density diagnostic PNG (methods vs truth).
    try:
        from density_plots import save_density_diag_png
        save_density_diag_png(
            out_path=f'{args.out}.density_diag.r{args.realization:03d}.png',
            r=args.realization,
            method_out=method_out,
            p_y0_true=p_y0_true, p_y1_true=p_y1_true,
            p_tau_true=p_tau_true, p_ate_true=p_ate_true,
            Y_CENTERS=Y_CENTERS, TAU_CENTERS=TAU_CENTERS,
            wb_fn=wasserstein_barycenter_1d,
            q_show=0,
        )
        print(f'[density-diag] saved r={args.realization}', flush=True)
    except Exception as e:
        print(f'[warn] density diag plot failed: {type(e).__name__}: {e}', flush=True)

    true_cate_raw = _np(cd.true_cate).reshape(-1)
    y_rng_over_2 = truth.y_rng / 2.0
    true_ate_raw = float(true_cate_raw.mean())

    out = dict(
        seed=np.int32(args.seed),
        d=np.int32(args.syn_d),
        n_train=np.int32(args.n_train),
        n_queries=np.int32(n_queries),
        p_y0_true=p_y0_true, p_y1_true=p_y1_true,
        p_tau_true=p_tau_true, p_ate_true=p_ate_true,
        true_cate_raw=true_cate_raw.astype(np.float32),
        true_ate_raw=np.float32(true_ate_raw),
        y_rng=np.float32(truth.y_rng),
    )
    tau_bin = float(TAU_CENTERS[1] - TAU_CENTERS[0])
    for name, d in method_out.items():
        p_ate = wasserstein_barycenter_1d(d['p_tau'], TAU_CENTERS)
        s = p_ate.sum() * tau_bin
        if s > 0: p_ate = p_ate / s

        cate_hat_scaled = (TAU_CENTERS[None, :] * d['p_tau']).sum(axis=1) * tau_bin
        cate_hat_raw = cate_hat_scaled * y_rng_over_2
        ate_hat_scaled = float((TAU_CENTERS * p_ate).sum() * tau_bin)
        ate_hat_raw = ate_hat_scaled * y_rng_over_2
        pehe = float(np.sqrt(np.mean((cate_hat_raw - true_cate_raw) ** 2)))
        eps_ate = float(abs(ate_hat_raw - true_ate_raw)
                        / max(abs(true_ate_raw), 1e-9))
        pehe_raw = eps_ate_raw = None
        if 'cate_raw_scaled' in d:
            cate_raw_raw = d['cate_raw_scaled'] * y_rng_over_2
            ate_raw_raw = float(cate_raw_raw.mean())
            pehe_raw = float(np.sqrt(np.mean((cate_raw_raw - true_cate_raw) ** 2)))
            eps_ate_raw = float(abs(ate_raw_raw - true_ate_raw)
                                / max(abs(true_ate_raw), 1e-9))

        def _malc_em_pehe(cate_em_scaled):
            arr = np.asarray(cate_em_scaled) * y_rng_over_2
            valid = np.isfinite(arr)
            if not np.any(valid):
                return None, None
            ate_em = float(arr[valid].mean())
            pehe = float(np.sqrt(np.mean((arr[valid] - true_cate_raw[valid]) ** 2)))
            eps  = float(abs(ate_em - true_ate_raw) / max(abs(true_ate_raw), 1e-9))
            return pehe, eps
        pehe_em_mix = eps_em_mix = None
        pehe_em_k1  = eps_em_k1  = None
        if 'cate_em_mix_scaled' in d:
            pehe_em_mix, eps_em_mix = _malc_em_pehe(d['cate_em_mix_scaled'])
        if 'cate_em_k1_scaled' in d:
            pehe_em_k1,  eps_em_k1  = _malc_em_pehe(d['cate_em_k1_scaled'])

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
        out[f'{name}__cate_hat_raw'] = cate_hat_raw.astype(np.float32)
        out[f'{name}__ate_hat_raw']  = np.float32(ate_hat_raw)
        out[f'{name}__pehe']         = np.float32(pehe)
        out[f'{name}__eps_ate']      = np.float32(eps_ate)
        if pehe_raw is not None:
            out[f'{name}__pehe_raw']    = np.float32(pehe_raw)
            out[f'{name}__eps_ate_raw'] = np.float32(eps_ate_raw)
        if pehe_em_mix is not None:
            out[f'{name}__pehe_em_mix']    = np.float32(pehe_em_mix)
            out[f'{name}__eps_ate_em_mix'] = np.float32(eps_em_mix)
        if pehe_em_k1 is not None:
            out[f'{name}__pehe_em_k1']     = np.float32(pehe_em_k1)
            out[f'{name}__eps_ate_em_k1']  = np.float32(eps_em_k1)
        print(f'[l2 ] {name:14s}  y0={l2_y0.mean():.4f}  y1={l2_y1.mean():.4f}  '
              f'tau={l2_tau.mean():.4f}  ate={l2_ate:.4f}', flush=True)

    shard = f'{args.out}.s{args.seed:04d}.npz'
    os.makedirs(os.path.dirname(shard) or '.', exist_ok=True)
    np.savez_compressed(shard, **out)
    print(f'[save] {shard}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
