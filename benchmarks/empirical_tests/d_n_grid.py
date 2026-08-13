"""d–N grid sweep on the linear SCM (ρ = 0).

For each d ∈ D_GRID (small values so we can hope to reach the theorem),
sweeps N ∈ N_GRID and records √PEHE for Ours(fn=50) and UWYK-NoAnc.
Goal: for each d, find the smallest N at which the observed √PEHE
ratio reaches (or crosses) the theorem's √2. Then a scaling law
N*(d) can be fit and extrapolated.

Reuses rho_scaling_linear.make_linear_scm / load_ours / load_uwyk.
4-way SLURM array (one d per task), resume-aware shards.
"""
from __future__ import annotations
import argparse, os, sys, types, traceback
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from rho_scaling_linear import make_linear_scm, load_ours, load_uwyk
from dopfn_helpers import load_dopfn_bb, load_dopfn, dopfn_predict_cate

DEVICE = torch.device('cpu')
D_GRID = tuple(range(2, 13))       # d = 2, 3, ..., 12
# Test the linear scaling hypothesis N*(d) ≈ 1250·d.
# One cell per d at the predicted knee; if the fit is correct, every
# cell's √PEHE ratio should sit near √2 = 1.414.
D_N_MAP = {d: (1250 * d,) for d in D_GRID}


def _pehe(true_cate, pred_cate):
    t = np.asarray(true_cate).reshape(-1)
    p = np.asarray(pred_cate).reshape(-1)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def _print_table(arr, num_key, den_key, num_label, den_label):
    """Per-d table with mean±SEM for numerator/denominator + ratio."""
    print()
    print(f'── {num_label} vs {den_label}  (N ≈ 1250·d) ──')
    print(f'{"d":>3} {"N":>6} {"n":>4} {num_label + " (mean±SEM)":>26s} '
          f'{den_label + " (mean±SEM)":>26s} '
          f'{"√PEHE ratio":>12} {"MSE ratio":>10}')
    print('-' * 96)
    for d in D_GRID:
        for N in D_N_MAP[d]:
            m = (arr['d'] == d) & (arr['N'] == N)
            if not m.any() or num_key not in arr or den_key not in arr: continue
            pu = arr[num_key][m]; pu = pu[np.isfinite(pu)]
            po = arr[den_key][m]; po = po[np.isfinite(po)]
            if pu.size == 0 or po.size == 0: continue
            sem_u = pu.std(ddof=1) / np.sqrt(pu.size) if pu.size > 1 else 0.0
            sem_o = po.std(ddof=1) / np.sqrt(po.size) if po.size > 1 else 0.0
            rat = pu.mean() / max(po.mean(), 1e-9)
            print(f'{d:>3} {N:>6} {int(m.sum()):>4} '
                  f'{pu.mean():>12.3f} ± {sem_u:>8.3f}   '
                  f'{po.mean():>12.3f} ± {sem_o:>8.3f}   '
                  f'{rat:>12.3f} {rat**2:>10.3f}')


def _plot_ratio(arr, num_key, den_key, num_label, den_label, out_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.axhline(np.sqrt(2), color='k', ls='--', lw=1.5,
                label=r'Theorem 3.2: $\sqrt{2}$')
    ax.axhline(1.0, color='r', ls=':', lw=1, alpha=0.6, label='no improvement')
    d_list, rats, sems = [], [], []
    for d in D_GRID:
        for N in D_N_MAP[d]:
            m = (arr['d'] == d) & (arr['N'] == N)
            if not m.any() or num_key not in arr or den_key not in arr: continue
            pu = arr[num_key][m]; po = arr[den_key][m]
            if pu.size == 0 or po.size == 0: continue
            rat = pu / np.maximum(po, 1e-9)
            d_list.append(d); rats.append(float(rat.mean()))
            sems.append(float(rat.std(ddof=1) / np.sqrt(rat.size))
                          if rat.size > 1 else 0.0)
    ax.errorbar(d_list, rats, yerr=sems, fmt='o-', color='#0F8A3C',
                  lw=2, markersize=8, capsize=4,
                  label='mean ± SEM')
    ax.set_xlabel(r'Covariate dimension $d$   (with $N \approx 1250\,d$)', fontsize=11)
    ax.set_ylabel(rf'$\sqrt{{\mathrm{{PEHE}}}}_{{\mathrm{{{num_label}}}}} / '
                    rf'\sqrt{{\mathrm{{PEHE}}}}_{{\mathrm{{{den_label}}}}}$',
                    fontsize=11)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight'); plt.close(fig)
    print(f'[save] {out_path}')


def _plot_aggregate(args):
    all_arr = None
    for idx in range(len(D_GRID)):
        shard = f'{args.out}.d{idx}.npz'
        if not os.path.exists(shard):
            print(f'[warn] missing shard {shard}'); continue
        with np.load(shard, allow_pickle=True) as f:
            d_ = {k: f[k] for k in f.files}
        if all_arr is None: all_arr = {k: [] for k in d_}
        for k, v in d_.items(): all_arr.setdefault(k, []).append(v)
    if all_arr is None or not len(all_arr.get('seed', [])):
        raise SystemExit('[error] no shards found')
    arr = {k: np.concatenate(v) for k, v in all_arr.items()}
    np.savez(args.out + '.npz', **arr)
    print(f'[save] {args.out}.npz  ({len(arr["seed"])} rows)')

    if 'pehe_uwyk' in arr and 'pehe_ours50' in arr:
        _print_table(arr, 'pehe_uwyk', 'pehe_ours50', 'UWYK', 'Ours(fn=50)')
        _plot_ratio(arr, 'pehe_uwyk', 'pehe_ours50', 'UWYK', 'Ours(fn{=}50)',
                     args.out + '.png')

    if 'pehe_dopfn' in arr and 'pehe_dopfnbb' in arr:
        _print_table(arr, 'pehe_dopfn', 'pehe_dopfnbb',
                       'Do-PFN', 'Ours-DoPFN-bb(200K)')
        _plot_ratio(arr, 'pehe_dopfn', 'pehe_dopfnbb',
                     'Do\\text{-}PFN', 'Ours\\text{-}DoPFN\\text{-}bb\\ (200K)',
                     args.out + '_dopfn.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo',                required=True)
    ap.add_argument('--checkpoint50',        required=True)
    ap.add_argument('--checkpoint-dopfn-bb', default='')
    ap.add_argument('--dopfn',               default='')
    ap.add_argument('--causalpfn',           default='')
    ap.add_argument('--uwyk-src',            required=True)
    ap.add_argument('--uwyk-ckpt-dir',       required=True)
    ap.add_argument('--K',                   type=int, default=15)
    ap.add_argument('--N-test',              type=int, default=50)
    ap.add_argument('--sigma-eps',           type=float, default=1.0)
    ap.add_argument('--out',                 default='d_n_grid.png')
    ap.add_argument('--d-index',             type=int, default=-1)
    ap.add_argument('--plot',                action='store_true')
    args = ap.parse_args()

    if args.plot:
        _plot_aggregate(args); return

    sys.path.insert(0, os.path.join(args.repo, 'benchmarks'))
    from methods.ours   import ours_pipeline
    from methods.uwyk   import uwyk_no_ancestral_pipeline

    print('[load] Ours fn=50', flush=True); o50 = load_ours(args, args.checkpoint50)
    print('[load] UWYK',       flush=True); uwyk = load_uwyk(args.uwyk_src, args.uwyk_ckpt_dir)

    run_dopfn = bool(args.checkpoint_dopfn_bb and args.dopfn)
    obb = None; DoPFNRegressor = None
    if run_dopfn:
        print('[load] Ours-DoPFN-bb (200K)', flush=True)
        obb = load_dopfn_bb(args, args.checkpoint_dopfn_bb)
        print('[load] Do-PFN', flush=True)
        DoPFNRegressor = load_dopfn(args)

    if args.d_index >= 0:
        d_targets = [D_GRID[args.d_index]]
        shard_path = f'{args.out}.d{args.d_index}.npz'
    else:
        d_targets = list(D_GRID)
        shard_path = args.out + '.npz'

    rows = []
    done: dict[tuple, set] = {}
    if os.path.exists(shard_path):
        with np.load(shard_path, allow_pickle=True) as f:
            for i in range(int(len(f['seed']))):
                r = {k: (float(f[k][i]) if f[k].dtype.kind == 'f' else int(f[k][i]))
                     for k in f.files}
                rows.append(r)
                key = (int(f['d'][i]), int(f['N'][i]))
                done.setdefault(key, set()).add(int(f['seed'][i]) - int(f['d'][i]) * 10_000 - int(f['N'][i]))
        print(f'[resume] {len(rows)} rows from {shard_path}', flush=True)

    _cwd = os.getcwd()
    for d in d_targets:
        for N in D_N_MAP[d]:
            for k in range(args.K):
                if k in done.get((d, N), set()): continue
                seed = d * 10_000 + N + k
                cd = make_linear_scm(seed=seed, n_context=N, n_test=args.N_test,
                                       rho=0.0, x_dim=d, sigma_eps=args.sigma_eps)
                true_cate = cd.true_cate.numpy()

                uwyk_pred = uwyk_no_ancestral_pipeline(uwyk, cd)
                pehe_uwyk = _pehe(true_cate, uwyk_pred)

                m50, edges50, J50, bw50, ctr50, NF50, wb50 = o50
                _ours_args = types.SimpleNamespace(repo=args.repo,
                                                      malc_B=30, malc_max_K=3,
                                                      n_eval=200, workers=8)
                ours50 = ours_pipeline(cd, m50, edges50, J50, bw50, NF50, ctr50,
                                         _ours_args, wb50)
                pehe_ours50 = _pehe(true_cate, ours50['ours_mean'])

                row = dict(d=d, N=N, seed=seed,
                            pehe_uwyk=pehe_uwyk, pehe_ours50=pehe_ours50)

                if run_dopfn:
                    os.chdir(args.dopfn)
                    dopfn_pred = dopfn_predict_cate(DoPFNRegressor, cd)
                    os.chdir(_cwd)
                    pehe_dopfn = _pehe(true_cate, dopfn_pred)
                    mbb, edgesbb, Jbb, bwbb, ctrbb, NFbb, wbbb = obb
                    ours_bb = ours_pipeline(cd, mbb, edgesbb, Jbb, bwbb, NFbb, ctrbb,
                                              _ours_args, wbbb)
                    pehe_dopfnbb = _pehe(true_cate, ours_bb['ours_mean'])
                    row['pehe_dopfn'] = pehe_dopfn
                    row['pehe_dopfnbb'] = pehe_dopfnbb
                    print(f'[scm] d={d} N={N:<5d} k={k}  '
                          f'UWYK={pehe_uwyk:.3f}  Ours50={pehe_ours50:.3f}  '
                          f'Do-PFN={pehe_dopfn:.3f}  Ours-DoPFN-bb={pehe_dopfnbb:.3f}',
                          flush=True)
                else:
                    print(f'[scm] d={d} N={N:<5d} k={k}  '
                          f'UWYK={pehe_uwyk:.3f}  Ours50={pehe_ours50:.3f}',
                          flush=True)
                rows.append(row)

    if not rows: print(f'[skip] {shard_path} unchanged'); return
    keys = sorted({k for r in rows for k in r.keys()})
    arr = {k: np.array([r.get(k, np.nan) for r in rows]) for k in keys}
    np.savez(shard_path, **arr)
    print(f'[save] {shard_path}  ({len(rows)} rows)')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
