"""Controlled d-scaling test on a strictly linear SCM.

Fixed ρ = 0 (noise-independent arms), fixed N_context = 200. Sweeps
covariate dimension d ∈ D_GRID and reports the √PEHE ratio and
stability ratio between UWYK-NoAnc and Ours(fn=50). Theorem 3.2
predicts the ratio does not depend on d.

Reuses rho_scaling_linear.make_linear_scm / load_ours / load_uwyk.
6-way SLURM array (one d per task) with resume-aware shards.
"""
from __future__ import annotations
import argparse, os, sys, types, traceback
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rho_scaling_linear import make_linear_scm, load_ours, load_uwyk

DEVICE = torch.device('cpu')
D_GRID = (5, 10, 20, 30, 50)


def _pehe(true_cate, pred_cate):
    t = np.asarray(true_cate).reshape(-1)
    p = np.asarray(pred_cate).reshape(-1)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def _plot_aggregate(args):
    import matplotlib.pyplot as plt
    all_arr = None
    for idx in range(len(D_GRID)):
        shard = f'{args.out}.d{idx}.npz'
        if not os.path.exists(shard):
            print(f'[warn] missing shard {shard}'); continue
        with np.load(shard, allow_pickle=True) as f:
            d = {k: f[k] for k in f.files}
        if all_arr is None: all_arr = {k: [] for k in d}
        for k, v in d.items(): all_arr[k].append(v)
    if all_arr is None or not len(all_arr.get('seed', [])):
        raise SystemExit('[error] no shards found')
    arr = {k: np.concatenate(v) for k, v in all_arr.items()}
    np.savez(args.out + '.npz', **arr)
    print(f'[save] {args.out}.npz  ({len(arr["seed"])} rows)')

    ratio = arr['pehe_uwyk'] / np.maximum(arr['pehe_ours50'], 1e-9)

    print()
    print(f'{"d":>4} {"n":>4} {"UWYK (mean±std)":>18} {"Ours (mean±std)":>18} '
          f'{"√PEHE ratio":>12} {"stability":>12}')
    print('-' * 78)
    for d in D_GRID:
        m = arr['d'] == d
        if not m.any(): continue
        pu, po = arr['pehe_uwyk'][m], arr['pehe_ours50'][m]
        print(f'{d:>4} {int(m.sum()):>4} '
              f'{pu.mean():>8.3f} ± {pu.std():>4.2f}   '
              f'{po.mean():>8.3f} ± {po.std():>4.2f}   '
              f'{np.median(ratio[m]):>12.3f} '
              f'{po.std()/max(pu.std(), 1e-9):>12.3f}')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(np.sqrt(2), color='k', ls='--', lw=1.5,
                label=r'Theory ceiling: $\sqrt{2}$')
    for d in D_GRID:
        m = arr['d'] == d
        if not m.any(): continue
        med = float(np.median(ratio[m])); std = float(ratio[m].std())
        ax.errorbar(d, med, yerr=std, fmt='o', color='#0F8A3C',
                     markersize=9, capsize=4, zorder=4,
                     label='empirical median ± std' if d == D_GRID[0] else None)
        ax.scatter(np.full(m.sum(), d), ratio[m], color='#2E4A6F',
                    alpha=0.4, s=32, zorder=3)
    ax.axhline(1.0, color='r', ls=':', lw=1, alpha=0.6)
    ax.set_xlabel(r'Covariate dimension $d$')
    ax.set_ylabel(r'$\sqrt{\mathrm{PEHE}}_{\mathrm{UWYK\ NoAnc}} / \sqrt{\mathrm{PEHE}}_{\mathrm{Ours\ (fn=50)}}$')
    ax.set_title(f'd-scaling on strictly linear SCM (K={args.K}/d, N={args.N_context}, ρ=0)')
    ax.set_xscale('log')
    ax.set_ylim(0, 3)
    ax.grid(alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'[save] {args.out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo',           required=True)
    ap.add_argument('--checkpoint50',   required=True)
    ap.add_argument('--uwyk-src',       required=True)
    ap.add_argument('--uwyk-ckpt-dir',  required=True)
    ap.add_argument('--K',              type=int, default=15)
    ap.add_argument('--N-context',      type=int, default=200)
    ap.add_argument('--N-test',         type=int, default=50)
    ap.add_argument('--sigma-eps',      type=float, default=1.0)
    ap.add_argument('--out',            default='d_scaling_linear.png')
    ap.add_argument('--d-index',        type=int, default=-1)
    ap.add_argument('--plot',           action='store_true')
    args = ap.parse_args()

    if args.plot:
        _plot_aggregate(args); return

    sys.path.insert(0, os.path.join(args.repo, 'benchmarks'))
    from methods.ours   import ours_pipeline
    from methods.uwyk   import uwyk_no_ancestral_pipeline

    print('[load] Ours fn=50', flush=True); o50 = load_ours(args, args.checkpoint50)
    print('[load] UWYK',       flush=True); uwyk = load_uwyk(args.uwyk_src, args.uwyk_ckpt_dir)

    if args.d_index >= 0:
        d_targets = [D_GRID[args.d_index]]
        shard_path = f'{args.out}.d{args.d_index}.npz'
    else:
        d_targets = list(D_GRID)
        shard_path = args.out + '.npz'

    rows = []
    done_ks: dict[int, set] = {d: set() for d in d_targets}
    if os.path.exists(shard_path):
        with np.load(shard_path, allow_pickle=True) as f:
            for i in range(int(len(f['seed']))):
                r = {k: (float(f[k][i]) if f[k].dtype.kind == 'f' else int(f[k][i]))
                     for k in f.files}
                rows.append(r)
                seed_i = int(f['seed'][i]); d_i = int(f['d'][i])
                done_ks.setdefault(d_i, set()).add(seed_i - d_i * 10_000)
        print(f'[resume] {len(rows)} rows from {shard_path}', flush=True)

    for d in d_targets:
        for k in range(args.K):
            if k in done_ks.get(d, set()): continue
            seed = d * 10_000 + k
            cd = make_linear_scm(seed=seed, n_context=args.N_context, n_test=args.N_test,
                                   rho=0.0, x_dim=d, sigma_eps=args.sigma_eps)
            true_cate = cd.true_cate.numpy()

            uwyk_pred = uwyk_no_ancestral_pipeline(uwyk, cd)
            pehe_uwyk = _pehe(true_cate, uwyk_pred)

            m50, edges50, J50, bw50, ctr50, NF50, wb50 = o50
            ours50 = ours_pipeline(cd, m50, edges50, J50, bw50, NF50, ctr50,
                                     types.SimpleNamespace(repo=args.repo,
                                                            malc_B=30, malc_max_K=3,
                                                            n_eval=200, workers=8),
                                     wb50)
            pehe_ours50 = _pehe(true_cate, ours50['ours_mean'])

            rows.append(dict(d=d, seed=seed,
                              pehe_uwyk=pehe_uwyk, pehe_ours50=pehe_ours50))
            print(f'[scm] d={d} k={k}  UWYK={pehe_uwyk:.3f}  Ours50={pehe_ours50:.3f}',
                  flush=True)

    if not rows: print(f'[skip] {shard_path} unchanged'); return
    keys = list(rows[0].keys())
    arr = {k: np.array([r[k] for r in rows]) for k in keys}
    np.savez(shard_path, **arr)
    print(f'[save] {shard_path}  ({len(rows)} rows)')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
