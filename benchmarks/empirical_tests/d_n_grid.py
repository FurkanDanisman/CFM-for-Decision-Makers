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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rho_scaling_linear import make_linear_scm, load_ours, load_uwyk

DEVICE = torch.device('cpu')
D_GRID = (2, 3, 4, 5, 8)
# Per-d N grids, targeted at each d's expected √2-crossing knee.
# One-shot exploration grid: few cells per d, run with K=1.
D_N_MAP = {
    2: (2000, 5000),
    3: (3000, 5000),
    4: (5000, 8000),
    5: (8000,),
    8: (12000,),
}


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
            d_ = {k: f[k] for k in f.files}
        if all_arr is None: all_arr = {k: [] for k in d_}
        for k, v in d_.items(): all_arr[k].append(v)
    if all_arr is None or not len(all_arr.get('seed', [])):
        raise SystemExit('[error] no shards found')
    arr = {k: np.concatenate(v) for k, v in all_arr.items()}
    np.savez(args.out + '.npz', **arr)
    print(f'[save] {args.out}.npz  ({len(arr["seed"])} rows)')

    print()
    print(f'{"d":>3} {"N":>6} {"n":>4} {"UWYK":>10} {"Ours":>10} '
          f'{"√PEHE ratio":>12} {"MSE ratio":>10}')
    print('-' * 66)
    for d in D_GRID:
        for N in D_N_MAP[d]:
            m = (arr['d'] == d) & (arr['N'] == N)
            if not m.any(): continue
            pu, po = arr['pehe_uwyk'][m], arr['pehe_ours50'][m]
            rat = pu.mean() / po.mean()
            print(f'{d:>3} {N:>6} {int(m.sum()):>4} '
                  f'{pu.mean():>10.3f} {po.mean():>10.3f} '
                  f'{rat:>12.3f} {rat**2:>10.3f}')

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.axhline(np.sqrt(2), color='k', ls='--', lw=1.5,
                label=r'Theorem 3.2: $\sqrt{2}$')
    ax.axhline(1.0, color='r', ls=':', lw=1, alpha=0.6, label='no improvement')
    colors = ['#0F8A3C', '#B84A2A', '#2E4A6F', '#8A4FBE']
    for c, d in zip(colors, D_GRID):
        Ns, rats = [], []
        for N in D_N_MAP[d]:
            m = (arr['d'] == d) & (arr['N'] == N)
            if not m.any(): continue
            pu, po = arr['pehe_uwyk'][m], arr['pehe_ours50'][m]
            Ns.append(N); rats.append(pu.mean() / po.mean())
        ax.plot(Ns, rats, 'o-', color=c, lw=2, markersize=9,
                 label=f'$d={d}$')
    ax.set_xscale('log')
    ax.set_xlabel(r'Context size $N$', fontsize=11)
    ax.set_ylabel(r'$\sqrt{\mathrm{PEHE}}_{\mathrm{UWYK}}/\sqrt{\mathrm{PEHE}}_{\mathrm{Ours(fn{=}50)}}$',
                    fontsize=11)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches='tight'); plt.close(fig)
    print(f'[save] {args.out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo',           required=True)
    ap.add_argument('--checkpoint50',   required=True)
    ap.add_argument('--uwyk-src',       required=True)
    ap.add_argument('--uwyk-ckpt-dir',  required=True)
    ap.add_argument('--K',              type=int, default=15)
    ap.add_argument('--N-test',         type=int, default=50)
    ap.add_argument('--sigma-eps',      type=float, default=1.0)
    ap.add_argument('--out',            default='d_n_grid.png')
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
                ours50 = ours_pipeline(cd, m50, edges50, J50, bw50, NF50, ctr50,
                                         types.SimpleNamespace(repo=args.repo,
                                                                malc_B=30, malc_max_K=3,
                                                                n_eval=200, workers=8),
                                         wb50)
                pehe_ours50 = _pehe(true_cate, ours50['ours_mean'])

                rows.append(dict(d=d, N=N, seed=seed,
                                  pehe_uwyk=pehe_uwyk, pehe_ours50=pehe_ours50))
                print(f'[scm] d={d} N={N:<5d} k={k}  '
                      f'UWYK={pehe_uwyk:.3f}  Ours50={pehe_ours50:.3f}',
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
