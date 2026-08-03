"""Controlled d-scaling test at CONSTANT N/d ratio (samples-per-feature).

Sweeps (d, N) pairs keeping N/d = 40 fixed:
    (d=5,  N=200),  (d=10, N=400),  (d=20, N=800),
    (d=30, N=1200), (d=50, N=2000)

Theorem 3.2's "d-agnostic" claim is a CR-limit statement, i.e.\ it
holds when both estimators are in their variance regime.
Section~\ref{subsec:scaling-d} showed that at fixed N = 200 the
ratio collapses to ~1 by d = 20 because bias dominates. This test
scales N with d to hold the sample-per-feature ratio constant; if
the theorem's d-agnosticism is realized in practice, the observed
$\sqrt{\mathrm{PEHE}}$ ratio should stay near the same value across
every (d, N) pair.

Reuses rho_scaling_linear.make_linear_scm / load_ours / load_uwyk.
5-way SLURM array (one d per task) with resume-aware shards.
"""
from __future__ import annotations
import argparse, os, sys, types, traceback
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rho_scaling_linear import make_linear_scm, load_ours, load_uwyk

DEVICE = torch.device('cpu')
# (d, N) pairs at constant N/d = 40 (samples per feature)
PAIRS = [(5, 200), (10, 400), (20, 800), (30, 1200), (50, 2000)]


def _pehe(true_cate, pred_cate):
    t = np.asarray(true_cate).reshape(-1)
    p = np.asarray(pred_cate).reshape(-1)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def _plot_aggregate(args):
    import matplotlib.pyplot as plt
    all_arr = None
    for idx in range(len(PAIRS)):
        shard = f'{args.out}.pair{idx}.npz'
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

    ratio = arr['pehe_uwyk'] / np.maximum(arr['pehe_ours50'], 1e-9)

    print()
    print(f'{"d":>4} {"N":>6} {"n":>4} {"UWYK (mean±std)":>18} '
          f'{"Ours (mean±std)":>18} {"√PEHE ratio":>12} {"MSE ratio":>10} {"stability":>10}')
    print('-' * 96)
    for d, N in PAIRS:
        m = (arr['d'] == d) & (arr['N'] == N)
        if not m.any(): continue
        pu, po = arr['pehe_uwyk'][m], arr['pehe_ours50'][m]
        sqrt_pehe_ratio = pu.mean() / po.mean()
        mse_ratio = (pu.mean() ** 2) / (po.mean() ** 2)
        stability = po.std() / max(pu.std(), 1e-9)
        print(f'{d:>4} {N:>6} {int(m.sum()):>4} '
              f'{pu.mean():>8.3f} ± {pu.std():>4.2f}   '
              f'{po.mean():>8.3f} ± {po.std():>4.2f}   '
              f'{sqrt_pehe_ratio:>12.3f} {mse_ratio:>10.3f} {stability:>10.3f}')

    d_vals = np.array([d for d, _ in PAIRS])
    ratios = np.zeros(len(PAIRS))
    stabilities = np.zeros(len(PAIRS))
    for i, (d, N) in enumerate(PAIRS):
        m = (arr['d'] == d) & (arr['N'] == N)
        if m.any():
            ratios[i] = arr['pehe_uwyk'][m].mean() / arr['pehe_ours50'][m].mean()
            stabilities[i] = arr['pehe_ours50'][m].std() / max(arr['pehe_uwyk'][m].std(), 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    ax.axhline(np.sqrt(2), color='k', ls='--', lw=1.6, label=r'Theorem 3.2: $\sqrt{2}$')
    ax.axhline(1.0, color='r', ls=':', lw=1.0, alpha=0.6, label='no improvement')
    ax.plot(d_vals, ratios, 'o-', color='#0F8A3C', lw=2.2, markersize=10)
    for xi, yi in zip(d_vals, ratios):
        ax.annotate(f'{yi:.3f}', xy=(xi, yi), xytext=(0, 8),
                     textcoords='offset points', ha='center', fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel(r'Covariate dimension $d$   (with $N = 40 d$)', fontsize=11)
    ax.set_ylabel(r'$\sqrt{\mathrm{PEHE}}$ ratio', fontsize=11)
    ax.set_ylim(0.9, 1.55)
    ax.grid(alpha=0.3, which='both')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_title(r'$\sqrt{\mathrm{PEHE}}_{\mathrm{UWYK}} / \sqrt{\mathrm{PEHE}}_{\mathrm{Ours(fn{=}50)}}$', fontsize=10.5)

    ax = axes[1]
    ax.axhline(1/np.sqrt(2), color='k', ls='--', lw=1.6, label=r'Cor 3.3: $1/\sqrt{2}$')
    ax.axhline(1.0, color='r', ls=':', lw=1.0, alpha=0.6, label='equal spread')
    ax.plot(d_vals, stabilities, 's-', color='#B84A2A', lw=2.2, markersize=10)
    for xi, yi in zip(d_vals, stabilities):
        ax.annotate(f'{yi:.3f}', xy=(xi, yi), xytext=(0, 8),
                     textcoords='offset points', ha='center', fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel(r'Covariate dimension $d$   (with $N = 40 d$)', fontsize=11)
    ax.set_ylabel('stability ratio', fontsize=11)
    ax.set_ylim(0.5, 1.35)
    ax.grid(alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title(r'$\mathrm{Std}_{\mathrm{Ours}} / \mathrm{Std}_{\mathrm{UWYK}}$ across SCMs', fontsize=10.5)

    fig.suptitle(r'Controlled linear SCM d-sweep at constant $N/d = 40$ ($K=' + str(args.K) + '$/pair)', fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches='tight')
    plt.close(fig)
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
    ap.add_argument('--out',            default='d_scaling_linear_ratio.png')
    ap.add_argument('--pair-index',     type=int, default=-1)
    ap.add_argument('--plot',           action='store_true')
    args = ap.parse_args()

    if args.plot:
        _plot_aggregate(args); return

    sys.path.insert(0, os.path.join(args.repo, 'benchmarks'))
    from methods.ours   import ours_pipeline
    from methods.uwyk   import uwyk_no_ancestral_pipeline

    print('[load] Ours fn=50', flush=True); o50 = load_ours(args, args.checkpoint50)
    print('[load] UWYK',       flush=True); uwyk = load_uwyk(args.uwyk_src, args.uwyk_ckpt_dir)

    if args.pair_index >= 0:
        pairs = [PAIRS[args.pair_index]]
        shard_path = f'{args.out}.pair{args.pair_index}.npz'
    else:
        pairs = list(PAIRS)
        shard_path = args.out + '.npz'

    rows = []
    done_ks: dict[tuple, set] = {p: set() for p in pairs}
    if os.path.exists(shard_path):
        with np.load(shard_path, allow_pickle=True) as f:
            for i in range(int(len(f['seed']))):
                r = {k: (float(f[k][i]) if f[k].dtype.kind == 'f' else int(f[k][i]))
                     for k in f.files}
                rows.append(r)
                seed_i = int(f['seed'][i]); d_i = int(f['d'][i]); N_i = int(f['N'][i])
                done_ks.setdefault((d_i, N_i), set()).add(seed_i - d_i * 10_000)
        print(f'[resume] {len(rows)} rows from {shard_path}', flush=True)

    for (d, N) in pairs:
        for k in range(args.K):
            if k in done_ks.get((d, N), set()): continue
            seed = d * 10_000 + k
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
            print(f'[scm] d={d} N={N} k={k}  UWYK={pehe_uwyk:.3f}  Ours50={pehe_ours50:.3f}',
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
