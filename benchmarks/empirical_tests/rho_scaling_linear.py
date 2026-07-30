"""Controlled ρ-scaling test on a strictly linear SCM.

Maximally controlled setup:
    X ~ N(0, I_d)
    (η_0, η_1) ~ N(0, [[σ², ρσ²], [ρσ², σ²]])
    Y_do0(x, η) = β_0 · x + η_0
    Y_do1(x, η) = β_1 · x + η_1

The conditional correlation given X:
    Corr(Y_do0, Y_do1 | X = x) = Corr(η_0, η_1) = ρ  exactly.

This is the theorem's ρ — no nonlinear damping, no X-mediated confound.
The x-axis of the plot is the TARGET ρ (which equals theorem's ρ by
construction), not the marginal empirical ρ (which conflates noise
correlation with X-mediated correlation).

Runs only Ours(fn=50) vs UWYK-NoAncestral per user request.
6-way SLURM array (one target ρ per task) with resume-aware shards.
"""
from __future__ import annotations
import argparse, gc, importlib, os, sys, time, traceback, types
import numpy as np
import torch

DEVICE = torch.device('cpu')
RHO_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)


def make_linear_scm(seed, n_context, n_test, rho, x_dim=5, sigma_eps=1.0):
    """Strictly linear SCM with imposed noise correlation ρ."""
    rng = np.random.default_rng(seed)
    N = n_context + n_test
    X = rng.standard_normal((N, x_dim)).astype(np.float32)
    w_T  = rng.standard_normal(x_dim) / np.sqrt(x_dim)
    w_Y0 = rng.standard_normal(x_dim) / np.sqrt(x_dim)
    w_Y1 = rng.standard_normal(x_dim) / np.sqrt(x_dim)
    mu0 = X @ w_Y0
    mu1 = X @ w_Y1
    Sigma = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    L = np.linalg.cholesky(Sigma + 1e-9 * np.eye(2))
    z = rng.standard_normal((N, 2))
    eta = z @ L.T
    y0 = (mu0 + sigma_eps * eta[:, 0]).astype(np.float32)
    y1 = (mu1 + sigma_eps * eta[:, 1]).astype(np.float32)
    logits = X @ w_T
    logits = (logits - logits.mean()) / (logits.std() + 1e-9)
    p_T = 1.0 / (1.0 + np.exp(-logits))
    T = rng.binomial(1, p_T).astype(np.float32)
    Y_obs = np.where(T > 0.5, y1, y0)
    idx = rng.permutation(N)
    ctx = idx[:n_context]; tst = idx[n_context:]
    class _CD: pass
    cd = _CD()
    cd.X_train = torch.from_numpy(X[ctx])
    cd.t_train = torch.from_numpy(T[ctx])
    cd.y_train = torch.from_numpy(Y_obs[ctx])
    cd.X_test  = torch.from_numpy(X[tst])
    cd.true_cate = torch.from_numpy((mu1[tst] - mu0[tst]).astype(np.float32))
    # residuals = the exogenous noise the theorem's ρ refers to. Since we
    # KNOW μ_t(X)=β_t·X exactly, the true residual is exactly η_t. Its
    # empirical correlation matches target ρ up to O(1/√N) sampling noise.
    cd._eta0_test = (y0[tst] - mu0[tst]).astype(np.float32)
    cd._eta1_test = (y1[tst] - mu1[tst]).astype(np.float32)
    cd._y0_test   = y0[tst]; cd._y1_test = y1[tst]
    return cd


def _pehe(true_cate, pred_cate):
    t = np.asarray(true_cate).reshape(-1)
    p = np.asarray(pred_cate).reshape(-1)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def load_ours(args, checkpoint_path):
    sys.path.insert(0, args.repo); sys.path.insert(0, os.path.join(args.repo, 'MALC'))
    from models.InterventionalPFN import InterventionalPFN
    _orig = torch.load
    def _p_load(*a, **kw):
        kw.setdefault('weights_only', False); return _orig(*a, **kw)
    torch.load = _p_load
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    cfg = ckpt['config']; J = cfg['J']
    edges = ckpt['edges'].cpu().numpy()
    bin_width = float(edges[1] - edges[0])
    centers = 0.5 * (edges[:-1] + edges[1:])
    NF = cfg['num_features']
    model = InterventionalPFN(
        num_features=NF, d_model=cfg['d_model'], depth=cfg['depth'],
        heads_feat=cfg['heads'], heads_samp=cfg['heads'], dropout=0.0,
        output_dim=J*J + 9 + 4, hidden_mult=cfg['hidden_mult'],
        normalize_features=True, normalize_treatment=False,
        use_treatment_in_query=False, use_checkpoint=False,
    ).to(DEVICE).eval()
    model.load_state_dict(ckpt['model_state_dict'])
    ot_dir = os.path.join(args.repo, 'MALC', 'Optimal_Transport')
    if ot_dir not in sys.path: sys.path.insert(0, ot_dir)
    from ot_barycenter import wasserstein_barycenter_1d
    return model, edges, J, bin_width, centers, NF, wasserstein_barycenter_1d


def load_uwyk(uwyk_src, uwyk_ckpt_dir):
    _saved = {}
    for name in list(sys.modules):
        if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
            _saved[name] = sys.modules.pop(name)
    sys.path.insert(0, uwyk_src)
    mod = importlib.import_module('models.PreprocessingGraphConditionedPFN')
    sys.path.remove(uwyk_src)
    for name in list(sys.modules):
        if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
            del sys.modules[name]
    sys.modules.update(_saved)
    return mod.PreprocessingGraphConditionedPFN(
        config_path=os.path.join(uwyk_ckpt_dir, 'best_model_config.yaml'),
        checkpoint_path=os.path.join(uwyk_ckpt_dir, 'best_model.pt'),
        device='cpu', verbose=False,
    ).load()


def _plot_aggregate(args):
    import matplotlib.pyplot as plt
    all_arr = None
    for idx in range(len(RHO_GRID)):
        shard = f'{args.out}.rho{idx}.npz'
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

    rho_plot = np.linspace(0, 0.98, 200)
    theory = np.sqrt(2 / (1 - rho_plot))
    ratio = arr['pehe_uwyk'] / np.maximum(arr['pehe_ours50'], 1e-9)
    emp = arr['emp_rho_cond']  # measured on true residuals → ≈ target ρ

    # sanity — how close is measured ρ to what we set?
    print('\n[ρ check] target → measured conditional (mean ± std across SCMs):')
    for target in RHO_GRID:
        mask = np.isclose(arr['target_rho'], target)
        if mask.any():
            m = emp[mask].mean(); s = emp[mask].std()
            print(f'  target ρ={target:+.2f}  →  emp ρ_cond = {m:+.3f} ± {s:.3f}   '
                  f'(n={int(mask.sum())} SCMs)')

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(rho_plot, theory, 'k--', lw=1.5, label=r'Theory: $\sqrt{2/(1-\rho)}$')
    for target in RHO_GRID:
        mask = np.isclose(arr['target_rho'], target)
        if not mask.any(): continue
        r_med = float(np.median(emp[mask]))  # measured, not target
        m = float(np.median(ratio[mask])); s = float(ratio[mask].std())
        ax.errorbar(r_med, m, yerr=s, fmt='o', color='#0F8A3C',
                     markersize=9, capsize=4, zorder=4,
                     label='empirical median ± std' if target == RHO_GRID[0] else None)
        ax.scatter(emp[mask], ratio[mask], color='#2E4A6F',
                    alpha=0.4, s=32, zorder=3)
    ax.axhline(1.0, color='r', ls=':', lw=1, alpha=0.6)
    ax.set_xlabel(r'Empirical $\rho$ (conditional, on true residuals)')
    ax.set_ylabel(r'$\sqrt{\mathrm{PEHE}}_{\mathrm{UWYK\ NoAnc}} / \sqrt{\mathrm{PEHE}}_{\mathrm{Ours\ (fn=50)}}$')
    ax.set_title(f'ρ-scaling on strictly linear SCM (K={args.K}/ρ, N={args.N_context})')
    ax.set_ylim(0, max(6, theory.max() * 1.05))
    ax.grid(alpha=0.3)
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
    ap.add_argument('--x-dim',          type=int, default=5)
    ap.add_argument('--sigma-eps',      type=float, default=1.0)
    ap.add_argument('--out',            default='rho_scaling_linear.png')
    ap.add_argument('--rho-index',      type=int, default=-1)
    ap.add_argument('--plot',           action='store_true')
    args = ap.parse_args()

    if args.plot:
        _plot_aggregate(args); return

    sys.path.insert(0, os.path.join(args.repo, 'benchmarks'))
    from methods.ours   import ours_pipeline
    from methods.uwyk   import uwyk_no_ancestral_pipeline

    print('[load] Ours fn=50', flush=True); o50 = load_ours(args, args.checkpoint50)
    print('[load] UWYK',        flush=True); uwyk = load_uwyk(args.uwyk_src, args.uwyk_ckpt_dir)

    if args.rho_index >= 0:
        rho_targets = [RHO_GRID[args.rho_index]]
        shard_path = f'{args.out}.rho{args.rho_index}.npz'
    else:
        rho_targets = list(RHO_GRID)
        shard_path = args.out + '.npz'

    rows = []
    done_ks: dict[float, set] = {r: set() for r in rho_targets}
    if os.path.exists(shard_path):
        with np.load(shard_path, allow_pickle=True) as f:
            for i in range(int(len(f['seed']))):
                r = {k: (float(f[k][i]) if f[k].dtype.kind == 'f' else int(f[k][i]))
                     for k in f.files}
                rows.append(r)
                seed_i = int(f['seed'][i]); target_r = float(f['target_rho'][i])
                inferred_k = seed_i - int(target_r * 10_000)
                done_ks.setdefault(target_r, set()).add(inferred_k)
        print(f'[resume] {len(rows)} rows from {shard_path}', flush=True)

    for target_rho in rho_targets:
        for k in range(args.K):
            if k in done_ks.get(target_rho, set()): continue
            seed = int(target_rho * 10_000) + k
            cd = make_linear_scm(seed, args.N_context, args.N_test,
                                   rho=target_rho, x_dim=args.x_dim,
                                   sigma_eps=args.sigma_eps)
            # theorem's ρ = Corr(noise) — measured on true residuals η_t
            emp_rho_cond = float(np.corrcoef(cd._eta0_test, cd._eta1_test)[0, 1])
            emp_rho_marg = float(np.corrcoef(cd._y0_test,   cd._y1_test  )[0, 1])
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

            rows.append(dict(target_rho=target_rho,
                              emp_rho_cond=emp_rho_cond,
                              emp_rho_marg=emp_rho_marg,
                              seed=seed, pehe_uwyk=pehe_uwyk,
                              pehe_ours50=pehe_ours50))
            print(f'[scm] target_ρ={target_rho:.2f} k={k}  '
                  f'emp_ρ_cond={emp_rho_cond:+.3f}  '
                  f'(marg={emp_rho_marg:+.3f})  '
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
