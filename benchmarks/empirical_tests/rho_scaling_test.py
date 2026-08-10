"""Controlled empirical test of theory_joint_advantage.tex Result B.

Result B predicts $\\Var(\\hat\\tau_{marg}) / \\Var(\\hat\\tau_{joint}) = 2/(1-\\rho)$
in the symmetric case. In terms of √PEHE (an estimate of the standard
deviation of $\\hat\\tau$ around $\\tau$), the ratio should be

    √PEHE_UWYK / √PEHE_Ours  ≈  √(2 / (1 - ρ))

for well-specified marginal-only and joint models trained on the same
data. This script sets up a **controlled test** where we sample K SCMs
whose true DGP-ρ we set ourselves, evaluate both R-PFN (fn=50 and fn=10
checkpoints) and UWYK / Do-PFN on each, and plot √PEHE ratio vs true ρ.

The SCM class is a bivariate location-scale family:

    Y_do0 = μ_0(X) + σ_ε · η_0
    Y_do1 = μ_1(X) + σ_ε · η_1
    (η_0, η_1) ~ N(0, [[1, ρ], [ρ, 1]])

with the mean surfaces μ_t drawn from a smooth polynomial prior — the
same functional form as CausalPFN's PolynomialDataset but with the
noise-coupling parameter ρ ∈ {0, 0.2, 0.4, 0.6, 0.8, 0.95} imposed
externally. For each ρ we sample K SCMs (default K=20 per ρ), evaluate
the four methods, and record √PEHE.

Runs on a single machine (no killarney needed) — the SCMs are small.

Usage
-----
    python rho_scaling_test.py \\
        --repo         $PWD \\
        --checkpoint50 $PWD/checkpoints/step_50000_final.pt \\
        --checkpoint10 $PWD/checkpoints_dopfn/step_50000_final.pt \\
        --uwyk-src     $PWD/../external/uwyk/src \\
        --uwyk-ckpt-dir $PWD/../external/uwyk/experiments/checkpoints/full_conditioned_model/final_earlytest_full_conditioning_16773252.0 \\
        --dopfn        $PWD/../external/dopfn \\
        --K            20 \\
        --N-context    200 \\
        --N-test       50 \\
        --out          rho_scaling_test.png
"""
from __future__ import annotations
import argparse, gc, importlib, os, sys, time, traceback, types
import numpy as np
import torch

DEVICE = torch.device('cpu')
RHO_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)


def make_polynomial_scm(seed: int, n_context: int, n_test: int,
                          rho: float, x_dim: int = 5, degree: int = 3,
                          sigma_eps: float = 1.0):
    """One SCM with imposed noise correlation ρ. Returns a CATE_Dataset
    view compatible with our pipelines, plus the true (Y_do0, Y_do1) for
    every test unit (needed for CATE ground truth)."""
    rng = np.random.default_rng(seed)
    N = n_context + n_test
    X = rng.standard_normal((N, x_dim)).astype(np.float32)
    # polynomial features up to `degree`
    feats = np.concatenate([X ** k for k in range(1, degree + 1)], axis=1)
    F = feats.shape[1]
    w_T  = rng.standard_normal(F) / np.sqrt(F)
    w_Y0 = rng.standard_normal(F) / np.sqrt(F)
    w_Y1 = rng.standard_normal(F) / np.sqrt(F)
    # true response surfaces
    mu0 = feats @ w_Y0
    mu1 = feats @ w_Y1
    # correlated bivariate noise per unit
    Sigma = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(2))
    z = rng.standard_normal((N, 2))
    eta = z @ L.T
    y0 = (mu0 + sigma_eps * eta[:, 0]).astype(np.float32)
    y1 = (mu1 + sigma_eps * eta[:, 1]).astype(np.float32)
    # treatment via sigmoid
    logits = (feats @ w_T)
    logits = (logits - logits.mean()) / (logits.std() + 1e-9)
    p_T = 1.0 / (1.0 + np.exp(-logits))
    T = rng.binomial(1, p_T).astype(np.float32)
    Y_obs = np.where(T > 0.5, y1, y0)
    # split into context + test
    idx = rng.permutation(N)
    ctx = idx[:n_context]; tst = idx[n_context:]
    class _CD: pass
    cd = _CD()
    cd.X_train = torch.from_numpy(X[ctx])
    cd.t_train = torch.from_numpy(T[ctx])
    cd.y_train = torch.from_numpy(Y_obs[ctx])
    cd.X_test  = torch.from_numpy(X[tst])
    cd.true_cate = torch.from_numpy((mu1[tst] - mu0[tst]).astype(np.float32))
    # for oracle ρ validation
    cd._y0_test = y0[tst]
    cd._y1_test = y1[tst]
    return cd


def _pehe(true_cate, pred_cate):
    t = np.asarray(true_cate).reshape(-1)
    p = np.asarray(pred_cate).reshape(-1)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def load_ours(args, checkpoint_path: str):
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


def load_uwyk(uwyk_src: str, uwyk_ckpt_dir: str, dopfn_path: str = ''):
    _saved = {}
    for name in list(sys.modules):
        if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
            _saved[name] = sys.modules.pop(name)
    _removed = False
    if dopfn_path and dopfn_path in sys.path:
        sys.path.remove(dopfn_path); _removed = True
    sys.path.insert(0, uwyk_src)
    mod = importlib.import_module('models.PreprocessingGraphConditionedPFN')
    sys.path.remove(uwyk_src)
    if _removed: sys.path.insert(0, dopfn_path)
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
    """Load all 6 per-ρ shards, concatenate, and produce the final plot."""
    import matplotlib.pyplot as plt
    all_arr = None
    for idx in range(len(RHO_GRID)):
        shard = f'{args.out}.rho{idx}.npz'
        if not os.path.exists(shard):
            print(f'[warn] missing shard {shard} — skipping'); continue
        with np.load(shard, allow_pickle=True) as f:
            d = {k: f[k] for k in f.files}
        if all_arr is None: all_arr = {k: [] for k in d}
        for k, v in d.items(): all_arr[k].append(v)
    if all_arr is None or not all_arr['seed']:
        raise SystemExit('[error] no shards found; run per-ρ jobs first')
    arr = {k: np.concatenate(v) for k, v in all_arr.items()}

    # Save the aggregated npz for downstream use.
    agg_path = args.out + '.npz'
    np.savez(agg_path, **arr)
    print(f'[save] {agg_path}  ({len(arr["seed"])} rows total)')

    # Two panels: √PEHE ratio (UWYK/Ours50) and (DoPFN/Ours10) vs ρ.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    rho_plot = np.linspace(0, 0.98, 200)
    theory = np.sqrt(2 / (1 - rho_plot))
    for ax, num_key, den_key, title in [
        (axes[0], 'pehe_uwyk',  'pehe_ours50',
         '√PEHE ratio (UWYK Ancestral / Ours fn=50) vs true ρ'),
        (axes[1], 'pehe_dopfn', 'pehe_ours10',
         '√PEHE ratio (Do-PFN / Ours fn=10) vs true ρ'),
    ]:
        ax.plot(rho_plot, theory, 'k--', lw=1.5,
                 label=r'Theory: $\sqrt{2/(1-\rho)}$')
        ratios = arr[num_key] / np.maximum(arr[den_key], 1e-9)
        for rho in RHO_GRID:
            mask = np.isclose(arr['rho'], rho)
            if not mask.any(): continue
            ax.scatter(arr['rho'][mask], ratios[mask], color='#2E4A6F',
                        alpha=0.35, s=32, zorder=3)
            m = float(ratios[mask].mean()); s = float(ratios[mask].std())
            ax.errorbar(rho, m, yerr=s, fmt='o', color='#0F8A3C',
                          markersize=8, capsize=4, zorder=4)
        ax.axhline(1.0, color='r', ls=':', lw=1, alpha=0.6)
        ax.set_xlabel('true DGP ρ')
        ax.set_ylabel('√PEHE ratio (marg / joint)')
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(6, theory.max() * 1.05))
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
    fig.suptitle('Test 1 (controlled) — √PEHE ratio vs true ρ',
                  fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'[save] {args.out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo',            required=True)
    ap.add_argument('--checkpoint50',    required=True)
    ap.add_argument('--checkpoint10',    required=True)
    ap.add_argument('--uwyk-src',        required=True)
    ap.add_argument('--uwyk-ckpt-dir',   required=True)
    ap.add_argument('--dopfn',           required=True)
    ap.add_argument('--K',               type=int, default=20,
                    help='SCMs per ρ value')
    ap.add_argument('--N-context',       type=int, default=200)
    ap.add_argument('--N-test',          type=int, default=50)
    ap.add_argument('--out',             default='rho_scaling_test.png')
    ap.add_argument('--rho-index',       type=int, default=-1,
                    help='If >=0, process only RHO_GRID[rho_index] and save '
                         'to <out>.rho<idx>.npz (used by the sbatch array).')
    ap.add_argument('--plot',            action='store_true',
                    help='Skip the SCM loop; just aggregate all per-rho .npz '
                         'shards from <out>.rho<idx>.npz for idx in 0..5 and '
                         'produce the final plot.')
    args = ap.parse_args()

    # Plot-only branch (used after the array finishes).
    if args.plot:
        _plot_aggregate(args); return

    sys.path.insert(0, os.path.join(args.repo, 'benchmarks'))
    from methods.ours   import ours_pipeline
    from methods.uwyk   import uwyk_ancestral_pipeline, uwyk_no_ancestral_pipeline
    from methods.dopfn  import dopfn_pipeline

    # Load models once
    print('[load] Ours fn=50', flush=True)
    o50 = load_ours(args, args.checkpoint50)
    print('[load] Ours fn=10', flush=True)
    o10 = load_ours(args, args.checkpoint10)
    print('[load] UWYK', flush=True)
    uwyk = load_uwyk(args.uwyk_src, args.uwyk_ckpt_dir, args.dopfn)
    print('[load] Do-PFN', flush=True)
    sys.path.insert(0, args.dopfn)
    _cwd_prev = os.getcwd(); os.chdir(args.dopfn)
    from scripts.transformer_prediction_interface.base import DoPFNRegressor
    os.chdir(_cwd_prev)

    if args.rho_index >= 0:
        rho_subset = [RHO_GRID[args.rho_index]]
        shard_path = f'{args.out}.rho{args.rho_index}.npz'
    else:
        rho_subset = list(RHO_GRID)
        shard_path = args.out + '.npz'

    # ── Resume from an existing shard if present ───────────────────────
    rows = []
    resumed_ks_by_rho: dict[float, set] = {rho: set() for rho in rho_subset}
    if os.path.exists(shard_path):
        with np.load(shard_path, allow_pickle=True) as f:
            n = int(len(f['seed']))
            for i in range(n):
                r = dict(rho=float(f['rho'][i]), rho_hat=float(f['rho_hat'][i]),
                         seed=int(f['seed'][i]),
                         pehe_uwyk=float(f['pehe_uwyk'][i]),
                         pehe_dopfn=float(f['pehe_dopfn'][i]),
                         pehe_ours50=float(f['pehe_ours50'][i]),
                         pehe_ours10=float(f['pehe_ours10'][i]))
                rows.append(r)
                # infer k from seed convention (seed = int(rho*10000) + k)
                inferred_k = r['seed'] - int(r['rho'] * 10_000)
                resumed_ks_by_rho.setdefault(r['rho'], set()).add(inferred_k)
        print(f'[resume] loaded {n} existing rows from {shard_path}', flush=True)

    for rho in rho_subset:
        for k in range(args.K):
            if k in resumed_ks_by_rho.get(rho, set()):
                continue  # already in shard — skip
            seed = int(rho * 10_000) + k
            cd = make_polynomial_scm(seed, args.N_context, args.N_test, rho)
            true_cate = cd.true_cate.numpy()
            # oracle ρ from actual paired samples (should match `rho`)
            rho_hat = float(np.corrcoef(cd._y0_test, cd._y1_test)[0, 1])

            # UWYK-Ancestral
            os.chdir(args.dopfn)  # no-op if already there
            # Fair marginal-only comparison — no DAG at inference.
            uwyk_pred = uwyk_no_ancestral_pipeline(uwyk, cd)
            pehe_uwyk = _pehe(true_cate, uwyk_pred)

            # Do-PFN
            dopfn_pred = dopfn_pipeline(cd, DoPFNRegressor)
            pehe_dopfn = _pehe(true_cate, dopfn_pred)
            os.chdir(_cwd_prev)

            # Ours fn=50
            m50, edges50, J50, bw50, ctr50, NF50, wb50 = o50
            ours50 = ours_pipeline(cd, m50, edges50, J50, bw50, NF50, ctr50,
                                     types.SimpleNamespace(repo=args.repo,
                                                            malc_B=30, malc_max_K=3,
                                                            n_eval=200, workers=8),
                                     wb50)
            pehe_ours50 = _pehe(true_cate, ours50['ours_mean'])

            # Ours fn=10
            m10, edges10, J10, bw10, ctr10, NF10, wb10 = o10
            ours10 = ours_pipeline(cd, m10, edges10, J10, bw10, NF10, ctr10,
                                     types.SimpleNamespace(repo=args.repo,
                                                            malc_B=30, malc_max_K=3,
                                                            n_eval=200, workers=8),
                                     wb10)
            pehe_ours10 = _pehe(true_cate, ours10['ours_mean'])

            rows.append(dict(rho=rho, rho_hat=rho_hat, seed=seed,
                              pehe_uwyk=pehe_uwyk, pehe_dopfn=pehe_dopfn,
                              pehe_ours50=pehe_ours50, pehe_ours10=pehe_ours10))
            print(f'[scm] ρ={rho:.2f} k={k} ρ̂={rho_hat:+.3f}  '
                  f'UWYK={pehe_uwyk:.3f} DoPFN={pehe_dopfn:.3f} '
                  f'Ours50={pehe_ours50:.3f} Ours10={pehe_ours10:.3f}',
                  flush=True)

    # ── Save shard (per-ρ) or aggregate (all-ρ) ─────────────────────────
    if not rows:
        print(f'[skip] no new rows to add; shard {shard_path} unchanged')
        return
    keys = list(rows[0].keys())
    arr = {k: np.array([r[k] for r in rows]) for k in keys}
    np.savez(shard_path, **arr)
    print(f'[save] {shard_path}  ({len(rows)} rows total)')

    # If we're in per-rho mode, stop here. The --plot aggregator picks up
    # all shards later.
    if args.rho_index >= 0:
        return

    import matplotlib.pyplot as plt
    raw = shard_path

    # Two panels: √PEHE ratio (UWYK/Ours50) and (DoPFN/Ours10) vs ρ.
    # Theory: ratio ≈ √(2/(1-ρ)) for well-specified estimators.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    rho_plot = np.linspace(0, 0.98, 200)
    theory = np.sqrt(2 / (1 - rho_plot))

    for ax, num_key, den_key, title in [
        (axes[0], 'pehe_uwyk',  'pehe_ours50',
         '√PEHE ratio (UWYK Ancestral / Ours fn=50) vs true ρ'),
        (axes[1], 'pehe_dopfn', 'pehe_ours10',
         '√PEHE ratio (Do-PFN / Ours fn=10) vs true ρ'),
    ]:
        ax.plot(rho_plot, theory, 'k--', lw=1.5,
                 label=r'Theory: $\sqrt{2/(1-\rho)}$')
        ratios = arr[num_key] / np.maximum(arr[den_key], 1e-9)
        for rho in RHO_GRID:
            mask = np.isclose(arr['rho'], rho)
            ax.scatter(arr['rho'][mask], ratios[mask], color='#2E4A6F',
                        alpha=0.35, s=32, zorder=3)
            m = float(ratios[mask].mean())
            s = float(ratios[mask].std())
            ax.errorbar(rho, m, yerr=s, fmt='o', color='#0F8A3C',
                          markersize=8, capsize=4, zorder=4)
        ax.axhline(1.0, color='r', ls=':', lw=1, alpha=0.6)
        ax.set_xlabel('true DGP ρ')
        ax.set_ylabel('√PEHE ratio (marg / joint)')
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(6, theory.max() * 1.05))
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)

    fig.suptitle('Controlled ρ-scaling test — theory prediction: '
                  r'$\sqrt{PEHE}_{marg}/\sqrt{PEHE}_{joint} \approx \sqrt{2/(1-\rho)}$',
                  fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'[save] {args.out}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
