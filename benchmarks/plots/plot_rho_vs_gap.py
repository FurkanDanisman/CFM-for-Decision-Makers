"""Empirical test of theory_joint_advantage.tex Results A-C.

Prediction from Propositions on scaling in ρ and unit-effect variance:
across Table-3 datasets, R-PFN's √PEHE improvement over UWYK-Ancestral
(and the fn=10 variant's improvement over Do-PFN) should be

  1. Positively correlated with the mean coupling ρ = Corr(Y_0, Y_1 | X)
     that R-PFN's joint output implies.
  2. Negatively correlated with the mean unit-level effect variance
     Var(Y_1 - Y_0 | X) that R-PFN's joint implies.

We compute those two quantities directly from Ours' 2D BarDistribution
output (p_mat) evaluated on each test query, average over queries per
dataset, then plot the two per-dataset scatters:

    Δ_marg vs ρ̄        (should be up-and-to-the-right)
    Δ_marg vs V̄        (should be down-and-to-the-right)

Both are read from the Table-3 npz files that live in results/ (fn=50
corpus) and results_dopfn/ (fn=10 corpus). This script runs on the
aggregate npz *if* the joint p_mat is cached there; otherwise it walks
back to the raw context files to recompute.

Inputs
------
    --results       ./results               # fn=50 corpus (must exist)
    --extra         ./results_dopfn         # fn=10 corpus (optional)
    --out           table3_rho_vs_gap.png

Outputs
-------
    <out>              — 2-panel scatter with Pearson r and 95% CI
    <out>.metadata.txt — per-dataset (ρ̄, V̄, Δ_marg, Δ_dopfn) table

Sign convention: Δ_marg > 0 means Ours beats UWYK-Ancestral on √PEHE.
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
import matplotlib.pyplot as plt

DATASETS = ['IHDP', 'ACIC', 'CPS', 'PSID', 'PSIDbal']


def _to_np(x):
    return np.asarray(x, dtype=np.float64)


def _rho_and_v_from_pmat(p_mat: np.ndarray, edges: np.ndarray) -> tuple[float, float]:
    """Discrete Pearson ρ and unit-effect variance from a normalised p_mat.

    p_mat is (J, J) with p_mat[j0, j1] the joint mass at bin (j0, j1),
    where axis 0 is Y_do0 and axis 1 is Y_do1. `edges` is the (J+1,)
    bin-edge vector on which the density was defined.
    """
    p = np.asarray(p_mat, dtype=np.float64)
    p /= max(p.sum(), 1e-12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    m0, m1 = centers[:, None], centers[None, :]
    ey0 = float((p * m0).sum())
    ey1 = float((p * m1).sum())
    v0 = float((p * (m0 - ey0) ** 2).sum())
    v1 = float((p * (m1 - ey1) ** 2).sum())
    cov = float((p * (m0 - ey0) * (m1 - ey1)).sum())
    rho = cov / (np.sqrt(v0 * v1) + 1e-12)
    var_tau = v0 + v1 - 2 * cov
    return rho, var_tau


def _extract_rho_v_per_dataset(results_dir: str) -> dict[str, tuple[float, float, int]]:
    """Walk the results directory and average ρ, V per dataset.

    Expects each npz to carry `rho_ours` and `var_tau_ours` — the two
    lightweight summary arrays populated by `backfill_rho_v.py`. Both
    are `(n_test,)` per realisation, so averaging across ALL
    realisations × ALL test queries per dataset gives the population
    mean ρ̄, V̄ needed for the theory-vs-empirics scatter.
    """
    per_ds = {}
    for ds in DATASETS:
        files = sorted(glob.glob(os.path.join(results_dir, f'{ds}_r*.npz')))
        rhos, vs = [], []
        n_realisations = 0
        for fn in files:
            f = np.load(fn, allow_pickle=True)
            if 'rho_ours' not in f.files or 'var_tau_ours' not in f.files:
                continue
            rhos.append(_to_np(f['rho_ours']))
            vs.append(_to_np(f['var_tau_ours']))
            n_realisations += 1
        if not rhos:
            continue
        rho_all = np.concatenate(rhos); v_all = np.concatenate(vs)
        rho_all = rho_all[np.isfinite(rho_all)]
        v_all   = v_all  [np.isfinite(v_all)]
        per_ds[ds] = (float(rho_all.mean()), float(v_all.mean()),
                       int(n_realisations))
    return per_ds


def _pehe_from_dir(results_dir: str, key: str) -> dict[str, float]:
    per_ds = {}
    for ds in DATASETS:
        files = sorted(glob.glob(os.path.join(results_dir, f'{ds}_r*.npz')))
        vals = []
        for fn in files:
            f = np.load(fn, allow_pickle=True)
            k = f'pehe_{key}'
            if k in f.files:
                vals.append(float(f[k]))
        if vals:
            per_ds[ds] = float(np.mean(vals))
    return per_ds


def _pearson_r_ci(x, y, alpha=0.05):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 3:
        return float('nan'), (float('nan'), float('nan'))
    r = float(np.corrcoef(x, y)[0, 1])
    z = 0.5 * np.log((1 + r) / (1 - r) + 1e-12)
    se = 1.0 / np.sqrt(n - 3)
    z_alpha = 1.96
    lo = np.tanh(z - z_alpha * se)
    hi = np.tanh(z + z_alpha * se)
    return r, (float(lo), float(hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True,
                    help='fn=50 corpus dir (Table-3 result npzs). '
                         'p_mats field must exist for the ρ/V computation.')
    ap.add_argument('--extra',   default=None,
                    help='fn=10 corpus dir (optional). If given, also '
                         'plots Δ_dopfn = pehe_dopfn - pehe_ours_mean.')
    ap.add_argument('--out',     default='table3_rho_vs_gap.png')
    args = ap.parse_args()

    print(f'[read] ρ, V per dataset from {args.results}...', flush=True)
    rho_v = _extract_rho_v_per_dataset(args.results)
    if not rho_v:
        print('[error] no npz in --results had both `edges` and `p_mats`. '
              'Rerun the Table-3 pipeline with the joint cache option to '
              'populate them, or point --results at a corpus that has them.',
              file=sys.stderr)
        sys.exit(1)

    print(f'[read] √PEHE per method per dataset...', flush=True)
    pehe_ours    = _pehe_from_dir(args.results, 'ours_mean')
    pehe_uwyk    = _pehe_from_dir(args.results, 'uwyk_ancestral')
    pehe_uwyk_na = _pehe_from_dir(args.results, 'uwyk_noanc')
    pehe_dopfn   = _pehe_from_dir(args.results, 'dopfn')

    pehe_ours_fn10 = None
    if args.extra and os.path.isdir(args.extra):
        pehe_ours_fn10 = _pehe_from_dir(args.extra, 'ours_mean')

    # ── Assemble per-dataset scatter data ────────────────────────────────
    rows = []
    for ds in DATASETS:
        if ds not in rho_v:            continue
        if ds not in pehe_ours:         continue
        rho, v, n = rho_v[ds]
        pehe_o    = pehe_ours[ds]
        pehe_u    = pehe_uwyk.get(ds, np.nan)
        pehe_d    = pehe_dopfn.get(ds, np.nan)
        pehe_o10  = (pehe_ours_fn10 or {}).get(ds, np.nan)
        rows.append(dict(ds=ds, rho=rho, v=v, n=n,
                          delta_marg_fn50 = pehe_u - pehe_o,
                          delta_dopfn_fn10 = (pehe_d - pehe_o10)
                                              if np.isfinite(pehe_o10) else np.nan))
    if not rows:
        print('[error] no datasets had both ρ and Δ_marg data.',
              file=sys.stderr)
        sys.exit(1)

    # ── Plot: 2 panels ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    def _panel(ax, xkey, xlabel, ykey, ylabel, title):
        x = np.array([r[xkey] for r in rows], dtype=np.float64)
        y = np.array([r[ykey] for r in rows], dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        ds  = [r['ds'] for r in rows if np.isfinite(r[xkey]) and np.isfinite(r[ykey])]
        if len(x) == 0:
            ax.set_visible(False); return
        ax.scatter(x, y, s=64, color='#2E4A6F', zorder=3)
        for xi, yi, di in zip(x, y, ds):
            ax.annotate(di, (xi, yi), textcoords='offset points',
                         xytext=(6, 6), fontsize=9)
        ax.axhline(0, color='k', lw=0.6, alpha=0.35)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        r, (lo, hi) = _pearson_r_ci(x, y)
        ax.text(0.02, 0.97,
                f"Pearson r = {r:+.2f}\n95% CI [{lo:+.2f}, {hi:+.2f}]",
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        ax.grid(alpha=0.3)

    _panel(axes[0], 'rho',
            r'$\bar\rho$   (mean coupling correlation, Ours joint)',
            'delta_marg_fn50',
            r'$\Delta_{\mathrm{marg}} = \sqrt{\mathrm{PEHE}}_{\mathrm{UWYK\,Anc}} - \sqrt{\mathrm{PEHE}}_{\mathrm{fn}=50}$',
            'Prediction: monotone increasing in $\\bar\\rho$')

    _panel(axes[1], 'v',
            r'$\bar{V} = \mathrm{Var}(Y_1 - Y_0 \mid X)$   (Ours joint)',
            'delta_marg_fn50',
            r'$\Delta_{\mathrm{marg}}$   ↑ = Ours better',
            'Prediction: monotone decreasing in $\\bar V$')

    fig.suptitle(
        'Empirical test of Results A–C: coupling ρ and unit-effect variance V '
        'vs. R-PFN(fn=50) improvement over UWYK-Ancestral on √PEHE.',
        fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'[save] {args.out}')

    # ── Also dump per-dataset numbers ────────────────────────────────────
    meta_path = args.out + '.metadata.txt'
    with open(meta_path, 'w') as fp:
        fp.write(f"{'dataset':<10} {'ρ̄':>8} {'V̄':>10} {'n_queries':>10}  "
                 f"{'Δ_marg(fn=50)':>16}  {'Δ_dopfn(fn=10)':>16}\n")
        for r in rows:
            fp.write(f"{r['ds']:<10} {r['rho']:>+8.3f} {r['v']:>10.3g} "
                      f"{r['n']:>10d}  {r['delta_marg_fn50']:>+16.3f}  "
                      f"{r['delta_dopfn_fn10']:>+16.3f}\n")
    print(f'[save] {meta_path}')


if __name__ == '__main__':
    main()
