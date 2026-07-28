"""Table-3 empirical √PEHE ratios vs theoretical √(2/(1-ρ)) prediction.

For each Table-3 dataset compute
    Δ_marg  = √PEHE_{UWYK-Anc} / √PEHE_{Ours (fn=50)}
    Δ_dopfn = √PEHE_{Do-PFN}   / √PEHE_{Ours (fn=10)}
and overlay against the theoretical prediction √(2/(1-ρ)). Since the
Table-3 simulators (Hill/RealCause) draw per-arm noise independently,
DGP-ρ ≈ 0 on all five datasets → theory predicts √2 ≈ 1.414.

Reports both the ρ=0 case AND the possibility that "effective ρ" is
non-zero if the models happen to induce some coupling. But the honest
prediction is √2 at these datasets.

Reads Table-3 npzs from --results (fn=50 corpus) and --extra (fn=10
corpus). Produces a two-panel figure + a text table.
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import matplotlib.pyplot as plt

DATASETS = ['IHDP', 'ACIC', 'CPS', 'PSID', 'PSIDbal']


def _pehe_per_dataset(results_dir: str, key: str) -> dict[str, tuple[float, float, int]]:
    per_ds = {}
    for ds in DATASETS:
        files = sorted(glob.glob(os.path.join(results_dir, f'{ds}_r*.npz')))
        vals = []
        for fn in files:
            f = np.load(fn, allow_pickle=True)
            k = f'pehe_{key}'
            if k in f.files:
                v = float(f[k])
                if np.isfinite(v): vals.append(v)
        if vals:
            per_ds[ds] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
    return per_ds


def _ratio_with_std(num_ds, den_ds):
    """Return (ratio, ratio_std, n) per dataset. Ratio std via delta method
    on the log-space std, propagating both sources of noise."""
    out = {}
    for ds in DATASETS:
        if ds not in num_ds or ds not in den_ds: continue
        m1, s1, n1 = num_ds[ds]; m2, s2, n2 = den_ds[ds]
        if m2 <= 0 or m1 <= 0: continue
        r = m1 / m2
        # log-space CLT approximation
        var_log = (s1 / m1) ** 2 / max(n1, 1) + (s2 / m2) ** 2 / max(n2, 1)
        r_std = r * np.sqrt(var_log)
        out[ds] = (float(r), float(r_std), int(min(n1, n2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, help='fn=50 corpus dir')
    ap.add_argument('--extra',   required=True, help='fn=10 corpus dir')
    ap.add_argument('--variant', default='ours_mean',
                    help='Ours variant to use for the ratio '
                         '(ours_mean, ours_malc_mode_msk, …)')
    ap.add_argument('--out',     default='table3_vs_theory.png')
    args = ap.parse_args()

    pehe_ours    = _pehe_per_dataset(args.results, args.variant)
    pehe_uwyk    = _pehe_per_dataset(args.results, 'uwyk_noanc')
    pehe_dopfn   = _pehe_per_dataset(args.results, 'dopfn')
    pehe_ours10  = _pehe_per_dataset(args.extra,   args.variant)

    ratios_marg  = _ratio_with_std(pehe_uwyk,  pehe_ours)
    ratios_dopfn = _ratio_with_std(pehe_dopfn, pehe_ours10)

    # ── Table dump ─────────────────────────────────────────────────────
    lines = []
    lines.append(f"{'dataset':<10}  {'UWYK/Ours(fn50)':>18}  {'Do-PFN/Ours(fn10)':>20}  {'theory (ρ=0)':>14}")
    lines.append('─' * 68)
    theory = float(np.sqrt(2.0))
    for ds in DATASETS:
        m1 = ratios_marg.get(ds, (np.nan, np.nan, 0))
        m2 = ratios_dopfn.get(ds, (np.nan, np.nan, 0))
        lines.append(f"{ds:<10}  {m1[0]:>10.3f} ± {m1[1]:.3f}  "
                      f"{m2[0]:>10.3f} ± {m2[1]:.3f}  {theory:>14.3f}")
    print('\n'.join(lines))
    with open(args.out + '.txt', 'w') as fp:
        fp.write('\n'.join(lines))
    print(f'[save] {args.out}.txt')

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, ratios, num_label, title in [
        (axes[0], ratios_marg,  'UWYK No-Ancestral', 'Δ_marg (marginal-only regime)'),
        (axes[1], ratios_dopfn, 'Do-PFN',         'Δ_dopfn (Do-PFN baseline)'),
    ]:
        xs = np.arange(len(DATASETS))
        rs = np.array([ratios.get(ds, (np.nan, np.nan, 0))[0] for ds in DATASETS])
        ss = np.array([ratios.get(ds, (np.nan, np.nan, 0))[1] for ds in DATASETS])
        ax.axhline(1.0, color='r', ls=':', lw=1, alpha=0.6,
                    label='ratio = 1 (no improvement)')
        ax.axhline(theory, color='k', ls='--', lw=1.5,
                    label=f'theory √(2/(1-ρ)) at ρ=0 = {theory:.3f}')
        ax.errorbar(xs, rs, yerr=ss, fmt='o', color='#2E4A6F',
                     markersize=8, capsize=4)
        ax.set_xticks(xs); ax.set_xticklabels(DATASETS)
        ax.set_ylabel(r'$\sqrt{\text{PEHE}}$ ratio (marg / joint)')
        ax.set_title(title, fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.5, max(1.6, theory * 1.1))
    fig.suptitle(f'Table-3 empirical ratios vs theoretical √2 (Ours variant: {args.variant})',
                  fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'[save] {args.out}')


if __name__ == '__main__':
    main()
