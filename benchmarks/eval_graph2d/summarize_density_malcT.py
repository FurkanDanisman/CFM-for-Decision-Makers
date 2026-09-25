"""Validated MALC-T summary for the eleven rows in FOR_FURKAN.md.

python benchmarks/eval_graph2d/summarize_density_malcT.py --write

The arm columns are the unchanged native arm scores. T transforms only tau.
ATE columns require COMPUTE_ATE=1 results; incomplete columns remain unfilled.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from summarize_density_marginals import ROWS
from refresh_density_tables import cell, table

DATASETS = {'IHDP': 100, 'ACIC': 10}
CONFIG = {'malc_variant': 'T', 'malc_B': 1000, 'malc_K': 1,
          'malc_seed': 0, 'malc_n_tau': 12001, 'n_y0': 4096}
BEGIN = '<!-- MALC_T_SUMMARY_BEGIN -->'
END = '<!-- MALC_T_SUMMARY_END -->'


def validate(z, pred, ds, r, shard):
    if str(z['dataset']) != ds or int(z['realization']) != r:
        raise ValueError(f'{z.zip.filename}: wrong dataset/realization')
    for key, value in CONFIG.items():
        if str(z[key]) != str(value):
            raise ValueError(f'{z.zip.filename}: unexpected {key}={z[key]}')
    suffix = f'/results_density_tauC/{shard}/{ds}/predictions/{ds}_r{r:03d}.npz'
    if not str(z['source_dump']).endswith(suffix):
        raise ValueError(f'{z.zip.filename}: wrong source prediction dump')
    for key in ('y_scale', 'y_shift', 'sigma_scaled', 'true_cate',
                'tau_star_scaled', 'n_context', 'n_y0'):
        np.testing.assert_allclose(z[key], pred[key], rtol=1e-12, atol=1e-12,
                                   err_msg=f'{z.zip.filename}: {key}')
    if int(z['n_queries']) != len(pred['mu0_scaled']):
        raise ValueError(f'{z.zip.filename}: wrong query count')
    if 'dopfn_malc_bins' in z.files and int(z['dopfn_malc_bins']) != 1024:
        raise ValueError(f'{z.zip.filename}: unexpected Do-PFN rebinning')


def collect(root, ate_root):
    data, coverage = {}, []
    for ds, n in DATASETS.items():
        expected = {f'{ds}_r{r:03d}.npz' for r in range(n)}
        for shard in dict.fromkeys(row[2] for row in ROWS):
            actual = {p.name for p in (root / shard / ds).glob(f'{ds}_r*.npz')}
            if actual != expected:
                raise ValueError(f'{root / shard / ds}: missing {sorted(expected - actual)}, '
                                 f'extra {sorted(actual - expected)}')
            extra = {p.name for p in (ate_root / shard / ds).glob(f'{ds}_r*.npz')} - expected
            if extra:
                raise ValueError(f'{ate_root / shard / ds}: extra realizations {sorted(extra)}')
        for label, run, shard, method in ROWS:
            ate_files = {p.name for p in (ate_root / shard / ds).glob(f'{ds}_r*.npz')}
            use_refit = ate_files == expected
            active_root = ate_root if use_refit else root
            values = {key: [] for key in ('arms', 'nll', 'l2', 'mass', 'ate')}
            fallback = queries = 0
            for r in range(n):
                name = f'{ds}_r{r:03d}.npz'
                with np.load(active_root / shard / ds / name) as z, np.load(
                        Path('results_density_tauC', shard, ds, 'predictions', name)) as pred:
                    validate(z, pred, ds, r, shard)
                    for key in ('nll', 'l2', 'mass'):
                        v = float(z[f'{key}_{method}'])
                        if not np.isfinite(v):
                            raise ValueError(f'{z.zip.filename}: nonfinite {key}_{method}')
                        values[key].append(v)
                    q = int(z['n_queries'])
                    mask = z[f'malc_fallback_{method}']
                    if (mask.shape != (q,) or not set(np.unique(mask)) <= {0, 1} or
                            mask.sum() != int(z[f'n_fallback_{method}']) or
                            not np.array_equal(np.flatnonzero(mask), z[f'malc_fallback_query_{method}'])):
                        raise ValueError(f'{z.zip.filename}: invalid fallback counts for {method}')
                    fallback += int(mask.sum())
                    queries += q
                    with np.load(Path('results_density_marginals', run, ds, name)) as marg:
                        if (str(marg['dataset']) != ds or int(marg['realization']) != r or
                                int(marg['n_queries']) != q or
                                not np.isclose(marg['y_scale'], pred['y_scale'], rtol=1e-12, atol=0)):
                            raise ValueError(f'{marg.zip.filename}: mismatched marginal source')
                        arm_sum = float(marg[f'nll_y0_{method}'] + marg[f'nll_y1_{method}'])
                        if not np.isfinite(arm_sum):
                            raise ValueError(f'{marg.zip.filename}: nonfinite arm NLL')
                        values['arms'].append(arm_sum)
                    ate_path = ate_root / shard / ds / name
                    if ate_path.exists():
                        with np.load(ate_path) as ate:
                            validate(ate, pred, ds, r, shard)
                            if str(ate['ate_density_source']) != 'malcT_post_fallback':
                                raise ValueError(f'{ate_path}: wrong ATE density source')
                            v = float(ate[f'nll_bary_{method}'])
                            if not np.isfinite(v) or v != float(ate[f'nll_mix_{method}']):
                                raise ValueError(f'{ate_path}: invalid ATE NLL')
                            values['ate'].append(v)
            coverage.append(f'{label}: {ds} CATE {n}/{n}, ATE {len(values["ate"])}/{n}'
                            + (' (CATE + ATE from refit)' if use_refit else ' (original CATE)'))
            data[ds, label] = {**{k: np.asarray(v) for k, v in values.items()},
                               'fallback': fallback, 'queries': queries, 'refit': use_refit}
    return data, coverage


def render(data):
    heads = ['Method'] + [f'{ds}: {metric} ↓' for ds in DATASETS
                         for metric in ('f_Y0 + f_Y1 (native)', 'f_τ (MALC-T)', 'f_ATE (MALC-T)')]
    rows, diagnostics = [], []
    for label, *_ in ROWS:
        row, diag = [label], [label]
        for ds, n in DATASETS.items():
            v = data[ds, label]
            row.extend([cell(v['arms']), cell(v['nll']),
                        cell(v['ate']) if len(v['ate']) == n else '—'])
            diag.extend([cell(v['l2']), cell(v['mass']),
                         f'{100 * v["fallback"] / v["queries"]:.2f}%'])
        rows.append(row)
        diagnostics.append(diag)
    pending = any(len(data[ds, label]['ate']) != n
                  for ds, n in DATASETS.items() for label, *_ in ROWS)
    joint_acic_mass = data['ACIC', 'Do-PFN 2D']['mass'].mean()
    status = ('**ATE computation pending.** `—` means that a complete MALC-T ATE run is\n'
              'not available; partial realization sets are never averaged into this table.\n'
              if pending else 'All MALC-T ATE columns have complete 100/10 coverage.\n')
    return f'''{BEGIN}
# MALC-T Summary Table

Variant **T** smooths the interior effect distribution in τ space after the
joint projection or independent-arm convolution, restoring its original
interior mass and retaining the tail contribution. The original CATE results
come from `results_density_tauC_malcT`, with **B=1000, K=1, seed=0**, 12001 smoothing
nodes and `n_y0=4096`. The job logs record 1024 rebinning bins for Do-PFN.
Its 1D arms are rebinned before MALC-T; comparisons with the exact native
density therefore include this rebinning step as well as smoothing.
The older MALC tables below use variant **J**, which fits on the joint plane.

All eleven rows use **100 IHDP realizations (r000–r099)** and **10 ACIC
realizations (r000–r009)**, matched to the native table's prediction dumps.
Cells are mean ± SE across realizations, on the same scaled axis.

{table(heads, rows)}

The **native** arm columns repeat the original arm NLLs: variant T fits no new
arm or joint density, so it does not define new marginal scores. These are
the input-model marginals, not marginals reconstructed from the smoothed τ.
The arm sum is formed per realization before calculating its SE.

{status}
For any family/dataset with a complete ATE refit in
`results_density_tauC_malcT_ate`, **both CATE and ATE columns and diagnostics**
use that refit. Until it is complete, the CATE column uses the full original
run and ATE stays blank. This keeps the two tiers on the same fitted densities;
MALC optimization can vary across software/numerical environments even with
the same sampling seed. The refresh command prints which source each row uses.

MALC-T ATE requires the W2 barycenter of the **smoothed per-query densities**.
The original MALC-T files save CATE metrics and means, but not those density
grids, so this requires repeating the CPU-only MALC fits from the saved model
predictions. `ate_abs_err_*` in those files is a point error in raw units;
it cannot substitute for ATE-density NLL. Native ATE scores cannot fill these
columns either.

## MALC-T CATE diagnostics

{table(['Method', 'IHDP: L2 ↓', 'IHDP: mass', 'IHDP: fallback',
        'ACIC: L2 ↓', 'ACIC: mass', 'ACIC: fallback'], diagnostics)}

Fallback is the percentage of queries scored with the raw density: failed
fits and queries whose observed τ* falls outside the fitted interior support
are retained using the evaluator's raw fallback. This rule uses the observed
evaluation outcome; these are scores of that existing hybrid rule. The
denominators are 7500 IHDP and 4810 ACIC queries per method. Mass is the
integral over [-3, 3], without renormalization; the Do-PFN 2D ACIC mean is
{joint_acic_mass:.4f}, a grid-mass shortfall of about {100 * (1 - joint_acic_mass):.2f}%.

**Provenance.** Do-PFN: `dopfn_refresh`; UWYK No-Anc: `5312884`; UWYK Anc
(v3a): `5312882`; CausalPFN: `5571187`. The summarizer checks realization
coverage, settings, source paths, truth/scaling, finite scores and fallbacks.

To compute the missing ATE columns on the cluster, from the repository root:

```bash
mkdir -p logs_density_ate_malcT
REPO="$PWD" sbatch benchmarks/cluster/submit_density_ate_malcT.sbatch
```

This writes a separate `results_density_tauC_malcT_ate` tree and verifies the
recomputed CATE metrics and fallback masks against the existing MALC-T run,
recording differences in the job logs and `reference_cate_matches` field.
The new files contain both tiers from the same fits; a mismatch does not
cause old CATE scores to be paired with new ATE scores.
After the jobs finish and the results are available locally, refresh with:

```bash
python benchmarks/eval_graph2d/summarize_density_malcT.py --write
```
{END}'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('results_density_tauC_malcT'))
    ap.add_argument('--ate-root', type=Path, default=Path('results_density_tauC_malcT_ate'))
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()
    data, coverage = collect(args.root, args.ate_root)
    section = render(data)
    if args.write:
        path = Path('FOR_FURKAN.md')
        text = path.read_text()
        if BEGIN in text:
            start, end = text.index(BEGIN), text.index(END) + len(END)
            text = text[:start] + section + text[end:]
        else:
            marker = '# UWYK — IHDP, all runs\n'
            if marker not in text:
                raise ValueError('Cannot locate insertion point in FOR_FURKAN.md')
            text = text.replace(marker, section + '\n\n\n' + marker, 1)
        path.write_text(text)
        print('\n'.join(coverage))
        print('Updated MALC-T section in FOR_FURKAN.md.')
    else:
        print(section)


if __name__ == '__main__':
    main()
