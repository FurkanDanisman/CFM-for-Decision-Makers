"""Four-row-per-dataset summary for Ours at MALC_B=500.

For each dataset prints 4 rows corresponding to different CATE point-estimate
recipes. Each row reports sqrt(PEHE) and eps_ATE.

    Raw-mean            : per-query E[Y_1] - E[Y_0] from raw p_mat marginals.
    MALC-CATE-mean      : per-query E[τ] from ∫ τ · p_MALC(τ) dτ
                          (MALC-smoothed joint density, diagonal-integrated).
    EM-mean-K1          : per-query τ from MALC forced to K=1, using
                          fit.fits[0].mu_hat = [E[Y_0], E[Y_1]] (EM-adjusted
                          single log-concave).
    EM-mean-Kselection  : per-query τ from MALC's BIC-selected K, using
                          Σ_k π_k · (mu_hat_k[1] - mu_hat_k[0]) across
                          mixture components.

Field mapping in the shard (written by benchmarks/run_one.py):
    pehe_ours_mean / err_ours_mean                       → Raw-mean
    pehe_ours_malc_mean / err_ours_malc_mean             → MALC-CATE-mean
    pehe_ours_em_k1_mean / err_ours_em_k1_mean           → EM-mean-K1
    pehe_ours_em_mix_mean / err_ours_em_mix_mean         → EM-mean-Kselection

Usage
-----
    python R-PFN/benchmarks/summary_table3_two_row.py \\
        --results /scratch/.../results_ours_only_B500
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

DATASETS = ['IHDP', 'ACIC', 'CPS', 'PSID', 'PSIDbal']

ROWS = [
    ('Raw-mean',           'pehe_ours_mean',            'err_ours_mean'),
    ('MALC-CATE-mean',     'pehe_ours_malc_mean',       'err_ours_malc_mean'),
    ('EM-mean-K1',         'pehe_ours_em_k1_mean',      'err_ours_em_k1_mean'),
    ('EM-mean-Kselection', 'pehe_ours_em_mix_mean',     'err_ours_em_mix_mean'),
]


def _agg(vals):
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float('nan'), float('nan'), 0
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0), int(arr.size)


def _fmt(m, s):
    if not np.isfinite(m):
        return '      —      '
    return f'{m:6.2f} ± {s:5.2f}'


def _collect_bucket(results_dir, only_keys=None):
    """Aggregate PEHE/eps_ATE per (dataset, ROW-key) from a shard directory."""
    bucket = {d: {k: {'pehe': [], 'eps': []} for _, k, _ in ROWS} for d in DATASETS}
    files = sorted(glob.glob(os.path.join(results_dir, '*_r*.npz')))
    for path in files:
        with np.load(path, allow_pickle=True) as f:
            try:
                dname = str(f['dataset'])
            except Exception:
                continue
            if dname not in bucket:
                continue
            for _, pehe_key, eps_key in ROWS:
                if only_keys is not None and pehe_key not in only_keys:
                    continue
                if pehe_key in f.files:
                    bucket[dname][pehe_key]['pehe'].append(float(f[pehe_key]))
                if eps_key in f.files:
                    bucket[dname][pehe_key]['eps'].append(float(f[eps_key]))
    return bucket


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True,
                    help='Directory of *_r*.npz shards (MALC + EM + raw variants)')
    ap.add_argument('--logy-results', default=None,
                    help='Optional directory of log-Y run shards. Only raw-mean '
                         'field is read from this dir and added as an extra '
                         '"Log-Y-mean" row per dataset.')
    args = ap.parse_args()

    bucket = _collect_bucket(args.results)
    if not any(any(x['pehe'] for x in dset.values()) for dset in bucket.values()):
        print(f'[fatal] no shards in {args.results}'); return 2

    # Optional log-Y bucket: only the raw-mean field is populated in log-Y
    # runs (MALC/EM auto-skipped). Read just that.
    logy_bucket = None
    if args.logy_results and os.path.isdir(args.logy_results):
        logy_bucket = _collect_bucket(args.logy_results, only_keys={'pehe_ours_mean'})

    header = f'{"Dataset":10s} {"Variant":22s}    {"sqrt(PEHE)":>14s}    {"eps_ATE":>14s}    {"n":>4s}'
    print()
    title = 'four point-estimate variants' + (' + Log-Y' if logy_bucket is not None else '')
    print(f'── Ours — Table 3 ({title}) ─────────────────────')
    print(header)
    print('-' * len(header))
    for d in DATASETS:
        first = True
        for label, pehe_key, _ in ROWS:
            pehe = bucket[d][pehe_key]['pehe']
            eps  = bucket[d][pehe_key]['eps']
            pm, ps, np_ = _agg(pehe)
            em, es, ne  = _agg(eps)
            n = min(np_, ne) if (np_ and ne) else 0
            ds_col = d if first else ''
            print(f'{ds_col:10s} {label:22s}    {_fmt(pm, ps):>14s}    '
                  f'{_fmt(em, es):>14s}    {n:>4d}')
            first = False
        # Log-Y row (only raw-mean is meaningful in log-Y runs)
        if logy_bucket is not None:
            pehe = logy_bucket[d]['pehe_ours_mean']['pehe']
            eps  = logy_bucket[d]['pehe_ours_mean']['eps']
            pm, ps, np_ = _agg(pehe)
            em, es, ne  = _agg(eps)
            n = min(np_, ne) if (np_ and ne) else 0
            print(f'{"":10s} {"Log-Y-mean":22s}    {_fmt(pm, ps):>14s}    '
                  f'{_fmt(em, es):>14s}    {n:>4d}')
        print('-' * len(header))
    print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
