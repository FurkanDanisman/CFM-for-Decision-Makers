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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, help='Directory of *_r*.npz shards')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results, '*_r*.npz')))
    if not files:
        print(f'[fatal] no shards in {args.results}'); return 2

    bucket = {d: {k: {'pehe': [], 'eps': []} for _, k, _ in ROWS} for d in DATASETS}
    for path in files:
        with np.load(path, allow_pickle=True) as f:
            try:
                dname = str(f['dataset'])
            except Exception:
                continue
            if dname not in bucket:
                continue
            for _, pehe_key, eps_key in ROWS:
                if pehe_key in f.files:
                    bucket[dname][pehe_key]['pehe'].append(float(f[pehe_key]))
                if eps_key in f.files:
                    bucket[dname][pehe_key]['eps'].append(float(f[eps_key]))

    header = f'{"Dataset":10s} {"Variant":22s}    {"sqrt(PEHE)":>14s}    {"eps_ATE":>14s}    {"n":>4s}'
    print()
    print('── Ours — Table 3 (four point-estimate variants) ─────────────────────')
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
        print('-' * len(header))
    print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
