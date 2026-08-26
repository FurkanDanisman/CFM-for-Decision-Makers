"""Aggregate the 4-row ACIC validation table.

Reads pickle-per-realization outputs from:
  - UWYK Predictive / No-anc / Anc  → $UWYK_ROOT/RealCauseEval/results/
                                          table1_{predictive,noanc,anc}_acic_valid_<jobid>/
  - ours fn=50                       → $TABLE1_OUT_ROOT/
                                          table1_ours_fn50_acic_<jobid>/

Prints one 4-row table with PEHE (mean ± SEM) and epsilon_ATE (mean ± SEM),
alongside the paper's Table 1 ACIC targets so any mismatch is immediately
visible.

Usage:
    JOB_ID=<slurm_array_job_id>  python3 aggregate_acic_4row.py
"""
from __future__ import annotations
import glob
import os
import pickle
import sys

import numpy as np


def _load_row(pattern: str, model_hint: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Load {pehe, ate_rel_err} arrays from every pkl matching `pattern`."""
    files = sorted(glob.glob(pattern))
    pehes, ates = [], []
    for fp in files:
        with open(fp, 'rb') as f:
            d = pickle.load(f)
        # UWYK format uses 'pehe'/'ate_rel_err'; ours matches.
        p = d.get('pehe')
        a = d.get('ate_rel_err')
        if p is not None and np.isfinite(p):
            pehes.append(float(p))
        if a is not None and np.isfinite(a):
            ates.append(float(a))
    return np.asarray(pehes), np.asarray(ates), len(files)


def _fmt(pehes: np.ndarray, ates: np.ndarray, n: int) -> str:
    if n == 0:
        return f'{"MISSING":>18s}   {"MISSING":>14s}   n=0'
    def _ms(x):
        if x.size == 0:
            return 'nan ± nan'
        sem = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
        return f'{x.mean():>8.3f} ± {sem:<6.3f}'
    return f'{_ms(pehes):>18s}   {_ms(ates):>14s}   n={n}'


def main():
    job_id = os.environ.get('JOB_ID')
    if job_id is None:
        # Best-effort: pick the newest matching directory across all jobs.
        print('[warn] JOB_ID env var not set; picking latest matching dirs.', file=sys.stderr)
    UWYK_ROOT = os.environ.get(
        'UWYK_ROOT', '/scratch/furkanbd/rpfn_bench_kit/external/uwyk')
    TABLE1_OUT_ROOT = os.environ.get(
        'TABLE1_OUT_ROOT', '/scratch/furkanbd/rpfn_bench_kit/results_table1_acic_valid')

    if job_id is not None:
        tag = f'acic_valid_{job_id}'
        rows = [
            ('UWYK Predictive',
             f'{UWYK_ROOT}/RealCauseEval/results/table1_predictive_{tag}/*ACIC*'),
            ('UWYK No-Anc',
             f'{UWYK_ROOT}/RealCauseEval/results/table1_noanc_{tag}/*ACIC*'),
            ('UWYK Anc',
             f'{UWYK_ROOT}/RealCauseEval/results/table1_anc_{tag}/*ACIC*'),
            ('ours fn=50 (pred-mirror)',
             f'{TABLE1_OUT_ROOT}/table1_ours_fn50_predstyle_acic_{job_id}/*ACIC*'),
            ('ours fn=50 (null-t)',
             f'{TABLE1_OUT_ROOT}/table1_ours_fn50_nullt_acic_{job_id}/*ACIC*'),
        ]
    else:
        # newest matching dir per prefix
        def _latest(prefix, root):
            cands = sorted(glob.glob(f'{root}/{prefix}*'), key=os.path.getmtime)
            return f'{cands[-1]}/*ACIC*' if cands else f'{root}/{prefix}*_NOT_FOUND/*ACIC*'
        rows = [
            ('UWYK Predictive', _latest('table1_predictive_acic_valid_',
                                         f'{UWYK_ROOT}/RealCauseEval/results')),
            ('UWYK No-Anc',     _latest('table1_noanc_acic_valid_',
                                         f'{UWYK_ROOT}/RealCauseEval/results')),
            ('UWYK Anc',        _latest('table1_anc_acic_valid_',
                                         f'{UWYK_ROOT}/RealCauseEval/results')),
            ('ours fn=50 (pred-mirror)', _latest('table1_ours_fn50_predstyle_acic_',
                                                   TABLE1_OUT_ROOT)),
            ('ours fn=50 (null-t)',      _latest('table1_ours_fn50_nullt_acic_',
                                                   TABLE1_OUT_ROOT)),
        ]

    paper = {
        'UWYK Predictive':          ('3.14 ± 0.47', '0.38 ± 0.06'),
        'UWYK No-Anc':              ('3.47 ± 0.47', '0.46 ± 0.09'),
        'UWYK Anc':                 ('2.79 ± 0.45', '0.17 ± 0.08'),
        'ours fn=50 (pred-mirror)': ('n/a',         'n/a'),
        'ours fn=50 (null-t)':      ('n/a',         'n/a'),
    }

    print(f'\n══ ACIC 4-row Table 1 validation ══')
    print(f'  UWYK_ROOT       = {UWYK_ROOT}')
    print(f'  TABLE1_OUT_ROOT = {TABLE1_OUT_ROOT}')
    print(f'  JOB_ID          = {job_id or "(newest match)"}')
    print()
    print(f'  {"method":<20s}   {"√PEHE ± SEM":>18s}   {"ε_ATE ± SEM":>14s}   n     '
          f'{"paper √PEHE":>14s}   {"paper ε_ATE":>14s}')
    print(f'  {"-"*20}   {"-"*18}   {"-"*14}   ---   {"-"*14}   {"-"*14}')
    for name, pat in rows:
        pehes, ates, n = _load_row(pat, name)
        p_ref, a_ref = paper[name]
        print(f'  {name:<20s}   {_fmt(pehes, ates, n)}     {p_ref:>14s}   {a_ref:>14s}')


if __name__ == '__main__':
    main()
