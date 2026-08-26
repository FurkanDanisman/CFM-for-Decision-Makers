"""Aggregate the 4-row ACIC reproduction table using UWYK's own scripts.

Reads pickle-per-realization outputs from:
  - UWYK Predictive / No-anc / Anc  → $UWYK_REPRO/RealCauseEval/results/
                                          table1_repro_{predictive,noanc,anc}_<jobid>/
  - ours fn=50                       → $TABLE1_OUT_ROOT/
                                          table1_repro_ours_fn50_<jobid>/

Usage:
    JOB_ID=<slurm_array_job_id>  python3 aggregate_acic_repro.py
"""
from __future__ import annotations
import glob
import os
import pickle
import sys

import numpy as np


def _load_row(pattern: str) -> tuple[np.ndarray, np.ndarray, int]:
    files = sorted(glob.glob(pattern))
    pehes, ates = [], []
    for fp in files:
        with open(fp, 'rb') as f:
            d = pickle.load(f)
        p, a = d.get('pehe'), d.get('ate_rel_err')
        if p is not None and np.isfinite(p): pehes.append(float(p))
        if a is not None and np.isfinite(a): ates.append(float(a))
    return np.asarray(pehes), np.asarray(ates), len(files)


def _fmt(x: np.ndarray) -> str:
    if x.size == 0: return 'nan ± nan'
    sem = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return f'{x.mean():>8.3f} ± {sem:<6.3f}'


def main():
    job_id = os.environ.get('JOB_ID')
    if job_id is None:
        print('[warn] JOB_ID env var not set; picking latest matching dirs.', file=sys.stderr)
    UWYK_REPRO = os.environ.get(
        'UWYK_REPRO', '/scratch/furkanbd/rpfn_bench_kit/external/uwyk_reproduce')
    TABLE1_OUT_ROOT = os.environ.get(
        'TABLE1_OUT_ROOT', '/scratch/furkanbd/rpfn_bench_kit/results_table1_ours_fn50')

    if job_id is not None:
        rows = [
            ('UWYK Predictive',
             f'{UWYK_REPRO}/RealCauseEval/results/table1_repro_predictive_{job_id}/*ACIC*'),
            ('UWYK No-Anc',
             f'{UWYK_REPRO}/RealCauseEval/results/table1_repro_noanc_{job_id}/*ACIC*'),
            ('UWYK Anc',
             f'{UWYK_REPRO}/RealCauseEval/results/table1_repro_anc_{job_id}/*ACIC*'),
            ('ours fn=50',
             f'{TABLE1_OUT_ROOT}/table1_repro_ours_fn50_{job_id}/*ACIC*'),
        ]
    else:
        def _latest(prefix, root):
            cands = sorted(glob.glob(f'{root}/{prefix}*'), key=os.path.getmtime)
            return f'{cands[-1]}/*ACIC*' if cands else f'{root}/{prefix}*_NOT_FOUND/*ACIC*'
        rows = [
            ('UWYK Predictive', _latest('table1_repro_predictive_',
                                         f'{UWYK_REPRO}/RealCauseEval/results')),
            ('UWYK No-Anc',     _latest('table1_repro_noanc_',
                                         f'{UWYK_REPRO}/RealCauseEval/results')),
            ('UWYK Anc',        _latest('table1_repro_anc_',
                                         f'{UWYK_REPRO}/RealCauseEval/results')),
            ('ours fn=50',      _latest('table1_repro_ours_fn50_', TABLE1_OUT_ROOT)),
        ]

    paper = {
        'UWYK Predictive':  ('3.14 ± 0.47', '0.38 ± 0.06'),
        'UWYK No-Anc':      ('3.47 ± 0.47', '0.46 ± 0.09'),
        'UWYK Anc':         ('2.79 ± 0.45', '0.17 ± 0.08'),
        'ours fn=50':       ('n/a',         'n/a'),
    }

    print(f'\n══ ACIC Table 1 reproduction (their scripts, their branch) ══')
    print(f'  UWYK_REPRO      = {UWYK_REPRO}')
    print(f'  TABLE1_OUT_ROOT = {TABLE1_OUT_ROOT}')
    print(f'  JOB_ID          = {job_id or "(newest match)"}')
    print()
    print(f'  {"method":<20s}   {"√PEHE ± SEM":>18s}   {"ε_ATE ± SEM":>14s}   n     '
          f'{"paper √PEHE":>14s}   {"paper ε_ATE":>14s}')
    print(f'  {"-"*20}   {"-"*18}   {"-"*14}   ---   {"-"*14}   {"-"*14}')
    for name, pat in rows:
        pehes, ates, n = _load_row(pat)
        p_ref, a_ref = paper[name]
        print(f'  {name:<20s}   {_fmt(pehes):>18s}   {_fmt(ates):>14s}   n={n:<3d}   '
              f'{p_ref:>14s}   {a_ref:>14s}')


if __name__ == '__main__':
    main()
