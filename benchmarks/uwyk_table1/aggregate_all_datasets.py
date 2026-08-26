"""Aggregate the 4-dataset × 4-method Table-1 reproduction.

Reads the pickle-per-realization outputs written by submit_table1_all_datasets.sbatch
under `table1_all_<DATASET>_<METHOD>_<jobid>/` in two roots:
  - UWYK Predictive/NoAnc/Anc rows → $UWYK_REPRO/RealCauseEval/results/
  - ours fn=50 rows                → $TABLE1_OUT_ROOT/

Usage:
    JOB_ID=<slurm_array_job_id>  python3 aggregate_all_datasets.py
"""
from __future__ import annotations
import glob
import os
import pickle
import sys

import numpy as np


DATASET_UWYK = {
    'IHDP':       'IHDP',
    'CPS':        'CPS',
    'PSID_unbal': 'PSID',
    'PSID_bal':   'PSID',
}

# Column keys are (dataset-label-for-print, DATASET_LOG-used-in-exp-name)
DATASETS = [
    ('IHDP',       'IHDP'),
    ('CPS',        'CPS'),
    ('PSID_unbal', 'PSID_unbal'),
    ('PSID_bal',   'PSID_bal'),
]

METHODS = [
    ('UWYK Predictive', 'predictive'),
    ('UWYK No-Anc',     'noanc'),
    ('UWYK Anc',        'anc'),
    ('ours fn=50',      'fn50'),
]

# Paper Table 1 targets. PSID_bal numbers come from the reproduce branch's
# own README (paper §F.3).
PAPER = {
    ('IHDP', 'predictive'):       (6.79, 0.81),
    ('IHDP', 'noanc'):            (6.28, 0.67),
    ('IHDP', 'anc'):              (5.49, 0.49),
    ('CPS',  'predictive'):       (11393, 0.78),
    ('CPS',  'noanc'):            (12800, 0.99),
    ('CPS',  'anc'):              (11213, 0.70),
    ('PSID_unbal', 'predictive'): (11820, 1.03),
    ('PSID_unbal', 'noanc'):      (13096, 0.98),
    ('PSID_unbal', 'anc'):        (12975, 1.09),
    ('PSID_bal', 'predictive'):   (22045, 0.945),
    ('PSID_bal', 'noanc'):        (21896, 0.936),
    ('PSID_bal', 'anc'):          (19711, 0.650),
}


def _load(pattern):
    files = sorted(glob.glob(pattern))
    pehes, ates = [], []
    for fp in files:
        try:
            with open(fp, 'rb') as f:
                d = pickle.load(f)
        except Exception:
            continue
        p, a = d.get('pehe'), d.get('ate_rel_err')
        if p is not None and np.isfinite(p): pehes.append(float(p))
        if a is not None and np.isfinite(a): ates.append(float(a))
    return np.asarray(pehes), np.asarray(ates), len(files)


def _fmt(x):
    if x.size == 0:
        return 'nan'
    sem = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    m = x.mean()
    if abs(m) >= 100:
        return f'{m:>7.0f} ± {sem:<5.0f}'
    return f'{m:>7.3f} ± {sem:<5.3f}'


def _fmt_paper(v):
    if v is None: return '---'
    p, a = v
    if abs(p) >= 100:
        return f'{p:>5.0f} / {a:<5.3f}'
    return f'{p:>5.2f} / {a:<5.3f}'


def main():
    job_id = os.environ.get('JOB_ID')
    if job_id is None:
        print('[warn] JOB_ID unset — using latest matching dir per method/dataset.', file=sys.stderr)
    UWYK_REPRO = os.environ.get(
        'UWYK_REPRO', '/scratch/furkanbd/rpfn_bench_kit/external/uwyk_reproduce')
    OURS_ROOT = os.environ.get(
        'TABLE1_OUT_ROOT', '/scratch/furkanbd/rpfn_bench_kit/results_table1_all')

    def pattern(ds_log, method):
        # each task's exp_name is table1_all_<DATASET_LOG>_<METHOD>_<JOB>
        if method == 'fn50':
            root = OURS_ROOT
        else:
            root = f'{UWYK_REPRO}/RealCauseEval/results'
        if job_id is not None:
            return f'{root}/table1_all_{ds_log}_{method}_{job_id}/*'
        # newest-match fallback
        cands = sorted(glob.glob(f'{root}/table1_all_{ds_log}_{method}_*'),
                        key=os.path.getmtime)
        return f'{cands[-1]}/*' if cands else f'{root}/NOT_FOUND/*'

    for ds_pretty, ds_log in DATASETS:
        print(f'\n══ {ds_pretty} ══')
        print(f'  {"method":<20s}   {"PEHE ± SEM":>18s}   {"eps_ATE ± SEM":>16s}   n     paper')
        print(f'  {"-"*20}   {"-"*18}   {"-"*16}   ---   {"-"*14}')
        # Predictive uses the dataset name UWYK saw (IHDP/CPS/PSID);
        # our exp_name uses the tag (IHDP/CPS/PSID_unbal/PSID_bal).
        for m_pretty, m_key in METHODS:
            # filter pkls to the specific dataset UWYK put in the filename
            uwyk_dataset = DATASET_UWYK.get(ds_log, ds_log)
            filter_hint = '' if m_key == 'fn50' else uwyk_dataset
            pat = pattern(ds_log, m_key)
            # for uwyk pkls the filename contains dataset; for ours too.
            # a wildcard match on '*' catches both.
            pehes, ates, n = _load(pat)
            paper_key = ds_log if ds_log in ('PSID_unbal', 'PSID_bal') else ds_log
            # rewrite key mapping to match PAPER dict
            if ds_log == 'IHDP':       paper_key = 'IHDP'
            elif ds_log == 'CPS':      paper_key = 'CPS'
            elif ds_log == 'PSID_unbal': paper_key = 'PSID_unbal'
            elif ds_log == 'PSID_bal':   paper_key = 'PSID_bal'
            paper_v = PAPER.get((paper_key, m_key))
            print(f'  {m_pretty:<20s}   {_fmt(pehes):>18s}   {_fmt(ates):>16s}   {n:<3d}   '
                  f'{_fmt_paper(paper_v):>14s}')


if __name__ == '__main__':
    main()
