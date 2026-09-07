"""UWYK RealCause eval — thin wrapper around UWYK's own reproduce-branch scripts.

Runs the identical scripts from
    https://github.com/ArikReuter/Graphs4CausalFoundationModels/tree/reproduce-realcause-results
with no modifications:

    IHDP / ACIC / CPS / PSID (unbalanced) → dofm_no_clustering.py
    PSID_bal                              → dofm_psid_balanced.py

Both invoked with --graph_mode all_unknown, both defaulting to the checkpoint
their scripts point at (best_model.pt in the full_conditioned_model dir).
All 5 datasets write per-realization pickles to a SINGLE folder
$UWYK_REPRO/results/$EXP_NAME/ — pickle filenames are
    dofm_noclust_{IHDP,ACIC,CPS,PSID}_{r}
    dofm_psid_balanced_PSID_{r}
so PSID vs PSID_bal is disambiguated by the model-name prefix (which is also
stored inside the pickle as record['model']).

Two modes
---------

Run one dataset:
    python realcause_eval/uwyk.py run \\
        --dataset PSID_bal \\
        --uwyk-repro /scratch/.../external/uwyk_reproduce \\
        --exp-name uwyk_realcause

Print the summary table for the results folder:
    python realcause_eval/uwyk.py summary \\
        --uwyk-repro /scratch/.../external/uwyk_reproduce \\
        --exp-name uwyk_realcause

For all 5 in parallel on Slurm, submit the array sbatch:
    sbatch benchmarks/cluster/submit_uwyk_reproduce_noanc.sbatch
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from collections import defaultdict

import numpy as np
from scipy import stats


DEFAULT_EXP_NAME = 'uwyk_realcause'

_DATASETS = ('IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal')

_RUN_SPEC = {
    'IHDP':     ('dofm_no_clustering.py', 'IHDP', 'dofm_noclust'),
    'ACIC':     ('dofm_no_clustering.py', 'ACIC', 'dofm_noclust'),
    'CPS':      ('dofm_no_clustering.py', 'CPS',  'dofm_noclust'),
    'PSID':     ('dofm_no_clustering.py', 'PSID', 'dofm_noclust'),
    'PSID_bal': ('dofm_psid_balanced.py', 'PSID', 'dofm_psid_balanced'),
}

_LABEL_FROM_MODEL = {
    'dofm_noclust':       {'IHDP': 'IHDP', 'ACIC': 'ACIC', 'CPS': 'CPS', 'PSID': 'PSID'},
    'dofm_psid_balanced': {'PSID': 'PSID_bal'},
}


def _run(args):
    script, uwyk_dataset, model_name = _RUN_SPEC[args.dataset]
    uwyk = os.path.abspath(args.uwyk_repro)
    script_path = os.path.join(uwyk, 'RealCauseEval', 'run_baselines', script)
    if not os.path.isfile(script_path):
        sys.exit(f'FATAL: UWYK script not found at {script_path}')

    ckpt_dir = os.path.join(uwyk, 'experiments', 'checkpoints', 'full_conditioned_model',
                            'final_earlytest_full_conditioning_16773252.0')
    for f in ('best_model.pt', 'best_model_config.yaml'):
        if not os.path.isfile(os.path.join(ckpt_dir, f)):
            sys.exit(f'FATAL: default UWYK checkpoint missing at {ckpt_dir}/{f} '
                     '— did you `git lfs pull`?')

    repo = args.repo or os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    shim = os.path.join(repo, 'benchmarks', 'uwyk_table1', 'shims')
    env['PYTHONPATH'] = (
        f'{uwyk}/RealCauseEval:{shim}' + (f':{env["PYTHONPATH"]}' if env.get('PYTHONPATH') else '')
    )

    cmd = [
        sys.executable, '-u', script_path,
        '--dataset',    uwyk_dataset,
        '--model',      model_name,
        '--exp_name',   args.exp_name,
        '--graph_mode', 'all_unknown',
    ]

    print(f'[uwyk] dataset={args.dataset}  script={script}  exp_name={args.exp_name}', flush=True)
    print(f'[uwyk] results dir: {uwyk}/results/{args.exp_name}/', flush=True)
    subprocess.run(cmd, env=env, check=True, cwd=uwyk)


def _load_pickles(folder: str):
    out = []
    if not os.path.isdir(folder):
        return out
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if not os.path.isfile(p) or name.endswith(('.csv', '.json', '.md')):
            continue
        try:
            with open(p, 'rb') as f:
                d = pickle.load(f)
            if isinstance(d, dict) and 'pehe' in d and 'ate_rel_err' in d \
               and 'dataset' in d and 'model' in d:
                out.append(d)
        except Exception:
            pass
    return out


def _label(rec: dict) -> str | None:
    return _LABEL_FROM_MODEL.get(rec['model'], {}).get(rec['dataset'])


def _mean_se(vals):
    n = len(vals)
    if n == 0:
        return float('nan'), float('nan'), 0
    arr = np.asarray(vals, dtype=float)
    m = float(arr.mean())
    se = float(stats.sem(arr)) if n > 1 else float('nan')
    return m, se, n


def _fmt(m, se, n, big):
    if n == 0 or not np.isfinite(m):
        return '—'
    if big:
        m_s, se_s = f'{m:,.2f}', (f'{se:,.2f}' if np.isfinite(se) else '—')
    else:
        m_s, se_s = f'{m:.4f}',  (f'{se:.4f}'  if np.isfinite(se) else '—')
    return f'{m_s} ± {se_s}  (n={n})'


def _summary(args):
    uwyk = os.path.abspath(args.uwyk_repro)
    folder = os.path.join(uwyk, 'results', args.exp_name)
    records = _load_pickles(folder)
    if not records:
        sys.exit(f'FATAL: no pickles found in {folder}')

    by_label = defaultdict(list)
    for rec in records:
        lab = _label(rec)
        if lab is not None:
            by_label[lab].append(rec)

    def cell(metric: str, label: str, big: bool) -> str:
        vals = [r[metric] for r in by_label.get(label, [])
                if r.get(metric) is not None and np.isfinite(r[metric])]
        return _fmt(*_mean_se(vals), big=big)

    lines = []
    lines.append(f'\nUWYK RealCause — {folder}\n')
    for metric, label, big_cols in (
        ('pehe',        '√PEHE',   {'CPS', 'PSID', 'PSID_bal'}),
        ('ate_rel_err', 'ε_ATE',   set()),
    ):
        lines.append(f'## {label} (mean ± SE)')
        lines.append('| ' + ' | '.join(_DATASETS) + ' |')
        lines.append('|' + '|'.join(['---'] * len(_DATASETS)) + '|')
        lines.append('| ' + ' | '.join(cell(metric, d, d in big_cols) for d in _DATASETS) + ' |')
        lines.append('')

    md = '\n'.join(lines)
    print(md)

    if args.out_md:
        with open(args.out_md, 'w') as f:
            f.write(md)
        print(f'wrote {args.out_md}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    r = sub.add_parser('run', help="Run UWYK's script for one dataset.")
    r.add_argument('--dataset', required=True, choices=list(_DATASETS))
    r.add_argument('--uwyk-repro', required=True,
                   help='UWYK reproduce-branch repo root.')
    r.add_argument('--exp-name', default=DEFAULT_EXP_NAME,
                   help=f'Single output folder for all 5 datasets. Default: {DEFAULT_EXP_NAME}.')
    r.add_argument('--repo', default=None, help='R-PFN repo root (for the sitecustomize shim).')
    r.set_defaults(func=_run)

    s = sub.add_parser('summary', help='Print the 5-column results table for one exp_name folder.')
    s.add_argument('--uwyk-repro', required=True)
    s.add_argument('--exp-name', default=DEFAULT_EXP_NAME)
    s.add_argument('--out-md', default=None,
                   help='Optional path to also write the markdown table to.')
    s.set_defaults(func=_summary)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
