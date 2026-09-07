#!/usr/bin/env python
"""Build the UWYK Table 3 reproduction summary from RealCauseEval/results/.

Walks every experiment folder under --results-dir, loads the per-realization
pickles produced by UWYK's dofm_*.py / predmodel_Slearner*.py, groups them
into paper Table 3's (row, column) cells by folder-name convention, and prints
+ writes a PEHE and an ATE table with paper targets alongside.

Row mapping (by folder-name substring, case-insensitive):
  Predictive  ← any folder containing 'predictive' or 'slearner'
  No-Anc      ← any folder containing 'noanc' or 'all_unknown'
  Anc         ← any folder containing 'anc' (and not 'noanc') or 'full_graph'

Column mapping (by folder name + pickle 'dataset' field):
  IHDP      ← IHDP records
  ACIC      ← ACIC records
  CPS       ← CPS records
  PSID_unbal← PSID records where folder name suggests unbalanced
              (contains 'unbal' or the pipeline is dofm_noclust / slearner_full)
  PSID_bal  ← PSID records where folder name suggests balanced
              (contains 'bal' but not 'unbal', or 'psid_balanced')

Ambiguity resolution: if the same (row, column) cell has >1 folder, prefer the
one with more realizations; ties broken alphabetically.

Usage:
    python realcause_eval/summarize_uwyk_reproduce.py \\
        --results-dir /scratch/.../external/uwyk_reproduce/RealCauseEval/results
    # writes uwyk_table3_summary.{md,csv} in --results-dir
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats


ROW_ORDER = ('Predictive', 'No-Anc', 'Anc')
COL_ORDER = ('IHDP', 'ACIC', 'CPS', 'PSID_unbal', 'PSID_bal')

# Paper Table 3 targets — (mean, stderr) per (metric, row, column).
PAPER = {
    ('pehe', 'Predictive', 'IHDP'):       ( 6.79,   0.81),
    ('pehe', 'Predictive', 'ACIC'):       ( 3.14,   0.47),
    ('pehe', 'Predictive', 'CPS'):        (11393.0, 31.0),
    ('pehe', 'Predictive', 'PSID_unbal'): (None, None),
    ('pehe', 'Predictive', 'PSID_bal'):   (22045.0, 136.0),
    ('pehe', 'No-Anc',     'IHDP'):       ( 6.28,   0.79),
    ('pehe', 'No-Anc',     'ACIC'):       ( 3.41,   0.52),
    ('pehe', 'No-Anc',     'CPS'):        (12792.0, 61.0),
    ('pehe', 'No-Anc',     'PSID_unbal'): (22435.0, 141.0),
    ('pehe', 'No-Anc',     'PSID_bal'):   (21896.0, 137.0),
    ('pehe', 'Anc',        'IHDP'):       ( 5.49,   0.78),
    ('pehe', 'Anc',        'ACIC'):       ( 2.79,   0.45),
    ('pehe', 'Anc',        'CPS'):        (11213.0, 60.0),
    ('pehe', 'Anc',        'PSID_unbal'): (None, None),
    ('pehe', 'Anc',        'PSID_bal'):   (19711.0, 230.0),

    ('ate_rel_err', 'Predictive', 'IHDP'):       (0.81,  0.11),
    ('ate_rel_err', 'Predictive', 'ACIC'):       (0.38,  0.06),
    ('ate_rel_err', 'Predictive', 'CPS'):        (0.78,  0.00),
    ('ate_rel_err', 'Predictive', 'PSID_unbal'): (None, None),
    ('ate_rel_err', 'Predictive', 'PSID_bal'):   (0.945, 0.006),
    ('ate_rel_err', 'No-Anc',     'IHDP'):       (0.67,  0.05),
    ('ate_rel_err', 'No-Anc',     'ACIC'):       (0.46,  0.09),
    ('ate_rel_err', 'No-Anc',     'CPS'):        (0.99,  0.01),
    ('ate_rel_err', 'No-Anc',     'PSID_unbal'): (None, None),
    ('ate_rel_err', 'No-Anc',     'PSID_bal'):   (0.936, 0.004),
    ('ate_rel_err', 'Anc',        'IHDP'):       (0.49,  0.08),
    ('ate_rel_err', 'Anc',        'ACIC'):       (0.17,  0.08),
    ('ate_rel_err', 'Anc',        'CPS'):        (0.70,  0.02),
    ('ate_rel_err', 'Anc',        'PSID_unbal'): (None, None),
    ('ate_rel_err', 'Anc',        'PSID_bal'):   (0.650, 0.018),
}


def _classify_row(folder: str) -> str | None:
    f = folder.lower()
    if 'noanc' in f or 'all_unknown' in f:
        return 'No-Anc'
    if 'predictive' in f or 'slearner' in f:
        return 'Predictive'
    if 'full_graph' in f or re.search(r'(?<!no)anc(?!estor)', f):
        return 'Anc'
    return None


def _classify_col(folder: str, dataset: str) -> str | None:
    f = folder.lower()
    d = dataset.upper()
    if d == 'IHDP':
        return 'IHDP'
    if d == 'ACIC':
        return 'ACIC'
    if d == 'CPS':
        return 'CPS'
    if d == 'PSID':
        if 'unbal' in f:
            return 'PSID_unbal'
        if 'bal' in f or 'balanced' in f:
            return 'PSID_bal'
        # dofm_no_clustering.py on PSID → unbalanced (per UWYK doc §3)
        if 'noclust' in f or 'no_clust' in f or 'slearner_full' in f:
            return 'PSID_unbal'
        return None
    return None


def _load_folder(path: str) -> list[dict]:
    out = []
    for name in os.listdir(path):
        p = os.path.join(path, name)
        if not os.path.isfile(p) or name.endswith(('.csv', '.json', '.md')):
            continue
        try:
            with open(p, 'rb') as f:
                d = pickle.load(f)
            if isinstance(d, dict) and 'pehe' in d and 'dataset' in d:
                out.append(d)
        except Exception:
            pass
    return out


def collect(results_dir: str, folder_filter: list[str] | None = None,
            ) -> dict[tuple[str, str], tuple[str, list[dict]]]:
    """Return {(row, col): (folder_name, records)} — one folder per cell.

    If folder_filter is given, only folders whose name contains any of the
    filter substrings are included.
    """
    candidates: dict[tuple[str, str], list[tuple[str, list[dict]]]] = defaultdict(list)
    for folder in sorted(os.listdir(results_dir)):
        sub = os.path.join(results_dir, folder)
        if not os.path.isdir(sub):
            continue
        if folder_filter and not any(sub_s in folder for sub_s in folder_filter):
            continue
        row = _classify_row(folder)
        if row is None:
            continue
        records = _load_folder(sub)
        if not records:
            continue
        by_col: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            col = _classify_col(folder, rec['dataset'])
            if col is not None:
                by_col[col].append(rec)
        for col, recs in by_col.items():
            candidates[(row, col)].append((folder, recs))

    picked: dict[tuple[str, str], tuple[str, list[dict]]] = {}
    for cell, opts in candidates.items():
        opts.sort(key=lambda t: (-len(t[1]), t[0]))
        picked[cell] = opts[0]
    return picked


def _mean_se(values: list[float]) -> tuple[float, float, int]:
    n = len(values)
    if n == 0:
        return float('nan'), float('nan'), 0
    arr = np.asarray(values, dtype=float)
    m = float(arr.mean())
    se = float(stats.sem(arr)) if n > 1 else float('nan')
    return m, se, n


def _fmt_cell(m: float, se: float, n: int, big: bool) -> str:
    if n == 0 or not np.isfinite(m):
        return '—'
    if big:
        m_s = f'{m:,.2f}'
        se_s = f'{se:,.2f}' if np.isfinite(se) else '—'
    else:
        m_s = f'{m:.4f}'
        se_s = f'{se:.4f}' if np.isfinite(se) else '—'
    return f'{m_s} ± {se_s}  (n={n})'


def _fmt_paper(m: float | None, se: float | None, big: bool) -> str:
    if m is None:
        return '—'
    if big:
        return f'{m:,.0f} ± {se:,.0f}'
    return f'{m:.3f} ± {se:.3f}'


def _big_col(col: str) -> bool:
    return col in ('CPS', 'PSID_unbal', 'PSID_bal')


def render(picked, metric: str, rows_present: list[str]) -> str:
    label = {'pehe': '√PEHE', 'ate_rel_err': 'ε_ATE'}[metric]
    lines = [f'\n## UWYK Table 3 — {label} (mean ± SE)\n']
    header = '| Row | ' + ' | '.join(COL_ORDER) + ' |'
    sep = '|' + '|'.join(['---'] * (1 + len(COL_ORDER))) + '|'
    lines += [header, sep]
    for row in rows_present:
        cells = [row]
        for col in COL_ORDER:
            big = _big_col(col) and metric == 'pehe'
            if (row, col) in picked:
                _, recs = picked[(row, col)]
                vals = [r[metric] for r in recs if metric in r
                        and r[metric] is not None
                        and np.isfinite(r[metric])]
                m, se, n = _mean_se(vals)
                cells.append(_fmt_cell(m, se, n, big))
            else:
                cells.append('—')
        lines.append('| ' + ' | '.join(cells) + ' |')
        paper_cells = [f'*{row} target*']
        for col in COL_ORDER:
            pm, pse = PAPER.get((metric, row, col), (None, None))
            paper_cells.append(_fmt_paper(pm, pse,
                                          big=_big_col(col) and metric == 'pehe'))
        lines.append('| ' + ' | '.join(paper_cells) + ' |')
    lines.append('')

    lines.append('Folder used per cell:')
    for row in rows_present:
        for col in COL_ORDER:
            if (row, col) in picked:
                folder, recs = picked[(row, col)]
                lines.append(f'  {row:<11} {col:<11} ← {folder}  (n={len(recs)})')
    return '\n'.join(lines)


def to_csv(picked, out_path: str) -> None:
    import csv
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'row', 'column', 'folder', 'n', 'mean', 'stderr',
                    'paper_mean', 'paper_stderr'])
        for metric in ('pehe', 'ate_rel_err'):
            for row in ROW_ORDER:
                for col in COL_ORDER:
                    folder = ''
                    m = se = float('nan')
                    n = 0
                    if (row, col) in picked:
                        folder, recs = picked[(row, col)]
                        vals = [r[metric] for r in recs if metric in r
                                and r[metric] is not None
                                and np.isfinite(r[metric])]
                        m, se, n = _mean_se(vals)
                    pm, pse = PAPER.get((metric, row, col), (None, None))
                    w.writerow([metric, row, col, folder, n, m, se,
                                '' if pm is None else pm,
                                '' if pse is None else pse])


OUR_RUN_FOLDERS = ('dofm_noclust_all_unknown', 'dofm_psid_balanced_all_unknown')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True,
                    help='UWYK RealCauseEval/results directory.')
    ap.add_argument('--folders', nargs='*', default=None,
                    help='Only include folders whose name contains any of these '
                         'substrings. Default: everything.')
    ap.add_argument('--our-run', action='store_true',
                    help=f'Shortcut for --folders {" ".join(OUR_RUN_FOLDERS)}. '
                         f'Shows only the No-Anc row our sbatch produced.')
    ap.add_argument('--out-md', default=None,
                    help='Markdown output path (default: <results-dir>/uwyk_table3_summary.md).')
    ap.add_argument('--out-csv', default=None,
                    help='CSV output path (default: <results-dir>/uwyk_table3_summary.csv).')
    args = ap.parse_args()

    if not os.path.isdir(args.results_dir):
        sys.exit(f'FATAL: results-dir not found: {args.results_dir}')

    folder_filter = args.folders
    if args.our_run:
        folder_filter = list(OUR_RUN_FOLDERS)

    picked = collect(args.results_dir, folder_filter=folder_filter)
    if not picked:
        where = ' + '.join(folder_filter) if folder_filter else args.results_dir
        sys.exit(f'FATAL: no recognizable pickles found in {where}')

    rows_present = [r for r in ROW_ORDER
                    if any((r, c) in picked for c in COL_ORDER)]

    md_pehe = render(picked, 'pehe',       rows_present)
    md_ate  = render(picked, 'ate_rel_err', rows_present)
    md = md_pehe + '\n' + md_ate + '\n'

    out_md  = args.out_md  or os.path.join(args.results_dir, 'uwyk_table3_summary.md')
    out_csv = args.out_csv or os.path.join(args.results_dir, 'uwyk_table3_summary.csv')
    with open(out_md, 'w') as f:
        f.write(md)
    to_csv(picked, out_csv)

    print(md)
    print(f'wrote {out_md}')
    print(f'wrote {out_csv}')


if __name__ == '__main__':
    main()
