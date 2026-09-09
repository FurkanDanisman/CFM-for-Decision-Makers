"""Aggregate the SCM case-study context-size sweep into PEHE + L1-ATE tables.

Reads $SWEEP/ctx<N>/<model>/<case>/... for the 6 models × 5 context sizes
× 6 case studies, computes per-(model, context, case):
    PEHE   = mean over realizations of pehe_raw
    L1_ATE = mean over realizations of |ate_pred - true_ate|
No outlier filtering (every realization counts).

Output-format handling (3 shapes):
  uniform  (dopfn_native, cpfn1d, cpfn2d, uwyk):
      per-realization r{NNN}.npz with pehe_raw + (ate_pred|ate_raw) + true_ate
  graph2d:
      per-realization <CASE>_r{NNN}.npz with pehe_raw_v3b + ate_raw_v3b + true_ate
      (v3b = ancestor-informed; 'raw' method = no EM/MALC smoothing)
  dopfn_bb:
      one summary.npz per case with arrays pehe[], ate_pred[], true_ate[]

Usage:
    python realcause_eval/aggregate_scm_ctx_sweep.py \\
        --sweep $DEPLOY_ROOT/results_scm_ctx_sweep \\
        --out   $DEPLOY_ROOT/results_scm_ctx_sweep/summary
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


MODELS   = ['dopfn_native', 'dopfn_bb', 'cpfn2d', 'cpfn1d', 'graph2d', 'uwyk']
CONTEXTS = [50, 100, 250, 500, 1000]
CASES    = ['Observed_Confounder', 'Observed_Mediator',
            'Observed_Mediator_and_Confounder', 'Unobserved_Confounder',
            'Frontdoor_Criterion', 'Backdoor_Criterion']


def _first(z, keys):
    for k in keys:
        if k in z.files:
            return np.asarray(z[k], dtype=np.float64)
    return None


def _cell_pehe_l1(sweep, ctx, model, case):
    """Return (pehe_array, l1_array) over realizations for one cell, or (None, None)."""
    cell = os.path.join(sweep, f'ctx{ctx}', model, case)
    if not os.path.isdir(cell):
        return None, None

    # ── dopfn_bb: single summary.npz with per-realization arrays.
    if model == 'dopfn_bb':
        cand = os.path.join(cell, 'summary.npz')
        if not os.path.isfile(cand):
            return None, None
        with np.load(cand, allow_pickle=True) as z:
            pehe = _first(z, ['pehe'])
            ate_pred = _first(z, ['ate_pred'])
            true_ate = _first(z, ['true_ate'])
        if pehe is None:
            return None, None
        l1 = (np.abs(ate_pred - true_ate) if (ate_pred is not None and true_ate is not None)
              else None)
        return pehe, l1

    # ── graph2d: <CASE>_r{NNN}.npz with per-tag keys (v3b ancestor, raw method).
    if model == 'graph2d':
        paths = sorted(glob.glob(os.path.join(cell, f'{case}_r*.npz')))
        pehe_l, l1_l = [], []
        for p in paths:
            with np.load(p, allow_pickle=True) as z:
                pe = _first(z, ['pehe_raw_v3b', 'pehe_full_v3b', 'pehe_em_v3b'])
                at = _first(z, ['ate_raw_v3b', 'ate_full_v3b', 'ate_em_v3b'])
                tr = _first(z, ['true_ate'])
            if pe is None:
                continue
            pehe_l.append(float(pe))
            if at is not None and tr is not None:
                l1_l.append(abs(float(at) - float(tr)))
        if not pehe_l:
            return None, None
        return np.array(pehe_l), (np.array(l1_l) if l1_l else None)

    # ── uniform: dopfn_native/uwyk write r{NNN}.npz; cpfn2d/cpfn1d write
    #    {CASE}_r{NNN}.npz (tag includes DATASET). Try both.
    paths = sorted(glob.glob(os.path.join(cell, 'r*.npz')))
    if not paths:
        paths = sorted(glob.glob(os.path.join(cell, f'{case}_r*.npz')))
    pehe_l, l1_l = [], []
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            pe = _first(z, ['pehe_raw'])
            at = _first(z, ['ate_pred', 'ate_raw'])
            tr = _first(z, ['true_ate'])
        if pe is None:
            continue
        pehe_l.append(float(pe))
        if at is not None and tr is not None:
            l1_l.append(abs(float(at) - float(tr)))
    if not pehe_l:
        return None, None
    return np.array(pehe_l), (np.array(l1_l) if l1_l else None)


def _fmt(vals):
    """median [IQR] (mean) — median is the primary statistic (matches the
    DoPFN paper's Fig 3 median bar-plots); mean shown in parens for
    reference. These case-study SCMs span many CATE scales, so the mean
    is dominated by a few high-scale realizations."""
    if vals is None or len(vals) == 0:
        return '—'
    med = float(np.median(vals))
    mean = float(np.mean(vals))
    q1, q3 = np.percentile(vals, [25, 75])
    return f'{med:.3f} [{q1:.3f}–{q3:.3f}] (μ={mean:.3f}, n={len(vals)})'


def _build_tables(sweep):
    lines_pehe = ['# PEHE — SCM case-study context-size sweep\n']
    lines_l1   = ['# L1-ATE (|ATE_pred − ATE_true|) — SCM case-study context-size sweep\n']

    for metric, lines, idx in (('PEHE', lines_pehe, 0), ('L1_ATE', lines_l1, 1)):
        for case in CASES:
            lines.append(f'\n## {case}\n')
            header = '| Model | ' + ' | '.join(f'ctx={c}' for c in CONTEXTS) + ' |'
            lines.append(header)
            lines.append('|' + '---|' * (len(CONTEXTS) + 1))
            for model in MODELS:
                cells = [model]
                for ctx in CONTEXTS:
                    pehe, l1 = _cell_pehe_l1(sweep, ctx, model, case)
                    cells.append(_fmt(pehe if idx == 0 else l1))
                lines.append('| ' + ' | '.join(cells) + ' |')

        # Case-averaged view: median over the 6 case studies of each cell's
        # median (macro-average of medians — matches the per-cell statistic).
        lines.append(f'\n## ALL CASES (macro-median over 6 case studies)\n')
        header = '| Model | ' + ' | '.join(f'ctx={c}' for c in CONTEXTS) + ' |'
        lines.append(header); lines.append('|' + '---|' * (len(CONTEXTS) + 1))
        for model in MODELS:
            cells = [model]
            for ctx in CONTEXTS:
                per_case_meds = []
                for case in CASES:
                    pehe, l1 = _cell_pehe_l1(sweep, ctx, model, case)
                    v = pehe if idx == 0 else l1
                    if v is not None and len(v):
                        per_case_meds.append(float(np.median(v)))
                cells.append(f'{np.median(per_case_meds):.3f}' if per_case_meds else '—')
            lines.append('| ' + ' | '.join(cells) + ' |')

    return '\n'.join(lines_pehe) + '\n\n' + '\n'.join(lines_l1) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', required=True,
                    help='Root holding ctx<N>/<model>/<case>/ dirs.')
    ap.add_argument('--out', default=None,
                    help='Output dir for the .md tables (default: <sweep>/summary).')
    args = ap.parse_args()

    md = _build_tables(args.sweep)
    print(md)
    out_dir = args.out or os.path.join(args.sweep, 'summary')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ctx_sweep_pehe_l1.md')
    with open(out_path, 'w') as f:
        f.write(md)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
