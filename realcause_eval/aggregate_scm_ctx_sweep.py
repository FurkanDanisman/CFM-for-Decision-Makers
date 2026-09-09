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


# Each spec: (display_name, dir_name, kind, tag)
#   kind ∈ {'uniform', 'graph2d', 'dopfn_bb'}
#   tag  : for graph2d, which ancestor variant's keys to read ('v3b' | 'noanc');
#          ignored otherwise.
# graph2d writes BOTH v3b and noanc keys in one npz (anc_mode=v3b_only), so
# graph2d_v3b / graph2d_noanc read the same dir with different key suffixes.
# uwyk_v3b reads the v3b run dir (uwyk/); uwyk_noanc reads a separate run
# dir (uwyk_noanc/) that must be produced with ANC_VARIANT=noanc.
MODEL_SPECS = [
    ('dopfn_native', 'dopfn_native', 'uniform',  None),
    ('dopfn_bb',     'dopfn_bb',     'dopfn_bb',  None),
    ('cpfn2d',       'cpfn2d',       'uniform',   None),
    ('cpfn1d',       'cpfn1d',       'uniform',   None),
    ('graph2d_noanc','graph2d',      'graph2d',   'noanc'),
    ('graph2d_v3b',  'graph2d',      'graph2d',   'v3b'),
    ('uwyk_noanc',   'uwyk_noanc',   'uniform',   None),
    ('uwyk_v3b',     'uwyk',         'uniform',   None),
]
CONTEXTS = [50, 100, 250, 500, 1000]
CASES    = ['Observed_Confounder', 'Observed_Mediator',
            'Observed_Mediator_and_Confounder', 'Unobserved_Confounder',
            'Frontdoor_Criterion', 'Backdoor_Criterion']


def _first(z, keys):
    for k in keys:
        if k in z.files:
            return np.asarray(z[k], dtype=np.float64)
    return None


def _cell_pehe_l1(sweep, ctx, spec, case, thr=float('inf')):
    """Return (pehe_array, l1_array, n_dropped) for one (spec, ctx, case).

    spec = (display_name, dir_name, kind, tag). Realizations with
    |true_ATE| > thr are dropped (outlier SCM draws); true_ATE is
    model-independent so the same realizations drop across all specs.
    """
    _name, dir_name, kind, tag = spec
    cell = os.path.join(sweep, f'ctx{ctx}', dir_name, case)
    if not os.path.isdir(cell):
        return None, None, 0

    # ── dopfn_bb: single summary.npz with per-realization arrays.
    if kind == 'dopfn_bb':
        cand = os.path.join(cell, 'summary.npz')
        if not os.path.isfile(cand):
            return None, None, 0
        with np.load(cand, allow_pickle=True) as z:
            pehe = _first(z, ['pehe'])
            ate_pred = _first(z, ['ate_pred'])
            true_ate = _first(z, ['true_ate'])
        if pehe is None:
            return None, None, 0
        l1 = (np.abs(ate_pred - true_ate) if (ate_pred is not None and true_ate is not None)
              else None)
        if true_ate is not None:
            keep = np.abs(true_ate) <= thr
            n_drop = int((~keep).sum())
            pehe = pehe[keep]
            if l1 is not None: l1 = l1[keep]
        else:
            n_drop = 0
        return pehe, l1, n_drop

    # ── graph2d: <CASE>_r{NNN}.npz; pick the tag's keys (v3b or noanc).
    if kind == 'graph2d':
        paths = sorted(glob.glob(os.path.join(cell, f'{case}_r*.npz')))
        pe_keys = [f'pehe_raw_{tag}', f'pehe_full_{tag}', f'pehe_em_{tag}']
        at_keys = [f'ate_raw_{tag}',  f'ate_full_{tag}',  f'ate_em_{tag}']
        pehe_l, l1_l, n_drop = [], [], 0
        for p in paths:
            with np.load(p, allow_pickle=True) as z:
                pe = _first(z, pe_keys); at = _first(z, at_keys); tr = _first(z, ['true_ate'])
            if pe is None:
                continue
            if tr is not None and abs(float(tr)) > thr:
                n_drop += 1; continue
            pehe_l.append(float(pe))
            if at is not None and tr is not None:
                l1_l.append(abs(float(at) - float(tr)))
        if not pehe_l:
            return None, None, n_drop
        return np.array(pehe_l), (np.array(l1_l) if l1_l else None), n_drop

    # ── uniform: dopfn_native/uwyk write r{NNN}.npz; cpfn2d/cpfn1d write
    #    {CASE}_r{NNN}.npz (tag includes DATASET). Try both.
    paths = sorted(glob.glob(os.path.join(cell, 'r*.npz')))
    if not paths:
        paths = sorted(glob.glob(os.path.join(cell, f'{case}_r*.npz')))
    pehe_l, l1_l, n_drop = [], [], 0
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            pe = _first(z, ['pehe_raw'])
            at = _first(z, ['ate_pred', 'ate_raw'])
            tr = _first(z, ['true_ate'])
        if pe is None:
            continue
        if tr is not None and abs(float(tr)) > thr:
            n_drop += 1; continue
        pehe_l.append(float(pe))
        if at is not None and tr is not None:
            l1_l.append(abs(float(at) - float(tr)))
    if not pehe_l:
        return None, None, n_drop
    return np.array(pehe_l), (np.array(l1_l) if l1_l else None), n_drop


def _fmt(vals, stat):
    """stat='mean' → mean ± SEM (n); stat='median' → median [Q1–Q3] (n)."""
    if vals is None or len(vals) == 0:
        return '—'
    n = len(vals)
    if stat == 'mean':
        m = float(np.mean(vals))
        sem = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else float('nan')
        sem_s = f'{sem:.3f}' if np.isfinite(sem) else '—'
        return f'{m:.3f} ± {sem_s} (n={n})'
    med = float(np.median(vals))
    q1, q3 = np.percentile(vals, [25, 75])
    return f'{med:.3f} [{q1:.3f}–{q3:.3f}] (n={n})'


def _macro(vals, stat):
    return float(np.mean(vals)) if stat == 'mean' else float(np.median(vals))


def _build_tables(sweep, thr=float('inf')):
    hdr_note = (f' (outliers |true_ATE|>{thr:g} dropped)' if np.isfinite(thr)
                else ' (no outlier filtering)')
    out_blocks = []

    for stat in ('mean', 'median'):
        stat_label = 'MEAN ± SEM' if stat == 'mean' else 'MEDIAN [Q1–Q3]'
        lines_pehe = [f'# PEHE ({stat_label}) — SCM case-study context sweep{hdr_note}\n']
        lines_l1   = [f'# L1-ATE ({stat_label}) — SCM case-study context sweep{hdr_note}\n']

        # Drop-count report (once, at the top of the mean block).
        _dn_spec = MODEL_SPECS[0]   # dopfn_native
        if np.isfinite(thr) and stat == 'mean':
            drop_report = ['\n## Dropped realizations per case (|true_ATE| > '
                           f'{thr:g})\n', '| Case | n_dropped / 100 |', '|---|---|']
            for case in CASES:
                _, _, nd = _cell_pehe_l1(sweep, CONTEXTS[0], _dn_spec, case, thr)
                drop_report.append(f'| {case} | {nd} |')
            lines_pehe = drop_report + [''] + lines_pehe

        for metric, lines, idx in (('PEHE', lines_pehe, 0), ('L1_ATE', lines_l1, 1)):
            for case in CASES:
                lines.append(f'\n## {case}\n')
                header = '| Model | ' + ' | '.join(f'ctx={c}' for c in CONTEXTS) + ' |'
                lines.append(header)
                lines.append('|' + '---|' * (len(CONTEXTS) + 1))
                for spec in MODEL_SPECS:
                    cells = [spec[0]]
                    for ctx in CONTEXTS:
                        pehe, l1, _ = _cell_pehe_l1(sweep, ctx, spec, case, thr)
                        cells.append(_fmt(pehe if idx == 0 else l1, stat))
                    lines.append('| ' + ' | '.join(cells) + ' |')

            # Case-averaged (macro over 6 case studies of each cell's stat).
            lines.append(f'\n## ALL CASES (macro-{stat} over 6 case studies)\n')
            header = '| Model | ' + ' | '.join(f'ctx={c}' for c in CONTEXTS) + ' |'
            lines.append(header); lines.append('|' + '---|' * (len(CONTEXTS) + 1))
            for spec in MODEL_SPECS:
                cells = [spec[0]]
                for ctx in CONTEXTS:
                    per_case = []
                    for case in CASES:
                        pehe, l1, _ = _cell_pehe_l1(sweep, ctx, spec, case, thr)
                        v = pehe if idx == 0 else l1
                        if v is not None and len(v):
                            per_case.append(_macro(v, stat))
                    cells.append(f'{_macro(per_case, stat):.3f}' if per_case else '—')
                lines.append('| ' + ' | '.join(cells) + ' |')

        out_blocks.append('\n'.join(lines_pehe) + '\n\n' + '\n'.join(lines_l1))

    return ('\n\n' + '=' * 70 + '\n\n').join(out_blocks) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', required=True,
                    help='Root holding ctx<N>/<model>/<case>/ dirs.')
    ap.add_argument('--out', default=None,
                    help='Output dir for the .md tables (default: <sweep>/summary).')
    ap.add_argument('--drop-abs-ate-above', type=float, default=None,
                    help='Drop realizations whose |true_ATE| exceeds this. '
                         'Same realizations dropped across all models (true_ATE '
                         'is model-independent). Default: no filtering.')
    args = ap.parse_args()

    thr = args.drop_abs_ate_above if args.drop_abs_ate_above is not None else float('inf')
    md = _build_tables(args.sweep, thr=thr)
    print(md)
    out_dir = args.out or os.path.join(args.sweep, 'summary')
    os.makedirs(out_dir, exist_ok=True)
    tag = f'_drop{args.drop_abs_ate_above:g}' if args.drop_abs_ate_above is not None else '_nofilter'
    out_path = os.path.join(out_dir, f'ctx_sweep_pehe_l1{tag}.md')
    with open(out_path, 'w') as f:
        f.write(md)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
