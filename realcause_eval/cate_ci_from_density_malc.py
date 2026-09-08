"""Aggregator for 2D-MALC CI dumps written by compute_malc_ci_cell.py.

Reads malc_ci_r<###>.npz files from each (method, dataset) cell and produces
a markdown table with PEHE / ε_ATE (stored point CATE from the mega-sbatch)
alongside 2D-MALC CI Coverage / Length.

PEHE / ε_ATE come from the ORIGINAL density NPZ (same values as
cate_ci_from_density.py). The MALC CI is on the MALC-smoothed p(τ) — a
different distribution than the raw joint — so its density mean can differ
from the stored point CATE by MALC's documented smoothing bias (~0.04-0.15
L2 on synthetic, per density_calc.md §7).

Layouts:
  inline (cpfn2d, graph2d): <method_dir>/<DATASET>/<DATASET>_r<###>.npz
                            <method_dir>/<DATASET>/malc_ci_r<###>.npz
  split  (dopfnbb):         <method_dir>/PSIDbal/density_r<###>.npz
                            <method_dir>/PSIDbal/malc_ci_r<###>.npz
                            <method_dir>/PSIDbal/summary.npz (pehe[], eps_ate[])

Usage:
    python realcause_eval/cate_ci_from_density_malc.py \\
        --out-root /scratch/.../results_rc_2d_density \\
        --methods cpfn2d graph2d dopfnbb
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


DATASETS = ('IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal')

# Point-CATE key resolution — same table as cate_ci_from_density.py.
_POINT_KEYS = {
    'cpfn2d':  ('pehe_raw',       'err_raw'),
    'graph2d': ('pehe_raw_noanc', 'err_raw_noanc'),
    'dopfnbb': ('pehe',           'eps_ate'),
}

_SPLIT_METHODS = {'dopfnbb'}
_DATASET_DIR_ALIASES = {
    'dopfnbb': {'PSID_bal': 'PSIDbal'},
}

_ALPHA = float(os.environ.get('ALPHA', '0.05'))   # 0.05 → 95% CI, 0.01 → 99% CI


def _winkler_is_vec(lo, hi, y, alpha=_ALPHA):
    """Vectorized IS_α over queries. Shapes (N,), (N,), (N,) → (N,)."""
    length = hi - lo
    return length + (2.0 / alpha) * np.maximum(lo - y, 0.0) \
                  + (2.0 / alpha) * np.maximum(y - hi, 0.0)


def _quantile_from_density_pq(p_pq, tau, level):
    """Linear-interp inverse-CDF at `level` per query.
    p_pq (N_q, T), tau (T,) uniform. Returns (N_q,)."""
    dtau = float(tau[1] - tau[0])
    cdf = np.cumsum(p_pq, axis=-1) * dtau
    cdf = cdf / cdf[:, -1:].clip(min=1e-12)
    below = cdf < level
    idx = np.argmax(~below, axis=-1)
    idx = np.where(cdf[..., -1] < level, cdf.shape[-1] - 1, idx)
    row = np.arange(cdf.shape[0])
    c_hi = cdf[row, idx]
    c_lo = np.where(idx > 0, cdf[row, np.maximum(idx - 1, 0)], 0.0)
    t_hi = tau[idx]
    t_lo = np.where(idx > 0, tau[np.maximum(idx - 1, 0)], tau[0])
    w = np.where(c_hi > c_lo, (level - c_lo) / (c_hi - c_lo), 0.0)
    return t_lo + w * (t_hi - t_lo)


def _crps_from_densities(p_tau, tau, y):
    """CRPS per query. p_tau: (N_q, T), tau: (T,) [scaled or raw], y: (N_q,).
    Returns CRPS in the SAME units as tau. Caller applies y_scale to convert.
    CRPS(F, y) = Σ (F(τ) − 1[τ ≥ y])² · dτ on a uniform τ grid."""
    dtau = float(tau[1] - tau[0])
    cdf = np.cumsum(p_tau, axis=-1) * dtau
    cdf = cdf / cdf[:, -1:].clip(min=1e-12)
    step = (tau[None, :] >= y[:, None]).astype(np.float64)
    return np.sum((cdf - step) ** 2, axis=-1) * dtau


def _mean_se(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float('nan'), float('nan')
    m  = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float('nan')
    return m, se


def _fmt(m, se, big=False):
    if not np.isfinite(m): return '—'
    if big:
        m_s  = f'{m:,.2f}'
        se_s = f'{se:,.2f}' if np.isfinite(se) else '—'
    else:
        m_s  = f'{m:.3f}'
        se_s = f'{se:.3f}' if np.isfinite(se) else '—'
    return f'{m_s} ± {se_s}'


def _load_dopfnbb_summary(dataset_dir, pehe_key, err_key):
    path = os.path.join(dataset_dir, 'summary.npz')
    if not os.path.isfile(path):
        return None, None
    try:
        with np.load(path, allow_pickle=True) as z:
            pehe_arr = np.asarray(z[pehe_key], dtype=np.float64) if pehe_key in z.files else None
            err_arr  = np.asarray(z[err_key],  dtype=np.float64) if err_key  in z.files else None
        return pehe_arr, err_arr
    except Exception as e:
        print(f'  [warn] {path}: {e}', file=sys.stderr)
        return None, None


def summarize_cell(method_dir, dataset, method, pehe_key, err_key, in_tag=''):
    dataset_dir_name = _DATASET_DIR_ALIASES.get(method, {}).get(dataset, dataset)
    dataset_dir      = os.path.join(method_dir, dataset_dir_name)
    is_split = (method in _SPLIT_METHODS)
    tag_prefix = f'{in_tag}_' if in_tag else ''
    malc_paths = sorted(glob.glob(os.path.join(dataset_dir, f'malc_ci_{tag_prefix}r*.npz')))
    if not malc_paths:
        return None
    # Inline path also holds the point CATE; split path pulls from summary.npz.
    if is_split:
        pehe_arr, err_arr = _load_dopfnbb_summary(dataset_dir, pehe_key, err_key)
    else:
        pehe_arr = err_arr = None

    pehes, errs, covs, lens = [], [], [], []
    winklers, crpses = [], []
    ate_diffs, fail_frac = [], []
    for r_idx, mp in enumerate(malc_paths):
        # Pair each malc_ci_<tag>r<xxx>.npz with its density sibling for the point row.
        base   = os.path.basename(mp)
        # Strip 'malc_ci_' and optional '{tag}_' prefix to isolate r<###>.
        stem   = base[len('malc_ci_'):-len('.npz')]
        r_tag  = stem[len(tag_prefix):] if in_tag else stem
        if is_split:
            density_sib = os.path.join(dataset_dir, f'density_{r_tag}.npz')
        else:
            density_sib = os.path.join(dataset_dir, f'{dataset_dir_name}_{r_tag}.npz')
        try:
            with np.load(mp, allow_pickle=True) as z:
                ate_malc  = float(z['ate_malc'])
                fails     = int(z['n_fails']) if 'n_fails' in z.files else 0
                N_q       = int(z['true_cate_per_query'].shape[0])
                # Winkler IS + CRPS per query (raw units).
                true_pq       = np.asarray(z['true_cate_per_query'], dtype=np.float64)
                p_taus_scaled = np.asarray(z['p_taus_scaled'], dtype=np.float64)  # (N_q, T) scaled
                tau_scaled    = np.asarray(z['tau_scaled'],    dtype=np.float64)
                y_scale       = float(z['y_scale'])
        except Exception as e:
            print(f'  [warn] {mp}: {e}', file=sys.stderr)
            continue
        # Recompute CI at env-var ALPHA (defaults 0.05). Level lo/hi.
        _lo, _hi = _ALPHA / 2.0, 1.0 - _ALPHA / 2.0
        tau_lo_pq = _quantile_from_density_pq(p_taus_scaled, tau_scaled, _lo) * y_scale
        tau_hi_pq = _quantile_from_density_pq(p_taus_scaled, tau_scaled, _hi) * y_scale
        cov       = float(np.mean((true_pq >= tau_lo_pq) & (true_pq <= tau_hi_pq)))
        length    = float(np.mean(tau_hi_pq - tau_lo_pq))
        covs.append(cov); lens.append(length)
        fail_frac.append(fails / max(N_q, 1))
        # IS per query (already in raw units since tau_lo/hi are raw).
        is_pq = _winkler_is_vec(tau_lo_pq, tau_hi_pq, true_pq)
        winklers.append(float(is_pq.mean()))
        # CRPS per query on scaled grid, then multiply by y_scale → raw CRPS.
        if os.environ.get('SKIP_CRPS', '0') == '1':
            crpses.append(float('nan'))
        else:
            crps_pq_scaled = _crps_from_densities(p_taus_scaled, tau_scaled,
                                                    true_pq / max(y_scale, 1e-12))
            crps_pq_raw = crps_pq_scaled * y_scale
            crpses.append(float(crps_pq_raw.mean()))

        # Stored PEHE / ε_ATE.
        if is_split:
            if pehe_arr is not None and r_idx < pehe_arr.size:
                pehes.append(float(pehe_arr[r_idx]))
            if err_arr is not None and r_idx < err_arr.size:
                errs.append(float(err_arr[r_idx]))
        else:
            if os.path.isfile(density_sib):
                try:
                    with np.load(density_sib, allow_pickle=True) as z_d:
                        if pehe_key in z_d.files and np.isfinite(z_d[pehe_key]):
                            pehes.append(float(z_d[pehe_key]))
                        if err_key in z_d.files and np.isfinite(z_d[err_key]):
                            errs.append(float(z_d[err_key]))
                        # Density sanity: MALC-mean vs stored point ate.
                        for k in ('ate_raw', 'ate_em', 'ate_dopfn', 'ate_full', 'ate_pred'):
                            if k in z_d.files:
                                ate_diffs.append(ate_malc - float(z_d[k]))
                                break
                except Exception as e:
                    print(f'  [warn] {density_sib}: {e}', file=sys.stderr)

    if not covs:
        return None
    return {
        'pehe':          _mean_se(pehes),
        'err':           _mean_se(errs),
        'cov':           _mean_se(covs),
        'len':           _mean_se(lens),
        'winkler':       _mean_se(winklers),
        'crps':          _mean_se(crpses),
        'n':             len(covs),
        'ate_bias_max':  float(np.max(np.abs(ate_diffs))) if ate_diffs else float('nan'),
        'fail_frac_max': float(np.max(fail_frac)) if fail_frac else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', required=True,
                    help='Parent dir; expects <out-root>/<method>/<DATASET>/malc_ci_r<###>.npz.')
    ap.add_argument('--methods', nargs='+',
                    default=['cpfn2d', 'graph2d', 'dopfnbb'],
                    help='Methods to include (one row each).')
    ap.add_argument('--datasets', nargs='+', default=None,
                    choices=list(DATASETS),
                    help='Restrict to a subset of datasets (default: all 5).')
    ap.add_argument('--graph2d-tag', default='noanc',
                    help='graph2d anc-tag for the PEHE / ε_ATE columns.')
    ap.add_argument('--dopfnbb-pehe-key', default='pehe', choices=['pehe', 'pehe_em'])
    ap.add_argument('--in-tag', default='',
                    help='Read tagged malc_ci_{tag}_r<###>.npz files (matches '
                         "compute_malc_ci_cell.py --out-tag). Default '' → untagged.")
    ap.add_argument('--out-md', default=None,
                    help='Also write the markdown table to this path.')
    args = ap.parse_args()

    if not os.path.isdir(args.out_root):
        sys.exit(f'FATAL: --out-root not found: {args.out_root}')

    point_keys = dict(_POINT_KEYS)
    point_keys['graph2d'] = (f'pehe_raw_{args.graph2d_tag}',
                              f'err_raw_{args.graph2d_tag}')
    point_keys['dopfnbb'] = (args.dopfnbb_pehe_key,
                              'eps_ate' if args.dopfnbb_pehe_key == 'pehe' else 'eps_ate_em')

    big  = {'CPS', 'PSID', 'PSID_bal'}
    dsets = tuple(args.datasets) if args.datasets else DATASETS

    header = '| Method | ' + ' | '.join(dsets) + ' |'
    sep    = '|' + '|'.join(['---'] * (1 + len(dsets))) + '|'
    lines = [
        f'\nRealCause density-CI, 2D-MALC-smoothed p(τ) — {args.out_root}',
        '',
        '(each cell, top → bottom: √PEHE / ε_ATE (from stored point CATE — '
        'matches realcause_eval mega-sbatch), Coverage / Length (95% CI from '
        'the 2D-MALC-smoothed p(τ), NOT the raw joint), Winkler IS_0.05 '
        '(length + 40·miss) and CRPS — both single-number combined '
        'coverage+length metrics, lower = better; n = realizations)',
        '',
        header, sep,
    ]
    sanity_lines = ['', '## Sanity — |mean(MALC p(τ)) − stored point ate| max, MALC fit-fail fraction max',
                    '(expect small non-zero bias from MALC smoothing; fit-fail should be ~0)',
                    '', header, sep]

    import time
    for method in args.methods:
        cells = [method]; sanity_cells = [method]
        method_dir = os.path.join(args.out_root, method)
        pehe_key, err_key = point_keys.get(method, ('pehe_raw', 'err_raw'))
        for d in dsets:
            _t0 = time.time()
            print(f'[{time.strftime("%H:%M:%S")}] {method:8s} / {d:9s} ...',
                  end='', flush=True)
            got = summarize_cell(method_dir, d, method, pehe_key, err_key,
                                   in_tag=args.in_tag)
            print(f' done in {time.time() - _t0:5.1f}s'
                  + (f' (n={got["n"]})' if got is not None else ' (no data)'),
                  flush=True)
            if got is None:
                cells.append('—'); sanity_cells.append('—'); continue
            pehe_s = _fmt(*got['pehe'], big=d in big)
            err_s  = _fmt(*got['err'],  big=False)
            cov_s  = _fmt(*got['cov'],  big=False)
            len_s  = _fmt(*got['len'],  big=d in big)
            is_s   = _fmt(*got['winkler'], big=d in big)
            crps_s = _fmt(*got['crps'],    big=d in big)
            n      = got['n']
            cells.append(
                f'PEHE {pehe_s}<br>'
                f'ε_ATE {err_s}<br>'
                f'Cov {cov_s}<br>'
                f'Len {len_s}<br>'
                f'IS {is_s}<br>'
                f'CRPS {crps_s} (n={n})'
            )
            bias = got['ate_bias_max']
            fail = got['fail_frac_max']
            bias_s = (f'{bias:.2e}' if abs(bias) < 1 else f'{bias:.4f}') if np.isfinite(bias) else '—'
            sanity_cells.append(f'|Δ| max = {bias_s}  fails≤{fail:.1%}')
        lines.append('| ' + ' | '.join(cells) + ' |')
        sanity_lines.append('| ' + ' | '.join(sanity_cells) + ' |')

    lines.extend(sanity_lines)
    md = '\n'.join(lines) + '\n'
    print(md)
    if args.out_md:
        with open(args.out_md, 'w') as f:
            f.write(md)
        print(f'wrote {args.out_md}')


if __name__ == '__main__':
    main()
