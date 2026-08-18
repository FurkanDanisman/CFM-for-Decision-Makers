"""FOR_LUKE.md-style L2 (+ optional KL) summary.

Reproduces the per-metric block layout used in FOR_LUKE.md:

    ── L2 y0  (mean ± SEM) ──
                          Method-A          Method-B          Method-C
                          X ± X             X ± X             X ± X

    ── L2 y1  (mean ± SEM) ──
    ...

Reads whatever methods have `{key}__l2_y0` etc. in the shards. Prefers
this order by default:

    Ours(fn=50), Ours(DoPFN-bb), Do-PFN, UWYK-NoAnc, UWYK-FullAnc

Add --with-kl to also print KL_fwd and KL_rev blocks (KL is stored under
`{key}__kl_y0_fwd` etc. in every shard produced by eval_realization.py).

Usage:
    python summary_like_luke.py \\
        --shards-glob "$DEPLOY_ROOT/ihdp_l2_dopfnbb_j10_s150k/out.r*.npz" \\
        --title "IHDP L2  n=100 realizations, density L2 (mean ± SEM, lower is better)" \\
        --with-kl
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np


# Default order + labels. Skipped silently if the key doesn't appear in any shard.
METHOD_ORDER_DEFAULT = ['ours_fn50', 'ours_dopfn_bb', 'dopfn', 'uwyk_noanc', 'uwyk_anc']
LABELS_DEFAULT = {
    'ours_fn50':     'Ours(fn=50) MALC-1D',
    'ours_dopfn_bb': 'Ours(DoPFN-bb) MALC-1D',
    'dopfn':         'Do-PFN',
    'uwyk_noanc':    'UWYK-NoAnc',
    'uwyk_anc':      'UWYK-FullAnc',
}


def _agg(vals):
    """Mean, SEM, n from a list of floats/arrays."""
    arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('nan'), float('nan'), 0
    m = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return m, sem, arr.size


def _load_metric(shards, key_prefix, metric_suffix):
    """Load metric_suffix values across all shards for key_prefix (a method).
    metric_suffix ∈ {'l2_y0', 'l2_y1', 'l2_tau', 'l2_ate', 'kl_y0_fwd', ...}."""
    per_query = ['l2_y0', 'l2_y1', 'l2_tau', 'kl_y0_fwd', 'kl_y0_rev',
                 'kl_y1_fwd', 'kl_y1_rev', 'kl_tau_fwd', 'kl_tau_rev']
    scalar = ['l2_ate', 'kl_ate_fwd', 'kl_ate_rev']
    all_vals = []
    for path in shards:
        with np.load(path) as f:
            k = f'{key_prefix}__{metric_suffix}'
            if k not in f.files:
                continue
            v = np.asarray(f[k])
            if metric_suffix in per_query:
                all_vals.extend(v.reshape(-1).tolist())
            elif metric_suffix in scalar:
                all_vals.append(float(v))
            else:
                all_vals.extend(np.atleast_1d(v).reshape(-1).tolist())
    return all_vals


def _print_block(title, methods, method_stats, col_width=30):
    print(f'── {title} ──')
    hdr = ' ' * 3
    row = ' ' * 3
    for m in methods:
        label = m['label']
        val, sem, n = method_stats[m['key']]
        cell = f'{val:8.4f} ± {sem:8.4f}' if np.isfinite(val) else '     —      '
        hdr += f'{label:>{col_width}}  '
        row += f'{cell:>{col_width}}  '
    print(hdr)
    print(row)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards-glob', required=True)
    ap.add_argument('--methods', default=','.join(METHOD_ORDER_DEFAULT),
                    help='Comma-separated method KEYS. Only keys present in the shards '
                         'are shown.')
    ap.add_argument('--title', default='',
                    help='Optional title line printed before the L2 blocks.')
    ap.add_argument('--with-kl', action='store_true',
                    help='Also print KL_fwd and KL_rev blocks.')
    ap.add_argument('--col-width', type=int, default=30,
                    help='Column width per method (default 30).')
    args = ap.parse_args()

    shards = sorted(glob.glob(args.shards_glob))
    if not shards:
        sys.exit(f'[fatal] no shards match {args.shards_glob}')

    requested_keys = [m.strip() for m in args.methods.split(',') if m.strip()]

    # Prune to methods actually present in the shards. Check the first shard.
    with np.load(shards[0]) as f0:
        present_keys = {k.split('__', 1)[0] for k in f0.files if '__l2_y0' in k}
    methods = []
    for k in requested_keys:
        if k in present_keys:
            methods.append({'key': k, 'label': LABELS_DEFAULT.get(k, k)})
    if not methods:
        sys.exit(f'[fatal] none of the requested methods present in shards. '
                  f'Available: {sorted(present_keys)}')

    print()
    if args.title:
        print(args.title)
        print()

    # ── L2 blocks ─────────────────────────────────────────
    for suffix, label in [('l2_y0',  'L2 y0  (mean ± SEM, lower is better)'),
                            ('l2_y1',  'L2 y1  (mean ± SEM, lower is better)'),
                            ('l2_tau', 'L2 tau (mean ± SEM, lower is better)'),
                            ('l2_ate', 'L2 ate (mean ± SEM, lower is better)')]:
        stats = {m['key']: _agg(_load_metric(shards, m['key'], suffix))
                 for m in methods}
        _print_block(label, methods, stats, col_width=args.col_width)

    if args.with_kl:
        for suffix, label in [('kl_y0_fwd', 'KL_fwd y0  (mean ± SEM, lower is better)'),
                                ('kl_y1_fwd', 'KL_fwd y1'),
                                ('kl_tau_fwd', 'KL_fwd tau'),
                                ('kl_ate_fwd', 'KL_fwd ate'),
                                ('kl_y0_rev', 'KL_rev y0'),
                                ('kl_y1_rev', 'KL_rev y1'),
                                ('kl_tau_rev', 'KL_rev tau'),
                                ('kl_ate_rev', 'KL_rev ate')]:
            stats = {m['key']: _agg(_load_metric(shards, m['key'], suffix))
                     for m in methods}
            _print_block(label, methods, stats, col_width=args.col_width)


if __name__ == '__main__':
    main()
