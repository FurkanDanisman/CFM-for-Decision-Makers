"""Two-block IHDP summary.

Block 1 (Table 3-style row):  sqrt(PEHE)  and  eps_ATE  for one method,
mean +/- std across realizations. Reported in raw Y units to match Table 3.

Block 2 (Density L2):  marginal L2 (p_y0, p_y1), CATE L2 (p_tau), and
ATE L2 (p_ate) for Do-PFN and one Ours method, mean +/- std.

Nothing else. Two blocks, minimal.

Usage
-----
    python R-PFN/benchmarks/l2_ihdp/summary_ihdp.py \\
        --ours-shards-glob "/scratch/furkanbd/rpfn_bench_kit/l2_ihdp/out_dopfn_bb.r*.npz" \\
        --dopfn-shards-glob "/scratch/furkanbd/rpfn_bench_kit/l2_ihdp/out.r*.npz" \\
        --ours-key ours_dopfn_bb \\
        --ours-label "Ours(DoPFN-bb)"
"""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np


def _agg(vals):
    arr = np.array(vals)
    return arr.mean(), arr.std(ddof=1) if arr.size > 1 else 0.0, arr.size


def _load_pehe_eps(shards, key):
    p, e = [], []
    for path in shards:
        with np.load(path) as f:
            if f'{key}__pehe' not in f.files:
                continue
            p.append(float(f[f'{key}__pehe']))
            e.append(float(f[f'{key}__eps_ate']))
    return p, e


def _load_l2(shards, key):
    y0, y1, tau, ate = [], [], [], []
    for path in shards:
        with np.load(path) as f:
            if f'{key}__l2_y0' not in f.files:
                continue
            y0.extend(np.atleast_1d(f[f'{key}__l2_y0']).tolist())
            y1.extend(np.atleast_1d(f[f'{key}__l2_y1']).tolist())
            tau.extend(np.atleast_1d(f[f'{key}__l2_tau']).tolist())
            ate.append(float(f[f'{key}__l2_ate']))
    return y0, y1, tau, ate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours-shards-glob',  required=True)
    ap.add_argument('--dopfn-shards-glob', required=True)
    ap.add_argument('--ours-key',   default='ours_dopfn_bb',
                    help='method key in the Ours shards (default ours_dopfn_bb)')
    ap.add_argument('--ours-label', default='Ours(DoPFN-bb)')
    args = ap.parse_args()

    ours_shards  = sorted(glob.glob(args.ours_shards_glob))
    dopfn_shards = sorted(glob.glob(args.dopfn_shards_glob))
    if not ours_shards:
        print(f'[fatal] no Ours shards match {args.ours_shards_glob}'); return 2
    if not dopfn_shards:
        print(f'[fatal] no Do-PFN shards match {args.dopfn_shards_glob}'); return 2

    # ── Block 1: Table 3 row for Ours ────────────────────────────────────
    p, e = _load_pehe_eps(ours_shards, args.ours_key)
    if not p:
        print(f'[fatal] no {args.ours_key}__pehe found in Ours shards'); return 2
    pm, ps, n = _agg(p)
    em, es, _ = _agg(e)

    print()
    print('── Table 3 addition — IHDP ────────────────────────────────────────')
    print(f'{"Method":22s} {"sqrt(PEHE)":>18s}   {"eps_ATE":>16s}')
    print('-' * 62)
    print(f'{args.ours_label:22s} {pm:>8.2f} ± {ps:<6.2f}    '
          f'{em:>6.2f} ± {es:<6.2f}    (n={n})')

    # ── Block 2: Density L2 — Do-PFN vs Ours ─────────────────────────────
    print()
    print('── Density L2 — IHDP ──────────────────────────────────────────────')
    print(f'{"Method":22s} {"p(Y_do0)":>13s} {"p(Y_do1)":>13s} '
          f'{"p(CATE)":>13s} {"p(ATE)":>13s}')
    print('-' * 78)
    for shards, key, label in [
        (dopfn_shards, 'dopfn',     'Do-PFN'),
        (ours_shards,  args.ours_key, args.ours_label),
    ]:
        y0, y1, tau, ate = _load_l2(shards, key)
        if not y0:
            print(f'{label:22s}  (no {key}__l2_* found in these shards)')
            continue
        y0m, y0s, _ = _agg(y0)
        y1m, y1s, _ = _agg(y1)
        tm,  ts,  _ = _agg(tau)
        am,  as_, na = _agg(ate)
        print(f'{label:22s} '
              f'{y0m:5.3f}±{y0s:5.3f} {y1m:5.3f}±{y1s:5.3f} '
              f'{tm:5.3f}±{ts:5.3f} {am:5.3f}±{as_:5.3f}   (ATE n={na})')

    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
