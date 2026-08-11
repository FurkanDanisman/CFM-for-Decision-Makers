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
    """Load PEHE + eps_ATE per shard, returning both MALC-mean (existing) and
    raw-mean (new) variants when available. Raw variant only exists for Ours
    methods; dopfn/uwyk_noanc shards will have empty raw lists."""
    p_malc, e_malc = [], []
    p_raw,  e_raw  = [], []
    for path in shards:
        with np.load(path) as f:
            if f'{key}__pehe' not in f.files:
                continue
            p_malc.append(float(f[f'{key}__pehe']))
            e_malc.append(float(f[f'{key}__eps_ate']))
            if f'{key}__pehe_raw' in f.files:
                p_raw.append(float(f[f'{key}__pehe_raw']))
                e_raw.append(float(f[f'{key}__eps_ate_raw']))
    return p_malc, e_malc, p_raw, e_raw


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
    p_malc, e_malc, p_raw, e_raw = _load_pehe_eps(ours_shards, args.ours_key)
    if not p_malc:
        print(f'[fatal] no {args.ours_key}__pehe found in Ours shards'); return 2
    pm_m, ps_m, n = _agg(p_malc)
    em_m, es_m, _ = _agg(e_malc)

    # Also load Do-PFN's PEHE/eps_ATE (only has "MALC" flavour since it doesn't
    # use MALC — the point estimate is a single method).
    p_dopfn_malc, e_dopfn_malc, _, _ = _load_pehe_eps(dopfn_shards, 'dopfn')
    dopfn_pm = dopfn_ps = dopfn_em = dopfn_es = None
    dopfn_n = 0
    if p_dopfn_malc:
        dopfn_pm, dopfn_ps, dopfn_n = _agg(p_dopfn_malc)
        dopfn_em, dopfn_es, _       = _agg(e_dopfn_malc)

    print()
    print('── Table 3 addition — IHDP ────────────────────────────────────────')
    print(f'{"Method":30s} {"sqrt(PEHE)":>18s}   {"eps_ATE":>16s}')
    print('-' * 70)
    if dopfn_pm is not None:
        print(f'{"Do-PFN":30s} {dopfn_pm:>8.2f} ± {dopfn_ps:<6.2f}    '
              f'{dopfn_em:>6.2f} ± {dopfn_es:<6.2f}    (n={dopfn_n})')
    print(f'{args.ours_label + " (MALC-mean)":30s} '
          f'{pm_m:>8.2f} ± {ps_m:<6.2f}    '
          f'{em_m:>6.2f} ± {es_m:<6.2f}    (n={n})')
    if p_raw:
        pm_r, ps_r, _ = _agg(p_raw)
        em_r, es_r, _ = _agg(e_raw)
        print(f'{args.ours_label + " (raw-mean)":30s} '
              f'{pm_r:>8.2f} ± {ps_r:<6.2f}    '
              f'{em_r:>6.2f} ± {es_r:<6.2f}    (n={len(p_raw)})')

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
