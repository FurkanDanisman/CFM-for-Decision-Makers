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
    """Load PEHE + eps_ATE per shard for four flavours (each optional):
      1. MALC (density-integrated)   __pehe / __eps_ate            (always for Ours)
      2. raw   (E[Y1]-E[Y0] marginals) __pehe_raw / __eps_ate_raw   (Ours after cate_raw_scaled shipped)
      3. EM mixture (fit.pi-weighted μ_hat) __pehe_em_mix / __eps_ate_em_mix
      4. EM K=1   (single-log-concave μ_hat) __pehe_em_k1  / __eps_ate_em_k1
    """
    p_malc,   e_malc   = [], []
    p_raw,    e_raw    = [], []
    p_em_mix, e_em_mix = [], []
    p_em_k1,  e_em_k1  = [], []
    for path in shards:
        with np.load(path) as f:
            if f'{key}__pehe' not in f.files:
                continue
            p_malc.append(float(f[f'{key}__pehe']))
            e_malc.append(float(f[f'{key}__eps_ate']))
            if f'{key}__pehe_raw' in f.files:
                p_raw.append(float(f[f'{key}__pehe_raw']))
                e_raw.append(float(f[f'{key}__eps_ate_raw']))
            if f'{key}__pehe_em_mix' in f.files:
                p_em_mix.append(float(f[f'{key}__pehe_em_mix']))
                e_em_mix.append(float(f[f'{key}__eps_ate_em_mix']))
            if f'{key}__pehe_em_k1' in f.files:
                p_em_k1.append(float(f[f'{key}__pehe_em_k1']))
                e_em_k1.append(float(f[f'{key}__eps_ate_em_k1']))
    return (p_malc, e_malc, p_raw, e_raw,
            p_em_mix, e_em_mix, p_em_k1, e_em_k1)


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
    (p_malc, e_malc, p_raw, e_raw,
     p_em_mix, e_em_mix, p_em_k1, e_em_k1) = _load_pehe_eps(ours_shards, args.ours_key)
    if not p_malc:
        print(f'[fatal] no {args.ours_key}__pehe found in Ours shards'); return 2
    pm_m, ps_m, n = _agg(p_malc)
    em_m, es_m, _ = _agg(e_malc)

    p_dopfn_malc, e_dopfn_malc, *_ = _load_pehe_eps(dopfn_shards, 'dopfn')
    dopfn_pm = dopfn_ps = dopfn_em = dopfn_es = None
    dopfn_n = 0
    if p_dopfn_malc:
        dopfn_pm, dopfn_ps, dopfn_n = _agg(p_dopfn_malc)
        dopfn_em, dopfn_es, _       = _agg(e_dopfn_malc)

    print()
    print('── Table 3 addition — IHDP ────────────────────────────────────────')
    print(f'{"Method":32s} {"sqrt(PEHE)":>18s}   {"eps_ATE":>16s}')
    print('-' * 74)
    if dopfn_pm is not None:
        print(f'{"Do-PFN":32s} {dopfn_pm:>8.2f} ± {dopfn_ps:<6.2f}    '
              f'{dopfn_em:>6.2f} ± {dopfn_es:<6.2f}    (n={dopfn_n})')
    print(f'{args.ours_label + " (MALC-mean, density)":32s} '
          f'{pm_m:>8.2f} ± {ps_m:<6.2f}    '
          f'{em_m:>6.2f} ± {es_m:<6.2f}    (n={n})')
    if p_raw:
        pm_r, ps_r, _ = _agg(p_raw)
        em_r, es_r, _ = _agg(e_raw)
        print(f'{args.ours_label + " (raw-mean)":32s} '
              f'{pm_r:>8.2f} ± {ps_r:<6.2f}    '
              f'{em_r:>6.2f} ± {es_r:<6.2f}    (n={len(p_raw)})')
    if p_em_mix:
        pmix_m, pmix_s, _ = _agg(p_em_mix)
        emix_m, emix_s, _ = _agg(e_em_mix)
        print(f'{args.ours_label + " (EM mixture)":32s} '
              f'{pmix_m:>8.2f} ± {pmix_s:<6.2f}    '
              f'{emix_m:>6.2f} ± {emix_s:<6.2f}    (n={len(p_em_mix)})')
    if p_em_k1:
        pk1_m, pk1_s, _ = _agg(p_em_k1)
        ek1_m, ek1_s, _ = _agg(e_em_k1)
        print(f'{args.ours_label + " (EM K=1)":32s} '
              f'{pk1_m:>8.2f} ± {pk1_s:<6.2f}    '
              f'{ek1_m:>6.2f} ± {ek1_s:<6.2f}    (n={len(p_em_k1)})')

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
