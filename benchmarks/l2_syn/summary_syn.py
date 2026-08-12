"""Two-block summary for linear-Gaussian synthetic density L2."""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np


def _agg(vals):
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float('nan'), float('nan'), 0
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0), arr.size


def _load_pehe_eps(shards, key):
    p_m, e_m, p_r, e_r = [], [], [], []
    p_em_mix, e_em_mix = [], []
    p_em_k1,  e_em_k1  = [], []
    for path in shards:
        with np.load(path) as f:
            if f'{key}__pehe' not in f.files: continue
            p_m.append(float(f[f'{key}__pehe']))
            e_m.append(float(f[f'{key}__eps_ate']))
            if f'{key}__pehe_raw' in f.files:
                p_r.append(float(f[f'{key}__pehe_raw']))
                e_r.append(float(f[f'{key}__eps_ate_raw']))
            if f'{key}__pehe_em_mix' in f.files:
                p_em_mix.append(float(f[f'{key}__pehe_em_mix']))
                e_em_mix.append(float(f[f'{key}__eps_ate_em_mix']))
            if f'{key}__pehe_em_k1' in f.files:
                p_em_k1.append(float(f[f'{key}__pehe_em_k1']))
                e_em_k1.append(float(f[f'{key}__eps_ate_em_k1']))
    return (p_m, e_m, p_r, e_r, p_em_mix, e_em_mix, p_em_k1, e_em_k1)


def _load_l2(shards, key):
    y0, y1, tau, ate = [], [], [], []
    for path in shards:
        with np.load(path) as f:
            if f'{key}__l2_y0' not in f.files: continue
            y0.extend(np.atleast_1d(f[f'{key}__l2_y0']).tolist())
            y1.extend(np.atleast_1d(f[f'{key}__l2_y1']).tolist())
            tau.extend(np.atleast_1d(f[f'{key}__l2_tau']).tolist())
            ate.append(float(f[f'{key}__l2_ate']))
    return y0, y1, tau, ate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards-glob', required=True)
    args = ap.parse_args()
    shards = sorted(glob.glob(args.shards_glob))
    if not shards:
        print(f'[fatal] no shards match {args.shards_glob}'); return 2

    METHODS = [
        ('Do-PFN',                     'dopfn'),
        ('Ours(DoPFN-bb 200K)',        'ours_dopfn_bb'),
        ('Ours(fn=50)',                'ours_fn50'),
    ]

    print()
    print(f'── Linear-Gaussian SCM (N=500, d=5, seeds={len(shards)}) ────────')
    print(f'{"Method":30s} {"sqrt(PEHE)":>16s}   {"eps_ATE":>14s}')
    print('-' * 78)
    for label, key in METHODS:
        (p_m, e_m, p_r, e_r,
         p_mix, e_mix, p_k1, e_k1) = _load_pehe_eps(shards, key)
        if not p_m: continue
        pm, ps, n = _agg(p_m); em, es, _ = _agg(e_m)
        suffix = ' (MALC-CATE-mean)' if p_r else ''
        print(f'{label + suffix:32s} {pm:>6.2f} ± {ps:<6.2f}   '
              f'{em:>5.2f} ± {es:<5.2f}   (n={n})')
        if p_r:
            pm_r, ps_r, _ = _agg(p_r); em_r, es_r, _ = _agg(e_r)
            print(f'{label + " (Raw-mean)":32s} {pm_r:>6.2f} ± {ps_r:<6.2f}   '
                  f'{em_r:>5.2f} ± {es_r:<5.2f}   (n={len(p_r)})')
        if p_k1:
            pm_k, ps_k, _ = _agg(p_k1); em_k, es_k, _ = _agg(e_k1)
            print(f'{label + " (EM-mean-K1)":32s} {pm_k:>6.2f} ± {ps_k:<6.2f}   '
                  f'{em_k:>5.2f} ± {es_k:<5.2f}   (n={len(p_k1)})')
        if p_mix:
            pm_x, ps_x, _ = _agg(p_mix); em_x, es_x, _ = _agg(e_mix)
            print(f'{label + " (EM-mean-Kselection)":32s} {pm_x:>6.2f} ± {ps_x:<6.2f}   '
                  f'{em_x:>5.2f} ± {es_x:<5.2f}   (n={len(p_mix)})')

    print()
    print('── Density L2 — Linear-Gaussian SCM ──────────────────────────────')
    print(f'{"Method":30s} {"p(Y_do0)":>13s} {"p(Y_do1)":>13s} '
          f'{"p(CATE)":>13s} {"p(ATE)":>13s}')
    print('-' * 88)
    for label, key in METHODS:
        y0, y1, tau, ate = _load_l2(shards, key)
        if not y0: continue
        y0m, y0s, _ = _agg(y0); y1m, y1s, _ = _agg(y1)
        tm, ts, _ = _agg(tau); am, as_, na = _agg(ate)
        print(f'{label:30s} '
              f'{y0m:5.3f}±{y0s:5.3f} {y1m:5.3f}±{y1s:5.3f} '
              f'{tm:5.3f}±{ts:5.3f} {am:5.3f}±{as_:5.3f}   (ATE n={na})')
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
