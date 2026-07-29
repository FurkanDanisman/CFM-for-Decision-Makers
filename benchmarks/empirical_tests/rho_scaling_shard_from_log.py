"""Parse a rho_scaling_test .out log for [scm] lines and save them as
per-ρ shard npz files that the resume path in rho_scaling_test.py can
pick up. Used to reuse SCMs completed by a job that timed out before
writing its aggregated npz.

Usage
-----
  python rho_scaling_shard_from_log.py --log LOG.out --out-prefix rho_scaling_test

Produces `rho_scaling_test.png.rho<i>.npz` for every ρ index present in
the log. Downstream: submit the array; each task's resume path skips
the k indices already in its shard.
"""
from __future__ import annotations
import argparse, os, re
import numpy as np

RHO_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)

# ρ=0.00 k=15 ρ̂=+0.032  UWYK=2.003 DoPFN=2.003 Ours50=2.294 Ours10=2.780
_LINE = re.compile(
    r'^\[scm\]\s+ρ=(?P<rho>[-+\d.]+)\s+k=(?P<k>\d+)\s+ρ̂=(?P<rhoh>[-+.\d]+)\s+'
    r'UWYK=(?P<uwyk>[-+\d.eE]+)\s+DoPFN=(?P<dopfn>[-+\d.eE]+)\s+'
    r'Ours50=(?P<o50>[-+\d.eE]+)\s+Ours10=(?P<o10>[-+\d.eE]+)'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log',        required=True,
                    help='.out log from a partial rho_scaling_test run')
    ap.add_argument('--out-prefix', default='rho_scaling_test.png',
                    help='write to <prefix>.rho<i>.npz')
    args = ap.parse_args()

    rows_by_rho: dict[float, list] = {rho: [] for rho in RHO_GRID}
    with open(args.log) as fp:
        for line in fp:
            m = _LINE.match(line.rstrip('\n'))
            if not m: continue
            rho = float(m.group('rho'))
            # snap to nearest grid entry
            rho_g = min(RHO_GRID, key=lambda r: abs(r - rho))
            if abs(rho_g - rho) > 1e-3: continue
            k = int(m.group('k'))
            seed = int(rho_g * 10_000) + k
            rows_by_rho[rho_g].append(dict(
                rho=rho_g, rho_hat=float(m.group('rhoh')), seed=seed,
                pehe_uwyk=float(m.group('uwyk')),
                pehe_dopfn=float(m.group('dopfn')),
                pehe_ours50=float(m.group('o50')),
                pehe_ours10=float(m.group('o10')),
            ))

    for idx, rho in enumerate(RHO_GRID):
        rows = rows_by_rho[rho]
        if not rows: continue
        out = f'{args.out_prefix}.rho{idx}.npz'
        # merge with any existing shard, dedup by (rho, seed)
        existing = []
        if os.path.exists(out):
            with np.load(out, allow_pickle=True) as f:
                for i in range(len(f['seed'])):
                    existing.append(dict(rho=float(f['rho'][i]),
                                          rho_hat=float(f['rho_hat'][i]),
                                          seed=int(f['seed'][i]),
                                          pehe_uwyk=float(f['pehe_uwyk'][i]),
                                          pehe_dopfn=float(f['pehe_dopfn'][i]),
                                          pehe_ours50=float(f['pehe_ours50'][i]),
                                          pehe_ours10=float(f['pehe_ours10'][i])))
        by_seed = {r['seed']: r for r in existing}
        for r in rows: by_seed[r['seed']] = r
        merged = sorted(by_seed.values(), key=lambda r: r['seed'])
        keys = list(merged[0].keys())
        arr = {k: np.array([r[k] for r in merged]) for k in keys}
        np.savez(out, **arr)
        print(f'[save] {out}  ({len(merged)} rows for ρ={rho:.2f})')


if __name__ == '__main__':
    main()
