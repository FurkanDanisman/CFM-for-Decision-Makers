"""Print a per-metric mean±SEM table from a Fig 2 shard (out.rhoN.npz).

Usage:
    python inspect_fig2_shard.py $DEPLOY_ROOT/fig2_pehe_l2_smoke/out.rho4.npz
"""
import os, sys, numpy as np

METHODS = ('uwyk_noanc', 'uwyk_anc', 'dopfn', 'ours_fn50')
LABEL = {
    'uwyk_noanc': 'UWYK-NoAnc', 'uwyk_anc': 'UWYK-FullAnc',
    'dopfn':      'Do-PFN',     'ours_fn50': 'Ours(fn=50)',
}
METRICS = [
    ('pehe',        'PEHE'),
    ('marg_l2',     'Marg-L2'),
    ('marg_kl_fwd', 'Marg-KLfwd'),
    ('marg_kl_rev', 'Marg-KLrev'),
    ('cate_l2',     'CATE-L2'),
    ('cate_kl_fwd', 'CATE-KLfwd'),
    ('cate_kl_rev', 'CATE-KLrev'),
    ('ate_l2',      'ATE-L2'),
    ('ate_kl_fwd',  'ATE-KLfwd'),
    ('ate_kl_rev',  'ATE-KLrev'),
]


def _fmt(v):
    v = v[np.isfinite(v)]
    if v.size == 0:
        return f'{"NaN":>15s}'
    m = v.mean()
    s = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
    return f'{m:7.3f} ± {s:5.3f}'


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: inspect_fig2_shard.py <path/to/out.rhoN.npz>')
    p = os.path.expandvars(sys.argv[1])
    with np.load(p) as f:
        d = {k: f[k] for k in f.files}
    print(f'shard: {p}   rows={len(d["rho"])}   rho={d["rho"][0]}\n')

    hdr = f'{"metric":>12s}   ' + '   '.join(f'{LABEL[m]:>15s}' for m in METHODS)
    print(hdr)
    print('-' * len(hdr))
    for prefix, mlabel in METRICS:
        cells = []
        for m in METHODS:
            v = d.get(f'{prefix}_{m}')
            cells.append(_fmt(v) if v is not None else f'{"—":>15s}')
        print(f'{mlabel:>12s}   ' + '   '.join(cells))


if __name__ == '__main__':
    main()
