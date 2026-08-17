"""Merge two Fig 2 shards (out.rhoN.npz) — new keys override old keys.

Use case: you re-ran fig2 with `--methods ours_fn50` (produces a shard with
only ours_fn50 metric keys). You want to combine it with a previous
"all four methods" shard so plotting + inspection sees a full row.

Merge policy per-key:
- If key exists only in `old`, keep old.
- If key exists only in `new`, add new.
- If key exists in both, use new (assumed to be the up-to-date version).

Row alignment: both shards MUST have the same `rho` and `seed` arrays.
That's the case as long as you regenerated with the SAME --K, --N-context,
--N-test, and --rho-index — the SCM seeds are `rho_idx*10000 + k` which is
deterministic. If seeds don't match, the merge aborts.

Usage:
    python merge_fig2_shards.py <old.npz> <new.npz> <out.npz>
    python merge_fig2_shards.py <old.npz> <new.npz> <out.npz> --diag  # merge .diag.npz variants (same paths but ending in .diag.npz)
"""
import argparse, os, sys, numpy as np


def _merge(old_path, new_path, out_path):
    with np.load(old_path, allow_pickle=True) as f:
        old = {k: f[k] for k in f.files}
    with np.load(new_path, allow_pickle=True) as f:
        new = {k: f[k] for k in f.files}

    # Sanity check row alignment (metric shards have 'rho' and 'seed').
    if 'rho' in old and 'rho' in new:
        if not np.array_equal(old['rho'], new['rho']):
            sys.exit(f'[error] rho arrays differ between {old_path} and {new_path}')
    if 'seed' in old and 'seed' in new:
        if not np.array_equal(old['seed'], new['seed']):
            sys.exit(f'[error] seed arrays differ between {old_path} and {new_path}')

    merged = dict(old)
    added, replaced = [], []
    for k, v in new.items():
        if k in merged:
            replaced.append(k)
        else:
            added.append(k)
        merged[k] = v

    np.savez(out_path, **merged)
    print(f'[merge] {old_path}   +   {new_path}   →   {out_path}')
    print(f'   added={len(added)}   replaced={len(replaced)}   kept={len(old)-len(replaced)}')
    if replaced:
        print(f'   replaced keys: {sorted(replaced)[:8]}{"..." if len(replaced)>8 else ""}')
    if added:
        print(f'   added keys:    {sorted(added)[:8]}{"..." if len(added)>8 else ""}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('old')
    ap.add_argument('new')
    ap.add_argument('out')
    ap.add_argument('--diag', action='store_true',
                    help='Also merge the corresponding .diag.npz shards '
                         '(same paths but .diag.npz suffix instead of .npz).')
    args = ap.parse_args()
    _merge(args.old, args.new, args.out)
    if args.diag:
        old_d = args.old.replace('.npz', '.diag.npz')
        new_d = args.new.replace('.npz', '.diag.npz')
        out_d = args.out.replace('.npz', '.diag.npz')
        if os.path.exists(old_d) and os.path.exists(new_d):
            _merge(old_d, new_d, out_d)
        else:
            print(f'[warn] skip diag merge — old_d exists={os.path.exists(old_d)}   new_d exists={os.path.exists(new_d)}')


if __name__ == '__main__':
    main()
