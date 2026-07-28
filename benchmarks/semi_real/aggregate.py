"""Aggregate the semi-real npz files into a per-dataset table.

Rows    = methods (Do-PFN, UWYK Ancestral, UWYK No-Ancestral, Ours variants).
Columns = √PEHE, ε_ATE per dataset.

Use --extra <dir>:<label> to append 7 additional Ours rows read from a
second results directory (e.g. the fn=10 checkpoint pass).
"""
import argparse, glob, os
import numpy as np


DATASETS = ['sales', 'law_race']

OURS_VARIANTS = [
    ('mean',              'ours_mean'),
    ('MALC-mean',         'ours_malc_mean'),
    ('MALC-mean-msk',     'ours_malc_mean_msk'),
    ('MALC-mode',         'ours_malc_mode'),
    ('MALC-mode-msk',     'ours_malc_mode_msk'),
    ('OT-mode',           'ours_ot_mode'),
    ('OT-mean',           'ours_ot_mean'),
]

BASELINES = [
    ('Do-PFN',              'dopfn'),
    ('UWYK Ancestral',      'uwyk_anc'),
    ('UWYK No-Ancestral',   'uwyk_noanc'),
]

METHODS = BASELINES + [(f'OURS {v}', k) for v, k in OURS_VARIANTS]


def _mean_std(a):
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    if a.size == 0: return np.nan, np.nan
    return float(a.mean()), float(a.std())


def _fmt(m, s):
    if np.isnan(m): return "        —       "
    if abs(m) >= 1000: return f"{m:7.0f} ± {s:5.0f}"
    return f"{m:6.2f} ± {s:5.2f}"


def _ingest(files, bucket_by_ds, methods, remap=None):
    if remap is None: remap = lambda k: k
    n_per = {d: 0 for d in DATASETS}
    for fn in files:
        f = np.load(fn, allow_pickle=True)
        try:
            dname = str(f['dataset'])
        except Exception:
            continue
        if dname not in bucket_by_ds: continue
        n_per[dname] += 1
        for _, key in methods:
            pk = f'pehe_{key}'; ek = f'err_{key}'
            store = bucket_by_ds[dname].setdefault(remap(key), {'pehe': [], 'err': []})
            if pk in f.files: store['pehe'].append(float(f[pk]))
            if ek in f.files: store['err'].append(float(f[ek]))
    return n_per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, help='Directory of *.npz')
    ap.add_argument('--out',     default=None, help='Optional output text file')
    ap.add_argument('--extra',   default=None,
                    help='Second results dir + label prefix, e.g. '
                         '"results_semi_real_fn10:OURS[fn=10]".')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results, '*.npz')))
    print(f"Found {len(files)} result files in {args.results}")

    method_list = list(METHODS)
    extra_dir = extra_label = None
    if args.extra:
        extra_dir, extra_label = args.extra.split(':', 1)
        if not os.path.isdir(extra_dir):
            raise SystemExit(f'--extra dir does not exist: {extra_dir}')
        method_list += [(f'{extra_label} {v}', f'extra_{k}') for v, k in OURS_VARIANTS]

    bucket_by_ds = {d: {} for d in DATASETS}
    n_per = _ingest(files, bucket_by_ds, METHODS)

    n_per_extra = {}
    if extra_dir is not None:
        extra_files = sorted(glob.glob(os.path.join(extra_dir, '*.npz')))
        print(f"Found {len(extra_files)} extra result files in {extra_dir}")
        n_per_extra = _ingest(extra_files, bucket_by_ds, OURS_VARIANTS,
                                remap=lambda k: f'extra_{k}')

    # ── Print table ─────────────────────────────────────────────────────────
    lines = []
    lines.append("")
    lines.append(f"{'':<28}  " + "  ".join(f"{d:^32}" for d in DATASETS))
    lines.append(f"{'method':<28}  " +
                 "  ".join(f"{'√PEHE ↓':^15} {'ε_ATE ↓':^15}" for _ in DATASETS))
    lines.append("─" * (30 + 34 * len(DATASETS)))

    for label, key in method_list:
        row = f"{label:<28}  "
        for d in DATASETS:
            store = bucket_by_ds[d].get(key, {'pehe': [], 'err': []})
            has_pehe = not (label.endswith('OT-mode') or label.endswith('OT-mean'))
            if has_pehe:
                m, s = _mean_std(store['pehe']); row += f"{_fmt(m, s):>15} "
            else:
                row += f"{'—':>15} "
            m, s = _mean_std(store['err'])
            row += f"{_fmt(m, s):>15} "
        lines.append(row)

    lines.append("─" * (30 + 34 * len(DATASETS)))
    lines.append(f"n_seeds per dataset (main): " +
                 " ".join(f"{d}={n_per[d]}" for d in DATASETS))
    if extra_dir is not None:
        lines.append(f"n_seeds per dataset (extra): " +
                     " ".join(f"{d}={n_per_extra[d]}" for d in DATASETS))

    out = "\n".join(lines)
    print(out)
    if args.out:
        with open(args.out, 'w') as fp: fp.write(out)
        print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
