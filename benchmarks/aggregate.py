"""
Read all per-job npz files in --results and print Table 3 in the paper's format.

Rows: DoPFN, UWYK, plus every OURS variant.
Columns: 5 datasets × (√PEHE, ε_ATE). OT-mode has empty PEHE cells on CATE
datasets.
"""
import argparse, glob, os
import numpy as np

DATASETS = ['IHDP', 'ACIC', 'CPS', 'PSID', 'PSIDbal']
PEHE_DATASETS = ['IHDP', 'ACIC', 'CPS', 'PSID', 'PSIDbal']   # paper Table 3 has all five

METHODS = [
    ('Do-PFN',                 'dopfn'),
    ('UWYK Ancestral',         'uwyk_anc'),
    ('UWYK Baseline',          'uwyk_baseline'),
    ('OURS mean',              'ours_mean'),
    ('OURS MALC-mean',         'ours_malc_mean'),
    ('OURS MALC-mean-msk',     'ours_malc_mean_msk'),
    ('OURS MALC-mode',         'ours_malc_mode'),
    ('OURS MALC-mode-msk',     'ours_malc_mode_msk'),
    ('OURS OT-mode',           'ours_ot_mode'),
]


def _mean_std(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0: return np.nan, np.nan
    return float(a.mean()), float(a.std())


def _fmt(m, s, big=False):
    if np.isnan(m): return "        —       "
    if big:  return f"{m:7.0f} ± {s:5.0f}"
    return f"{m:6.2f} ± {s:5.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, help='Directory of *.npz')
    ap.add_argument('--out',     default=None, help='Optional output text file')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results, '*_r*.npz')))
    print(f"Found {len(files)} result files in {args.results}")

    # Bucket: dataset → method_key → (list of pehe, list of relerr)
    bucket = {d: {k: {'pehe': [], 'relerr': []} for _, k in METHODS} for d in DATASETS}
    n_per = {d: 0 for d in DATASETS}
    for fn in files:
        f = np.load(fn, allow_pickle=True)
        try:
            dname = str(f['dataset'])
        except Exception:
            continue
        if dname not in bucket: continue
        n_per[dname] += 1
        for _, key in METHODS:
            pk = f'pehe_{key}'; ek = f'err_{key}'
            if pk in f.files: bucket[dname][key]['pehe'].append(float(f[pk]))
            if ek in f.files: bucket[dname][key]['relerr'].append(float(f[ek]))

    # ── Print table ─────────────────────────────────────────────────────────
    lines = []
    lines.append(f"\n{'':<22}  " + "  ".join(f"{d:^32}" for d in DATASETS))
    lines.append(f"{'method':<22}  " + "  ".join(f"{'√PEHE ↓':^15} {'ε_ATE ↓':^15}" for _ in DATASETS))
    lines.append("─" * (24 + 34 * len(DATASETS)))

    for label, key in METHODS:
        row = f"{label:<22}  "
        for d in DATASETS:
            pehe = bucket[d][key]['pehe']; relerr = bucket[d][key]['relerr']
            big_pehe = d in ('CPS', 'PSID', 'PSIDbal')   # PEHE in raw y units
            if d in PEHE_DATASETS and label != 'OURS OT-mode':
                m, s = _mean_std(pehe); row += f"{_fmt(m, s, big=big_pehe):>15} "
            else:
                row += f"{'—':>15} "
            m, s = _mean_std(relerr)
            row += f"{_fmt(m, s):>15} "
        lines.append(row)

    lines.append("─" * (24 + 34 * len(DATASETS)))
    lines.append(f"n_realizations per dataset: " +
                 " ".join(f"{d}={n_per[d]}" for d in DATASETS))

    out = "\n".join(lines)
    print(out)
    if args.out:
        with open(args.out, 'w') as fp: fp.write(out)
        print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
