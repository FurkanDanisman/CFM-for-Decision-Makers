"""Aggregate the context-sweep npz files into a table.

Columns  = context sizes N.
Rows     = OURS variants.
Cells    = mean ± std across SCMs (per metric: PEHE or ε_ATE).

Two tables per invocation — one for each source (prior / poly).
"""
import argparse, glob, os
import numpy as np


VARIANTS = [
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


def _fmt(m, s):
    if np.isnan(m): return "        —       "
    if abs(m) >= 1000: return f"{m:8.0f} ± {s:6.0f}"
    return f"{m:7.2f} ± {s:5.2f}"


def _one_table(bucket, n_values, metric_key):
    """metric_key = 'pehe' or 'err'. bucket[N][variant_key] = list of values."""
    lines = []
    header = f"{'method':<24} " + "  ".join(f"{'N=' + str(N):^15}" for N in n_values)
    lines.append(header)
    lines.append("─" * len(header))
    for label, vkey in VARIANTS:
        row = f"{label:<24} "
        for N in n_values:
            vals = bucket[N].get(vkey, {}).get(metric_key, [])
            m, s = _mean_std(vals)
            row += f"{_fmt(m, s):>15}  "
        lines.append(row)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, help='dir of *.npz')
    ap.add_argument('--out',     default=None, help='optional output text file')
    ap.add_argument('--metric',  choices=['pehe', 'err', 'both'], default='both')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results, '*.npz')))
    print(f"Found {len(files)} npz files.")

    # source → N → variant → metric → [values]
    per_source = {'prior': {}, 'poly': {}}
    for fn in files:
        f = np.load(fn, allow_pickle=True)
        try:
            src = str(f['source']); N = int(f['n_context'])
        except Exception:
            continue
        if src not in per_source: continue
        bucket = per_source[src].setdefault(N, {})
        for _, vkey in VARIANTS:
            store = bucket.setdefault(vkey, {'pehe': [], 'err': []})
            for m in ('pehe', 'err'):
                key = f'{m}_{vkey}'
                if key in f.files:
                    store[m].append(float(f[key]))

    text = []
    for src, buckets in per_source.items():
        n_values = sorted(buckets.keys())
        if not n_values: continue
        n_scms = min(len(buckets[N].get('ours_mean', {}).get('pehe', [])) for N in n_values)
        text.append(f"\n{'='*90}\n{'SOURCE: ' + src:>90}\n"
                     f"(N SCMs per column ≈ {n_scms})\n{'='*90}")
        if args.metric in ('pehe', 'both'):
            text.append("\n√PEHE ↓  (mean ± std across SCMs)\n")
            text.append(_one_table(buckets, n_values, 'pehe'))
        if args.metric in ('err', 'both'):
            text.append("\nε_ATE ↓  (mean ± std across SCMs)\n")
            text.append(_one_table(buckets, n_values, 'err'))

    joined = "\n".join(text)
    print(joined)
    if args.out:
        with open(args.out, 'w') as fp: fp.write(joined)
        print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
