"""Aggregate the context-sweep npz files into a table.

Columns  = context sizes N.
Rows     = OURS variants.
Cells    = mean ± std across SCMs (per metric: PEHE or ε_ATE).

Two tables per invocation — one for each source (prior / poly).

For the **prior** source, ε_ATE is recomputed from stored ATEs with a tolerance
in the denominator to avoid divide-by-zero on SCMs with true ATE ≈ 0:

    ε_ATE_prior = |τ̂ − τ| / (|τ| + PRIOR_ATE_TOL)

Poly source keeps the standard |τ̂ − τ| / |τ| formula recorded in the npz.
"""
import argparse, glob, os
import numpy as np

PRIOR_ATE_TOL = 1e-4


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
    ('Do-PFN',                 'dopfn'),
    ('UWYK Full Ancestral',    'uwyk_anc'),
    ('UWYK No-Ancestral',      'uwyk_noanc'),
    ('UWYK Baseline',          'uwyk_baseline'),
]

VARIANTS = BASELINES + [(f'OURS {v}', k) for v, k in OURS_VARIANTS]


def _mean_std(a):
    """Mean ± std over the finite entries of `a`. Returns (mean, std, n_valid)."""
    a = np.asarray(a, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.nan, np.nan, 0
    return float(finite.mean()), float(finite.std()), int(finite.size)


def _median(a):
    """Median over the finite entries of `a`. Returns (median, n_valid)."""
    a = np.asarray(a, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.nan, 0
    return float(np.median(finite)), int(finite.size)


def _fmt(m, s):
    if np.isnan(m): return "        —       "
    if abs(m) >= 1000: return f"{m:8.0f} ± {s:6.0f}"
    return f"{m:7.2f} ± {s:5.2f}"


def _one_table(bucket, n_values, metric_key, variants, show_n=False, use_median=False):
    """metric_key = 'pehe' or 'err'. bucket[N][variant_key] = list of values."""
    lines = []
    header = f"{'method':<28} " + "  ".join(f"{'N=' + str(N):^15}" for N in n_values)
    lines.append(header)
    lines.append("─" * len(header))
    for label, vkey in variants:
        row = f"{label:<28} "
        for N in n_values:
            vals = bucket[N].get(vkey, {}).get(metric_key, [])
            if use_median:
                m, nv = _median(vals); s = 0.0
                cell = "        —       " if np.isnan(m) else (
                    f"{m:8.0f}  n={nv:3d}" if abs(m) >= 1000 else f"{m:7.2f}  n={nv:3d}"
                )
            else:
                m, s, nv = _mean_std(vals)
                cell = _fmt(m, s) if not show_n else (
                    _fmt(m, s) + f" [{nv}]"
                )
            row += f"{cell:>15}  "
        lines.append(row)
    return "\n".join(lines)


def _ingest(files, per_source, variants, remap_key=None):
    """Populate per_source buckets from a directory of npz files.

    remap_key: optional function key -> stored_key (used for --extra so we
    keep the fn=10 Ours numbers under distinct bucket keys).
    """
    if remap_key is None:
        remap_key = lambda k: k
    for fn in files:
        f = np.load(fn, allow_pickle=True)
        try:
            src = str(f['source']); N = int(f['n_context'])
        except Exception:
            continue
        if src not in per_source: continue
        bucket = per_source[src].setdefault(N, {})
        true_ate = float(f['true_ate']) if 'true_ate' in f.files else None
        for _, vkey in variants:
            store = bucket.setdefault(remap_key(vkey), {'pehe': [], 'err': []})
            pk = f'pehe_{vkey}'
            if pk in f.files:
                store['pehe'].append(float(f[pk]))
            if src == 'prior' and true_ate is not None and f'ate_{vkey}' in f.files:
                ate_pred = float(f[f'ate_{vkey}'])
                err = abs(ate_pred - true_ate) / (abs(true_ate) + PRIOR_ATE_TOL)
                store['err'].append(err)
            else:
                ek = f'err_{vkey}'
                if ek in f.files:
                    store['err'].append(float(f[ek]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results',    required=True, help='dir of *.npz')
    ap.add_argument('--out',        default=None, help='optional output text file')
    ap.add_argument('--metric',     choices=['pehe', 'err', 'both'], default='both')
    ap.add_argument('--show-n',     action='store_true',
                    help='append [n_valid] to each mean±std cell')
    ap.add_argument('--median',     action='store_true',
                    help='also show the median (robust to inf on near-zero ATE)')
    ap.add_argument('--extra',      default=None,
                    help='Second results dir + label prefix (e.g. '
                         '"results_sweep_dopfn:OURS[fn=10]"). Appends 7 more '
                         'Ours rows so both checkpoints appear side by side.')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results, '*.npz')))
    print(f"Found {len(files)} npz files.")

    # Assemble the final variants list, adding extra Ours rows if requested
    variants_all = list(VARIANTS)
    extra_dir = extra_label = None
    if args.extra:
        extra_dir, extra_label = args.extra.split(':', 1)
        if not os.path.isdir(extra_dir):
            raise SystemExit(f'--extra dir does not exist: {extra_dir}')
        variants_all += [(f'{extra_label} {v}', f'extra_{k}') for v, k in OURS_VARIANTS]

    per_source = {'prior': {}, 'poly': {}}
    _ingest(files, per_source, VARIANTS)

    if extra_dir is not None:
        extra_files = sorted(glob.glob(os.path.join(extra_dir, '*.npz')))
        print(f"Found {len(extra_files)} extra npz files in {extra_dir}")
        # For extra, only ingest Ours variants and remap the bucket key.
        _ingest(extra_files, per_source, OURS_VARIANTS,
                 remap_key=lambda k: f'extra_{k}')

    text = []
    for src, buckets in per_source.items():
        n_values = sorted(buckets.keys())
        if not n_values: continue
        # Count how many SCMs actually landed per (variant, N)
        per_N_counts = []
        n_scms = 0
        for N in n_values:
            got = len(buckets[N].get('ours_mean', {}).get('pehe', []))
            per_N_counts.append((N, got))
            n_scms = max(n_scms, got)
        counts_str = ", ".join(f"N={N}: {got}" for N, got in per_N_counts)
        text.append(f"\n{'='*90}\n{'SOURCE: ' + src:>90}\n"
                     f"raw SCM counts per N — {counts_str}\n"
                     f"non-finite ε_ATE entries (SCMs with true ATE ≈ 0) are filtered per cell.\n"
                     f"{'='*90}")

        if args.metric in ('pehe', 'both'):
            text.append("\n√PEHE ↓  (mean ± std across SCMs, non-finite filtered)\n")
            text.append(_one_table(buckets, n_values, 'pehe', variants_all, show_n=args.show_n))
        if args.metric in ('err', 'both'):
            text.append("\nε_ATE ↓  (mean ± std across SCMs, non-finite filtered)\n")
            text.append(_one_table(buckets, n_values, 'err', variants_all, show_n=args.show_n))
            if args.median:
                text.append("\nε_ATE ↓  MEDIAN (robust to near-zero-ATE SCMs)\n")
                text.append(_one_table(buckets, n_values, 'err', variants_all, use_median=True))

    joined = "\n".join(text)
    print(joined)
    if args.out:
        with open(args.out, 'w') as fp: fp.write(joined)
        print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
