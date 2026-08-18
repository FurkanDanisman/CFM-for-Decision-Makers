"""Aggregate all completed eval_dopfn_bb_raw runs into a single table.

Parses every .out log in $DEPLOY_ROOT/logs matching the known prefixes
(rel_, j10_s*, fn50_) and prints one section per checkpoint (25k, 50k,
100k, 150k, fn=50).

Deduplicates by (checkpoint, dataset, scheme) keeping the newest .out
file (in case a job was re-submitted). Silently skips logs without a
final summary block (still running or crashed).

Usage:
    python aggregate_eval_table.py
    python aggregate_eval_table.py --deploy /scratch/furkanbd/rpfn_bench_kit
"""
from __future__ import annotations
import argparse, glob, os, re, sys


def parse_name(basename: str):
    b = basename[:-4] if basename.endswith('.out') else basename
    scheme_re = r'(min_max|std|trim5|trim10|log_transform)'
    # rel_${ds}_${tag}_%j.out  → J=10 at 150k
    m = re.match(rf'^rel_([A-Za-z_]+?)_{scheme_re}_\d+$', b)
    if m:
        return ('J=10 s150k', m.group(1), m.group(2))
    # j10_s${STEP}_${ds}_${tag}_%j.out  → J=10 at any step
    m = re.match(rf'^j10_s(\d+)_([A-Za-z_]+?)_{scheme_re}_\d+$', b)
    if m:
        return (f'J=10 s{int(m.group(1)) // 1000}k', m.group(2), m.group(3))
    # fn50_${ds}_${tag}_%j.out  → fn=50 (also matches fn50big_/fn50L_ retries
    # which share the same output-path pattern in our sbatch)
    m = re.match(rf'^fn50_([A-Za-z_]+?)_{scheme_re}_\d+$', b)
    if m:
        return ('fn=50', m.group(1), m.group(2))
    # alsoD_ (Do-PFN reference)
    m = re.match(rf'^alsoD_([A-Za-z_]+?)_{scheme_re}_\d+$', b)
    if m:
        return ('Do-PFN ref', m.group(1), m.group(2))
    return None


def parse_stats(path: str) -> dict:
    """Extract summary means. Returns {} if the summary block hasn't printed."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, errors='replace') as f:
        for line in f:
            for prefix, key in [
                ('PEHE (inner)',        'pehe_i'),
                ('PEHE (full 9-reg)',   'pehe_f'),
                ('eps_ATE (inner)',     'ate_i'),
                ('eps_ATE (full 9-reg)', 'ate_f'),
                ('eps_ATE (Do-PFN)',    'ate_dopfn'),
            ]:
                if line.startswith(prefix):
                    m = re.search(r'mean=([\-\d.]+)', line)
                    if m:
                        try:
                            out[key] = float(m.group(1))
                        except ValueError:
                            pass
    return out


CKPT_ORDER = ['J=10 s25k', 'J=10 s50k', 'J=10 s100k', 'J=10 s150k',
              'fn=50', 'Do-PFN ref']
DATASETS   = ['IHDP', 'ACIC', 'CPS', 'PSID', 'PSIDbal', 'law_race', 'sales']
SCHEMES    = ['min_max', 'std', 'trim5', 'trim10', 'log_transform']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deploy', default=os.environ.get('DEPLOY_ROOT'))
    args = ap.parse_args()
    if not args.deploy:
        sys.exit('DEPLOY_ROOT not set; pass --deploy or export it')

    log_dir = os.path.join(args.deploy, 'logs')
    files = glob.glob(os.path.join(log_dir, '*.out'))
    rows = {}   # (ckpt, ds, tag) -> (mtime, stats)
    for f in files:
        parsed = parse_name(os.path.basename(f))
        if parsed is None:
            continue
        stats = parse_stats(f)
        if not stats:
            continue
        key = parsed
        m = os.path.getmtime(f)
        if key not in rows or m > rows[key][0]:
            rows[key] = (m, stats)

    n_done = len(rows)
    print(f'[aggregate] parsed {len(files)} .out files, {n_done} completed rows\n')

    for ckpt in CKPT_ORDER:
        has_any = any((ckpt, ds, tag) in rows for ds in DATASETS for tag in SCHEMES)
        if not has_any:
            continue
        print(f'========== {ckpt} ==========')
        print(f'{"dataset":10s} {"scheme":<14s} {"PEHE_inner":>14s} '
              f'{"PEHE_full":>14s} {"eps_ATE_i":>10s} {"eps_ATE_f":>10s}')
        print('-' * 80)
        for ds in DATASETS:
            for tag in SCHEMES:
                key = (ckpt, ds, tag)
                if key not in rows:
                    continue
                s = rows[key][1]
                pi = f'{s.get("pehe_i", float("nan")):14.4f}'
                pf = f'{s.get("pehe_f", float("nan")):14.4f}'
                ei = f'{s.get("ate_i",  float("nan")):10.4f}'
                ef = f'{s.get("ate_f",  float("nan")):10.4f}'
                print(f'{ds:10s} {tag:<14s} {pi} {pf} {ei} {ef}')
        print()


if __name__ == '__main__':
    main()
