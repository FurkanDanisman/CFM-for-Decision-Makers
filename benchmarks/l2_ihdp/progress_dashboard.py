"""One-shot progress dashboard for the 6 downstream experiments.

For each task shows: expected total jobs, currently R/PD in queue, completed
with output shards, and any FAILED / TIMEOUT / OOM per sacct.

Detects tasks by their job-name pattern and their expected output paths.
Runs on the login node; needs only sacct + filesystem access.

Usage:
    python progress_dashboard.py
    python progress_dashboard.py --hours 24    # widen sacct window
"""
from __future__ import annotations
import argparse, glob, os, subprocess, sys, re


# One row per experiment: (label, job-name pattern, expected count, output-glob)
# Job names use `startswith` matching, output globs are absolute (with $DEPLOY_ROOT).
TASKS = [
    # (label,                    job_name_prefix,       n_expected, output_glob_relative_to_deploy_root)
    ('(2) d-scaling linear N=200',    'd-linear-d',          5,   'd_scaling_linear_dopfnbb_j10s150k.png.d*.npz'),
    ('(4) d,N grid (N=1250·d)',        'dn-grid-',            11,  'd_n_grid_dopfnbb_j10s150k.png.d*.npz'),
    # 500 seeds × 5 N values × 1 source (poly) = 2500 shards
    ('(5) context sweep (poly)',       'rpfn-ctxsweep',       2500, 'results_sweep_dopfnbb_j10s150k/poly_seed*_N*.npz'),
    ('(7) IHDP L2 density (std)',      'rpfn-ihdp-l2',        100, 'ihdp_l2_dopfnbb_j10_s150k_std/out.r*.npz'),
    ('(8) ACIC L2 density (std)',      'rpfn-acic-l2',        10,  'acic_l2_dopfnbb_j10_s150k_std/out.r*.npz'),
    ('(9) d=6 density DoPFN vs bb',    'd6-density',          1,   'd6_density_dopfnbb_j10s150k.png.d*.npz'),
    # J=10 s200k CPS sweep (side-quest)
    ('J=10 s200k CPS eval',            'j10s200000-CPS',      5,   'eval_dopfn_bb_raw/j10_s200000_CPS_*.npz'),
]


def _squeue_state_counts(job_name_prefix: str) -> tuple:
    """Returns (n_R, n_PD, n_other) for jobs whose name starts with prefix."""
    try:
        out = subprocess.check_output(
            ['squeue', '--me', '-h', '-o', '%j %t'], text=True, timeout=30
        )
    except Exception:
        return (0, 0, 0)
    r = pd = other = 0
    for line in out.splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        name, state = parts[0], parts[1]
        if not name.startswith(job_name_prefix):
            continue
        if state == 'R':
            r += 1
        elif state == 'PD':
            pd += 1
        else:
            other += 1
    return (r, pd, other)


def _sacct_recent(job_name_prefix: str, hours: int) -> dict:
    """State histogram from sacct for jobs starting with prefix in the last N hours.
    Returns {state: count}. Only counts top-level jobs (not .batch/.extern steps)."""
    try:
        since = subprocess.check_output(
            ['date', '-d', f'{hours} hours ago', '+%Y-%m-%dT%H:%M:%S'], text=True
        ).strip()
        out = subprocess.check_output(
            ['sacct', '-u', os.environ.get('USER', 'furkanbd'),
             '--starttime', since,
             '--format=JobID,JobName%40,State',
             '--noheader', '--parsable2'], text=True, timeout=60
        )
    except Exception:
        return {}
    hist = {}
    for line in out.splitlines():
        parts = line.split('|')
        if len(parts) < 3:
            continue
        jobid, name, state = parts[0], parts[1], parts[2]
        # Skip step rows (contain a dot in JobID: X.batch, X.extern)
        if '.' in jobid:
            continue
        if not name.startswith(job_name_prefix):
            continue
        # State may include trailing " by ..." for cancelled — normalize
        s = state.split()[0] if state else 'UNKNOWN'
        hist[s] = hist.get(s, 0) + 1
    return hist


def _count_outputs(deploy: str, glob_pat: str) -> int:
    return len(glob.glob(os.path.join(deploy, glob_pat)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deploy', default=os.environ.get('DEPLOY_ROOT'))
    ap.add_argument('--hours', type=int, default=12,
                    help='sacct window in hours (default 12).')
    args = ap.parse_args()
    if not args.deploy:
        sys.exit('DEPLOY_ROOT not set; pass --deploy or export it')

    print(f'{"task":40s}  {"expected":>8s}  {"R":>3s} {"PD":>3s}  {"outputs":>7s}  {"sacct (last %dh)":<40s}' % args.hours)
    print('-' * 130)
    for label, prefix, n_expected, out_glob in TASKS:
        r, pd, other = _squeue_state_counts(prefix)
        n_out = _count_outputs(args.deploy, out_glob)
        sacct_hist = _sacct_recent(prefix, args.hours)
        sacct_str = ' '.join(f'{s}={n}' for s, n in sorted(sacct_hist.items()))
        status = ''
        if n_out >= n_expected:
            status = ' ✓ done'
        elif r + pd > 0:
            status = f' ↻ in flight'
        elif n_out == 0 and not sacct_hist:
            status = ' — not submitted'
        else:
            status = f' ⚠ partial ({n_out}/{n_expected})'
        print(f'{label:40s}  {n_expected:>8d}  {r:>3d} {pd:>3d}  {n_out:>7d}  {sacct_str:<40s}{status}')


if __name__ == '__main__':
    main()
