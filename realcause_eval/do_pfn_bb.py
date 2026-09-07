"""Do-PFN-bb RealCause eval — wrapper around l2_ihdp/eval_dopfn_bb_raw.py.

Reproduces Table 3 "Do-PFN 2D" row (paper: IHDP 5.06 ACIC 3.15 CPS 10715
PSID 18617 PSID_bal 18711) using the working script at:
    benchmarks/l2_ihdp/eval_dopfn_bb_raw.py

Paper settings (from task #118 "Run 6 downstream experiments with DoPFN-bb-j10
@150k std" — traced via `benchmarks/l2_ihdp/table3_row.py` which reads shards
named `rel_${DATASET}_std.npz`):

  Checkpoint : dopfn_bb_j10_step_150000.pt   (J=10, step 150k)
  Y-scaling  : --y-scaling std --std-target 0.3
  Context    : default full context (--n-context 0)
               eval_dopfn_bb_raw.py has no per-query loops without
               --malc-upsample, so full context on IHDP/ACIC/PSID/PSIDbal
               runs at DoPFN-native speed. For CPS (24k rows) pass
               --n-context 1000 if wall-clock matters.

Usage — one dataset:
    python realcause_eval/do_pfn_bb.py \\
        --dataset IHDP \\
        --outdir /scratch/.../results_repro/do_pfn_bb/IHDP \\
        --ckpt   /scratch/.../Required_checkpoints/dopfn_bb_j10_step_150000.pt \\
        --repo   /scratch/.../R-PFN \\
        --dopfn  /scratch/.../external/dopfn \\
        --causalpfn /scratch/.../external/causalpfn

Per-realization npzs land at OUTDIR/density_r<###>.npz (density-dump path)
plus an aggregate OUTDIR/summary.npz.

To sweep all 5 datasets, invoke once per dataset or use the existing
benchmarks/cluster/submit_eval_dopfn_bb_realcause.sbatch with the
--y-scaling / --std-target / --checkpoint overrides shown below.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True,
                    choices=['IHDP', 'ACIC', 'CPS', 'PSID', 'PSIDbal'])
    p.add_argument('--outdir', required=True,
                    help='Per-realization density npzs land at OUTDIR/density_r<###>.npz. '
                         'Aggregate summary at OUTDIR/summary.npz.')
    p.add_argument('--ckpt', required=True,
                    help='DoPFN-backbone-with-2D-head checkpoint (.pt). '
                         'Paper: dopfn_bb_j10_step_150000.pt (J=10, step 150k).')
    p.add_argument('--repo', required=True, help='R-PFN repo root.')
    p.add_argument('--dopfn', required=True, help='dopfn upstream repo root.')
    p.add_argument('--causalpfn', required=True, help='CausalPFN repo root.')
    p.add_argument('--y-scaling', default='std',
                    choices=['std', 'min_max', 'iqr', 'trim_min_max'],
                    help="Paper: 'std' (Y ≈ N(0, std/std_target)).")
    p.add_argument('--std-target', type=float, default=0.3,
                    help='Target scaled-σ for --y-scaling std. Paper: 0.3.')
    p.add_argument('--n-context', type=int, default=0,
                    help='0 = full context (paper default). Set to 1000 to '
                         'random-subsample context per realization if CPS wall-clock '
                         'is a concern.')
    p.add_argument('--extra-args', nargs='*', default=[],
                    help='Extra args forwarded to eval_dopfn_bb_raw.py.')
    return p.parse_args()


def main():
    args = _parse_args()

    script = os.path.join(args.repo, 'benchmarks', 'l2_ihdp', 'eval_dopfn_bb_raw.py')
    os.makedirs(args.outdir, exist_ok=True)

    cmd = [
        sys.executable, '-u', script,
        '--repo',       args.repo,
        '--dopfn',      args.dopfn,
        '--causalpfn',  args.causalpfn,
        '--checkpoint', args.ckpt,
        '--dataset',    args.dataset,
        '--out',        os.path.join(args.outdir, 'summary.npz'),
        '--y-scaling',  args.y_scaling,
        '--std-target', str(args.std_target),
        '--n-context',  str(args.n_context),
    ] + list(args.extra_args)

    env = os.environ.copy()
    env.setdefault('DENSITY_DUMP', '1')  # save per-realization density_r<###>.npz

    print(f'[do_pfn_bb] dataset={args.dataset}  y_scaling={args.y_scaling}  '
          f'std_target={args.std_target}  n_context={args.n_context}  '
          f'ckpt={os.path.basename(args.ckpt)}', flush=True)

    # DoPFN-bb needs to run from the dopfn upstream root (for datasets/ import etc.)
    subprocess.run(cmd, env=env, check=True, cwd=args.dopfn)


if __name__ == '__main__':
    main()
