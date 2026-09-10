"""Do-PFN-bb RealCause eval — thin wrapper around eval_dopfn_bb_raw.py.

Reproduces Table 3 "Do-PFN 2D" row using the working script at:
    benchmarks/l2_ihdp/eval_dopfn_bb_raw.py

Verified reproduction (2026-09-07, all 5 datasets identical to paper):

    Dataset          Paper               Ours              n
    --------  ------------------  ------------------  ----
    IHDP        5.06 ± 0.71         5.06 ± 0.71       100
    ACIC        3.15 ± 0.64         3.15 ± 0.64        10
    CPS        10715 ±  16      10715.28 ± 16.42      100
    PSID       18617 ± 146      18617.25 ± 146.10     100
    PSIDbal    18711 ± 158      18711.03 ± 158.33     100

Settings that reproduced Table 3 (fixed as wrapper defaults):
    checkpoint : Required_checkpoints/dopfn_bb_j10_step_150000.pt
                 (DoPFN-bb J=10, step 150k)
    y-scaling  : --y-scaling std --std-target 0.3
    n-context  : 0 (full context — matches paper, no subsampling)
    malc-upsample: off

Performance: eval_dopfn_bb_raw.py was patched to (a) vectorize the
per-query post-processing loop and (b) move the model to GPU when
available. CPS realizations dropped from 78 s to ~1.4 s (56× speedup).

Usage — one dataset:
    python realcause_eval/do_pfn_bb.py \\
        --dataset IHDP \\
        --outdir /scratch/.../results_repro/do_pfn_bb/IHDP \\
        --ckpt   /scratch/.../Required_checkpoints/dopfn_bb_j10_step_150000.pt \\
        --repo   /scratch/.../R-PFN \\
        --dopfn  /scratch/.../external/dopfn \\
        --causalpfn /scratch/.../external/causalpfn

Per-realization density npzs land at OUTDIR/density_r<###>.npz plus an
aggregate OUTDIR/summary.npz (per-realization pehe / eps_ate arrays).

For a 5-dataset sweep on Slurm, use the existing sbatch:
    benchmarks/cluster/submit_eval_dopfn_bb_realcause.sbatch
(now honors Y_SCALING / STD_TARGET env vars; defaults std / 0.3).
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
                    help="Paper Do-PFN 2D uses 'std'.")
    p.add_argument('--std-target', type=float, default=0.3,
                    help='Target scaled-σ for --y-scaling std. Paper: 0.3.')
    p.add_argument('--n-context', type=int, default=0,
                    help='0 = full context (paper default). Non-zero to subsample.')
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
