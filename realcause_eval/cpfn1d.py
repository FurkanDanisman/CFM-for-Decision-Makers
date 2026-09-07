"""cpfn1d (a.k.a. causalpfn-orig-headrand) RealCause eval — thin wrapper.

Runs a plain CausalPFN checkpoint (1D bar-dist head, e.g. j1024_headrand)
on the 5 RealCause datasets using
    benchmarks/eval_causalpfn2d/eval_causalpfn_v0_realcause.py
verbatim. Per-realization NPZs land at OUTDIR/<DATASET>_r<###>.npz.

Defaults match the graph2d and cpfn2d wrappers so numbers are directly
comparable:
    EVAL_MAX_CONTEXT   = 1000       (random-subsample train context)
    EVAL_CONTEXT_SEED  = 43         (deterministic subsample seed)
    STD_MODE           = per_arm    (per-arm Y standardization)
    COMPILE            = 0

Usage:
    python realcause_eval/cpfn1d.py \\
        --dataset IHDP \\
        --outdir /scratch/.../results/cpfn1d/IHDP \\
        --ckpt /scratch/.../causalpfn_j1024_headrand_A1_h100/step_checkpoints/run/step_0050000.pt \\
        --repo /scratch/.../R-PFN \\
        --causalpfn /scratch/.../external/causalpfn

For all 5 datasets in one array job, use:
    sbatch benchmarks/cluster/submit_realcause_cpfn1d_step50k.sbatch
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True,
                    choices=['IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal'])
    p.add_argument('--outdir', required=True,
                    help='Per-realization npzs land at OUTDIR/<DATASET>_r<###>.npz.')
    p.add_argument('--ckpt', required=True,
                    help='CausalPFN 1D checkpoint (.pt).')
    p.add_argument('--repo', required=True, help='R-PFN repo root.')
    p.add_argument('--causalpfn', required=True,
                    help='CausalPFN repo root (for benchmarks / dataset loaders).')
    p.add_argument('--eval-max-context', type=int, default=1000,
                    help='Random-subsample train context to this cap. Default: 1000.')
    p.add_argument('--eval-context-seed', type=int, default=43,
                    help='Deterministic subsampling seed. Default: 43.')
    p.add_argument('--std-mode', default='per_arm',
                    choices=['pooled', 'per_arm', 'log', 'log_per_arm', 'log_winsor'],
                    help='Y standardization for eval. Default: per_arm.')
    p.add_argument('--compile', default='0', choices=['0', '1'],
                    help='torch.compile toggle. Default: 0.')
    p.add_argument('--skip-em', default='1', choices=['0', '1'],
                    help='Skip EM step in density post-processing. Default: 1.')
    p.add_argument('--density-dump', default='0', choices=['0', '1'],
                    help='Dump per-realization density npzs alongside pehe. Default: 0.')
    return p.parse_args()


def main():
    args = _parse_args()

    env = os.environ.copy()
    env.update({
        'DATASET':           args.dataset,
        'OUT':               args.outdir,
        'CKPT':              args.ckpt,
        'CAUSALPFN':         args.causalpfn,
        'EVAL_MAX_CONTEXT':  str(args.eval_max_context),
        'EVAL_CONTEXT_SEED': str(args.eval_context_seed),
        'STD_MODE':          args.std_mode,
        'COMPILE':           args.compile,
        'SKIP_EM':           args.skip_em,
        'DENSITY_DUMP':      args.density_dump,
    })

    script = os.path.join(args.repo, 'benchmarks', 'eval_causalpfn2d',
                          'eval_causalpfn_v0_realcause.py')
    os.makedirs(args.outdir, exist_ok=True)

    print(f'[cpfn1d] dataset={args.dataset}  std_mode={args.std_mode}  '
          f'eval_max_context={args.eval_max_context}  '
          f'seed={args.eval_context_seed}  ckpt={os.path.basename(args.ckpt)}',
          flush=True)

    subprocess.run(
        [sys.executable, '-u', script, '--dataset', args.dataset],
        env=env, check=True,
    )


if __name__ == '__main__':
    main()
