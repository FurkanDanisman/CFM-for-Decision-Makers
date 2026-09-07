"""cpfn2d RealCause eval — thin wrapper around eval_cpfn2d_realcause.py.

Runs the CausalPFN-2D-head (cpfn2d, e.g. j32_random) checkpoint on the 5
RealCause datasets using
    benchmarks/eval_causalpfn2d/eval_cpfn2d_realcause.py
verbatim. Per-realization NPZs land at OUTDIR/<DATASET>_r<###>.npz and a
summary is printed at the end.

Defaults (paper CausalPFN-C 2D was trained to consume full context):
    EVAL_MAX_CONTEXT   = 0          (no cap — feed the full training set)
    EVAL_CONTEXT_SEED  = 1          (only used if EVAL_MAX_CONTEXT > 0)
    STD_MODE           = per_arm    (per-arm Y standardization)
    COMPILE            = 0          (torch.compile off — cpfn2d bug)

Usage:
    python realcause_eval/cpfn2d.py \\
        --dataset IHDP \\
        --outdir /scratch/.../results/cpfn2d/IHDP \\
        --ckpt /scratch/.../cpfn2d_j32_random_A1_h100/step_checkpoints/run/step_0050000.pt \\
        --repo /scratch/.../R-PFN \\
        --causalpfn /scratch/.../external/causalpfn

For all 5 datasets in one array job, use:
    sbatch benchmarks/cluster/submit_realcause_cpfn2d_step50k.sbatch
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
                    help='cpfn2d checkpoint (.pt with model_state_dict + config + edges).')
    p.add_argument('--repo', required=True, help='R-PFN repo root.')
    p.add_argument('--causalpfn', required=True,
                    help='CausalPFN repo root (for benchmarks / dataset loaders).')
    p.add_argument('--eval-max-context', type=int, default=0,
                    help='Cap on training context (random subsample if > 0). '
                         'Default: 0 = no cap (CausalPFN uses full context).')
    p.add_argument('--eval-context-seed', type=int, default=1,
                    help='Subsample seed (only used if --eval-max-context > 0). '
                         'Default: 1.')
    p.add_argument('--std-mode', default='per_arm',
                    choices=['pooled', 'per_arm', 'log', 'log_per_arm',
                             'log_winsor', 'std_target'],
                    help='Y standardization for eval. Default: per_arm '
                         '(context Y standardized by its own arm mean/std). '
                         'std_target = DoPFN-bb-style scaling (σ = --std-target).')
    p.add_argument('--std-target', type=float, default=1.0,
                    help='Only used with --std-mode std_target. cpfn2d has '
                         'bar-dist edges [-10, +10] and J=32 (bin_width=0.625), '
                         'so σ must satisfy σ > 0.625 (else bulk collapses to '
                         'one bin) AND 3σ < 10 (else tails clipped). Sensible '
                         'range: 0.7-3.3. Default 1.0 matches CausalPFN training '
                         'convention (σ_scaled ≈ 1). Do NOT use dopfn-bb\'s 0.3 '
                         '— cpfn2d\'s wider head makes it inappropriate.')
    p.add_argument('--compile', default='0', choices=['0', '1'],
                    help='torch.compile toggle (default 0 — cpfn2d has a compile bug).')
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
        # Empty EVAL_MAX_CONTEXT disables the cap (feed full context).
        'EVAL_MAX_CONTEXT':  str(args.eval_max_context) if args.eval_max_context > 0 else '',
        'EVAL_CONTEXT_SEED': str(args.eval_context_seed),
        'STD_MODE':          args.std_mode,
        'STD_TARGET':        str(args.std_target),
        'COMPILE':           args.compile,
        'SKIP_EM':           args.skip_em,
        'DENSITY_DUMP':      args.density_dump,
    })

    script = os.path.join(args.repo, 'benchmarks', 'eval_causalpfn2d',
                          'eval_cpfn2d_realcause.py')
    os.makedirs(args.outdir, exist_ok=True)

    cap_str = args.eval_max_context if args.eval_max_context > 0 else 'none (full context)'
    print(f'[cpfn2d] dataset={args.dataset}  std_mode={args.std_mode}  '
          f'eval_max_context={cap_str}  '
          f'seed={args.eval_context_seed}  ckpt={os.path.basename(args.ckpt)}',
          flush=True)

    subprocess.run(
        [sys.executable, '-u', script, '--dataset', args.dataset],
        env=env, check=True,
    )


if __name__ == '__main__':
    main()
