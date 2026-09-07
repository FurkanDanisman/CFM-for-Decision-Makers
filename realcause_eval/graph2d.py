"""graph2d RealCause eval — thin wrapper around eval_graph2d_realcause.py.

Reproduces Table 3 "UWYK No-Anc 2D" row (paper: IHDP 4.52 ACIC 2.75 CPS 12299
PSID 21434 PSID_bal 20051) using the working script at
`benchmarks/eval_graph2d/eval_graph2d_realcause.py` verbatim.

Verified reproduction (job 5286314, aggregated via summarize_realcause.py):
    IHDP 4.519  ACIC 2.750  CPS 12299.16  PSID 21430.06  PSID_bal 20046.90
identical to paper within rounding.

Settings required for the match:
  EVAL_MAX_CONTEXT=1000     random-subsample context to 1000 rows/realization
  EVAL_CONTEXT_SEED=43      deterministic subsampling seed
  ANC_MODE=noanc            zero-adjacency (no ancestor hint at inference)
  DENSITY_PRIMARY_MODE=noanc mirror noanc-mode densities to un-suffixed keys

Usage:
    python realcause_eval/graph2d.py \\
        --dataset IHDP \\
        --outdir /scratch/.../results/graph2d/IHDP \\
        --ckpt /scratch/.../graph2d_step_50000.pt \\
        --repo /scratch/.../R-PFN \\
        --causalpfn /scratch/.../external/causalpfn \\
        --uwyk /scratch/.../external/uwyk_reproduce

To sweep all 5 datasets, invoke once per dataset (or use the existing
benchmarks/cluster/submit_eval_graph2d_realcause.sbatch which does the
5-task array).
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
    p.add_argument('--ckpt', required=True, help='graph2d checkpoint (.pt).')
    p.add_argument('--repo', required=True,
                    help='R-PFN repo root (contains benchmarks/).')
    p.add_argument('--causalpfn', required=True,
                    help='CausalPFN repo root (for benchmarks / dataset loaders).')
    p.add_argument('--uwyk', required=True,
                    help='UWYK reproduce repo root (for graph-conditioned wrapper).')
    p.add_argument('--eval-max-context', type=int, default=1000,
                    help='Random-subsample train context to this cap (paper: 1000).')
    p.add_argument('--eval-context-seed', type=int, default=43,
                    help='Deterministic subsampling seed (paper: 43).')
    p.add_argument('--anc-mode', default='v3b_only',
                    help='Adjacency variant. Default v3b_only = v3a (T→Y=+1, '
                         'X→T=+1, X→Y=+1) with symmetric -1 completions '
                         '(diagonals 0). Alternatives: noanc, v6a_only, focus4, etc.')
    p.add_argument('--density-primary-mode', default=None,
                    help='Which anc tag mirrors to unsuffixed pehe_raw/err_raw '
                         'keys. Default: same as --anc-mode variant tag '
                         "('v3b' for v3b_only, 'noanc' for noanc, etc.).")
    return p.parse_args()


_PRIMARY_TAG_FROM_MODE = {
    'v3b_only':  'v3b',
    'v6a_only':  'v6a',
    'v5a_only':  'v5a',
    'v5b_only':  'v5b',
    'v4a_only':  'v4a',
    'v6b_only':  'v6b',
    'ty_only':   'ty',
    'noanc':     'noanc',
}


def main():
    args = _parse_args()

    primary = args.density_primary_mode or _PRIMARY_TAG_FROM_MODE.get(
        args.anc_mode, args.anc_mode)

    env = os.environ.copy()
    env.update({
        'DATASET':              args.dataset,
        'OUT':                  args.outdir,
        'CKPT':                 args.ckpt,
        'REPO':                 args.repo,
        'CAUSALPFN':            args.causalpfn,
        'UWYK':                 args.uwyk,
        'EVAL_MAX_CONTEXT':     str(args.eval_max_context),
        'EVAL_CONTEXT_SEED':    str(args.eval_context_seed),
        'ANC_MODE':             args.anc_mode,
        'DENSITY_PRIMARY_MODE': primary,
    })

    script = os.path.join(args.repo, 'benchmarks', 'eval_graph2d',
                          'eval_graph2d_realcause.py')
    os.makedirs(args.outdir, exist_ok=True)

    print(f'[graph2d] dataset={args.dataset}  anc_mode={args.anc_mode}  '
          f'eval_max_context={args.eval_max_context}  '
          f'seed={args.eval_context_seed}', flush=True)

    subprocess.run(
        [sys.executable, '-u', script, '--dataset', args.dataset],
        env=env, check=True,
    )


if __name__ == '__main__':
    main()
