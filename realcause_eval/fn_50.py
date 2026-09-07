"""fn=50 (Ours) RealCause eval — wrapper around ours_fn50_no_clustering.py.

Reproduces Table 3 "UWYK No-Graph 2D" (aka fn=50 in your terminology) using
the working pipeline at:
    benchmarks/uwyk_table1/ours_fn50_no_clustering.py

Per-dataset flags (matches the paper's job submissions):
  IHDP / ACIC / PSID (unbal):  single forward, no clustering
  CPS:                          --subsample --n-repeats 20 --max-n-train 1000
                                (bagged random subsets, preserves marginal X)
  PSID_bal:                     --psid-balanced
                                (keep all T=1, subsample 500 T=0 per realization)

Paper targets (Table 3 UWYK No-Graph 2D row):
    IHDP 4.32   ACIC 2.86   CPS 12688   PSID 21293   PSID_bal 18500

Verified reproduction (aggregate_all_datasets.py output) matches within
rounding. Pickle-per-realization output at
`$TABLE1_OUT_ROOT/<exp_name>/ours_fn50_<DATASET>_<r>` with keys:
    {model, dataset, realization, pehe, ate_rel_err, cate_preds, true_ate,
     n_queries, n_context}

Usage — one dataset:
    python realcause_eval/fn_50.py \\
        --dataset IHDP \\
        --exp-name my_fn50_run \\
        --out /scratch/.../results_table1_ours_fn50 \\
        --ckpt /scratch/.../checkpoints/step_50000_final.pt \\
        --causalpfn /scratch/.../external/causalpfn

The wrapper picks the right flags for each dataset automatically. Override
via extra CLI args if needed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


# Per-dataset flag templates.
_DATASET_FLAGS = {
    'IHDP':     [],
    'ACIC':     [],
    'PSID':     [],   # PSID unbalanced — single forward
    'CPS':      ['--subsample', '--n-repeats', '20', '--max-n-train', '1000'],
    'PSID_bal': ['--psid-balanced'],
}


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True,
                    choices=list(_DATASET_FLAGS.keys()),
                    help='RealCause dataset. Wrapper auto-selects the correct '
                         'inference-mode flags per dataset (see module docstring).')
    p.add_argument('--exp-name', required=True,
                    help='Experiment name — output goes to $out/<exp_name>/.')
    p.add_argument('--out', required=True,
                    help='TABLE1_OUT_ROOT — per-realization pickles land at '
                         'OUT/<exp_name>/ours_fn50_<DATASET>_<r>.')
    p.add_argument('--ckpt', required=True, help='fn=50 checkpoint (.pt).')
    p.add_argument('--causalpfn', required=True,
                    help='CausalPFN repo root (for benchmarks / dataset loaders).')
    p.add_argument('--repo', default=None,
                    help='R-PFN repo root. Default: two levels up from this file.')
    p.add_argument('--extra-args', nargs='*', default=[],
                    help='Extra args forwarded to ours_fn50_no_clustering.py. '
                         'These override the wrapper-chosen per-dataset flags.')
    return p.parse_args()


def main():
    args = _parse_args()

    repo = args.repo or os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..'))
    script = os.path.join(repo, 'benchmarks', 'uwyk_table1',
                          'ours_fn50_no_clustering.py')

    # PSID_bal → runs on the PSID loader with --psid-balanced set. The
    # underlying script only accepts {IHDP, ACIC, CPS, PSID}, so translate.
    underlying_dataset = 'PSID' if args.dataset == 'PSID_bal' else args.dataset

    env = os.environ.copy()
    env['CAUSALPFN_ROOT']    = args.causalpfn
    env['OURS_CKPT']         = args.ckpt
    env['TABLE1_OUT_ROOT']   = args.out

    cmd = [
        sys.executable, '-u', script,
        '--dataset',  underlying_dataset,
        '--model',    'ours_fn50',
        '--exp_name', args.exp_name,
    ] + _DATASET_FLAGS[args.dataset] + list(args.extra_args)

    print(f'[fn_50] dataset={args.dataset}  '
          f'underlying_dataset={underlying_dataset}  '
          f'flags={_DATASET_FLAGS[args.dataset]}  '
          f'extra_args={args.extra_args}', flush=True)

    subprocess.run(cmd, env=env, check=True)


if __name__ == '__main__':
    main()
