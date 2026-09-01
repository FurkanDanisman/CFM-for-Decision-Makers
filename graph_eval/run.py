"""CLI wrapper around benchmarks/eval_graph2d/eval_graph2d_realcause.py.

Take a graph2d checkpoint + a RealCause dataset name, produce per-realization
anc / noanc PEHE + err_ATE. Wraps the existing eval by setting env vars, so any
bug fix in the eval script automatically applies here too.

See ./README.md for setup and examples.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Evaluate anc vs noanc for a graph2d checkpoint on a RealCause dataset.'
    )
    parser.add_argument('--ckpt', required=True,
                        help='Path to graph2d checkpoint (.pt)')
    parser.add_argument('--dataset', required=True,
                        choices=('IHDP', 'ACIC', 'CPS', 'PSID', 'PSID_bal'),
                        help='RealCause dataset name')
    parser.add_argument('--out', required=True,
                        help='Output directory for per-realization *.npz files')
    parser.add_argument('--max-context', type=int, default=1000,
                        help='Context subsample cap per realization (default 1000)')
    parser.add_argument('--propagate', type=int, default=1, choices=(0, 1),
                        help='Apply propagate_ancestor_knowledge to anc matrix (default 1)')
    parser.add_argument('--max-realizations', type=int, default=None,
                        help='Cap number of realizations (default: all)')
    parser.add_argument('--anc-mode', default='full',
                        choices=('full', 'ty_only', 'ty_antisym', 'all_variants'),
                        help='Anc matrix content variant (default: full)')
    parser.add_argument('--uwyk',
                        default=os.environ.get('UWYK'),
                        help='Path to UWYK repo root (or set UWYK env var)')
    parser.add_argument('--causalpfn',
                        default=os.environ.get('CAUSALPFN'),
                        help='Path to CausalPFN repo root (or set CAUSALPFN env var)')
    args = parser.parse_args()

    if not args.uwyk:
        sys.exit('ERROR: pass --uwyk or set UWYK env var to your UWYK repo root')
    if not args.causalpfn:
        sys.exit('ERROR: pass --causalpfn or set CAUSALPFN env var to your CausalPFN repo root')

    if not os.path.isfile(args.ckpt):
        sys.exit(f'ERROR: checkpoint not found at {args.ckpt}')

    os.makedirs(args.out, exist_ok=True)

    # Route env vars to the underlying eval script.
    os.environ['CKPT']              = os.path.abspath(args.ckpt)
    os.environ['DATASET']           = args.dataset
    os.environ['OUT']               = os.path.abspath(args.out)
    os.environ['UWYK']              = os.path.abspath(args.uwyk)
    os.environ['CAUSALPFN']         = os.path.abspath(args.causalpfn)
    os.environ['EVAL_MAX_CONTEXT']  = str(args.max_context)
    os.environ['PROPAGATE_ANC']     = str(args.propagate)
    os.environ['ANC_MODE']          = args.anc_mode
    if args.max_realizations is not None:
        os.environ['MAX_REAL'] = str(args.max_realizations)

    # Set up sys.path so the eval script's own imports resolve. Path lookup
    # mirrors what the sbatch does on the cluster.
    HERE = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.abspath(os.path.join(HERE, os.pardir))
    for p in (
        REPO,                                                 # so `benchmarks.*`, `training_graph2d.*` resolve
        os.path.join(REPO, 'benchmarks', 'uwyk_table1', 'shims'),  # optional wandb/faiss shims
    ):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    # Now delegate to the actual eval.
    from benchmarks.eval_graph2d import eval_graph2d_realcause  # noqa: F401
    # The eval script runs its main() at import time via `if __name__ == '__main__'`
    # only if imported directly. To make sure it runs, call main() explicitly.
    if hasattr(eval_graph2d_realcause, 'main'):
        eval_graph2d_realcause.main()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
