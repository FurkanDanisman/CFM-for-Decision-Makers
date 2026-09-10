"""UWYK-1D RealCause eval through OUR graph2d harness.

Loads UWYK's own 1D bar-distribution checkpoint (best_model.pt from the
full_conditioned_model dir) and runs it through
    benchmarks/eval_graph2d/eval_uwyk1d_realcause.py
which shares dataset loading, PSID balancing, context handling, PAM
construction, and npz layout with the graph2d harness. Emits BOTH v3b
(v3a + symmetric -1 completions) and noanc keys per realization NPZ:
    pehe_raw_v3b, err_raw_v3b, pehe_raw_noanc, err_raw_noanc

Used for the mega comparison table where we want an ancestor-info variant
for UWYK's 1D checkpoint (UWYK's own dofm_no_clustering.py / dofm_psid_balanced.py
only expose graph_mode ∈ {all_unknown, full_graph}, not v3b).

Defaults:
    ANC_MODE           = v3b_only    (emits both v3b + noanc)
    T_ENCODING         = target      (what UWYK's own dofm_no_clustering.py does)
    EVAL_MAX_CONTEXT   = 1000        (matches UWYK's training regime)
    EVAL_CONTEXT_SEED  = 43
    X_CLIP_QUANTILE    = 0.99        (matches UWYK's best_model_config.yaml
                                      remove_outliers=true, outlier_quantile=0.99;
                                      the checkpoint expects clipped X)

Usage:
    python realcause_eval/uwyk1d.py \\
        --dataset IHDP \\
        --outdir /scratch/.../results/uwyk1d/IHDP \\
        --uwyk-repro /scratch/.../external/uwyk_reproduce \\
        --repo /scratch/.../R-PFN \\
        --causalpfn /scratch/.../external/causalpfn
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
    p.add_argument('--uwyk-repro', required=True,
                    help='UWYK reproduce-branch repo root (holds best_model.pt).')
    p.add_argument('--repo', required=True, help='R-PFN repo root.')
    p.add_argument('--causalpfn', required=True,
                    help='CausalPFN repo root (for benchmarks / dataset loaders).')
    p.add_argument('--ckpt', default=None,
                    help='UWYK checkpoint (.pt). Default: full_conditioned_model/best_model.pt.')
    p.add_argument('--config', default=None,
                    help='UWYK config YAML. Default: full_conditioned_model/best_model_config.yaml.')
    p.add_argument('--anc-mode', default='v3b_only',
                    help='Adjacency variant. Default v3b_only emits BOTH v3b + noanc.')
    p.add_argument('--t-encoding', default='target', choices=['binary', 'target'],
                    help="'target' = feed T ← mean(Y|T) (matches UWYK dofm scripts). "
                         "'binary' = feed T ∈ {0,1}. Default: target.")
    p.add_argument('--eval-max-context', type=int, default=1000,
                    help='Cap on training context. Default: 1000 (matches UWYK training).')
    p.add_argument('--eval-context-seed', type=int, default=43,
                    help='Deterministic subsampling seed. Default: 43.')
    p.add_argument('--x-clip-quantile', type=float, default=0.99,
                    help='Per-column 99th-quantile clip on X before standardization. '
                         'Matches UWYK config remove_outliers=true, outlier_quantile=0.99. '
                         'Default: 0.99. Pass 0 to disable.')
    p.add_argument('--density-anc-tag', default='noanc',
                    help='Which mode\'s p_y0/p_y1 marginals to persist as the '
                         "unsuffixed density in the NPZ (default 'noanc', matches "
                         "paper Table 3 UWYK-NoAnc). Set to 'v3b' when you want "
                         "density-CI / density-metrics on the v3b variant.")
    return p.parse_args()


def main():
    args = _parse_args()

    uwyk_root = os.path.abspath(args.uwyk_repro)
    ckpt_dir = os.path.join(
        uwyk_root, 'experiments', 'checkpoints', 'full_conditioned_model',
        'final_earlytest_full_conditioning_16773252.0',
    )
    ckpt   = args.ckpt   or os.path.join(ckpt_dir, 'best_model.pt')
    config = args.config or os.path.join(ckpt_dir, 'best_model_config.yaml')
    for f, label in ((ckpt, 'checkpoint'), (config, 'config')):
        if not os.path.isfile(f):
            sys.exit(f'FATAL: UWYK {label} not found at {f} — did you `git lfs pull`?')

    os.makedirs(args.outdir, exist_ok=True)

    env = os.environ.copy()
    env.update({
        'DATASET':           args.dataset,
        'OUT':               args.outdir,
        'CKPT':              ckpt,
        'CONFIG':            config,
        'UWYK':              uwyk_root,
        'CAUSALPFN':         args.causalpfn,
        'ANC_MODE':          args.anc_mode,
        'T_ENCODING':        args.t_encoding,
        'EVAL_MAX_CONTEXT':  str(args.eval_max_context),
        'EVAL_CONTEXT_SEED': str(args.eval_context_seed),
        'X_CLIP_QUANTILE':   str(args.x_clip_quantile) if args.x_clip_quantile > 0 else '',
        'DENSITY_ANC_TAG':   args.density_anc_tag,
        'PYTHONUNBUFFERED':  '1',
    })
    shim = os.path.join(args.repo, 'benchmarks', 'uwyk_table1', 'shims')
    env['PYTHONPATH'] = shim + (f':{env["PYTHONPATH"]}' if env.get('PYTHONPATH') else '')

    script = os.path.join(args.repo, 'benchmarks', 'eval_graph2d',
                          'eval_uwyk1d_realcause.py')

    print(f'[uwyk1d] dataset={args.dataset}  anc_mode={args.anc_mode}  '
          f'T_encoding={args.t_encoding}  '
          f'eval_max_context={args.eval_max_context}  '
          f'seed={args.eval_context_seed}  '
          f'x_clip_quantile={args.x_clip_quantile}', flush=True)

    subprocess.run([sys.executable, '-u', script], env=env, check=True)


if __name__ == '__main__':
    main()
