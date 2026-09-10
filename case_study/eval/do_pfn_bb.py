"""Do-PFN-bb case-study eval — sweeps (case study × N) through the SAME
underlying script Table1/do_pfn_bb.py uses (benchmarks/l2_ihdp/eval_dopfn_bb_raw.py),
with our npz loader shadowed in via PYTHONPATH. See _common.py.

Unlike the cpfn/graph wrappers this underlying script is CLI-arg-driven (not
env DATASET/OUT) and must run from the dopfn upstream root.

Usage:
    python case_study/eval/do_pfn_bb.py \\
        --data-root case_study/data_shift+2 --context-sizes 200 500 1000 \\
        --outdir /scratch/.../cs_eval/do_pfn_bb \\
        --ckpt /scratch/.../Required_checkpoints/dopfn_bb_j10_step_150000.pt \\
        --dopfn /scratch/.../external/dopfn --causalpfn /scratch/.../external/causalpfn
"""
from __future__ import annotations

import argparse
import os

import _common as C


def _args():
    p = argparse.ArgumentParser(); C.add_common_args(p)
    p.add_argument("--ckpt", required=True, help="DoPFN-bb 2D-head checkpoint (.pt).")
    p.add_argument("--dopfn", required=True, help="dopfn upstream repo root.")
    p.add_argument("--causalpfn", required=True, help="CausalPFN repo root.")
    p.add_argument("--y-scaling", default="std",
                   choices=["std", "min_max", "iqr", "trim_min_max"])
    p.add_argument("--std-target", type=float, default=0.3)
    p.add_argument("--n-context", type=int, default=0, help="0 = full context.")
    p.add_argument("--extra-args", nargs="*", default=[])
    return p.parse_args()


def main():
    a = _args()
    script = os.path.join(a.repo, "benchmarks", "l2_ihdp", "eval_dopfn_bb_raw.py")
    for case, n in C.cells(a.cases, a.context_sizes):
        out = C.cell_outdir(a.outdir, case, n)
        cmd = ["python", "-u", script,
               "--repo", a.repo, "--dopfn", a.dopfn, "--causalpfn", a.causalpfn,
               "--checkpoint", a.ckpt, "--dataset", case,
               "--out", os.path.join(out, "summary.npz"),
               "--y-scaling", a.y_scaling, "--std-target", str(a.std_target),
               "--n-context", str(a.n_context)] + list(a.extra_args)
        # eval_dopfn_bb_raw.py runs from the dopfn root; the shim still resolves
        # via PYTHONPATH (absolute EVAL_DIR), so cwd does not matter for it.
        env = C.cell_env(a.data_root, n, {"DENSITY_DUMP": os.environ.get("DENSITY_DUMP", "1")})
        print(f"[do_pfn_bb] {case} N={n} y_scaling={a.y_scaling} "
              f"std_target={a.std_target} ckpt={os.path.basename(a.ckpt)}", flush=True)
        C.launch(cmd, env, cwd=a.dopfn, dry_run=a.dry_run,
                 label=f"do_pfn_bb {case} N={n}")


if __name__ == "__main__":
    main()
