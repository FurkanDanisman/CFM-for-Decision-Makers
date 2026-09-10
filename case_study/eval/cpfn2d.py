"""cpfn2d case-study eval — sweeps (case study × N) through the SAME underlying
script Table1/cpfn2d.py uses (benchmarks/eval_causalpfn2d/eval_cpfn2d_realcause.py),
with our npz loader shadowed in via PYTHONPATH. See _common.py.

Usage:
    python case_study/eval/cpfn2d.py \\
        --data-root case_study/data_shift+2 --context-sizes 200 500 1000 \\
        --outdir /scratch/.../cs_eval/cpfn2d \\
        --ckpt /scratch/.../cpfn2d_j32_random_.../step_0050000.pt \\
        --causalpfn /scratch/.../external/causalpfn
"""
from __future__ import annotations

import argparse
import os

import _common as C


def _args():
    p = argparse.ArgumentParser(); C.add_common_args(p)
    p.add_argument("--ckpt", required=True, help="cpfn2d checkpoint (.pt).")
    p.add_argument("--causalpfn", required=True, help="CausalPFN repo root.")
    p.add_argument("--std-mode", default="per_arm",
                   choices=["pooled", "per_arm", "log", "log_per_arm", "log_winsor",
                            "std_target", "per_arm_std_target"])
    p.add_argument("--std-target", type=float, default=3.0)
    p.add_argument("--compile", default="0", choices=["0", "1"])
    p.add_argument("--skip-em", default="1", choices=["0", "1"])
    p.add_argument("--density-dump", default="0", choices=["0", "1"])
    return p.parse_args()


def main():
    a = _args()
    script = os.path.join(a.repo, "benchmarks", "eval_causalpfn2d",
                          "eval_cpfn2d_realcause.py")
    for case, n in C.cells(a.cases, a.context_sizes):
        out = C.cell_outdir(a.outdir, case, n)
        env = C.cell_env(a.data_root, n, {
            "DATASET": case, "OUT": out, "CKPT": a.ckpt, "CAUSALPFN": a.causalpfn,
            "EVAL_MAX_CONTEXT": "", "EVAL_CONTEXT_SEED": "1",
            "STD_MODE": a.std_mode, "STD_TARGET": a.std_target,
            "COMPILE": a.compile, "SKIP_EM": a.skip_em, "DENSITY_DUMP": a.density_dump,
        })
        print(f"[cpfn2d] {case} N={n} std_mode={a.std_mode} "
              f"ckpt={os.path.basename(a.ckpt)}", flush=True)
        C.launch(["python", "-u", script, "--dataset", case], env,
                 dry_run=a.dry_run, label=f"cpfn2d {case} N={n}")


if __name__ == "__main__":
    main()
