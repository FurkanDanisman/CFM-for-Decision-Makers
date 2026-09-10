"""UWYK-1D case-study eval — sweeps (case study × N) through the SAME underlying
script Table1/uwyk1d.py uses (benchmarks/eval_graph2d/eval_uwyk1d_realcause.py),
with our npz loader shadowed in via PYTHONPATH. See _common.py.

PYTHONPATH carries both our loader shim (first) and UWYK's own shim dir
(benchmarks/uwyk_table1/shims), exactly as Table1/uwyk1d.py sets up.

Usage:
    python case_study/eval/uwyk1d.py \\
        --data-root case_study/data_shift+2 --context-sizes 200 500 1000 \\
        --outdir /scratch/.../cs_eval/uwyk1d \\
        --uwyk-repro /scratch/.../external/uwyk_reproduce \\
        --causalpfn /scratch/.../external/causalpfn
"""
from __future__ import annotations

import argparse
import os
import sys

import _common as C


def _args():
    p = argparse.ArgumentParser(); C.add_common_args(p)
    p.add_argument("--uwyk-repro", required=True,
                   help="UWYK reproduce-branch repo root (holds best_model.pt).")
    p.add_argument("--causalpfn", required=True, help="CausalPFN repo root.")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--anc-mode", default="v3b_only")
    p.add_argument("--t-encoding", default="target", choices=["binary", "target"])
    p.add_argument("--eval-max-context", type=int, default=1000)
    p.add_argument("--eval-context-seed", type=int, default=43)
    p.add_argument("--x-clip-quantile", type=float, default=0.99)
    p.add_argument("--density-anc-tag", default="noanc")
    return p.parse_args()


def main():
    a = _args()
    uwyk_root = os.path.abspath(a.uwyk_repro)
    ckpt_dir = os.path.join(uwyk_root, "experiments", "checkpoints",
                            "full_conditioned_model",
                            "final_earlytest_full_conditioning_16773252.0")
    ckpt = a.ckpt or os.path.join(ckpt_dir, "best_model.pt")
    config = a.config or os.path.join(ckpt_dir, "best_model_config.yaml")
    for f, label in ((ckpt, "checkpoint"), (config, "config")):
        if not a.dry_run and not os.path.isfile(f):
            sys.exit(f"FATAL: UWYK {label} not found at {f} — did you `git lfs pull`?")

    script = os.path.join(a.repo, "benchmarks", "eval_graph2d",
                          "eval_uwyk1d_realcause.py")
    uwyk_shim = os.path.join(a.repo, "benchmarks", "uwyk_table1", "shims")
    for case, n in C.cells(a.cases, a.context_sizes):
        out = C.cell_outdir(a.outdir, case, n)
        env = C.cell_env(a.data_root, n, {
            "DATASET": case, "OUT": out, "CKPT": ckpt, "CONFIG": config,
            "UWYK": uwyk_root, "CAUSALPFN": a.causalpfn,
            "ANC_MODE": a.anc_mode, "T_ENCODING": a.t_encoding,
            "EVAL_MAX_CONTEXT": a.eval_max_context,
            "EVAL_CONTEXT_SEED": a.eval_context_seed,
            "X_CLIP_QUANTILE": a.x_clip_quantile if a.x_clip_quantile > 0 else "",
            "DENSITY_ANC_TAG": a.density_anc_tag, "PYTHONUNBUFFERED": "1",
        }, extra_pythonpath=[uwyk_shim])
        print(f"[uwyk1d] {case} N={n} anc_mode={a.anc_mode} "
              f"t_encoding={a.t_encoding} ctx_cap={a.eval_max_context}", flush=True)
        C.launch(["python", "-u", script], env, dry_run=a.dry_run,
                 label=f"uwyk1d {case} N={n}")


if __name__ == "__main__":
    main()
