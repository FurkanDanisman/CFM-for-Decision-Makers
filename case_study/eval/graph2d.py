"""graph2d case-study eval — sweeps (case study × N) through the SAME underlying
script Table1/graph2d.py uses (benchmarks/eval_graph2d/eval_graph2d_realcause.py),
with our npz loader shadowed in via PYTHONPATH. See _common.py.

Note: eval_graph2d_realcause.py builds the adjacency hint (PAM) from the case
study's own graph when DATASET is a case study, so the ancestor-info variants
(--anc-mode) use the correct case topology automatically.

Usage:
    python case_study/eval/graph2d.py \\
        --data-root case_study/data_shift+2 --context-sizes 200 500 1000 \\
        --outdir /scratch/.../cs_eval/graph2d \\
        --ckpt /scratch/.../graph2d_step_50000.pt \\
        --causalpfn /scratch/.../external/causalpfn --uwyk /scratch/.../external/uwyk_reproduce
"""
from __future__ import annotations

import argparse
import os

import _common as C

_PRIMARY_TAG = {"v3b_only": "v3b", "v6a_only": "v6a", "v5a_only": "v5a",
                "v5b_only": "v5b", "v4a_only": "v4a", "v6b_only": "v6b",
                "ty_only": "ty", "noanc": "noanc"}


def _args():
    p = argparse.ArgumentParser(); C.add_common_args(p)
    p.add_argument("--ckpt", required=True, help="graph2d checkpoint (.pt).")
    p.add_argument("--causalpfn", required=True, help="CausalPFN repo root.")
    p.add_argument("--uwyk", required=True, help="UWYK reproduce repo root.")
    p.add_argument("--eval-max-context", type=int, default=1000)
    p.add_argument("--eval-context-seed", type=int, default=43)
    p.add_argument("--x-clip-quantile", type=float, default=0.99)
    p.add_argument("--anc-mode", default="v3b_only")
    p.add_argument("--density-primary-mode", default=None)
    p.add_argument("--y-scaling", default="minmax", choices=["minmax", "std"])
    p.add_argument("--std-target", type=float, default=0.3)
    p.add_argument("--density-dump", default="0", choices=["0", "1"])
    return p.parse_args()


def main():
    a = _args()
    primary = a.density_primary_mode or _PRIMARY_TAG.get(a.anc_mode, a.anc_mode)
    script = os.path.join(a.repo, "benchmarks", "eval_graph2d",
                          "eval_graph2d_realcause.py")
    for case, n in C.cells(a.cases, a.context_sizes):
        out = C.cell_outdir(a.outdir, case, n)
        env = C.cell_env(a.data_root, n, {
            "DATASET": case, "OUT": out, "CKPT": a.ckpt, "REPO": a.repo,
            "CAUSALPFN": a.causalpfn, "UWYK": a.uwyk,
            "EVAL_MAX_CONTEXT": a.eval_max_context,
            "EVAL_CONTEXT_SEED": a.eval_context_seed,
            "ANC_MODE": a.anc_mode, "DENSITY_PRIMARY_MODE": primary,
            "X_CLIP_QUANTILE": a.x_clip_quantile if a.x_clip_quantile > 0 else "",
            "Y_SCALING": a.y_scaling, "STD_TARGET": a.std_target,
            "DENSITY_DUMP": a.density_dump,
        })
        print(f"[graph2d] {case} N={n} anc_mode={a.anc_mode} "
              f"ctx_cap={a.eval_max_context}", flush=True)
        C.launch(["python", "-u", script, "--dataset", case], env,
                 dry_run=a.dry_run, label=f"graph2d {case} N={n}")


if __name__ == "__main__":
    main()
