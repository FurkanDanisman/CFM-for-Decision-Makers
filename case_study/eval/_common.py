"""Shared plumbing for the case-study eval wrappers.

Each wrapper mirrors its `realcause_eval/Table1/` counterpart but, instead of a
single `--dataset`, sweeps the grid (case study × context size N) of npz cells
produced by `case_study/generation.py`. Every underlying `benchmarks/eval_*`
script already resolves case studies through a bare
`from scm_case_study_dataset import SCMCaseStudyDataset`, so we only have to:

  1. put THIS folder first on PYTHONPATH (shadows the DoPFN-pkl loader with our
     npz-backed `scm_case_study_dataset.py`), and
  2. export CASE_STUDY_DATA_ROOT / CASE_STUDY_N so the shim knows which cell.

The underlying scripts then run their real model logic unchanged and already
report the case-study ATE metric as L1 |ate_hat - true_ate|.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(EVAL_DIR))          # .../R-PFN
TABLE1_DIR = os.path.join(REPO, "realcause_eval", "Table1")

CASE_STUDIES = (
    "Observed_Confounder",
    "Backdoor_Criterion",
    "Observed_Mediator",
    "Observed_Mediator_and_Confounder",
    "Unobserved_Confounder",
    "Frontdoor_Criterion",
)


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--data-root", required=True,
                   help="generation.py output root, e.g. case_study/data_shift+2")
    p.add_argument("--context-sizes", nargs="*", type=int, default=[200, 500, 1000],
                   help="Context-size cells N to evaluate (default 200 500 1000).")
    p.add_argument("--cases", nargs="*", default=list(CASE_STUDIES),
                   choices=list(CASE_STUDIES),
                   help="Case studies to evaluate (default: all six).")
    p.add_argument("--outdir", required=True,
                   help="Per-realization npzs land at OUTDIR/<Case>_N<N>/<Case>_r<###>.npz.")
    p.add_argument("--repo", default=REPO, help="R-PFN repo root (has benchmarks/).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the command + key env for each cell instead of running.")
    return p


def cells(cases, ns):
    for case in cases:
        for n in ns:
            yield case, n


def cell_env(data_root: str, n: int, extra: dict | None = None,
             extra_pythonpath: list[str] | None = None) -> dict:
    """os.environ + the case-study selection + shim on PYTHONPATH (first)."""
    env = os.environ.copy()
    env["CASE_STUDY_DATA_ROOT"] = os.path.abspath(data_root)
    env["CASE_STUDY_N"] = str(n)
    parts = [EVAL_DIR] + (extra_pythonpath or [])
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(parts)
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def cell_outdir(outdir: str, case: str, n: int) -> str:
    d = os.path.join(outdir, f"{case}_N{n}")
    os.makedirs(d, exist_ok=True)
    return d


def launch(cmd, env, cwd=None, dry_run=False, label=""):
    if dry_run:
        keys = ("DATASET", "OUT", "CKPT", "CASE_STUDY_DATA_ROOT", "CASE_STUDY_N",
                "EVAL_MAX_CONTEXT", "ANC_MODE", "STD_MODE", "Y_SCALING")
        shown = {k: env[k] for k in keys if k in env}
        print(f"[dry-run] {label}\n          cmd: {' '.join(cmd)}\n"
              f"          cwd: {cwd or os.getcwd()}\n          env: {shown}", flush=True)
        return
    subprocess.run(cmd, env=env, cwd=cwd, check=True)
