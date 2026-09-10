"""Do-PFN (native) case-study eval — sweeps (case study × N).

Table1/do_pfn.py is self-contained (no underlying benchmarks script) and its
loader only knows the RealCause datasets, so here we import its verbatim
pipeline + regressor builder + PEHE helper and drive them with OUR npz loader.

ATE error is reported as L1 |ate_hat - true_ate| — the convention every other
case-study eval script uses (Table1/do_pfn used a relative error, which is for
RealCause; case studies use L1, and our shift keeps true_ate away from 0).

Usage:
    python case_study/eval/do_pfn.py \\
        --data-root case_study/data_shift+2 --context-sizes 200 500 1000 \\
        --outdir /scratch/.../cs_eval/do_pfn \\
        --dopfn /scratch/.../external/dopfn --causalpfn /scratch/.../external/causalpfn
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

import _common as C


def _args():
    p = argparse.ArgumentParser(); C.add_common_args(p)
    p.add_argument("--dopfn", required=True, help="dopfn upstream repo root.")
    p.add_argument("--causalpfn", required=True, help="CausalPFN repo root.")
    p.add_argument("--max-real", type=int, default=0, help="Cap realizations (0=all).")
    return p.parse_args()


def main():
    a = _args()
    grid = list(C.cells(a.cases, a.context_sizes))

    if a.dry_run:
        for case, n in grid:
            out = C.cell_outdir(a.outdir, case, n)
            print(f"[dry-run] do_pfn {case} N={n} -> {out}  "
                  f"(root={os.path.abspath(a.data_root)})", flush=True)
        return

    # Heavy deps only for a real run. Reuse Table1/do_pfn's verbatim pipeline.
    sys.path.insert(0, a.causalpfn)
    sys.path.insert(0, a.causalpfn + "/src")
    sys.path.insert(0, a.dopfn)
    sys.path.insert(0, C.TABLE1_DIR)   # Table1/do_pfn.py
    sys.path.insert(0, C.EVAL_DIR)     # our loader shim
    import torch
    import do_pfn as t1                                  # realcause_eval/Table1/do_pfn.py
    from scm_case_study_dataset import SCMCaseStudyDataset

    t1._install_check_array_shim()
    prev_cwd = os.getcwd()
    os.chdir(a.dopfn)
    try:
        from scripts.transformer_prediction_interface.base import DoPFNRegressor
        reg = t1._build_dopfn_regressor(DoPFNRegressor)   # build ONCE, reuse
    finally:
        os.chdir(prev_cwd)

    os.environ["CASE_STUDY_DATA_ROOT"] = os.path.abspath(a.data_root)
    t0 = time.time()
    for case, n in grid:
        os.environ["CASE_STUDY_N"] = str(n)
        ds = SCMCaseStudyDataset(case)
        out = C.cell_outdir(a.outdir, case, n)
        n_real = ds.n_tables if a.max_real <= 0 else min(ds.n_tables, a.max_real)
        print(f"[do_pfn] {case} N={n}  n_reals={n_real} -> {out}", flush=True)
        for r in range(n_real):
            cd, _ = ds[r]
            true_cate = np.asarray(cd.true_cate).reshape(-1)
            os.chdir(a.dopfn)
            try:
                cate_pred = t1.dopfn_pipeline(cd, reg)
            finally:
                os.chdir(prev_cwd)
            pehe = t1._pehe(true_cate, cate_pred)
            true_ate = float(true_cate.mean())
            ate_hat = float(np.asarray(cate_pred).mean())
            err = abs(ate_hat - true_ate)             # L1 (case-study convention)
            np.savez(
                os.path.join(out, f"{case}_r{r:03d}.npz"),
                dataset=case, realization=r, n_context=n,
                pehe_dopfn=np.float64(pehe), err_dopfn=np.float64(err),
                true_ate=np.float64(true_ate), ate_pred=np.float64(ate_hat),
                true_cate=true_cate.astype(np.float32),
                cate_pred=np.asarray(cate_pred, dtype=np.float32),
            )
            print(f"  r={r:03d}  pehe={pehe:7.3f}  errL1={err:6.3f}  "
                  f"(true_ate={true_ate:+.3f}, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
