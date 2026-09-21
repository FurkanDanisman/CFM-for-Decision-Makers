#!/usr/bin/env python
"""Build a ComplexMech cell containing ONLY strongly-coupled realizations:
generate in batches, keep those with arm-noise rho > threshold, stop at a target
count. Output is an ordinary cmech data root, so every downstream eval works on it
unchanged.

WHY. ComplexMech scores coverage against `true_cate = Y_do1 - Y_do0`, a NOISY tau,
while every other benchmark uses the noiseless `mu_1 - mu_0`. The shared noise
cancels between the paired passes exactly when the arms are coupled, so at rho -> 1
the target collapses onto the noiseless CATE a CATE-predicting head actually
predicts. Restricting to rho > 0.99 therefore isolates the head's calibration from
the benchmark's target definition: whatever undercoverage SURVIVES here cannot be
blamed on the noisy target.

The bucketed run already showed coverage rising while interval LENGTH falls (e.g.
dopfn_repro_joint2d n=20: Cov 0.7585 -> 0.9554 as Len 0.4612 -> 0.3201), which no
model can do by predicting narrower -- that is the target shrinking. This script
takes it to the limit with a purpose-built sample rather than a 37-realization
tail of the existing one.

rho is `ols-on-X`, an UPPER bound on |rho|: nonlinearity the linear fit misses
stays in the residual. At a 0.99 threshold that matters -- some accepted
realizations will have a true rho below 0.99 -- so the manifest records every
accepted rho and the acceptance rate, and `--rho-report` prints the distribution
so the cut can be judged rather than assumed.

    python benchmarks/make_cmech_rho_filtered.py --nodes 5 --target-real 100 \
        --n-test 100 --rho-min 0.99 --out-root $SCRATCH/cmech_data_rho99
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_GEN = os.path.join(_REPO, "UWYK_Fig3_4", "generate_pehe_benchmark.py")


def _ols_resid(X, y):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, float).ravel()
    A = np.hstack([np.ones((X.shape[0], 1)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def rho_of(path):
    """(rho, n_queries) for one realization file; rho is nan if not computable."""
    try:
        z = np.load(path, allow_pickle=True)
    except Exception:
        return float("nan"), 0
    if not {"Y_do0", "Y_do1", "X_test"} <= set(z.files):
        return float("nan"), 0
    y0 = np.asarray(z["Y_do0"], float).ravel()
    y1 = np.asarray(z["Y_do1"], float).ravel()
    X = np.asarray(z["X_test"], float)
    n = y0.size
    if n < 8 or X.shape[0] != n:
        return float("nan"), n
    e0, e1 = _ols_resid(X, y0), _ols_resid(X, y1)
    if e0.std() <= 0 or e1.std() <= 0:
        return float("nan"), n
    return float(np.corrcoef(e0, e1)[0, 1]), n


def cell_dir(root, prior, n, regime, hide):
    return os.path.join(root, prior, f"{n}node", regime, f"hide_{hide}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=5)
    ap.add_argument("--target-real", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--rho-min", type=float, default=0.99)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--stage-root", default=None)
    ap.add_argument("--batch", type=int, default=100,
                    help="realizations generated per batch")
    ap.add_argument("--max-batches", type=int, default=40)
    ap.add_argument("--prior", default="complexmech")
    ap.add_argument("--regime", default="path_TY")
    ap.add_argument("--hide", type=float, default=0.0)
    ap.add_argument("--seed-base", type=int, default=100000)
    ap.add_argument("--keep-stage", action="store_true",
                    help="keep the raw generated batches instead of deleting them")
    ap.add_argument("--rho-report", action="store_true", default=True)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra args passed through to generate_pehe_benchmark")
    a = ap.parse_args()

    out_cell = cell_dir(a.out_root, a.prior, a.nodes, a.regime, a.hide)
    os.makedirs(out_cell, exist_ok=True)
    # Per-node stage dir. Batch dirs are named b0, b1, ... and the cleanup
    # rmtree()s them, so a shared _stage would let one node count delete
    # another's in-flight batch when several run concurrently.
    stage_root = a.stage_root or os.path.join(a.out_root, "_stage",
                                              f"n{a.nodes}")

    # Resume: count what is already accepted so a re-run tops up rather than restarts.
    kept = sorted(glob.glob(os.path.join(out_cell, "r*.npz")),
                  key=lambda p: int(os.path.basename(p)[1:-4]))
    n_kept = len(kept)
    if n_kept:
        print(f"resuming: {n_kept} realization(s) already accepted in {out_cell}")

    # Which SOURCE files were already accepted. Without this a resume that reuses
    # kept stage batches would copy the same realization twice, since the output
    # names are renumbered and carry no trace of where they came from.
    mp = os.path.join(a.out_root, f"manifest_rho{a.rho_min}_n{a.nodes}.json")
    used_src: set = set()
    prev_rho: list = []
    if os.path.isfile(mp):
        try:
            with open(mp) as fh:
                _prev = json.load(fh)
            used_src = set(_prev.get("accepted_sources", []))
            # Carry the earlier run's rho values forward too, so the manifest
            # describes the whole cell and not just this invocation.
            prev_rho = [float(x) for x in _prev.get("accepted_rho", [])]
            if used_src:
                print(f"resuming: {len(used_src)} source file(s) already consumed")
        except Exception:
            pass

    seen_rho, accepted_rho, n_gen = [], list(prev_rho), 0
    t0 = time.time()
    for b in range(a.max_batches):
        if n_kept >= a.target_real:
            break
        stage = os.path.join(stage_root, f"b{b}")
        s_cell = cell_dir(stage, a.prior, a.nodes, a.regime, a.hide)
        if not glob.glob(os.path.join(s_cell, "r*.npz")):
            cmd = [sys.executable, "-u", _GEN,
                   "--prior", a.prior,
                   "--nodes", str(a.nodes),
                   "--regimes", a.regime,
                   "--hide-fractions", str(a.hide),
                   "--n-realizations", str(a.batch),
                   "--seed-base", str(a.seed_base + b * 7919),
                   "--out-dir", stage]
            if a.n_test:
                cmd += ["--n-test", str(a.n_test)]
            cmd += [x for x in a.extra if x != "--"]
            print(f"\n[batch {b}] generating {a.batch} realization(s) "
                  f"(seed-base {a.seed_base + b * 7919}) ...", flush=True)
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"[batch {b}] generator exited {r.returncode}; stopping",
                      file=sys.stderr)
                break

        files = sorted(glob.glob(os.path.join(s_cell, "r*.npz")),
                       key=lambda p: int(os.path.basename(p)[1:-4]))
        n_gen += len(files)
        n_new = 0
        for f in files:
            if n_kept >= a.target_real:
                break
            if os.path.abspath(f) in used_src:
                continue
            rho, nq = rho_of(f)
            if not np.isfinite(rho):
                continue
            seen_rho.append(rho)
            if rho <= a.rho_min:
                continue
            if a.n_test and nq < a.n_test:
                continue                      # too few queries to meet the spec
            dst = os.path.join(out_cell, f"r{n_kept}.npz")
            shutil.copy2(f, dst)
            accepted_rho.append(rho)
            used_src.add(os.path.abspath(f))
            n_kept += 1
            n_new += 1
        rate = ((len(accepted_rho) - len(prev_rho)) / len(seen_rho)) if seen_rho else 0.0
        print(f"[batch {b}] +{n_new} accepted  |  total {n_kept}/{a.target_real}"
              f"  |  acceptance {rate:.1%} over {len(seen_rho)} scored"
              f"  |  {time.time() - t0:.0f}s", flush=True)
        if not a.keep_stage:
            shutil.rmtree(stage, ignore_errors=True)

    man = {"nodes": a.nodes, "prior": a.prior, "regime": a.regime, "hide": a.hide,
           "rho_min": a.rho_min, "n_test": a.n_test,
           "target_real": a.target_real, "accepted": n_kept,
           "generated_scored": len(seen_rho),
           "acceptance_rate": ((len(accepted_rho) - len(prev_rho)) / len(seen_rho))
           if seen_rho else None,
           "accepted_rho": accepted_rho, "cell_dir": out_cell,
           "accepted_sources": sorted(used_src),
           "rho_estimator": "ols-on-X (upper bound on |rho|)"}
    with open(mp, "w") as fh:
        json.dump(man, fh, indent=2)

    print(f"\naccepted {n_kept}/{a.target_real} into {out_cell}")
    print(f"manifest: {mp}")
    if a.rho_report and seen_rho:
        s = np.asarray(seen_rho)
        print(f"\nrho over {s.size} generated realization(s):")
        for q in (10, 25, 50, 75, 90, 99):
            print(f"  p{q:<3d} {np.percentile(s, q):+.4f}")
        for t in (0.8, 0.9, 0.95, 0.99, 0.999):
            print(f"  frac > {t:<6} {float((s > t).mean()):.1%}")
    if accepted_rho:
        ar = np.asarray(accepted_rho)
        print(f"\naccepted rho: min {ar.min():.5f}  median {np.median(ar):.5f}  "
              f"max {ar.max():.5f}")
    if n_kept < a.target_real:
        print(f"\nSHORT by {a.target_real - n_kept}. Raise --max-batches or lower "
              f"--rho-min; the acceptance rate above says how many more are needed.",
              file=sys.stderr)
        return 1
    print(f"\nNext: point the evals at this root --\n"
          f"  export UWYK_FIG34_DATA={a.out_root}\n"
          f"  (dump all models for CMECH_n{a.nodes}, then score as usual)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
