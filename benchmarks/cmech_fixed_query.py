#!/usr/bin/env python
"""ComplexMech with the SAME queries under repeated observational samples.

One accepted realization gives its SCM, its query units and their potential
outcomes; only the training sample is redrawn, DRAWS times. Every query therefore
keeps a fixed true effect, and the fraction of a model's intervals that cover it is
frequentist coverage for a single estimand -- not coverage averaged over queries and
DGP draws, which is what the main tables report.

WHY NOT REPRODUCE AN EXISTING REALIZATION. Rebuilding one from its npz would need
the original --seed-base AND the SCMSampler seed, and the file records neither. So
fresh realizations are generated here under the same rho > 0.99 filter that built
the existing cells; nothing is re-derived and nothing has to match an old run.

THE UNITS PROBLEM, and why true_cate is not used. ComplexMech's stored true_cate is
in PROCESSED units: the outcome is squashed toward [-1, 1] by a scaling the
processor refits on whatever training sample it is handed. Redraw that sample and
the same real effect is recorded as a different number -- the estimand moves, and
"coverage of one fixed value" becomes meaningless. generate_realization now also
stores `true_cate_raw` in the SCM's own units, which does NOT move, plus the
recovered processed = a + b*raw affine. Coverage is taken in raw units: a model's
interval, which it reports in processed units, is mapped back by (x - a)/b.

That recovery is exact only while the transform IS affine. Outlier clipping at the
0.99 quantile is not, so `target_affine_resid` is checked and a realization whose
residual is large is REJECTED rather than scaled by a fitted b that does not apply.

    python benchmarks/cmech_fixed_query.py --nodes 5 --target-real 10 \
        --draws 100 --n-test 100 --queries 10 --rho-min 0.99 \
        --out-root $SCRATCH/cmech_fq
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "UWYK_Fig3_4"))

from generate_pehe_benchmark import (                      # noqa: E402
    _import_uwyk, _save, generate_realization, resolve_config,
)
sys.path.insert(0, _HERE)
from make_cmech_rho_filtered import _ols_resid             # noqa: E402


def rho_of_rec(rec):
    """Arm-noise rho for one record, the same `ols-on-X` upper bound as the cells."""
    y0 = np.asarray(rec["Y_do0"], float).ravel()
    y1 = np.asarray(rec["Y_do1"], float).ravel()
    X = np.asarray(rec["X_test"], float)
    if y0.size < 8 or X.shape[0] != y0.size:
        return float("nan")
    e0, e1 = _ols_resid(X, y0), _ols_resid(X, y1)
    if e0.std() <= 0 or e1.std() <= 0:
        return float("nan")
    return float(np.corrcoef(e0, e1)[0, 1])


def reorder(rec, qidx):
    """Move the chosen query rows to positions 0..K-1, in every test-side array.

    The eval harnesses take the FIRST n_query test rows, so a random choice of
    queries only takes effect if those rows are moved to the front. Train-side
    arrays are untouched -- they are the sample being resampled, not the queries.
    """
    n = np.asarray(rec["Y_do0"]).ravel().size
    order = np.r_[qidx, np.delete(np.arange(n), qidx)]
    out = dict(rec)
    for k in ("X_test", "T_test_do0", "T_test_do1", "Y_do0", "Y_do1", "true_cate",
              "Y_do0_raw", "Y_do1_raw", "true_cate_raw"):
        if k in out:
            v = np.asarray(out[k])
            if v.shape and v.shape[0] == n:
                out[k] = v[order]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=5)
    ap.add_argument("--regime", default="path_TY")
    ap.add_argument("--hide", type=float, default=0.0)
    ap.add_argument("--prior", default="complexmech")
    ap.add_argument("--target-real", type=int, default=10,
                    help="accepted realizations to keep")
    ap.add_argument("--draws", type=int, default=100,
                    help="observational samples per realization")
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--queries", type=int, default=10,
                    help="query rows to keep, chosen at random and moved to front")
    ap.add_argument("--rho-min", type=float, default=0.99)
    ap.add_argument("--max-affine-resid", type=float, default=1e-3,
                    help="reject a realization whose processed-vs-raw map is not "
                         "affine to this tolerance (outlier clipping breaks it)")
    ap.add_argument("--min-tau-het", type=float, default=0.05,
                    help="same band the existing cells use; 0 would admit "
                         "realizations where every query shares one truth")
    ap.add_argument("--max-tau-het", type=float, default=float("inf"))
    ap.add_argument("--seed-base", type=int, default=7_000_000)
    ap.add_argument("--max-attempts", type=int, default=2000)
    ap.add_argument("--query-seed", type=int, default=12345)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()

    cfg, cfg_path, synth = resolve_config(a.prior, a.nodes, a.regime, a.hide)
    uwyk = _import_uwyk()
    SCMSampler = uwyk[0]
    cell = os.path.join(a.out_root, a.prior, f"{a.nodes}node", a.regime,
                        f"hide_{a.hide}")
    os.makedirs(cell, exist_ok=True)
    print(f"[cmech-fq] nodes={a.nodes} regime={a.regime} hide={a.hide}"
          f"{' (synth cfg)' if synth else ''}")
    print(f"  target {a.target_real} realizations x {a.draws} samples, "
          f"{a.queries} of {a.n_test} queries, rho >= {a.rho_min}")
    print(f"  out: {cell}")

    qrng = np.random.default_rng(a.query_seed)
    kept, rejected = [], {"rho": 0, "affine": 0, "error": 0, "moving_truth": 0}
    attempt = 0
    while len(kept) < a.target_real and attempt < a.max_attempts:
        seed = a.seed_base + 1_000_000 * attempt
        attempt += 1
        sampler = SCMSampler(cfg["scm_config"], seed=seed * 31 + 17)
        try:
            recs = generate_realization(
                cfg, sampler, uwyk, a.regime, seed=seed, hide_fraction=a.hide,
                test_feature_mask_fraction=0.0, n_test_override=a.n_test,
                min_tau_het=a.min_tau_het, max_tau_het=a.max_tau_het,
                n_train_draws=a.draws)
        except Exception as exc:                                  # noqa: BLE001
            rejected["error"] += 1
            print(f"  seed {seed}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if not isinstance(recs, list):
            recs = [recs]

        rho = rho_of_rec(recs[0])
        if not np.isfinite(rho) or rho < a.rho_min:
            rejected["rho"] += 1
            continue
        resid = max(float(r["target_affine_resid"]) for r in recs)
        if resid > a.max_affine_resid:
            rejected["affine"] += 1
            print(f"  seed {seed}: affine residual {resid:.2e} -- clipped, rejected",
                  flush=True)
            continue
        # The point of the whole exercise: the truth must NOT move between samples.
        t = np.stack([np.asarray(r["true_cate_raw"], float).ravel() for r in recs])
        spread = float(np.abs(t - t[0]).max())
        if spread > 1e-9:
            rejected["moving_truth"] += 1
            print(f"  seed {seed}: true_cate_raw moves by {spread:.2e} -- rejected",
                  flush=True)
            continue

        qidx = np.sort(qrng.choice(a.n_test, size=min(a.queries, a.n_test),
                                   replace=False))
        ri = len(kept)
        for d, rec in enumerate(recs):
            _save(reorder(rec, qidx), os.path.join(cell, f"r{ri}_d{d}.npz"))
        b = float(recs[0]["target_affine_b"])
        kept.append({"realization": ri, "seed": int(seed), "rho": rho,
                     "affine_resid": resid, "affine_b": b,
                     "query_index": [int(v) for v in qidx],
                     "true_cate_raw": [float(v) for v in t[0][qidx]]})
        print(f"  kept r{ri}: rho={rho:+.4f} resid={resid:.1e} b={b:.4g} "
              f"tau_raw min {t[0][qidx].min():+.4g} max {t[0][qidx].max():+.4g}",
              flush=True)

    man = {"prior": a.prior, "nodes": a.nodes, "regime": a.regime, "hide": a.hide,
           "draws": a.draws, "n_test": a.n_test, "queries": a.queries,
           "rho_min": a.rho_min, "attempts": attempt, "rejected": rejected,
           "cfg_path": cfg_path, "synth_cfg": bool(synth),
           "accepted": kept,
           "note": "coverage must be scored in RAW units: true_cate_raw is fixed "
                   "across the d0..d{draws-1} replicates, true_cate is NOT."}
    with open(os.path.join(cell, "manifest_fq.json"), "w") as fh:
        json.dump(man, fh, indent=2)

    print(f"\nkept {len(kept)}/{a.target_real} in {attempt} attempts; "
          f"rejected {rejected}")
    print(f"wrote {len(kept) * a.draws} npz to {cell}")
    if len(kept) < a.target_real:
        print("FAILED to reach the target -- widen --max-attempts or lower "
              "--rho-min", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
