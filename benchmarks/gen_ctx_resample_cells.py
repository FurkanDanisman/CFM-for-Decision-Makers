"""Context-resampling cells: fixed queries, fixed SCM, 100 independent contexts.

WHY THIS EXISTS. Coverage is a property of a procedure under repeated sampling of
the DATA with the target HELD FIXED. Neither of the three benchmarks supports
that as shipped: ComplexMech and the case studies redraw the SCM every
realization (no fixed target), and tau is deterministic given x there (rho = 1,
shared arm noise), so a predictive interval has no 95% level to converge to
either. RealCause resamples covariates, so its targets do not repeat.

In-context models let us construct the missing experiment directly: hold one
query point fixed and resample only the CONTEXT. Then

    target   tau(x_q) = mu_1(x_q) - mu_0(x_q)   -- fixed, one SCM, one x_q
    data     100 independent context sets from the same SCM
    coverage fraction of the 100 intervals containing tau(x_q)

which is the textbook construction, and it asks the model's interval to behave
as a CONFIDENCE interval for a fixed quantity. A model whose interval carries
aleatoric arm noise that tau does not have will over-cover; one that captures
the rho = 1 coupling and emits estimation uncertainty alone lands near 0.95.

COVARIATES ARE ENDOGENOUS, which is why only the context is resampled. X are
nodes of the SCM, generated from the exogenous noise, so redrawing the noise
moves every row's x. The query rows are therefore taken ONCE and copied
verbatim into all 100 cells; only the context half differs between them. Each
output file is a valid case-study realization npz, so the existing dump
harnesses and scorer run against this tree unchanged -- no new per-model predict
paths, which is where convention bugs come from.

Usage:
    python benchmarks/gen_ctx_resample_cells.py \\
        --case Observed_Confounder --d 5 --n-context 1000 --n-query 10 \\
        --n-contexts 100 --out $SCRATCH/ctxresample/shift0/d5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO, "case_study"), os.path.join(_REPO, "case_study", "d_variation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generation import _SampledSCM, Realization, save_realization   # noqa: E402
from generation_d import build_dag_d                                # noqa: E402


def redraw_noise(scm, rng):
    """Fresh exogenous draw for every node, same structural equations.

    _noise is sampled once in __init__ and reused by every forward(do_T=...) --
    that reuse is exactly what makes the two arms share noise (rho = 1). Writing
    new arrays of the same shapes gives an independent dataset from the SAME
    SCM: same graph, same weights, same activations, new randomness.
    """
    for k, v in scm._noise.items():
        scm._noise[k] = rng.normal(0.0, scm.noise_std, size=v.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="Observed_Confounder")
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--n-context", type=int, default=1000)
    ap.add_argument("--n-query", type=int, default=10)
    ap.add_argument("--n-contexts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0,
                    help="fixes the SCM (weights, activations) AND the query rows")
    ap.add_argument("--cate-shift", type=float, default=0.0)
    ap.add_argument("--out", required=True,
                    help="cell dir; files land at <out>/<case>/N<n_context>/<case>_<r>.npz")
    a = ap.parse_args()

    N = a.n_context + a.n_query
    nodes = build_dag_d(a.case, a.d)
    scm = _SampledSCM(nodes, N=N, rng=np.random.default_rng(a.seed),
                      cate_shift=a.cate_shift)

    # ── the FIXED half: one draw, last n_query rows become the queries ──────
    base = scm.forward()
    mu0 = scm.forward(do_T=0.0, y_noiseless=True)[scm.y_name]
    mu1 = scm.forward(do_T=1.0, y_noiseless=True)[scm.y_name]
    feat = [n.name for n in nodes if not n.is_outcome and not n.is_treatment]
    stack = lambda d: np.column_stack([d[f] for f in feat]).astype(np.float64)

    q = slice(a.n_context, N)
    Xq, Tq, Yq = stack(base)[q], base[scm.t_name][q], base[scm.y_name][q]
    mu0q, mu1q = mu0[q], mu1[q]
    tau_q = mu1q - mu0q

    print(f"[fixed] case={a.case} d={a.d} n_context={a.n_context} "
          f"n_query={a.n_query} seed={a.seed}")
    print(f"[fixed] tau(x_q) = {np.array2string(tau_q, precision=5)}")
    print(f"[fixed] these {a.n_query} query rows are identical in all "
          f"{a.n_contexts} cells; only the context half is resampled.")

    out_dir = os.path.join(a.out, a.case, f"N{a.n_context}")
    os.makedirs(out_dir, exist_ok=True)

    for r in range(a.n_contexts):
        # Independent context draw. Seed offset by 1 so r=0 is NOT the draw the
        # queries came from -- otherwise one of the 100 contexts would share its
        # randomness with the target and that cell would not be exchangeable
        # with the rest.
        redraw_noise(scm, np.random.default_rng([a.seed, 1, r]))
        d_ = scm.forward()
        c = slice(0, a.n_context)
        Xc, Tc, Yc = stack(d_)[c], d_[scm.t_name][c], d_[scm.y_name][c]
        m0c = scm.forward(do_T=0.0, y_noiseless=True)[scm.y_name][c]
        m1c = scm.forward(do_T=1.0, y_noiseless=True)[scm.y_name][c]

        rec = Realization(
            case_study=a.case, n_context=a.n_context, seed=int(r),
            exo_std=float(scm.exo_std), noise_std=float(scm.noise_std),
            feature_names=feat,
            X=np.vstack([Xc, Xq]).astype(np.float32),
            T=np.concatenate([Tc, Tq]).astype(np.float32),
            Y=np.concatenate([Yc, Yq]).astype(np.float32),
            cate=np.concatenate([m1c - m0c, tau_q]).astype(np.float32),
            mu_0=np.concatenate([m0c, mu0q]).astype(np.float32),
            mu_1=np.concatenate([m1c, mu1q]).astype(np.float32),
            graph_edges=scm.graph_edges(),
            cate_shift=float(a.cate_shift),
        )
        save_realization(rec, os.path.join(out_dir, f"{a.case}_{r}.npz"))

    meta = dict(case=a.case, d=a.d, n_context=a.n_context, n_query=a.n_query,
                n_contexts=a.n_contexts, seed=a.seed,
                tau_query=[float(x) for x in tau_q],
                design="fixed SCM, fixed query rows, resampled context")
    with open(os.path.join(a.out, "ctxresample_manifest.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[done] {a.n_contexts} cells -> {out_dir}")
    print(f"[done] manifest -> {a.out}/ctxresample_manifest.json")


if __name__ == "__main__":
    main()
