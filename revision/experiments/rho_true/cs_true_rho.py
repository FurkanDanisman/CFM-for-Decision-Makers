"""True cross-world correlation rho(x) on the case studies, exactly from the generator.
python cs_true_rho.py DATA_ROOT OUT_PREFIX [--q 100] [--k 2000] [--shifts 0 +2 -2] [--ds 5 ... 50] [--n 1000]
    DATA_ROOT = $KIT/case_study_data/d_variation  (layout shift*/d{D}/{Case}/N{N}/{Case}_{r}.npz)

rho(x) = corr(Y^do(1), Y^do(0) | X = x)  (paper Eq. eq:rho; Bodik, Huang & Yu 2026). Computed only for the
scenarios whose covariates are all pre-treatment (Observed_Confounder, Backdoor_Criterion,
Frontdoor_Criterion, Unobserved_Confounder): there the observed covariates are roots, so the remaining
randomness (hidden roots U, the noise of every structural node) is independent of X, and fixing x and
redrawing it gives rho(x) exactly. The two arms share every draw, as in the benchmark.

Per file: the SCM is rebuilt from its stored seed (case_study/d_variation/generation_d.py, same RNG
stream), checked to reproduce the stored X, Y, mu_0, mu_1; then for the first Q stored units, x is fixed
and K draws of the rest give Y(0), Y(1) and their correlation. Per dataset: median of rho(x) over the Q
units, and the share of Var(Y(1) - Y(0) | x) relative to Var(Y(0) | x) (how much the ITE varies at fixed x).
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(REPO, "case_study"))
sys.path.insert(0, os.path.join(REPO, "case_study", "d_variation"))
from generation import ACTIVATIONS, _SampledSCM          # noqa: E402
from generation_d import build_dag_d                      # noqa: E402

CASES = ("Observed_Confounder", "Backdoor_Criterion", "Frontdoor_Criterion", "Unobserved_Confounder")


def rebuild(f):
    z = np.load(f, allow_pickle=True)
    case, seed, n = str(z["case_study"]), int(z["seed"]), int(z["n_context"])
    d = int(z["X"].shape[1])
    shift = float(z["cate_shift"])
    nodes = build_dag_d(case, d)
    scm = _SampledSCM(nodes, N=n, rng=np.random.default_rng(seed), cate_shift=shift)
    obs = scm.forward()
    X = np.stack([obs[m.name] for m in nodes if m.observed], axis=-1)
    mu0 = scm.forward(do_T=0.0, y_noiseless=True)[scm.y_name]
    mu1 = scm.forward(do_T=1.0, y_noiseless=True)[scm.y_name]
    ok = (np.allclose(X, z["X"], atol=1e-4) and np.allclose(obs[scm.y_name], z["Y"], atol=1e-4)
          and np.allclose(mu0, z["mu_0"], atol=1e-4) and np.allclose(mu1, z["mu_1"], atol=1e-4))
    return case, d, scm, z["X"].astype(np.float64), ok


def redraw(scm, x, K, t, rng_draws):
    """Y(t) for K draws at fixed covariates x; rng_draws holds the shared draws (same for both arms)."""
    vals = {}
    for n in scm.nodes:
        if n.observed:
            vals[n.name] = np.full(K, x[n.name])
        elif n.is_treatment:
            vals[n.name] = np.full(K, float(t))
        elif n.kind == "root_normal":                       # hidden root (U)
            vals[n.name] = rng_draws[n.name]
        else:                                               # hidden mediator or outcome
            w = scm._weights[n.name]
            lin = np.stack([vals[p] for p in n.parents], axis=-1) @ w
            out = ACTIVATIONS[scm._activation[n.name]](lin) + rng_draws[n.name]
            if n.is_outcome and scm._shift_outcome:
                out = out + scm._shift * float(t)
            vals[n.name] = out
    return vals[scm.y_name]


def per_file(f, Q, K, seed):
    case, d, scm, Xs, ok = rebuild(f)
    names = [n.name for n in scm.nodes if n.observed]
    rng = np.random.default_rng(seed)
    rhos, vfrac = [], []
    for i in range(min(Q, Xs.shape[0])):
        x = dict(zip(names, Xs[i]))
        draws = {}
        for n in scm.nodes:
            if n.observed or n.is_treatment:
                continue
            sd = scm.exo_std if n.kind == "root_normal" else scm.noise_std
            draws[n.name] = rng.normal(0.0, sd, size=K)
        y0, y1 = redraw(scm, x, K, 0.0, draws), redraw(scm, x, K, 1.0, draws)
        rhos.append(np.corrcoef(y0, y1)[0, 1])
        vfrac.append(np.var(y1 - y0) / np.var(y0))
    return case, d, ok, np.array(rhos), np.array(vfrac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out")
    ap.add_argument("--q", type=int, default=100); ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--shifts", nargs="+", default=["0", "+2", "-2"])   # the paper's three shifts
    ap.add_argument("--ds", nargs="+", type=int, default=[5, 10, 20, 30, 40, 50])
    ap.add_argument("--n", type=int, default=1000)                       # the paper's context size
    a = ap.parse_args()
    rows = []
    for case in CASES:
        files = [f for s_ in a.shifts for d_ in a.ds
                 for f in glob.glob(os.path.join(a.root, f"shift{s_}", f"d{d_}", case, f"N{a.n}", f"{case}_*.npz"))]
        for f in sorted(files):
            parts = f.split(os.sep)
            shift, dd, nn = parts[-5], parts[-4], parts[-2]
            r = int(re.search(r"_(\d+)\.npz$", f).group(1))
            c, d, ok, rho, vf = per_file(f, a.q, a.k, seed=r)
            rows.append(dict(case=c, shift=shift, d=d, N=nn, r=r, rebuilt_ok=int(ok),
                             rho_median=float(np.nanmedian(rho)), rho_min=float(np.nanmin(rho)),
                             var_ite_over_var_y0=float(np.nanmedian(vf))))
        print(f"{case}: {sum(1 for x in rows if x['case'] == case)} files", flush=True)
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    md = ["# True ρ(x) on the case studies (exact, from the generator)", "",
          "Per dataset: median over its first Q units of ρ(x) = corr(Y(1), Y(0) | X = x), K redraws of all "
          "noise and hidden roots with x fixed (shared across arms). `rebuilt` = SCMs that reproduce the stored "
          "X, Y, μ0, μ1 from their seed.", "",
          "| scenario | d | datasets | rebuilt | ρ median (5%, 95%) | share ρ > 0.99 | Var(ITE|x)/Var(Y(0)|x) median |",
          "|---|---|---|---|---|---|---|"]
    for case in CASES:
        for d in sorted({x["d"] for x in rows if x["case"] == case}):
            sel = [x for x in rows if x["case"] == case and x["d"] == d]
            v = np.array([x["rho_median"] for x in sel]); vf = np.array([x["var_ite_over_var_y0"] for x in sel])
            md.append(f"| {case} | {d} | {len(sel)} | {sum(x['rebuilt_ok'] for x in sel)} | "
                      f"{np.median(v):+.3f} ({np.quantile(v, 0.05):+.3f}, {np.quantile(v, 0.95):+.3f}) | "
                      f"{np.mean(v > 0.99):.2f} | {np.median(vf):.3g} |")
    open(a.out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
