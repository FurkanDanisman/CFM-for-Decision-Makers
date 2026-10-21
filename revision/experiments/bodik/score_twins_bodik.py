"""Twins in Bodik's setup (every family queried, d = 10): predict the other twin from x and the known twin.
python score_twins_bodik.py DATA_ROOT OUT_PREFIX

  Do-PFN and CausalPFN-C models: context = all 11,984 families (twins_full); one evaluation, SE over the families.
  UWYK models: context = 1000 random families, 10 draws (twins_sub); every metric averaged over the draws, SE over
               the draws.
Per query, the predicted distribution of the other twin's outcome (raw units): 2D, the learned joint conditioned on
the known twin's outcome; 1D, the other arm's marginal. Metrics (Bodik's definitions, the paper's hist read-out):
  MSE      (mean - outcome)^2, mean = the distribution's mean
  coverage, length, IS_0.05 of the 95% interval [2.5%, 97.5% quantiles], endpoints on bin edges
  CRPS     integral (G(x) - 1{x >= outcome})^2 dx, G the predicted CDF
"""
import csv
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bodik_cf_readouts as R                                    # noqa: E402
import bodik_crps as B                                           # noqa: E402

MODELS = [("Do-PFN", "dopfn1d", None, "full"), ("Do-PFN 2D", "dopfn2d", None, "full"),
          ("Do-PFN (matched resolution)", "dopfn1d_mr", None, "full"),
          ("CausalPFN-C", "cpfn1d", None, "full"), ("CausalPFN-C 2D", "cpfn2d", None, "full"),
          ("CausalPFN-C (matched resolution)", "cpfn1d_mr", None, "full"),
          ("UWYK No-Anc", "uwyk1d", "noanc", "sub"), ("UWYK Anc", "uwyk1d", "v3a", "sub"),
          ("UWYK 2D No-Anc", "graph2d", "noanc", "sub"), ("UWYK 2D Anc", "graph2d", "v3a", "sub"),
          ("UWYK No-Anc (matched resolution)", "uwyk1d_mr", "noanc", "sub"),
          ("UWYK Anc (matched resolution)", "uwyk1d_mr", "v3a", "sub")]
NB = 6                                                           # query blocks per context


def per_query(D, part, f, tag, is2d):
    k = int(re.search(r"r(\d+)\.npz$", f).group(1))
    z = np.load(os.path.join(D, part, "d10", "truths", f"r{k:03d}.npz"))
    t = np.rint(z["t_test"]).astype(int)
    y0, y1 = z["y0"].astype(float), z["y1"].astype(float)
    yf, y = np.where(t == 1, y1, y0), np.where(t == 1, y0, y1)
    E, W, T, _ = R.cond_hist(f, tag, t, yf)
    mean, L, U = R.readouts(E, W, T)["A"]
    _, crps = B.run_file(f, tag, t, yf, y, "2d" if is2d else "1d", 0.0)
    cov = ((L <= y + 1e-9) & (y <= U + 1e-9)).astype(float)
    IS = (U - L) + 40 * np.maximum(0, L - y) + 40 * np.maximum(0, y - U)
    return k, np.stack([(mean - y) ** 2, cov, U - L, IS, crps], 1)


def main(D, out):
    names = ("mse", "coverage", "length", "IS", "crps")
    recs = []
    for label, mdir, tag, part in MODELS:
        files = sorted(glob.glob(os.path.join(D, "dumps", mdir, f"twins_{part}", "d10", "*r[0-9]*.npz")))
        if not files:
            print("missing", label); continue
        got = [per_query(D, f"twins_{part}", f, tag, "2D" in label) for f in files]
        if part == "full":
            V = np.concatenate([v for _, v in got])
            m, se = V.mean(0), V.std(0, ddof=1) / np.sqrt(V.shape[0])
            n = f"{V.shape[0]} families"
        else:
            draws = {}
            for k, v in got:
                draws.setdefault(k // NB, []).append(v)
            M = np.array([np.concatenate(v).mean(0) for v in draws.values()])   # metric per context draw
            m, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(M.shape[0])
            n = f"{M.shape[0]} draws"
        r = dict(model=label, context="all 11,984" if part == "full" else "1000, 10 draws", n=n)
        for j, nm in enumerate(names):
            r[nm], r[nm + "_se"] = float(m[j]), float(se[j])
        r["abs_cov_err"] = abs(r["coverage"] - 0.95)
        recs.append(r)
        print(f"{label:34s} " + "  ".join(f"{nm} {r[nm]:.4f}±{r[nm + '_se']:.4f}" for nm in names), flush=True)
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    f = lambda r, k: f"{r[k]:.4f} ± {r[k + '_se']:.4f}"
    md = ["# Twins (d = 10) in Bodik's setup: predict the other twin, every family queried", "",
          "Do-PFN / CausalPFN-C: context = all families (± over families). UWYK: context = 1000 random families, "
          "10 draws (mean ± SE over draws). 2D: learned joint conditioned on the known twin; 1D: the other arm's "
          "marginal. Lower is better (coverage: closest to 0.95).", "",
          "| model | context | MSE | |ĉ − 0.95| | ĉ | length | IS₀.₀₅ | CRPS |", "|---|---|---|---|---|---|---|---|"]
    for r in recs:
        md.append(f"| {r['model']} | {r['context']} | {f(r, 'mse')} | {r['abs_cov_err']:.3f} | {f(r, 'coverage')} | "
                  f"{f(r, 'length')} | {f(r, 'IS')} | {f(r, 'crps')} |")
    open(out + ".md", "w").write("\n".join(md) + "\n")
    print(f"wrote {out}.csv / .md")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
