"""IHDP and ACIC with a controlled cross-world correlation of the outcome noise.
python gen_rc_rho.py OUT_ROOT --causalpfn $CAUSALPFN [--draws-acic 10]

Both benchmarks generate outcomes as Y(t) = mu_t(x) + eps_t with eps_t ~ N(0, 1) drawn separately per arm
(IHDP: vdorie/npci; ACIC 2016: vdorie/aciccomp; checked on the stored files: residual sd ~ 1, cross-arm
correlation ~ 0). Here everything is kept exactly as in CausalPFN's loaders (benchmarks/ihdp.py,
benchmarks/acic2016.py: covariates, treatments, context/query split, mu_0, mu_1), and only the noise pair is
redrawn with corr(eps_0, eps_1) = rho. The noise is additive and independent of x, so rho(x) = rho EXACTLY;
rho is drawn uniformly within each bin, so every dataset lands in its bin.

  IHDP: the 100 replicates, once per bin (context = the train file, queries = the test file).
  ACIC: the 10 settings of the loader (seeded 90/10 split), each with `draws-acic` coupled noise draws per bin.

Layout (the format of revision/experiments/E2_in_prior/common.write_world, read with DATASET=CMECH_n5):
    OUT_ROOT/bin{b}/{IHDP,ACIC}/shard0/complexmech/5node/path_TY/hide_0.0/r###.npz   (harness input)
    OUT_ROOT/bin{b}/{IHDP,ACIC}/truths/r###.npz   delta_shared = realized ITE Y(1) - Y(0) of the queries,
        cate = mu_1 - mu_0, y0, y1, t_test, rho (= rho_true)
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(REPO, "revision", "experiments", "E2_in_prior"))
from common import write_world                                 # noqa: E402

BINS = ((-0.5, 0.0), (0.0, 0.5), (0.5, 1.0))
SEED = 20261014


def coupled(rng, n, rho):
    z0, z1 = rng.standard_normal(n), rng.standard_normal(n)
    return z0, rho * z0 + np.sqrt(max(1.0 - rho * rho, 0.0)) * z1


def ihdp(out, causalpfn):
    tr = np.load(os.path.join(causalpfn, "benchmarks", "IHDP", "ihdp_npci_1-100.train.npz"))
    te = np.load(os.path.join(causalpfn, "benchmarks", "IHDP", "ihdp_npci_1-100.test.npz"))
    for b, (lo, hi) in enumerate(BINS):
        root = os.path.join(out, f"bin{b}", "IHDP")
        for idx in range(100):
            rng = np.random.default_rng([SEED, 1, b, idx])
            rho = float(rng.uniform(lo, hi))
            Xc, Tc = tr["x"][..., idx], tr["t"][..., idx]
            e0, e1 = coupled(rng, Xc.shape[0], rho)
            Yc = np.where(Tc == 1, tr["mu1"][..., idx] + e1, tr["mu0"][..., idx] + e0)
            Xq, mu0, mu1 = te["x"][..., idx], te["mu0"][..., idx], te["mu1"][..., idx]
            q0, q1 = coupled(rng, Xq.shape[0], rho)
            y0, y1 = mu0 + q0, mu1 + q1
            write_world(root, idx, Xc, Tc, Yc, Xq,
                        {"delta_shared": y1 - y0, "cate": mu1 - mu0, "y0": y0, "y1": y1,
                         "t_test": te["t"][..., idx], "rho": np.full(Xq.shape[0], rho)})
        print(f"IHDP bin{b}: 100 datasets in {root}", flush=True)


def acic_tables(n_tables=10, test_ratio=0.1, seed=42):
    """benchmarks/acic2016.py ACIC2016Dataset._get_data, verbatim up to the outcome, plus mu_0/mu_1 per split."""
    import pandas as pd
    X_URL = ("https://raw.githubusercontent.com/BiomedSciAI/causallib/master/causallib/datasets/data/"
             "acic_challenge_2016/x.csv")
    ZY_URL = ("https://raw.githubusercontent.com/BiomedSciAI/causallib/master/causallib/datasets/data/"
              "acic_challenge_2016/zymu_{}.csv")
    x_data = pd.read_csv(X_URL)
    rngs = [np.random.default_rng(seed + i) for i in range(n_tables)]
    out = []
    for idx in range(n_tables):
        sim = pd.read_csv(ZY_URL.format(idx + 1))
        x_data.columns = [f"x_{i+1}" for i in range(x_data.shape[1])]
        sim.columns = ["z", "y0", "y1", "mu0", "mu1"]
        cat = ["x_2", "x_21", "x_24"]
        num = [f"x_{i+1}" for i in range(x_data.shape[1]) if f"x_{i+1}" not in cat]
        x_data["x_2_numeric"] = x_data["x_2"].astype("category").cat.codes
        x_data["x_21_numeric"] = x_data["x_21"].astype("category").cat.codes
        x_data["x_24_numeric"] = x_data["x_24"].astype("category").cat.codes
        num = num + ["x_2_numeric", "x_21_numeric", "x_24_numeric"]
        x_data = x_data.loc[:, num]
        X = x_data.values.astype(np.float32)
        T = sim["z"].values.astype(np.float32)
        mu0, mu1 = sim["mu0"].values.astype(np.float64), sim["mu1"].values.astype(np.float64)
        r0 = sim["y0"].values - mu0
        r1 = sim["y1"].values - mu1
        ind = rngs[idx].permutation(X.shape[0])
        cut = int(len(ind) * (1 - test_ratio))
        out.append((X, T, mu0, mu1, ind[:cut], ind[cut:], float(np.std(r0)), float(np.std(r1)),
                    float(np.corrcoef(r0, r1)[0, 1])))
    return out


def acic(out, draws):
    tables = acic_tables()
    for i, t in enumerate(tables):
        print(f"ACIC setting {i}: stored noise sd {t[6]:.3f} / {t[7]:.3f}, cross-arm corr {t[8]:+.3f}", flush=True)
    for b, (lo, hi) in enumerate(BINS):
        root = os.path.join(out, f"bin{b}", "ACIC")
        r = 0
        for i, (X, T, mu0, mu1, tr, te, *_ ) in enumerate(tables):
            for k in range(draws):
                rng = np.random.default_rng([SEED, 2, b, i, k])
                rho = float(rng.uniform(lo, hi))
                e0, e1 = coupled(rng, X.shape[0], rho)
                y0, y1 = mu0 + e0, mu1 + e1
                Y = np.where(T == 1, y1, y0)
                write_world(root, r, X[tr], T[tr], Y[tr], X[te],
                            {"delta_shared": (y1 - y0)[te], "cate": (mu1 - mu0)[te], "y0": y0[te], "y1": y1[te],
                             "t_test": T[te], "rho": np.full(len(te), rho)})
                r += 1
        print(f"ACIC bin{b}: {r} datasets in {root}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--causalpfn", required=True)
    ap.add_argument("--draws-acic", type=int, default=10)
    ap.add_argument("--only", choices=["IHDP", "ACIC"], default=None)
    a = ap.parse_args()
    if a.only in (None, "IHDP"):
        ihdp(a.out, a.causalpfn)
    if a.only in (None, "ACIC"):
        acic(a.out, a.draws_acic)


if __name__ == "__main__":
    main()
