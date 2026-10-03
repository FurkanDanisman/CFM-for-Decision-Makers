"""E1 truths: realized delta = y1 - y0 at every IHDP / ACIC query, beside tau = mu1 - mu0.

The dumps carry tau only. The query units are recovered exactly as CausalPFN's
loaders pick them, and tau rebuilt from the raw files must equal the loader's
true_cate on every unit before delta is written, so the two files describe the
same units in the same order.

  IHDP  test units in file order; y1/y0 from yf/ycf by t (NPCI, independent unit noise)
  ACIC  test = default_rng(42 + idx).permutation(n)[int(0.9 n):]; y0/y1 from zymu_<idx+1>.csv

Writes <out>/<DATASET>_r###.npz with keys delta, tau.

  python revision/experiments/E1_E8_rescoring/build_delta_truth.py \
      --causalpfn $CAUSALPFN --out $SCRATCH/revision_e1/truth
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--causalpfn", default=os.environ.get("CAUSALPFN"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.causalpfn)
    sys.path.insert(0, os.path.join(a.causalpfn, "src"))
    from benchmarks import IHDPDataset, ACIC2016Dataset
    import benchmarks.acic2016 as acic_mod
    import benchmarks.ihdp as ihdp_mod
    import pandas as pd
    os.makedirs(a.out, exist_ok=True)

    def put(ds, r, delta, tau, ref):
        if not np.array_equal(tau, np.asarray(ref, dtype=np.float32)):
            raise SystemExit(f"{ds} r{r}: rebuilt tau != loader true_cate")
        np.savez(os.path.join(a.out, f"{ds}_r{r:03d}.npz"), delta=delta, tau=tau)

    # IHDP
    ihdp = IHDPDataset()
    te = np.load(os.path.join(os.path.dirname(ihdp_mod.__file__), "IHDP",
                              "ihdp_npci_1-100.test.npz"))
    for r in range(ihdp.n_tables):
        t = te["t"][..., r].astype(np.float32)
        yf = te["yf"][..., r].astype(np.float32)
        ycf = te["ycf"][..., r].astype(np.float32)
        y1, y0 = np.where(t == 1, yf, ycf), np.where(t == 1, ycf, yf)
        tau = te["mu1"][..., r].astype(np.float32) - te["mu0"][..., r].astype(np.float32)
        put("IHDP", r, (y1 - y0).astype(np.float32), tau, ihdp[r][0].true_cate)
    print(f"IHDP: {ihdp.n_tables} realizations")

    # ACIC
    acic = ACIC2016Dataset()
    for r in range(acic.n_tables):
        sim = pd.read_csv(acic_mod.ZY_CSV_URL(r + 1))
        sim.columns = ["z", "y0", "y1", "mu0", "mu1"]
        n = len(sim)
        idx = np.random.default_rng(42 + r).permutation(n)
        test = idx[int(n * (1 - acic.test_ratio)):]
        f = lambda c: sim[c].values.astype(np.float32)[test]
        put("ACIC", r, f("y1") - f("y0"), f("mu1") - f("mu0"), acic[r][0].true_cate)
    print(f"ACIC: {acic.n_tables} realizations")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
