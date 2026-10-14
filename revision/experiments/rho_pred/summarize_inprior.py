"""Summarise the in-prior predicted rho of the 2D heads.
python summarize_inprior.py RES_DIR OUT_PREFIX            -> OUT_PREFIX.csv (per dataset), OUT_PREFIX.md

Per dataset: the median over its queries of the predicted rho. Per model: the distribution of those
dataset medians (quantiles, shares), and of all queries pooled.
"""
import csv
import glob
import os
import sys

import numpy as np

MODELS = [("dopfn2d", "Do-PFN 2D"), ("graph2d", "UWYK 2D"), ("cpfn2d", "CausalPFN-C 2D")]
Q = (0.05, 0.25, 0.5, 0.75, 0.95)


def load(res, m):
    per_ds, pooled = [], []
    for f in sorted(glob.glob(os.path.join(res, m, "shard*.npz"))):
        z = np.load(f, allow_pickle=True)
        shard = int(os.path.basename(f)[5:-4])
        for i, r in zip(z["idx"], z["rho"]):
            r = np.asarray(r, dtype=np.float64)
            r = r[np.isfinite(r)]
            if r.size:
                per_ds.append((shard, int(i), float(np.median(r)), int(r.size)))
                pooled.append(r)
    return per_ds, (np.concatenate(pooled) if pooled else np.array([]))


def row(name, v):
    if v.size == 0:
        return f"| {name} | 0 |" + " — |" * (len(Q) + 5)
    qs = " | ".join(f"{x:+.3f}" for x in np.quantile(v, Q))
    return (f"| {name} | {v.size} | {qs} | {v.mean():+.3f} | {np.mean(v > 0.99):.3f} | {np.mean(v > 0.9):.3f} | "
            f"{np.mean(v < 0.5):.3f} | {np.mean(v < 0):.3f} |")


def main(res, out):
    hdr = ("| model | n | 5% | 25% | 50% | 75% | 95% | mean | > 0.99 | > 0.9 | < 0.5 | < 0 |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|---|")
    md = ["# In-prior predicted ρ of the 2D heads", "",
          "Each 2D head on 1000 datasets of its own training prior (E2 new-world data, shards 0–9), the paper's "
          "harness. Predicted ρ per query = correlation of the head's predicted joint on the interior grid "
          "(piecewise-uniform cells). It is the model's prediction, not the prior's true ρ, and includes the model's "
          "uncertainty about the means.", "",
          "## Per dataset (median over its queries)", "", hdr]
    pooled_rows, rows = [], []
    for m, name in MODELS:
        per_ds, pooled = load(res, m)
        md.append(row(name, np.array([p[2] for p in per_ds])))
        pooled_rows.append(row(name, pooled))
        rows += [dict(model=name, shard=s, dataset=i, rho_median=r, n_queries=n) for s, i, r, n in per_ds]
    md += ["", "## All queries pooled", "", hdr] + pooled_rows
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "shard", "dataset", "rho_median", "n_queries"])
        w.writeheader()
        w.writerows(rows)
    open(out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
