"""Per-query mean and standard deviation of each predicted outcome, from one shard of 1D Do-PFN dumps.
python score_moments.py DUMP_DIR OUT_NPZ
Dump: p_y0_scaled, p_y1_scaled (q, K) on raw bin edges `edges` (K+1), with Do-PFN's own bucket means
`bucket_means` (K; half-normal tail means in the two outer buckets). Mean = sum_k p_k m_k; variance =
sum_k p_k (m_k^2 + w_k^2 / 12) - mean^2, i.e. uniform within each bucket of width w_k.
Also the 95% predictive interval of each outcome: lo/hi = lower edge of the bucket holding the 2.5% quantile and
upper edge of the bucket holding the 97.5% quantile (the paper's hist read-out).
Saves idx (dataset index within the shard) and mu0, sd0, mu1, sd1, lo0, hi0, lo1, hi1 of shape (n_datasets, n_queries)."""
import glob, os, re, sys
import numpy as np

dump_dir, out = sys.argv[1], sys.argv[2]
idx, M = [], {k: [] for k in ("mu0", "sd0", "mu1", "sd1", "lo0", "hi0", "lo1", "hi1")}
for f in sorted(glob.glob(os.path.join(dump_dir, "*r[0-9][0-9][0-9].npz"))):
    with np.load(f, allow_pickle=True) as z:
        if "p_y0_scaled" not in z.files:
            continue
        e = np.asarray(z["edges"], np.float64).reshape(-1)
        m = np.asarray(z["bucket_means"], np.float64).reshape(-1)
        w2 = np.diff(e) ** 2 / 12.0
        for t in (0, 1):
            p = np.asarray(z[f"p_y{t}_scaled"], np.float64)
            p = p / p.sum(1, keepdims=True)
            mu = p @ m
            M[f"mu{t}"].append(mu); M[f"sd{t}"].append(np.sqrt(np.maximum(p @ (m ** 2 + w2) - mu ** 2, 0.0)))
            F = np.cumsum(p, 1)                                     # CDF at the upper edge of every bucket
            K = p.shape[1]
            b_lo = np.clip((F < 0.025).sum(1), 0, K - 1); b_hi = np.clip((F < 0.975).sum(1), 0, K - 1)
            M[f"lo{t}"].append(e[b_lo]); M[f"hi{t}"].append(e[b_hi + 1])
    idx.append(int(re.search(r"r(\d+)\.npz$", f).group(1)))
np.savez(out, idx=np.array(idx), **{k: np.array(v, dtype=np.float32) for k, v in M.items()})
print(f"{out}: {len(idx)} datasets")
