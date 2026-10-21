"""Pack every model's Twins predictions (d = 10 and d = 1) into small files for local analysis.
python pack_twins.py DATA_ROOT OUT_DIR

One file per (model directory, d): OUT_DIR/<dir>_d<d>.npz. For every query block r and every density key of the
dump (p_y0_scaled[_tag], p_y1_scaled[_tag], p_joint_scaled[_tag]) the array is stored SPARSE: the flat indices and
float32 values of the entries above 1e-7 of their query's total (a 0/1 outcome puts its mass in a few bins), plus
the full shape. Every scalar / small array the scorers read (edges, y_shift, y_scale, arm maps, UWYK tail scales,
true_cate_per_query) is copied as is, and the block's truths (y0, y1, t_test, delta_shared).
Dropping entries below 1e-7 of a query's mass changes no probability by more than that.
"""
import glob
import os
import re
import sys

import numpy as np

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
DIRS = ["dopfn1d", "dopfn2d", "uwyk1d", "graph2d", "cpfn1d", "cpfn2d", "dopfn1d_mr", "dopfn1d_ml", "uwyk1d_mr",
        "uwyk1d_ml", "cpfn1d_mr", "cpfn1d_ml", "cpfn_v0", "uwyk_bin"]
DENS = re.compile(r"^p_(y0|y1|joint)_scaled(_\w+)?$")
SKIP_TAG = ("_v3b",)                                            # no scorer reads the v3b mode
for m in DIRS:
    for d in (10, 1):
        files = sorted(glob.glob(os.path.join(D, "dumps", m, "twins", f"d{d}", "*r[0-9]*.npz")))
        if not files:
            print("missing", m, d); continue
        out, parts = {}, []

        def flush(tag=""):
            p = os.path.join(OUT, f"{m}_d{d}{tag}.npz")
            np.savez_compressed(p, **out)
            parts.append(p)
            print(f"{p}: {os.path.getsize(p) / 1e6:.1f} MB", flush=True)
        for f in files:
            r = int(re.search(r"r(\d+)\.npz$", f).group(1))
            with np.load(f, allow_pickle=True) as z:
                tagged = any(DENS.match(k) and k.endswith(("_noanc", "_v3a")) for k in z.files)
                for k in z.files:
                    if k.endswith(SKIP_TAG) or k.endswith(tuple(s + "_v3b" for s in ("sL_raw_0", "sR_raw_0", "sL_raw_1", "sR_raw_1"))):
                        continue
                    if tagged and DENS.match(k) and not k.endswith(("_noanc", "_v3a")):
                        continue                                # the untagged copy duplicates DENSITY_ANC_TAG (noanc)
                    v = np.asarray(z[k])
                    if DENS.match(k) and v.ndim >= 2:
                        x = v.astype(np.float64).reshape(v.shape[0], -1)
                        keep = x > 1e-7 * x.sum(1, keepdims=True)
                        idx = np.flatnonzero(keep.ravel())
                        out[f"r{r}|{k}|shape"] = np.array(v.shape)
                        out[f"r{r}|{k}|idx"] = idx.astype(np.int64)
                        out[f"r{r}|{k}|val"] = x.ravel()[idx].astype(np.float32)
                    elif v.size <= 5000 and v.dtype.kind in "fiub":
                        out[f"r{r}|{k}"] = v
            tz = np.load(os.path.join(D, "twins", f"d{d}", "truths", f"r{r:03d}.npz"))
            for k in ("y0", "y1", "t_test", "delta_shared"):
                out[f"r{r}|truth_{k}"] = tz[k]
            if sum(v.nbytes for v in out.values()) > 150e6:           # keep every pushed file under GitHub's 100 MB
                flush(f"_part{len(parts)}"); out = {}
        if out:
            flush(f"_part{len(parts)}" if parts else "")
        big = [p for p in parts if os.path.getsize(p) > 95e6]
        if big:
            print("WARNING: over 95 MB (GitHub refuses > 100 MB):", big)
