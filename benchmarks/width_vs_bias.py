"""Does the interval width know when the model is wrong?

Per realization: mean interval width, and mean bias (predicted mean - truth).
A calibrated interval must be WIDE where the bias is large -- that is the only
way per-realization coverage can sit at its nominal level instead of flipping
between 0 and 1. If the correlation is ~0, the width carries no information
about the error and nominal coverage is unreachable by construction.

Uses the scorer's own per-query output, so no key-name assumptions and no
circular proxy (PEHE is an error measure, not a width).
"""
import glob, os, sys
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "UWYK_Fig3_4"))
from cate_density_metrics import METHODS, score_file, _resolve_dir

import argparse
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--root", default=os.path.join(os.environ.get("SCRATCH", ""),
                                               "cmech_dens", "N1000"))
_ap.add_argument("--dataset", default="CMECH_n5_nonzero")
_A = _ap.parse_args()
root = _A.root
print(f"{'method':17s} {'n':>4s}  {'spearman(width,|bias|)':>22s}  "
      f"{'median width':>12s}  {'median |bias|':>13s}")
print("-" * 78)
for label, subdir, tag in METHODS:
    d = _resolve_dir(root, subdir, _A.dataset)
    if not d:
        continue
    W, B = [], []
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        if os.path.basename(f) == "summary.npz":
            continue
        got = score_file(f, tag)
        if not got:
            continue
        W.append(float(np.mean([g["length"] for g in got])))
        B.append(float(abs(np.mean([g["bias"] for g in got]))))
    if len(W) > 5:
        r = spearmanr(W, B).statistic
        print(f"{label:17s} {len(W):4d}  {r:+22.3f}  {np.median(W):12.4f}  "
              f"{np.median(B):13.4f}")
