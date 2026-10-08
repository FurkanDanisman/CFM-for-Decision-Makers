"""Run Do-PFN 1D on case-study npz files written by generation.py and score the CATE.

    python run_dopfn.py --data-root data --cases Observed_Confounder --n 1000 --max-real 5

Context = all N rows; queries = the same units; truth = the npz `cate`.
Prints PEHE and L1-ATE per realization and the mean per case.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DOPFN_ROOT = os.path.join(HERE, "dopfn")

p = argparse.ArgumentParser()
p.add_argument("--data-root", required=True)
p.add_argument("--cases", nargs="+", required=True)
p.add_argument("--n", type=int, default=1000, help="context-size subfolder N{n}")
p.add_argument("--max-real", type=int, default=0, help="0 = all realizations")
a = p.parse_args()
data_root = os.path.abspath(a.data_root)

# Do-PFN loads artifacts/ by relative path, so build it from inside DOPFN_ROOT.
os.chdir(DOPFN_ROOT)
sys.path.insert(0, DOPFN_ROOT)
from scripts.transformer_prediction_interface.base import DoPFNRegressor  # noqa: E402

reg = DoPFNRegressor()
reg.device = "cuda" if torch.cuda.is_available() else "cpu"


def dopfn_cate(X, T, Y):
    x_ctx = np.concatenate([T[:, None], X], 1).astype(np.float32)   # T in column 0
    reg.fit(torch.tensor(x_ctx), torch.tensor(Y.astype(np.float32)))
    X0 = np.concatenate([np.zeros((len(X), 1)), X], 1).astype(np.float32)
    X1 = X0.copy(); X1[:, 0] = 1.0
    f0 = reg.predict_full(torch.tensor(X0))   # p(y | do(T=0), x)
    f1 = reg.predict_full(torch.tensor(X1))   # p(y | do(T=1), x)
    return np.asarray(f1["mean"]) - np.asarray(f0["mean"])


for case in a.cases:
    files = sorted(glob.glob(os.path.join(data_root, case, f"N{a.n}", f"{case}_*.npz")),
                   key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]))
    if a.max_real:
        files = files[:a.max_real]
    pehe, ate = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        cate_hat = dopfn_cate(d["X"], d["T"], d["Y"])
        pehe.append(np.sqrt(np.mean((cate_hat - d["cate"]) ** 2)))
        ate.append(abs(cate_hat.mean() - d["cate"].mean()))
        print(f"{os.path.basename(f)}  PEHE={pehe[-1]:.4f}  L1-ATE={ate[-1]:.4f}", flush=True)
    print(f"{case} N={a.n}  realizations={len(files)}  "
          f"mean PEHE={np.mean(pehe):.4f}  mean L1-ATE={np.mean(ate):.4f}")
