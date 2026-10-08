"""Experiment A: reproduce Do-PFN v1 in-prior results on two case studies.

    python exp_A.py --data-root data --cases Observed_Confounder Unobserved_Confounder \
        --ns 200 500 1000 --out results/exp_A

Per dataset: normalized CID MSE with the paper protocol (Fig. 3 row 1, App. D.2), MSE_CATE = mean (tau_hat - tau)^2, the paper's normalized MSE
(divided by (max tau - min tau)^2, Appendix D.2), and Rel. ATE error
|ATE_hat - ATE| / |ATE|. Summary: median and IQR over datasets.
Context = all N rows; queries = the same units; truth = npz `cate`.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DOPFN_ROOT = os.path.join(HERE, "dopfn")

p = argparse.ArgumentParser()
p.add_argument("--data-root", required=True)
p.add_argument("--cases", nargs="+", required=True)
p.add_argument("--ns", nargs="+", type=int, default=[1000])
p.add_argument("--max-real", type=int, default=0, help="0 = all realizations")
p.add_argument("--out", default="results/exp_A")
a = p.parse_args()
data_root = os.path.abspath(a.data_root)
out = os.path.abspath(a.out)
os.makedirs(out, exist_ok=True)

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
    return np.asarray(f0["mean"]), np.asarray(f1["mean"])


rows = []
for case in a.cases:
    for n in a.ns:
        files = sorted(glob.glob(os.path.join(data_root, case, f"N{n}", f"{case}_*.npz")),
                       key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]))
        if a.max_real:
            files = files[:a.max_real]
        for f in files:
            d = np.load(f, allow_pickle=True)
            tau = d["cate"].astype(np.float64)
            m0, m1 = (m.astype(np.float64) for m in dopfn_cate(d["X"], d["T"], d["Y"]))
            tau_hat = m1 - m0
            # CID, paper protocol (Fig. 3 row 1, App. D.2; released Do-PFN data/code):
            # random 50/50 split; context = train half (T, X, Y); queries = test units
            # at the opposite arm t = 1 - T; truth = counterfactual y = mu_t + eps_Y
            # (same eps_Y as the factual row); normalized by (max y - min y)^2.
            X, T, Y = d["X"], d["T"], d["Y"]
            mu0, mu1 = d["mu_0"].astype(np.float64), d["mu_1"].astype(np.float64)
            eps_y = Y - np.where(T == 1, mu1, mu0)
            perm = np.random.default_rng(int(d["seed"])).permutation(len(Y))
            tr, te = perm[: len(Y) // 2], perm[len(Y) // 2:]
            reg.fit(torch.tensor(np.c_[T[tr], X[tr]].astype(np.float32)),
                    torch.tensor(Y[tr].astype(np.float32)))
            t_cf = 1.0 - T[te]
            y_pred = np.asarray(reg.predict_full(torch.tensor(np.c_[t_cf, X[te]].astype(np.float32)))["mean"],
                                dtype=np.float64)
            y_int = np.where(t_cf == 1, mu1[te], mu0[te]) + eps_y[te]
            nmse_cid = np.mean((y_pred - y_int) ** 2) / (y_int.max() - y_int.min()) ** 2
            mse = np.mean((tau_hat - tau) ** 2)
            rng = tau.max() - tau.min()
            ate, ate_hat = tau.mean(), tau_hat.mean()
            rows.append(dict(
                case=case, n=n, r=int(f.rsplit("_", 1)[1].split(".")[0]), seed=int(d["seed"]),
                nmse_cid=nmse_cid,
                mse_cate=mse,
                nmse_cate=mse / rng ** 2 if rng > 0 else np.nan,
                rel_ate=abs(ate_hat - ate) / abs(ate) if ate != 0 else np.nan,
                abs_ate=abs(ate_hat - ate), ate=ate, ate_hat=ate_hat,
                noise_std=float(d["noise_std"]), exo_std=float(d["exo_std"])))
            print(f"{case} N={n} {os.path.basename(f)}  nMSE_CID={nmse_cid:.4g}  MSE={mse:.4g}  "
                  f"relATE={rows[-1]['rel_ate']:.4g}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(out, "per_dataset.csv"), index=False)

df = pd.DataFrame(rows)


def q(s, k):
    return s.quantile(k)


summ = df.groupby(["case", "n"]).agg(
    datasets=("mse_cate", "size"),
    nmse_cid_med=("nmse_cid", "median"),
    nmse_cid_q25=("nmse_cid", lambda s: q(s, .25)), nmse_cid_q75=("nmse_cid", lambda s: q(s, .75)),
    mse_med=("mse_cate", "median"), mse_q25=("mse_cate", lambda s: q(s, .25)),
    mse_q75=("mse_cate", lambda s: q(s, .75)), mse_mean=("mse_cate", "mean"),
    nmse_med=("nmse_cate", "median"),
    nmse_q25=("nmse_cate", lambda s: q(s, .25)), nmse_q75=("nmse_cate", lambda s: q(s, .75)),
    rel_ate_med=("rel_ate", "median"),
    rel_ate_q25=("rel_ate", lambda s: q(s, .25)), rel_ate_q75=("rel_ate", lambda s: q(s, .75)),
).reset_index()
summ.to_csv(os.path.join(out, "summary.csv"), index=False)
pd.set_option("display.width", 250)
print(summ.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
