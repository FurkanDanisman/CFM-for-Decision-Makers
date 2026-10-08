"""Experiment A diagnostic: run Do-PFN v1 on the paper's RELEASED case-study datasets
(Do-PFN repo, data/prior_sampling/<Case>/<Case>_<i>.pkl) and score the normalized CID MSE two ways:

  paper protocol : generate_valid_split(n_splits=2); fit on train (x_obs, y_obs); query the
                   test units at the opposite arm (x_int, t_int = 1 - t_obs); truth y_int;
                   normalized by (max y - min y)^2 of the truth.
  our protocol   : fit on all rows; query every unit at both arms; truth = y_obs for the
                   factual arm, y_int for the counterfactual arm; same normalization.

    python exp_A_released.py --upstream <path to Do-PFN repo> --out results/exp_A_released
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--upstream", required=True)
p.add_argument("--cases", nargs="+", default=["Observed_Confounder", "Unobserved_Confounder"])
p.add_argument("--out", default="results/exp_A_released")
a = p.parse_args()
upstream, out = os.path.abspath(a.upstream), os.path.abspath(a.out)
os.makedirs(out, exist_ok=True)

# Unpickling needs the upstream `datasets` package; the model comes from our dopfn/.
# Our dopfn/ goes first so `model` / `scripts` resolve to our (patched) copy.
os.chdir(os.path.join(HERE, "dopfn"))
sys.path.insert(0, os.path.join(HERE, "dopfn"))
sys.path.insert(1, upstream)
import datasets  # noqa: E402,F401
from scripts.transformer_prediction_interface.base import DoPFNRegressor  # noqa: E402

reg = DoPFNRegressor()
reg.device = "cuda" if torch.cuda.is_available() else "cpu"


def nmse(pred, true):
    return float(np.mean((pred - true) ** 2) / (true.max() - true.min()) ** 2)


def predict_mean(x_ctx, y_ctx, x_q):
    reg.fit(torch.as_tensor(x_ctx, dtype=torch.float32), torch.as_tensor(y_ctx, dtype=torch.float32))
    return np.asarray(reg.predict_full(torch.as_tensor(x_q, dtype=torch.float32))["mean"], dtype=np.float64)


rows = []
for case in a.cases:
    files = sorted(glob.glob(os.path.join(upstream, "data", "prior_sampling", case, f"{case}_*.pkl")),
                   key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]))
    for f in files:
        d = pickle.load(open(f, "rb"))
        # paper protocol
        tr, te = d.generate_valid_split(n_splits=2)
        pred = predict_mean(tr.x_obs, tr.y_obs, te.x_int)
        paper = nmse(pred, te.y_int.numpy().astype(np.float64))
        # our protocol on the same data
        x_obs, y_obs = d.x_obs.numpy(), d.y_obs.numpy().astype(np.float64)
        x_int, y_int = d.x_int.numpy(), d.y_int.numpy().astype(np.float64)
        reg.fit(torch.as_tensor(x_obs, dtype=torch.float32), torch.as_tensor(y_obs, dtype=torch.float32))
        p_f = np.asarray(reg.predict_full(torch.as_tensor(x_obs, dtype=torch.float32))["mean"], dtype=np.float64)
        p_c = np.asarray(reg.predict_full(torch.as_tensor(x_int, dtype=torch.float32))["mean"], dtype=np.float64)
        ours = nmse(np.concatenate([p_f, p_c]), np.concatenate([y_obs, y_int]))
        rows.append(dict(case=case, file=os.path.basename(f), n=len(y_obs),
                         p_treated=float(x_obs[:, 0].mean()), nmse_paper_protocol=paper,
                         nmse_our_protocol=ours))
        print(f"{case} {os.path.basename(f)} n={len(y_obs)}  paper-protocol={paper:.4g}  "
              f"our-protocol={ours:.4g}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "per_dataset.csv"), index=False)

df = pd.DataFrame(rows)
print(df.groupby("case")[["nmse_paper_protocol", "nmse_our_protocol"]].median()
      .to_string(float_format=lambda v: f"{v:.4g}"))
