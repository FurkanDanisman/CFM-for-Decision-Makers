"""Merge the shape-path runs: realizations 0-99 (results/exp_B_shapes/<scale>) and the
extension 100-999 (results/exp_B_shapes_ext/<scale>_a, <scale>_b) into
results/exp_B_shapes_1000/<scale>/per_dataset.csv. Safe to rerun while the extension is
still going: plot_exp_B_shapes.py only draws conditions whose realizations all exist.

    python merge_exp_B_shapes.py
"""
import glob
import os

import pandas as pd

KEY = ["case", "dial", "level", "target", "r"]
for scale in ("scaled", "unscaled"):
    parts = [f"results/exp_B_shapes/{scale}/per_dataset.csv"] + \
        sorted(glob.glob(f"results/exp_B_shapes_ext/{scale}_*/per_dataset.csv"))
    df = pd.concat([pd.read_csv(p) for p in parts if os.path.exists(p)], ignore_index=True)
    df = df.drop_duplicates(KEY).sort_values(KEY)
    out = f"results/exp_B_shapes_1000/{scale}/per_dataset.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    n = df.groupby(["case", "dial", "level", "target"]).size()
    print(f"{scale}: {len(df)} rows; conditions with 1000 data sets: {(n == 1000).sum()} / {len(n)}")
