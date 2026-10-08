"""Experiment E analysis: Do-PFN vs the best baseline on the B and C data sets.

    python exp_E_analysis.py --e-res results/exp_E --b-res results/exp_B --c-res results/exp_C

Per data set: Delta = error(Do-PFN) - error(best baseline), best = the lower of the
S-learner (TabPFN) and causal forest DML errors on that data set. Breakdown level: the
smallest level (in-prior first) at which the bootstrap 95% interval of the median Delta
is no longer entirely below zero. Writes delta_summary.csv, breakdown.csv and figures
(median error per method against the level, breakdown level marked).
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--e-res", default="results/exp_E")
p.add_argument("--b-res", default="results/exp_B")
p.add_argument("--c-res", default="results/exp_C")
p.add_argument("--boot", type=int, default=2000)
a = p.parse_args()
rng = np.random.default_rng(0)
KEY = ["case", "dial", "level", "target", "r"]
METRICS = {"mse_cate": "MSE (CATE)", "rel_ate": "Rel. ATE error"}
METHODS = {"Do-PFN (v1)": "#e34948", "S-learner (TabPFN)": "#2a78d6", "Causal forest DML": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
# level order per dial, in-prior first (B3: c=1 is in-prior and sits between 0.5 and 2)
ORDER = {"B1_tails": [0, 30, 10, 5, 3], "B2_skew": [0, 16, 4, 1], "B3_scale": [0, 0.5, 2, 4, 8],
         "B4_hetero": [0, 0.5, 1, 2]}
for g in ("sin", "sign", "cubic"):
    ORDER[g] = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def load_dopfn(res, exp):
    d = pd.read_csv(os.path.join(res, "per_dataset.csv"))
    d["exp"], d["method"] = exp, "Do-PFN (v1)"
    return d[["exp", "method"] + KEY + list(METRICS)]


base = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(a.e_res, "*", "*.csv"))])
dop = pd.concat([load_dopfn(r, e) for r, e in ((a.b_res, "B"), (a.c_res, "C"))
                 if os.path.exists(os.path.join(r, "per_dataset.csv"))])
allm = pd.concat([dop, base[["exp", "method"] + KEY + list(METRICS)]], ignore_index=True)
allm.to_csv(os.path.join(a.e_res, "all_methods_per_dataset.csv"), index=False)


def rows_for(exp, case, dial, target, level):
    """Per-data-set rows of one condition; level 0 = that experiment's in-prior run."""
    if level == 0:
        q = (allm.exp == exp) & (allm.case == case) & (allm.dial == "in_prior")
    else:
        q = (allm.exp == exp) & (allm.case == case) & (allm.dial == dial) & (allm.level == level) \
            & (allm.target == target)
    return allm[q]


def med_ci(x):
    x = x[np.isfinite(x)]
    b = np.median(x[rng.integers(0, len(x), (a.boot, len(x)))], axis=1)
    return np.median(x), *np.quantile(b, [0.025, 0.975])


summ, brk = [], []
for exp, targets in (("B", ("all", "outcome")), ("C", ("outcome", "treatment", "both"))):
    dials = [d for d in ORDER if (d.startswith("B")) == (exp == "B")]
    for case in sorted(allm.case.unique()):
        for dial in dials:
            for target in targets:
                for metric in METRICS:
                    first_break = None
                    for level in ORDER[dial]:
                        d = rows_for(exp, case, dial, target, level)
                        if d.empty:
                            continue
                        w = d.pivot_table(index="r", columns="method", values=metric)
                        if "Do-PFN (v1)" not in w or w.shape[1] < 2:
                            continue
                        best = w.drop(columns="Do-PFN (v1)").min(axis=1, skipna=True)
                        delta = (w["Do-PFN (v1)"] - best).values
                        m, lo, hi = med_ci(delta)
                        meds = {k: float(np.nanmedian(w[k])) for k in w.columns}
                        summ.append(dict(exp=exp, case=case, dial=dial, target=target, metric=metric,
                                         level=level, delta_med=m, delta_lo=lo, delta_hi=hi,
                                         **{f"median {k}": v for k, v in meds.items()}))
                        if first_break is None and not hi < 0:
                            first_break = level
                    brk.append(dict(exp=exp, case=case, dial=dial, target=target, metric=metric,
                                    breakdown_level=first_break))
summ, brk = pd.DataFrame(summ), pd.DataFrame(brk)
summ.to_csv(os.path.join(a.e_res, "delta_summary.csv"), index=False)
brk.to_csv(os.path.join(a.e_res, "breakdown.csv"), index=False)
print(brk[brk.metric == "mse_cate"].to_string(index=False))

for exp, targets in (("B", ("all", "outcome")), ("C", ("outcome", "treatment", "both"))):
    if not (summ.exp == exp).any():
        continue
    dials = [d for d in ORDER if (d.startswith("B")) == (exp == "B")]
    cases = sorted(summ.case.unique())
    rows = [(c, t) for c in cases for t in targets]
    fig, axes = plt.subplots(len(rows), len(dials), figsize=(3.6 * len(dials), 2.6 * len(rows)),
                             squeeze=False)
    for i, (case, target) in enumerate(rows):
        for j, dial in enumerate(dials):
            ax = axes[i, j]
            s = summ[(summ.exp == exp) & (summ.case == case) & (summ.dial == dial)
                     & (summ.target == target) & (summ.metric == "mse_cate")]
            x = [ORDER[dial].index(l) for l in s.level]
            for meth, c in METHODS.items():
                col = f"median {meth}"
                if col in s:
                    ax.plot(x, s[col], "-o", color=c, lw=2, ms=4, mec="white", label=meth)
            b = brk[(brk.exp == exp) & (brk.case == case) & (brk.dial == dial) & (brk.target == target)
                    & (brk.metric == "mse_cate")].breakdown_level
            if len(b) and pd.notna(b.iloc[0]):
                ax.axvline(ORDER[dial].index(b.iloc[0]), color=MUTED, ls="--", lw=1.2)
            ax.set_xticks(range(len(ORDER[dial])), ["in-prior" if l == 0 else f"{l:g}" for l in ORDER[dial]],
                          fontsize=7.5)
            ax.set_yscale("log")
            ax.grid(axis="y", color=GRID, lw=0.8, which="both")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(dial, loc="left", fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(f"{case.split('_')[0]} Conf.\n{target}\nmedian MSE (CATE)", fontsize=8.5, color=INK)
    axes[0, -1].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Experiment E ({exp} data sets): median MSE (CATE) per method; dashed = breakdown level",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(a.e_res, f"exp_E_{exp}.png"), dpi=180, facecolor="white")
print("done", a.e_res)
