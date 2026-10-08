"""Aggregate Experiment D (plan, Section 4) from exp_D.py outputs and plot.

    python exp_D_aggregate.py --res results/exp_D

conv: for one fixed SCM, with tau_bar(x_i) the mean of the R estimates at x_i,
    Var   = (1/m) sum_i 1/(R-1) sum_r (tau_hat_r(x_i) - tau_bar(x_i))^2
    Bias2 = (1/m) sum_i (tau_bar(x_i) - tau(x_i))^2 - Var / R
rate: d_r(x_i) = [tau_hat_{r,lambda}(x_i) - tau_lambda(x_i)] - [tau_hat_{r,0}(x_i) - tau_0(x_i)],
    E(lambda) = (1/m) sum_i d_bar(x_i)^2 - (1/m) sum_i s_d^2(x_i) / R
Summaries: mean over the K SCMs with a bootstrap 95% interval over SCMs; the fitted
log-log slope of E(lambda) on lambda in {0.1, 0.2, 0.3}, bootstrapped over SCMs.
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
p.add_argument("--res", default="results/exp_D")
p.add_argument("--boot", type=int, default=2000)
a = p.parse_args()
rng = np.random.default_rng(0)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def var_bias(est, tau):
    """est (R, m), tau (m,) -> (Var, Bias2) for one SCM."""
    R = est.shape[0]
    var = est.var(axis=0, ddof=1).mean()
    return var, ((est.mean(axis=0) - tau) ** 2).mean() - var / R


def boot_mean(x):
    x = np.asarray(x, dtype=float)
    b = x[rng.integers(0, len(x), (a.boot, len(x)))].mean(axis=1)
    return x.mean(), *np.quantile(b, [0.025, 0.975])


# ---- convergence in n ----
rows = []
for f in glob.glob(os.path.join(a.res, "conv", "*", "*", "k*.npz")):
    d = np.load(f)
    for i, n in enumerate(d["n_grid"]):
        v, b2 = var_bias(d["tau_hat"][i], d["tau"])
        rows.append(dict(case=str(d["case"]), setting=str(d["setting"]), k=int(d["k"]), n=int(n),
                         var=v, bias2=b2))
if rows:
    conv = pd.DataFrame(rows)
    conv.to_csv(os.path.join(a.res, "conv_per_scm.csv"), index=False)
    summ = []
    for (case, setting, n), g in conv.groupby(["case", "setting", "n"]):
        for q in ("var", "bias2"):
            m, lo, hi = boot_mean(g[q])
            summ.append(dict(case=case, setting=setting, n=n, quantity=q, mean=m, lo=lo, hi=hi, K=len(g)))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(a.res, "conv_summary.csv"), index=False)
    cases = sorted(summ.case.unique())
    settings = list(dict.fromkeys(summ.setting))
    fig, axes = plt.subplots(len(cases), 2, figsize=(12, 4.2 * len(cases)), squeeze=False)
    for i, case in enumerate(cases):
        for j, q in enumerate(("var", "bias2")):
            ax = axes[i, j]
            for c, setting in zip(PALETTE, settings):
                s = summ[(summ.case == case) & (summ.setting == setting) & (summ.quantity == q)].sort_values("n")
                pos = s["mean"] > 0
                ax.fill_between(s.n[pos], s.lo[pos].clip(lower=1e-12), s.hi[pos], color=c, alpha=0.10, lw=0)
                ax.plot(s.n[pos], s["mean"][pos], "-o", color=c, lw=2, ms=4, mec="white", label=setting)
            ax.set_xscale("log")
            if (summ[(summ.case == case) & (summ.quantity == q)]["mean"] > 0).any():
                ax.set_yscale("log")
            ax.grid(color=GRID, lw=0.8, which="both")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_xlabel("context size n", color=INK)
            ax.set_title(f"{case.replace('_', ' ')}: {'variance' if q == 'var' else 'squared bias'}",
                         loc="left", fontsize=10, color=INK)
    axes[0, 1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(a.res, "exp_D_convergence.png"), dpi=200, facecolor="white")

# ---- rate in lambda ----
rows = []
for f in glob.glob(os.path.join(a.res, "rate", "*", "k*.npz")):
    d = np.load(f)
    est, tau, lams = d["tau_hat"], d["tau"], d["lambdas"]           # (L, R, m), (L, m)
    err0 = est[0] - tau[0]
    R = est.shape[1]
    for li, lam in enumerate(lams):
        dd = (est[li] - tau[li]) - err0                                # (R, m)
        e = (dd.mean(axis=0) ** 2).mean() - dd.var(axis=0, ddof=1).mean() / R
        rows.append(dict(g=str(d["g"]), target=str(d["target"]), k=int(d["k"]), lam=float(lam), E=e))
if rows:
    rate = pd.DataFrame(rows)
    rate.to_csv(os.path.join(a.res, "rate_per_scm.csv"), index=False)
    fit_l = (0.1, 0.2, 0.3)
    slopes, summ = [], []
    for (g, target), gr in rate.groupby(["g", "target"]):
        piv = gr.pivot(index="k", columns="lam", values="E")
        for lam in piv.columns:
            m, lo, hi = boot_mean(piv[lam])
            summ.append(dict(g=g, target=target, lam=lam, mean=m, lo=lo, hi=hi, K=len(piv)))

        def slope(sub):
            y = sub[list(fit_l)].mean(axis=0).values
            return np.polyfit(np.log(fit_l), np.log(y), 1)[0] if np.all(y > 0) else np.nan
        bs = [slope(piv.iloc[rng.integers(0, len(piv), len(piv))]) for _ in range(a.boot)]
        slopes.append(dict(g=g, target=target, slope=slope(piv),
                           lo=np.nanquantile(bs, 0.025), hi=np.nanquantile(bs, 0.975),
                           frac_boot_nan=float(np.mean(np.isnan(bs)))))
    pd.DataFrame(summ).to_csv(os.path.join(a.res, "rate_summary.csv"), index=False)
    slopes = pd.DataFrame(slopes)
    slopes.to_csv(os.path.join(a.res, "rate_slopes.csv"), index=False)
    print(slopes.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    summ = pd.DataFrame(summ)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, g in zip(axes, ("sin", "sign", "cubic")):
        any_pos = False
        for c, (target, ls) in zip(PALETTE, (("outcome", "-"), ("both", "--"))):
            s = summ[(summ.g == g) & (summ.target == target) & (summ.lam > 0)].sort_values("lam")
            pos = s["mean"] > 0
            any_pos |= bool(pos.any())
            sl = slopes[(slopes.g == g) & (slopes.target == target)]
            lab = f"{target}: slope {sl.slope.iloc[0]:.2f}" if len(sl) else target
            ax.fill_between(s.lam[pos], s.lo[pos].clip(lower=1e-12), s.hi[pos], color=c, alpha=0.10, lw=0)
            ax.plot(s.lam[pos], s["mean"][pos], ls, marker="o", color=c, lw=2, ms=4, mec="white", label=lab)
        s = summ[(summ.g == g) & (summ.lam == 0.1)]
        if len(s) and s["mean"].max() > 0:
            ref = s["mean"].max()
            lam = np.array([0.1, 1.0])
            ax.plot(lam, ref * (lam / 0.1) ** 2, ":", color=MUTED, lw=1.5, label="slope 2 reference")
        if any_pos:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(color=GRID, lw=0.8, which="both")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("λ", color=INK)
        ax.set_title(f"g = {g}", loc="left", fontsize=10, color=INK)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("excess squared bias E(λ)", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(a.res, "exp_D_rate.png"), dpi=200, facecolor="white")
print("done", a.res)
