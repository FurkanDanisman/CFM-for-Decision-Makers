"""In-prior PEHE: every pair of 1D, matched label and 2D on the same R = 1000 datasets, for all three families.
PEHE = RMSE(mu1 - mu0, delta_shared) / sigma_Y per dataset, as in inprior_pehe.py. Per pair: mean PEHE (SE) of each
model, mean difference with its 95% CI and the paired t-test, and the Wilcoxon signed-rank test on the same pairs;
Holm over the 3 pairs x 2 tests of each family, alpha 0.05.
python inprior_pairs.py DATA RES CROSS OUT_TXT   (same DATA / RES / CROSS as inprior_pehe.py)"""
import glob, os, re, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SHARD

DATA, RES, CROSS, OUT = sys.argv[1:5]
NW = 1000
CD = lambda prior: os.path.join(CROSS, f"on_{prior}")
FAMILIES = [  # (family, data prior, [(label, dir, model)] in the order 1D, matched label, 2D)
    ("Do-PFN", "dopfn", [("1D", RES, "dopfn_native"), ("matched label", CD("dopfn"), "dopfn1d_ml"),
                         ("2D", RES, "dopfn_joint2d_262k")]),
    ("UWYK", "uwyk_2d", [("1D", CD("uwyk_2d"), "uwyk1d"), ("matched label", CD("uwyk_2d"), "uwyk1d_ml"),
                         ("2D", RES, "graph2d")]),
    ("CausalPFN-C", "cpfn_2d", [("1D", CD("cpfn_2d"), "cpfn1d"), ("matched label", CD("cpfn_2d"), "cpfn1d_ml"),
                                ("2D", RES, "cpfn2d")])]
PAIRS = [(0, 1), (0, 2), (1, 2)]
lines = []
say = lambda s="": (print(s), lines.append(s))


def load(d, model):
    out = {}
    for f in glob.glob(os.path.join(d, model, "outall_shard*.npz")):
        k = int(re.search(r"shard(\d+)\.npz$", f).group(1)); z = np.load(f)
        for j, i in enumerate(z["idx"]):
            r = k * 100 + int(i)
            if r < NW:
                out[r] = z["mu1"][j].astype(np.float64) - z["mu0"][j].astype(np.float64)
    return out


def truth(prior, r):
    ds = np.load(os.path.join(DATA, prior, "truths", f"r{r:03d}.npz"))["delta_shared"].astype(np.float64)
    sy = float(np.std(np.load(os.path.join(DATA, prior, f"shard{r // SHARD}", "complexmech", "5node", "path_TY",
                                           "hide_0.0", f"r{r:03d}.npz"))["Y_train"]))
    return ds, sy


def holm(ps):
    order = np.argsort(ps); rej = np.zeros(len(ps), bool)
    for k, i in enumerate(order):
        if ps[i] <= 0.05 / (len(ps) - k): rej[i] = True
        else: break
    return rej


say("In-prior PEHE / sigma_Y, every pair of 1D, matched label and 2D on the same datasets")
say("diff = mean over datasets of (B - A) with 95% CI; paired t-test and Wilcoxon; Holm over the 6 tests per family")
for fam, prior, models in FAMILIES:
    say(); say(f"== {fam}  (data: {prior}; " + ", ".join(f"{l} = {m}" for l, _, m in models) + ")")
    P = [load(d, m) for _, d, m in models]
    if not all(P):
        say("   missing predictions: " + ", ".join(l for (l, _, _), p in zip(models, P) if not p)); continue
    common = sorted(set(P[0]) & set(P[1]) & set(P[2]))
    T = {r: truth(prior, r) for r in common}
    E = [np.array([np.sqrt(np.mean((p[r] - T[r][0]) ** 2)) / T[r][1] for r in common]) for p in P]
    say(f"   {len(common)} datasets; mean PEHE (SE) [median]: " + "; ".join(
        f"{l} {e.mean():.4f} ({e.std(ddof=1) / np.sqrt(len(e)):.4f}) [{np.median(e):.4f}]" for (l, _, _), e in zip(models, E)))
    R = []
    for a, b in PAIRS:
        d = E[b] - E[a]; m = d.mean(); h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        R.append((models[a][0], models[b][0], m, m - h, m + h, stats.ttest_rel(E[b], E[a]).pvalue,
                  stats.wilcoxon(E[b], E[a]).pvalue, float(np.mean(d < 0)), float(np.median(d))))
    rej = holm(np.array([p for r in R for p in (r[5], r[6])])).reshape(-1, 2)
    say(f"   {'A':14s} {'B':14s} {'diff B - A':>11s} {'95% CI':>21s} {'t-test p':>9s} {'Wilcoxon p':>11s} "
        f"{'B better in':>11s} {'median diff':>11s}  result")
    for (a, b, m, lo, hi, pt, pw, share, med), (rt, rw) in zip(R, rej):
        def verdict(rj, sgn): return (f"{b} better" if sgn < 0 else f"{b} worse") if rj else "n.s."
        say(f"   {a:14s} {b:14s} {m:11.4f} {f'[{lo:.4f}, {hi:.4f}]':>21s} {pt:9.2g} {pw:11.2g} {share:11.2f} {med:11.4f}"
            f"  t: {verdict(rt, m)}; Wilcoxon: {verdict(rw, med)}")
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote", OUT)
