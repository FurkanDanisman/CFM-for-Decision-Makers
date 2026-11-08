"""In-prior point estimation: tests on the AVERAGE error, 2D vs each baseline (1D, matched resolution, matched label) on
the same R = 1000 datasets. Per dataset: RMSE over its queries / sigma_Y (as in inprior_pehe.py). Per comparison: mean of
each model, mean difference 2D - baseline with its 95% CI, and the paired t-test on the per-dataset differences; Holm
over the tests of each family, alpha 0.05. Outcomes: Y0, Y1, ITE against the shared- and independent-noise truths; for
CausalPFN-C also m0, m1 (the noiseless E_y0, E_y1 its 1D baselines are trained on; m1 - m0 = ITE shared there).
python inprior_mean_tests.py DATA RES CROSS OUT_TXT   (same DATA / RES / CROSS as inprior_pehe.py)"""
import os, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SHARD

DATA, RES, CROSS, OUT = sys.argv[1:5]
NW = 1000
CD = lambda prior: os.path.join(CROSS, f"on_{prior}")
FAMILIES = [
    ("Do-PFN", "dopfn", [("1D", RES, "dopfn_native"), ("matched resolution", CD("dopfn"), "dopfn1d_mr"),
                         ("matched label", CD("dopfn"), "dopfn1d_ml")], (RES, "dopfn_joint2d_262k")),
    ("UWYK", "uwyk_2d", [("1D", CD("uwyk_2d"), "uwyk1d"), ("matched resolution", CD("uwyk_2d"), "uwyk1d_mr"),
                         ("matched label", CD("uwyk_2d"), "uwyk1d_ml"), ("UWYK-B", CD("uwyk_2d"), "uwykB")], (RES, "graph2d")),
    ("CausalPFN-C", "cpfn_2d", [("1D", CD("cpfn_2d"), "cpfn1d"), ("matched resolution", CD("cpfn_2d"), "cpfn1d_mr"),
                                ("matched label", CD("cpfn_2d"), "cpfn1d_ml")], (RES, "cpfn2d"))]
lines = []
say = lambda s="": (print(s), lines.append(s))


def load(d, model):
    import glob, re
    out = {}
    for f in glob.glob(os.path.join(d, model, "outall_shard*.npz")):
        k = int(re.search(r"shard(\d+)\.npz$", f).group(1)); z = np.load(f)
        for j, i in enumerate(z["idx"]):
            r = k * 100 + int(i)
            if r < NW:
                out[r] = (z["mu0"][j].astype(np.float64), z["mu1"][j].astype(np.float64))
    return out


def truths(prior, r):
    ty = np.load(os.path.join(DATA, prior, "truths_y", f"r{r:03d}.npz"))
    tt = np.load(os.path.join(DATA, prior, "truths", f"r{r:03d}.npz"))
    sy = float(np.std(np.load(os.path.join(DATA, prior, f"shard{r // SHARD}", "complexmech", "5node", "path_TY",
                                           "hide_0.0", f"r{r:03d}.npz"))["Y_train"]))
    t = {"Y0 shared": ("0", ty["y0"]), "Y1 shared": ("1", ty["y1"]), "ITE shared": ("d", tt["delta_shared"]),
         "Y0 indep": ("0", ty["y0_ind"]), "Y1 indep": ("1", ty["y1_ind"]), "ITE indep": ("d", tt["delta_indep"])}
    if "m0" in ty.files:
        t["m0"], t["m1"] = ("0", ty["m0"]), ("1", ty["m1"])
    return {k: (a, v.astype(np.float64)) for k, (a, v) in t.items()}, sy


def score(prior, pred, r):
    (mu0, mu1), (T, sy) = pred, truths(prior, r)
    est = {"0": mu0, "1": mu1, "d": mu1 - mu0}
    return {k: float(np.sqrt(np.mean((est[a] - v) ** 2))) / sy for k, (a, v) in T.items()}


def holm(ps):
    order = np.argsort(ps); rej = np.zeros(len(ps), bool)
    for k, i in enumerate(order):
        if ps[i] <= 0.05 / (len(ps) - k): rej[i] = True
        else: break
    return rej


say("In-prior point estimation: paired t-test on the AVERAGE error / sigma_Y, 2D vs each baseline, same datasets")
say("diff = mean over datasets of (2D - baseline), 95% CI from the paired differences; Holm per family, alpha 0.05")
for fam, prior, bases, (d2, m2) in FAMILIES:
    say(); say(f"== {fam}  (data: {prior}; 2D = {m2}; baselines: " + ", ".join(f"{l} = {m}" for l, _, m in bases) + ")")
    p2 = load(d2, m2)
    have = [(l, load(d, m)) for l, d, m in bases]
    for l, p in have:
        if not p: say(f"   missing predictions for {l}")
    have = [(l, p) for l, p in have if p]
    if not p2 or not have:
        say("   missing 2D or baseline predictions"); continue
    common = sorted(set(p2).intersection(*[set(p) for _, p in have]))
    S2 = [score(prior, p2[r], r) for r in common]
    outs = list(S2[0])
    A2 = {o: np.array([s[o] for s in S2]) for o in outs}
    T = []
    for l, p in have:
        SB = [score(prior, p[r], r) for r in common]
        for o in outs:
            b = np.array([s[o] for s in SB]); dlt = A2[o] - b
            m, h = dlt.mean(), stats.t.ppf(0.975, len(dlt) - 1) * dlt.std(ddof=1) / np.sqrt(len(dlt))
            T.append((l, o, len(dlt), b.mean(), A2[o].mean(), m, m - h, m + h, stats.ttest_rel(A2[o], b).pvalue))
    rej = holm(np.array([t[-1] for t in T]))
    say(f"   {len(common)} datasets; {len(T)} tests (Holm)")
    say(f"   {'baseline':20s} {'outcome':11s} {'baseline':>9s} {'2D':>9s} {'diff 2D-base':>13s} {'95% CI':>21s} {'p':>9s}  result")
    for (l, o, n, mb, m2_, d, lo, hi, p), rj in zip(T, rej):
        res = ("2D better" if d < 0 else "2D worse") if rj else "not significant"
        say(f"   {l:20s} {o:11s} {mb:9.4f} {m2_:9.4f} {d:13.4f} {f'[{lo:.4f}, {hi:.4f}]':>21s} {p:9.2g}  {res}")
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote", OUT)
