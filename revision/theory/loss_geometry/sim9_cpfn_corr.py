"""CausalPFN toy with the prior's measured numbers (cpfn_u_check: u ~2.2x eta, corr(u0, u1) ~0.3-0.4, realised
corr(Y0, Y1 | x) ~0.5). Same functions and network as sim5_track.py (x ~ U[-2, 2], width 6, Adam, 4000 steps, batch 256).
  u   : sd 2.2, corr(u0, u1) = 0.4                         (kept by both targets)
  eta : shared by the arms (eta1 = eta0), sd drawn per unit from the network's range, independently of x
  1D       : target m + u,        separate loss
  2D-full  : target m + u + eta,  joint loss, correlation learned freely
  2D-weak  : target m + u + eta,  joint loss, learned correlation capped at |rho| <= 0.1 (the weak level CausalPFN-C 2D
             reaches: predicted ~0.08 against realised ~0.5)
PEHE against tau(x) on a grid of x; 30 seeds; eta ranges sd in [0.25, 0.75), [0.75, 1.25), [1.25, 1.75].
   .venv/bin/python revision/theory/loss_geometry/sim9_cpfn_corr.py OUT_DIR
"""
import importlib.util, json, os, sys
import numpy as np
import torch
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s5", os.path.join(HERE, "sim5_track.py"))
S5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S5)
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
STEPS, BATCH, SEEDS, WIDTH = 4000, 256, range(30), 6
SD_U, RHO_U, CAP = 2.2, 0.4, 0.1
RANGES = {"low": (0.25, 0.75), "medium": (0.75, 1.25), "high": (1.25, 1.75)}
MODELS = ("1D", "2D-full", "2D-weak")
m0, tau, XG = S5.m0, S5.tau, S5.XG


def draw(x, rng, model, g):
    n = len(x); lo, hi = RANGES[rng]
    z0, z1 = torch.randn(n, generator=g), torch.randn(n, generator=g)
    u0, u1 = SD_U * z0, SD_U * (RHO_U * z0 + (1 - RHO_U ** 2) ** 0.5 * z1)
    y = torch.stack([m0(x)[:, 0] + u0, (m0(x) + tau(x))[:, 0] + u1], 1)
    if model != "1D":
        eta = (lo + (hi - lo) * torch.rand(n, generator=g)) * torch.randn(n, generator=g)
        y = y + eta[:, None]
    return y


def loss(o, y, model):
    if model == "1D":
        return S5.nll(o, y, "separate")
    a, b, s0, s1 = o[:, 0], o[:, 1], o[:, 2].exp(), o[:, 3].exp()
    r = (CAP if model == "2D-weak" else 0.999) * torch.tanh(o[:, 4])
    z0, z1 = (y[:, 0] - a) / s0, (y[:, 1] - b) / s1
    return ((z0 ** 2 - 2 * r * z0 * z1 + z1 ** 2) / (2 * (1 - r ** 2)) + s0.log() + s1.log() + 0.5 * torch.log(1 - r ** 2)).mean()


def run(args):
    rng, seed, model = args
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(1000 + seed); torch.manual_seed(seed)
    f = S5.net(WIDTH); opt = torch.optim.Adam(f.parameters(), lr=1e-3)
    for _ in range(STEPS):
        x = torch.rand(BATCH, 1, generator=g) * 4 - 2
        l = loss(f(x), draw(x, rng, model, g), model)
        opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        o = f(XG); a, b = o[:, 0], o[:, 1]
        r = (CAP if model == "2D-weak" else 0.999) * torch.tanh(o[:, 4])
        return args, dict(pehe=((b - a) - tau(XG)[:, 0]).pow(2).mean().sqrt().item(), rho_hat=r.mean().item())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [(r, s, m) for r in RANGES for s in SEEDS for m in MODELS]
    with Pool(8) as pool:
        res = dict(pool.map(run, jobs))
    json.dump({"|".join(map(str, k)): v for k, v in res.items()}, open(os.path.join(OUT, "sim9_results.json"), "w"))
    from scipy import stats
    L = ["CausalPFN toy with the prior's numbers (u sd 2.2, corr(u) 0.4, eta shared); PEHE vs tau(x), mean (SE) over 30 seeds"]
    for r in RANGES:
        P = {m: np.array([res[(r, s, m)]["pehe"] for s in SEEDS]) for m in MODELS}
        se = lambda v: v.std(ddof=1) / np.sqrt(len(v))
        L.append(f"eta sd {RANGES[r]}: " + "  ".join(f"{m} {P[m].mean():.4f} ({se(P[m]):.4f})" for m in MODELS))
        for m in ("2D-full", "2D-weak"):
            q = P[m] / P["1D"]
            L.append(f"   {m} / 1D {q.mean():.3f}  2D better in {int((P[m] < P['1D']).sum())}/30  t p {stats.ttest_rel(P[m], P['1D']).pvalue:.2g}  "
                     f"learned rho {np.mean([res[(r, s, m)]['rho_hat'] for s in SEEDS]):.2f}")
    open(os.path.join(OUT, "sim9_summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
