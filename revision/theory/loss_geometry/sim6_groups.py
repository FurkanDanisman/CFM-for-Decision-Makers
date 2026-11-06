"""Toy where the correlation rho = corr(Y0, Y1 | x) varies from unit to unit, independently of x. Same functions as sim5_track.py (x ~ U[-2, 2],
m0 = sin(2x) + 0.5x, tau = 0.5 sin(3x)), same network (width 6, outputs m0, m1, two sds and a correlation, all learned),
Adam, 4000 steps, batch 256, 10 seeds; a 1D (separate loss) and a 2D (joint loss) network per setting:
  each unit's pair (Y0, Y1) gets its own correlation rho = corr(Y0, Y1 | x), drawn from U[lo, hi] independently of x,
  for the ranges low [0.1, 0.4), medium [0.4, 0.7), high [0.7, 0.95] and full [0.1, 0.95]
Tracked every 100 steps: training loss on fixed test units; the loss of a perfect prediction (true m0, m1, sd 1, the
unit's true correlation) on the same units; after training: PEHE against tau(x) on a grid of x.
   .venv/bin/python revision/theory/loss_geometry/sim6_groups.py OUT_DIR
"""
import importlib.util, json, os, sys
import numpy as np
import torch
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s5", os.path.join(HERE, "sim5_track.py"))
S5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S5)
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
STEPS, BATCH, EVERY, SEEDS, WIDTH = 4000, 256, 100, range(10), 6
RANGES = {"low": (0.1, 0.4), "medium": (0.4, 0.7), "high": (0.7, 0.95), "full": (0.1, 0.95)}
SETTINGS = list(RANGES)
m0, tau, XG = S5.m0, S5.tau, S5.XG


def corr(setting, n, g):
    lo, hi = RANGES[setting]
    return lo + (hi - lo) * torch.rand(n, generator=g)


def draw(x, setting, g):
    r = corr(setting, len(x), g)
    z0, z1 = torch.randn(len(x), generator=g), torch.randn(len(x), generator=g)
    return torch.stack([m0(x)[:, 0] + z0, (m0(x) + tau(x))[:, 0] + r * z0 + torch.sqrt(1 - r ** 2) * z1], 1), r


def test_units(setting):
    g = torch.Generator().manual_seed(12345)
    x = torch.rand(30000, 1, generator=g) * 4 - 2
    y, r = draw(x, setting, g)
    return x, y, r


def run(args):
    setting, seed, loss = args
    torch.set_num_threads(1)
    XT, YT, _ = test_units(setting)
    g = torch.Generator().manual_seed(1000 + seed); torch.manual_seed(seed)
    f = S5.net(WIDTH); opt = torch.optim.Adam(f.parameters(), lr=1e-3)
    track = []
    for step in range(1, STEPS + 1):
        x = torch.rand(BATCH, 1, generator=g) * 4 - 2
        l = S5.nll(f(x), draw(x, setting, g)[0], loss)
        opt.zero_grad(); l.backward(); opt.step()
        if step % EVERY == 0:
            with torch.no_grad():
                track.append(dict(step=step, loss=S5.nll_units(f(XT), YT, loss).mean().item()))
    with torch.no_grad():
        a, b, _, _, rh = S5.heads(f(XG))
        pehe = ((b - a) - tau(XG)[:, 0]).pow(2).mean().sqrt().item()
    return args, dict(track=track, pehe=pehe, rho_hat=rh.mean().item())


def perfect(setting, loss):
    XT, YT, RT = test_units(setting); n = len(XT)
    o = torch.stack([m0(XT)[:, 0], (m0(XT) + tau(XT))[:, 0], torch.zeros(n), torch.zeros(n), torch.atanh(RT / 0.999)], 1)
    return S5.nll_units(o, YT, loss).mean().item()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [(st, s, l) for st in SETTINGS for s in SEEDS for l in ("separate", "joint")]
    with Pool(8) as pool:
        res = dict(pool.map(run, jobs))
    out = {"ranges": RANGES, "perfect": {f"{st}|{l}": perfect(st, l) for st in SETTINGS for l in ("separate", "joint")},
           "runs": {"|".join(map(str, k)): v for k, v in res.items()}}
    json.dump(out, open(os.path.join(OUT, "sim6_results.json"), "w"))
    for st in SETTINGS:
        row = []
        for l in ("separate", "joint"):
            P = np.array([res[(st, s, l)]["pehe"] for s in SEEDS])
            row.append(f"{l} PEHE {P.mean():.4f} ({P.std(ddof=1) / np.sqrt(len(P)):.4f})")
        rh = np.mean([res[(st, s, 'joint')]["rho_hat"] for s in SEEDS])
        print(f"{st:6s}: " + "   ".join(row) + f"   2D learned rho {rh:.2f}")
